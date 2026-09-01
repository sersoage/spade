"""Adversarial offline tests for the prospective v5 outcome replay."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spade.core import proofpack_bridge as bridge
from tools import run_spade_agy_experiment as base
from tools import run_spade_agy_outcome_replay as replay


@dataclass
class _Clause:
    clause_id: str
    status: str = "pass"
    summary: str = "passed"


class _Report:
    def __init__(
        self, code: str, *, seeds: list[int], timeout_seconds: float, max_turns: int
    ) -> None:
        self.schema_version = bridge.QUALIFICATION_SCHEMA
        self.passed = True
        self.environment_name = "OfflineReplayEnv"
        self.environment_digest = "sha256:" + hashlib.sha256(code.encode()).hexdigest()
        self.clauses = {clause_id: _Clause(clause_id) for clause_id in bridge.EXPECTED_CLAUSES}
        self.metadata = {
            "action_format": "boxed",
            "seeds": seeds,
            "max_turns": max_turns,
            "timeout_seconds": timeout_seconds,
            "execution_boundary": bridge.EXPECTED_EXECUTION_BOUNDARY,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "passed": self.passed,
                "environment_name": self.environment_name,
                "environment_digest": self.environment_digest,
                "clauses": {key: vars(value) for key, value in sorted(self.clauses.items())},
                "metadata": self.metadata,
            },
            sort_keys=True,
        )


class _Env:
    def __init__(self, code: str, max_turns: int) -> None:
        match = re.search(r"environment-([0-9]+)", code)
        assert match is not None
        self.number = int(match.group(1))
        self.max_turns = max_turns
        self.turn = 0
        self.seed = 0

    def reset(self, seed: int | None = None):
        self.turn = 0
        self.seed = 0 if seed is None else seed
        return f"puzzle environment={self.number} seed={self.seed}", {"seed": self.seed}

    def solution(self):
        return [f"a{index}" for index in range(1, 6)] if self.number == 8 else "ok"

    def step(self, _action: str):
        self.turn += 1
        required = 5 if self.number == 8 else 1
        terminated = self.turn >= required
        truncated = self.turn >= self.max_turns and (not terminated or self.number in {3, 4})
        return (
            f"state environment={self.number} seed={self.seed} turn={self.turn}",
            1.0 if terminated else 0.0,
            terminated,
            truncated,
            {"turn": self.turn},
        )

    def close(self) -> None:
        pass


class _Target:
    def __init__(self, code: str, max_turns: int) -> None:
        self.code = code
        self.max_turns = max_turns

    def instantiate(self):
        return _Env(self.code, self.max_turns)


class _ArtifactReport:
    release_authorized = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": False,
            "release_authorized": False,
            "decision_reason": "offline replay test",
        }


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"], text=True
    ).strip()


def _runtime(tmp_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    agy_bin = (tmp_path / "agy-test-bin").resolve()
    agy_bin.write_text("offline agy fixture\n", encoding="utf-8")
    source_paths = {
        "spade_live_runner": Path(inspect.getsourcefile(base.live.call_llm) or "").resolve(),
        "proofpack_qualifier": Path(
            inspect.getsourcefile(base.live.qualify_spade_environment) or ""
        ).resolve(),
        "proofpack_receipt_validator": Path(
            inspect.getsourcefile(base.validate_positive_proofpack_receipt) or ""
        ).resolve(),
        "assay_writer": Path(
            inspect.getsourcefile(base.live.write_spade_evaluation) or ""
        ).resolve(),
    }
    revisions = {
        "spade": _git_head(source_paths["spade_live_runner"]),
        "proofpack": _git_head(source_paths["proofpack_qualifier"]),
        "assay": _git_head(source_paths["assay_writer"]),
    }
    python = Path(sys.executable).resolve()
    runtime = {
        "runner_digest": replay._file_digest(Path(base.__file__).resolve()),
        "python_implementation": "CPython",
        "python_version": "offline-test",
        "python_executable": str(python),
        "python_executable_digest": replay._file_digest(python),
        "platform": "offline-test",
        "agy_executable": str(agy_bin),
        "agy_executable_digest": replay._file_digest(agy_bin),
        "agy_version": "agy offline-test",
        "imported_sources": {
            name: {"path": str(path), "digest": replay._file_digest(path)}
            for name, path in source_paths.items()
        },
    }
    return revisions, runtime


def _dependencies(
    revisions: dict[str, str],
    runtime: dict[str, Any],
    *,
    llm_call=None,
    assay_calls: list[dict[str, Any]] | None = None,
    target_factory=None,
    emit_model_lock: bool = False,
) -> base.RunnerDependencies:
    designer = 0

    async def default_call(_client, _model, prompt, **_kwargs):
        nonlocal designer
        if "Environment Designer" in prompt:
            designer += 1
            return f"```python\n# environment-{designer}\nclass PuzzleEnv: pass\n```"
        if "expert tutor" in prompt:
            return "Use a general elimination strategy."
        return r"\boxed{ok}"

    def qualify(code, *, seeds, timeout_seconds, max_turns):
        return _Report(
            code,
            seeds=seeds,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
        )

    calls = assay_calls if assay_calls is not None else []

    def assay_writer(**kwargs):
        calls.append(kwargs)
        report = _ArtifactReport()
        evaluation = {
            "schema_version": "assay-spade-evaluation/v2",
            "curriculum_manifest_digest": "sha256:" + "8" * 64,
            "observations_digest": "sha256:" + "9" * 64,
            "report": report.to_dict(),
        }
        certification = {
            "schema_version": "assay-spade-certification/v2",
            "evaluation_digest": base._digest(evaluation),
            "release_authorized": False,
            "artifact_digest": "sha256:" + "0" * 64,
        }
        bundle_digest = base._digest(certification)
        certification["artifact_digest"] = bundle_digest
        output = Path(kwargs["output_dir"])
        artifact = output / "spade" / bundle_digest.removeprefix("sha256:")
        artifact.mkdir(parents=True, exist_ok=True)
        evaluation_path = artifact / "evaluation.json"
        certification_path = artifact / "certification.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        certification_path.write_text(
            json.dumps(certification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence = output / "evidence" / "records.jsonl"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text('{"schema_version":"assay-evidence/v2"}\n', encoding="utf-8")
        if emit_model_lock:
            (output / "model.lock").write_text("forbidden\n", encoding="utf-8")
        return SimpleNamespace(
            report=report,
            bundle_digest=bundle_digest,
            artifact_dir=artifact,
            evaluation_path=evaluation_path,
            certification_path=certification_path,
            model_lock_path=None,
        )

    return base.RunnerDependencies(
        llm_call=llm_call or default_call,
        qualify=qualify,
        target_factory=target_factory
        or (lambda code, **kwargs: _Target(code, int(kwargs["max_turns"]))),
        assay_writer=assay_writer,
        task_factory=lambda **kwargs: kwargs,
        cluster_factory=lambda **kwargs: kwargs,
        run_metadata_factory=lambda **kwargs: kwargs,
        client_or_bin=object(),
        source_revisions=revisions,
        runtime_identity=runtime,
    )


@dataclass
class _Fixture:
    source_run: Path
    output_root: Path
    plan: dict[str, Any]
    plan_path: Path
    revisions: dict[str, str]
    runtime: dict[str, Any]


def _fixture(tmp_path: Path, *, cap: int | None = None) -> _Fixture:
    revisions, runtime = _runtime(tmp_path)
    dependencies = _dependencies(revisions, runtime)
    source_root = (tmp_path / "source-runs").resolve()
    source_plan = base.build_plan(
        experiment_id="offline-v4-source",
        model=replay.CANONICAL_MODEL,
        skills=base.PILOT_SKILLS,
        difficulties=base.PILOT_DIFFICULTIES,
        qualification_seeds=base.live.QUALIFIED_SEEDS,
        evaluation_seeds=base.live.QUALIFIED_SEEDS,
        total_call_cap=300,
        reserve_count=9,
        max_turns=5,
        design_attempts_per_slot=3,
        hint_attempts=2,
        llm_timeout_seconds=180.0,
        qualification_timeout_seconds=5.0,
        minimum_certification_clusters=18,
        alpha=0.05,
        non_inferiority_margin=0.10,
        source_revisions=revisions,
        runtime_identity=runtime,
        protocol_id="spade-agy-18-cluster-pilot/v1",
        run_output_root=source_root,
    )
    source_plan_path = tmp_path / "source-plan.json"
    base.write_plan(source_plan_path, source_plan)
    source_run = base.derive_run_dir(source_root, source_plan)
    source_engine = base._Engine(
        source_plan, base._pretty_json(source_plan), source_run, dependencies
    )
    with base._single_writer(source_run):
        source_engine.initialize()
        asyncio.run(source_engine.prepare_cohort())

    # Deliberately add actor evidence to prove the v5 allowlist excludes it.
    asyncio.run(
        source_engine.call(
            purpose={
                "phase": "actor",
                "outcome_id": source_plan["outcome_schedule"][0]["outcome_id"],
                "cluster_id": source_plan["outcome_schedule"][0]["cluster_id"],
                "seed": source_plan["outcome_schedule"][0]["seed"],
                "arm": source_plan["outcome_schedule"][0]["arm"],
                "turn": 1,
            },
            prompt="actor evidence excluded from v5",
        )
    )
    actor_outcome = source_run / "outcomes" / "excluded" / "outcome.json"
    actor_outcome.parent.mkdir(parents=True)
    actor_outcome.write_text("{}\n", encoding="utf-8")

    output_root = (tmp_path / "v5-runs").resolve()
    plan = replay.build_outcome_replay_plan(
        experiment_id="offline-v5-replay",
        source_run_dir=source_run,
        run_output_root=output_root,
        expected_source_plan_digest=source_plan["plan_digest"],
        expected_source_cohort_digest=json.loads(
            (source_run / "cohort-lock.json").read_text(encoding="utf-8")
        )["cohort_digest"],
        total_call_cap=cap,
        dependencies=dependencies,
        source_revisions=revisions,
        runtime_identity=runtime,
    )
    plan_path = tmp_path / "v5-plan.json"
    replay.write_plan(plan_path, plan)
    return _Fixture(source_run, output_root, plan, plan_path, revisions, runtime)


def _redigest_plan(plan: dict[str, Any]) -> None:
    plan["plan_digest"] = base._digest(
        {key: value for key, value in plan.items() if key != "plan_digest"}
    )


def test_plan_seals_horizons_budget_and_excludes_v4_actor_evidence(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    plan = fixture.plan
    horizons = {item["cluster_id"]: item["horizon"] for item in plan["cluster_horizons"]}

    assert horizons["c003-mathematical-reasoning-medium"] == 2
    assert horizons["c004-mathematical-reasoning-hard"] == 2
    assert horizons["c008-strategic-planning-hard"] == 5
    assert set(horizons.values()) == {1, 2, 5}
    assert sum(horizons.values()) == 24
    assert plan["configuration"]["computed_call_ceiling"] == 288
    assert plan["configuration"]["total_call_cap"] == 272
    assert plan["budget_context"] == {
        "prior_charged_calls": 178,
        "authorized_global_call_cap": 450,
        "planned_max_global_calls": 450,
        "headroom_calls": 0,
        "canonical_authorized_run": True,
    }
    assert [item["ordinal"] for item in plan["outcome_schedule"]] == list(range(1, 109))
    assert len(plan["pair_schedule"]) == 54
    viability = {item["cluster_id"]: item for item in plan["horizon_viability"]}
    for cluster_id in (
        "c003-mathematical-reasoning-medium",
        "c004-mathematical-reasoning-hard",
    ):
        record = viability[cluster_id]
        assert record["searched_horizons"] == [1, 2]
        assert record["searches"][0]["all_seeds_viable"] is False
        assert record["searches"][1]["all_seeds_viable"] is True
        for seed_receipt in record["searches"][0]["seeds"].values():
            assert seed_receipt["replay_count"] == 2
            receipt = seed_receipt["deterministic_receipt"]
            assert receipt["status"] == "not-viable"
            assert receipt["trace"] is not None
            final = receipt["trace"]["steps"][-1]
            assert final["terminated"] is True
            assert final["truncated"] is True
    assert viability["c008-strategic-planning-hard"]["searched_horizons"] == [5]
    imported = [item["path"] for item in plan["source_evidence"]["import_manifest"]]
    assert not any(path.startswith("outcomes/") for path in imported)
    for path in imported:
        if path.startswith("calls/") and path.endswith("request.json"):
            request = json.loads((fixture.source_run / path).read_text(encoding="utf-8"))
            assert request["purpose"]["phase"] in {"designer", "hint"}


def test_partial_import_is_completed_idempotently_without_provider_calls(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    provider_calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("source import attempted a provider call")

    run_dir = replay.derive_run_dir(fixture.output_root, fixture.plan)
    import_root = run_dir / replay.IMPORT_ROOT
    partial = fixture.plan["source_evidence"]["import_manifest"][:17]
    for entry in partial:
        source = fixture.source_run / entry["path"]
        target = import_root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    dependencies = _dependencies(fixture.revisions, fixture.runtime, llm_call=forbidden)
    engine = replay._ReplayEngine(
        fixture.plan, base._pretty_json(fixture.plan), run_dir, dependencies
    )
    with base._single_writer(run_dir):
        engine.initialize()
        engine.initialize()

    imported = replay._safe_tree(import_root)
    assert len(imported) == len(fixture.plan["source_evidence"]["import_manifest"])
    engine._validate_import_manifest_bytes(import_root)
    assert provider_calls == 0


def test_rejected_horizon_trace_must_be_deterministic(tmp_path) -> None:
    fixture = _fixture(tmp_path)

    class _NondeterministicRejectedEnv(_Env):
        nonce = 0

        def step(self, action: str):
            result = list(super().step(action))
            if self.number == 3 and self.max_turns == 1:
                type(self).nonce += 1
                result[4] = {"turn": self.turn, "nonce": type(self).nonce}
            return tuple(result)

    def target_factory(code, **kwargs):
        return SimpleNamespace(
            instantiate=lambda: _NondeterministicRejectedEnv(code, int(kwargs["max_turns"]))
        )

    dependencies = _dependencies(fixture.revisions, fixture.runtime, target_factory=target_factory)
    snapshot = replay._validate_source_snapshot(fixture.source_run, dependencies=dependencies)
    bounds = replay._derive_cluster_oracle_bounds(snapshot.plan, snapshot.selections)
    with pytest.raises(base.ExperimentError, match="search is not deterministic"):
        replay._search_horizon_viability(snapshot, bounds, dependencies)


def test_whole_pair_retry_discards_sibling_and_resume_makes_no_duplicate_calls(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path)
    workdirs: list[Path] = []
    failed = False
    assay_calls: list[dict[str, Any]] = []

    async def actor(_client, _model, prompt, **kwargs):
        nonlocal failed
        workdir = Path(kwargs["workdir"])
        assert workdir.is_dir()
        workdirs.append(workdir)
        if "Privileged strategy hint" in prompt and not failed:
            failed = True
            raise base.live.LiveEvalError("agy returned an empty response", 4)
        return r"\boxed{ok}"

    dependencies = _dependencies(
        fixture.revisions, fixture.runtime, llm_call=actor, assay_calls=assay_calls
    )
    result = asyncio.run(
        replay.run_experiment(
            fixture.plan_path,
            fixture.output_root,
            execute=True,
            acknowledged_call_cap=272,
            dependencies=dependencies,
        )
    )
    assert result.status == "complete"
    assert result.call_count == 134
    assert len(workdirs) == len({str(path) for path in workdirs})
    assert all(not path.exists() for path in workdirs)
    first_pair = fixture.plan["pair_schedule"][0]
    resolution = json.loads(
        (result.run_dir / "pair-resolutions" / f"{first_pair['pair_id']}.json").read_text()
    )
    assert resolution["selected_pair_attempt"] == 2
    assert len(resolution["excluded_attempts"]) == 1
    assert (
        result.run_dir
        / "pair-attempts"
        / first_pair["pair_id"]
        / "attempt-01"
        / first_pair["arm_order"][0]
        / "outcome.json"
    ).is_file()
    assert len(assay_calls) == 1

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("completed provider call was duplicated")

    resumed = asyncio.run(
        replay.run_experiment(
            fixture.plan_path,
            fixture.output_root,
            execute=True,
            acknowledged_call_cap=272,
            dependencies=_dependencies(
                fixture.revisions,
                fixture.runtime,
                llm_call=forbidden,
                assay_calls=[],
            ),
        )
    )
    assert resumed.call_count == result.call_count


def test_generic_nonzero_exit_is_fatal_without_pair_retry_or_assay(tmp_path) -> None:
    fixture = _fixture(tmp_path, cap=10)
    calls = 0
    assay_calls: list[dict[str, Any]] = []

    async def fatal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise base.live.LiveEvalError("agy failed with exit 2: permission denied", 4)

    with pytest.raises(base.ExperimentIncomplete, match="exhausted after fatal_transport"):
        asyncio.run(
            replay.run_experiment(
                fixture.plan_path,
                fixture.output_root,
                execute=True,
                acknowledged_call_cap=10,
                dependencies=_dependencies(
                    fixture.revisions,
                    fixture.runtime,
                    llm_call=fatal,
                    assay_calls=assay_calls,
                ),
            )
        )
    run_dir = replay.derive_run_dir(fixture.output_root, fixture.plan)
    assert calls == 1
    assert not (
        run_dir / "pair-attempts" / fixture.plan["pair_schedule"][0]["pair_id"] / "attempt-02"
    ).exists()
    assert not (run_dir / "assay-request.json").exists()
    assert assay_calls == []


def test_redigested_schedule_cap_and_import_tampering_fail_validation(tmp_path) -> None:
    fixture = _fixture(tmp_path)

    bad_cap = json.loads(json.dumps(fixture.plan))
    bad_cap["configuration"]["computed_call_ceiling"] = 287
    _redigest_plan(bad_cap)
    with pytest.raises(base.ExperimentError, match="computed call ceiling"):
        replay.validate_plan(bad_cap)

    bad_schedule = json.loads(json.dumps(fixture.plan))
    bad_schedule["outcome_schedule"][0]["arm"] = "hinted"
    bad_schedule["outcome_schedule"][1]["arm"] = "unhinted"
    _redigest_plan(bad_schedule)
    with pytest.raises(base.ExperimentError, match="not deterministic"):
        replay.validate_plan(bad_schedule)

    bad_import = json.loads(json.dumps(fixture.plan))
    bad_import["source_evidence"]["import_manifest"].append(
        {"path": "outcomes/x.json", "digest": "sha256:" + "0" * 64, "size_bytes": 3}
    )
    bad_import["source_evidence"]["import_manifest"].sort(key=lambda item: item["path"])
    bad_import["source_evidence"]["import_manifest_digest"] = base._digest(
        bad_import["source_evidence"]["import_manifest"]
    )
    _redigest_plan(bad_import)
    with pytest.raises(base.ExperimentError, match="actor/outcome/Assay"):
        replay.validate_plan(bad_import)

    bad_rejected_trace = json.loads(json.dumps(fixture.plan))
    viability_record = bad_rejected_trace["horizon_viability"][2]
    seed_record = next(iter(viability_record["searches"][0]["seeds"].values()))
    receipt = seed_record["deterministic_receipt"]
    assert receipt["status"] == "not-viable"
    receipt["trace"]["solution"] = "tampered rejected-horizon solution"
    receipt["trace"]["oracle_actions"][0] = r"\boxed{tampered rejected-horizon solution}"
    receipt["trace"]["steps"][0]["action"] = r"\boxed{tampered rejected-horizon solution}"
    receipt["trace_digest"] = base._digest(receipt["trace"])
    seed_record["receipt_digest"] = base._digest(receipt)
    viability_body = {
        key: value for key, value in viability_record.items() if key != "record_digest"
    }
    viability_record["record_digest"] = base._digest(viability_body)
    bad_rejected_trace["horizon_viability_digest"] = base._digest(
        bad_rejected_trace["horizon_viability"]
    )
    _redigest_plan(bad_rejected_trace)
    with pytest.raises(base.ExperimentError, match="locked probe digests"):
        replay.validate_plan(bad_rejected_trace)

    bad_rejected_action = json.loads(json.dumps(fixture.plan))
    viability_record = bad_rejected_action["horizon_viability"][2]
    seed_record = next(iter(viability_record["searches"][0]["seeds"].values()))
    receipt = seed_record["deterministic_receipt"]
    receipt["trace"]["oracle_actions"][0] = r"\boxed{evil}"
    receipt["trace"]["steps"][0]["action"] = r"\boxed{evil}"
    receipt["trace_digest"] = base._digest(receipt["trace"])
    seed_record["receipt_digest"] = base._digest(receipt)
    viability_body = {
        key: value for key, value in viability_record.items() if key != "record_digest"
    }
    viability_record["record_digest"] = base._digest(viability_body)
    bad_rejected_action["horizon_viability_digest"] = base._digest(
        bad_rejected_action["horizon_viability"]
    )
    _redigest_plan(bad_rejected_action)
    with pytest.raises(base.ExperimentError, match="locked solution"):
        replay.validate_plan(bad_rejected_action)

    bad_rejected_chain = json.loads(json.dumps(fixture.plan))
    viability_record = bad_rejected_chain["horizon_viability"][2]
    seed_record = next(iter(viability_record["searches"][0]["seeds"].values()))
    receipt = seed_record["deterministic_receipt"]
    receipt["trace"]["steps"][0]["pre_observation"] = "disconnected"
    receipt["trace_digest"] = base._digest(receipt["trace"])
    seed_record["receipt_digest"] = base._digest(receipt)
    viability_body = {
        key: value for key, value in viability_record.items() if key != "record_digest"
    }
    viability_record["record_digest"] = base._digest(viability_body)
    bad_rejected_chain["horizon_viability_digest"] = base._digest(
        bad_rejected_chain["horizon_viability"]
    )
    _redigest_plan(bad_rejected_chain)
    with pytest.raises(base.ExperimentError, match="observation chain"):
        replay.validate_plan(bad_rejected_chain)


def test_exact_retry_classifier_is_narrow() -> None:
    assert (
        replay._ReplayEngine._failure_category(
            base.live.LiveEvalError("agy returned an empty response", 4)
        )
        == "empty_response"
    )
    assert (
        replay._ReplayEngine._failure_category(
            base.live.LiveEvalError("agy call timed out after 180s", 4)
        )
        == "provider_timeout"
    )
    assert (
        replay._ReplayEngine._failure_category(
            base.live.LiveEvalError(
                "agy failed with exit 1: Error: timeout waiting for response", 4
            )
        )
        == "provider_timeout"
    )
    assert (
        replay._ReplayEngine._failure_category(
            base.live.LiveEvalError(
                "agy failed with exit 1: Error: timeout waiting for response after 180s", 4
            )
        )
        == "fatal_transport"
    )
    assert replay._ReplayEngine._failure_category(OSError("could not start")) == "fatal_transport"


def test_candidate_canary_and_orphan_preoutcome_call_are_rejected(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    canary = next((fixture.source_run / "candidates").glob("*/*")) / "actor-outcome.json"
    canary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(base.ExperimentError, match="unexpected source candidate artifact"):
        replay.build_outcome_replay_plan(
            experiment_id="rejected-canary",
            source_run_dir=fixture.source_run,
            run_output_root=(tmp_path / "canary-runs").resolve(),
            expected_source_plan_digest=fixture.plan["source_evidence"]["source_plan_digest"],
            expected_source_cohort_digest=fixture.plan["source_evidence"]["source_cohort_digest"],
            dependencies=_dependencies(fixture.revisions, fixture.runtime),
            source_revisions=fixture.revisions,
            runtime_identity=fixture.runtime,
        )


def test_fatal_environment_is_terminal_and_resume_spends_zero_calls(tmp_path) -> None:
    fixture = _fixture(tmp_path, cap=10)
    calls = 0

    async def actor(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return r"\boxed{ok}"

    class _FailingEnv(_Env):
        def step(self, _action: str):
            raise RuntimeError("deterministic local failure")

    def actor_failing_factory():
        instantiations = 0

        def factory(code, **kwargs):
            nonlocal instantiations
            instantiations += 1
            environment_type = _FailingEnv if instantiations > 120 else _Env
            return SimpleNamespace(
                instantiate=lambda: environment_type(code, int(kwargs["max_turns"]))
            )

        return factory

    dependencies = _dependencies(
        fixture.revisions,
        fixture.runtime,
        llm_call=actor,
        target_factory=actor_failing_factory(),
    )
    with pytest.raises(replay._ArmExecutionFailure, match="deterministic local failure"):
        asyncio.run(
            replay.run_experiment(
                fixture.plan_path,
                fixture.output_root,
                execute=True,
                acknowledged_call_cap=10,
                dependencies=dependencies,
            )
        )
    run_dir = replay.derive_run_dir(fixture.output_root, fixture.plan)
    summary = json.loads(
        next((run_dir / "pair-attempts").glob("*/attempt-01/attempt.json")).read_text()
    )
    assert summary["status"] == "fatal_failure"
    assert summary["failure_category"] == "fatal_environment"
    assert summary["failed_call_id"] is not None
    assert calls == 1

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("terminal local failure retried the provider")

    with pytest.raises(base.ExperimentIncomplete, match="terminal fatal_failure"):
        asyncio.run(
            replay.run_experiment(
                fixture.plan_path,
                fixture.output_root,
                execute=True,
                acknowledged_call_cap=10,
                dependencies=_dependencies(
                    fixture.revisions,
                    fixture.runtime,
                    llm_call=forbidden,
                    target_factory=actor_failing_factory(),
                ),
            )
        )
    assert calls == 1
    assert not (run_dir / "assay-request.json").exists()


def test_unknown_run_leaf_is_rejected_before_resume_call(tmp_path) -> None:
    fixture = _fixture(tmp_path, cap=1)
    actor_calls = 0

    async def actor(*_args, **_kwargs):
        nonlocal actor_calls
        actor_calls += 1
        return r"\boxed{ok}"

    with pytest.raises(base.CallCapExceeded):
        asyncio.run(
            replay.run_experiment(
                fixture.plan_path,
                fixture.output_root,
                execute=True,
                acknowledged_call_cap=1,
                dependencies=_dependencies(fixture.revisions, fixture.runtime, llm_call=actor),
            )
        )
    run_dir = replay.derive_run_dir(fixture.output_root, fixture.plan)
    (run_dir / "unledgered.bin").write_bytes(b"canary")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("provider called with unknown run artifact")

    with pytest.raises(base.ExperimentError, match="unexpected top-level replay artifact"):
        asyncio.run(
            replay.run_experiment(
                fixture.plan_path,
                fixture.output_root,
                execute=True,
                acknowledged_call_cap=1,
                dependencies=_dependencies(fixture.revisions, fixture.runtime, llm_call=forbidden),
            )
        )
    assert actor_calls == 1


def test_physical_model_lock_forces_failed_assay_result(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    assay_calls: list[dict[str, Any]] = []
    with pytest.raises(base.ExperimentIncomplete, match="forbidden model.lock"):
        asyncio.run(
            replay.run_experiment(
                fixture.plan_path,
                fixture.output_root,
                execute=True,
                acknowledged_call_cap=272,
                dependencies=_dependencies(
                    fixture.revisions,
                    fixture.runtime,
                    assay_calls=assay_calls,
                    emit_model_lock=True,
                ),
            )
        )
    run_dir = replay.derive_run_dir(fixture.output_root, fixture.plan)
    result = json.loads((run_dir / "assay-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert (run_dir / "assay" / "model.lock").is_file()
    assert len(assay_calls) == 1
