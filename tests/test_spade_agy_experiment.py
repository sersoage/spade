"""Adversarial offline tests for the sealed multi-environment agy runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spade.core import proofpack_bridge as bridge
from tools import run_spade_agy_experiment as agy


REVISIONS = {"spade": "a" * 40, "proofpack": "b" * 40, "assay": "c" * 40}
RUNTIME = {
    "runner_digest": "sha256:" + "d" * 64,
    "python_implementation": "CPython",
    "python_version": "3.12.0",
    "python_executable": "/test/python",
    "python_executable_digest": "sha256:" + "e" * 64,
    "platform": "test-platform",
    "agy_executable": "/test/agy",
    "agy_executable_digest": "sha256:" + "f" * 64,
    "agy_version": "agy test",
    "imported_sources": {
        name: {"path": f"/test/{name}.py", "digest": "sha256:" + char * 64}
        for name, char in zip(
            (
                "spade_live_runner",
                "proofpack_qualifier",
                "proofpack_receipt_validator",
                "assay_writer",
            ),
            "1234",
        )
    },
}


@dataclass
class _Clause:
    clause_id: str
    status: str = "pass"
    summary: str = "passed"


class _Report:
    def __init__(
        self,
        code: str,
        *,
        seeds: list[int],
        timeout_seconds: float,
        max_turns: int,
        valid: bool = True,
    ) -> None:
        digest_code = code if valid else code + "-mismatch"
        self.schema_version = bridge.QUALIFICATION_SCHEMA
        self.passed = True
        self.environment_name = "OfflinePuzzleEnv"
        self.environment_digest = "sha256:" + hashlib.sha256(
            digest_code.encode("utf-8")
        ).hexdigest()
        self.clauses = {
            clause_id: _Clause(clause_id) for clause_id in bridge.EXPECTED_CLAUSES
        }
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
                "clauses": {
                    key: vars(value) for key, value in sorted(self.clauses.items())
                },
                "metadata": self.metadata,
            },
            sort_keys=True,
        )


class _Env:
    def __init__(self, code: str, *, fail_probe: bool = False) -> None:
        self.code = code
        self.fail_probe = fail_probe

    def reset(self, seed: int | None = None):
        if self.fail_probe:
            raise RuntimeError("probe failed")
        return f"puzzle for {self.code} seed={seed}", {}

    def solution(self):
        return "42"

    def step(self, action: str):
        assert action.startswith(r"\boxed{")
        return "complete", 1.0, True, False, {}

    def close(self) -> None:
        pass


class _Target:
    def __init__(self, code: str, *, fail_probe: bool = False) -> None:
        self.code = code
        self.fail_probe = fail_probe

    def instantiate(self):
        return _Env(self.code, fail_probe=self.fail_probe)


class _ArtifactReport:
    release_authorized = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": False,
            "release_authorized": False,
            "decision_reason": "test",
        }


def _plan(
    *,
    cluster_count: int = 4,
    cap: int = 100,
    reserve_count: int = 0,
    design_attempts: int = 1,
    hint_attempts: int = 1,
    run_output_root: Path = Path("/private/tmp/spade-agy-test-runs"),
) -> dict[str, Any]:
    return agy.build_plan(
        experiment_id="offline-pilot",
        model="agy-explicit-test-model",
        skills=[f"skill-{index}" for index in range(cluster_count)],
        difficulties=["medium"],
        qualification_seeds=[0, 1, 42],
        evaluation_seeds=[0],
        total_call_cap=cap,
        reserve_count=reserve_count,
        max_turns=1,
        design_attempts_per_slot=design_attempts,
        hint_attempts=hint_attempts,
        source_revisions=REVISIONS,
        runtime_identity=RUNTIME,
        run_output_root=run_output_root,
    )


def _write_plan(tmp_path: Path, plan: dict[str, Any]) -> Path:
    return agy.write_plan(tmp_path / "plan.json", plan)


def _dependencies(
    run_dir: Path,
    *,
    llm_call=None,
    qualify=None,
    target_factory=None,
    assay_calls: list[dict[str, Any]] | None = None,
) -> agy.RunnerDependencies:
    counters = {"designer": 0}

    async def default_call(_client, _model, prompt, **_kwargs):
        if "Environment Designer" in prompt:
            counters["designer"] += 1
            return f"```python\n# environment-{counters['designer']}\nclass PuzzleEnv: pass\n```"
        if "expert tutor" in prompt:
            return "Use a general elimination strategy."
        assert (run_dir / "cohort-lock.json").is_file()
        return r"\boxed{ok}"

    def default_qualify(code, *, seeds, timeout_seconds, max_turns):
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
            "evaluation_digest": agy._digest(evaluation),
            "release_authorized": False,
            "artifact_digest": "sha256:" + "0" * 64,
        }
        bundle_digest = agy._digest(certification)
        certification["artifact_digest"] = bundle_digest
        output_dir = Path(kwargs["output_dir"])
        artifact_dir = output_dir / "spade" / bundle_digest.removeprefix("sha256:")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = artifact_dir / "evaluation.json"
        certification_path = artifact_dir / "certification.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        certification_path.write_text(
            json.dumps(certification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence_path = output_dir / "evidence" / "records.jsonl"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text('{"schema_version":"assay-evidence/v2"}\n', encoding="utf-8")
        return SimpleNamespace(
            report=report,
            bundle_digest=bundle_digest,
            artifact_dir=artifact_dir,
            evaluation_path=evaluation_path,
            certification_path=certification_path,
            model_lock_path=None,
        )

    return agy.RunnerDependencies(
        llm_call=llm_call or default_call,
        qualify=qualify or default_qualify,
        target_factory=target_factory or (lambda code, **_kwargs: _Target(code)),
        assay_writer=assay_writer,
        task_factory=lambda **kwargs: kwargs,
        cluster_factory=lambda **kwargs: kwargs,
        run_metadata_factory=lambda **kwargs: kwargs,
        client_or_bin=object(),
        source_revisions=REVISIONS,
        runtime_identity=RUNTIME,
    )


def test_pilot_template_is_deterministic_explicit_and_counterbalanced(monkeypatch) -> None:
    monkeypatch.setattr(agy, "_source_revisions", lambda: REVISIONS)
    monkeypatch.setattr(agy, "_runtime_identity", lambda: RUNTIME)
    first = agy.build_pilot_plan(
        experiment_id="agy-pilot",
        model="agy-model-x",
        run_output_root="/private/tmp/spade-agy-pilot",
    )
    second = agy.build_pilot_plan(
        experiment_id="agy-pilot",
        model="agy-model-x",
        run_output_root="/private/tmp/spade-agy-pilot",
    )

    assert first == second
    assert len(first["cluster_schedule"]) == 18
    assert sum(len(item["candidate_slots"]) for item in first["cluster_schedule"]) == 27
    assert len(first["outcome_schedule"]) == 18 * 3 * 2
    assert first["backend_identity_attested"] is False
    assert first["route_authority"] == "requested-route-only"
    first_pair = first["outcome_schedule"][:2]
    second_pair = first["outcome_schedule"][2:4]
    assert [item["arm"] for item in first_pair] == ["unhinted", "hinted"]
    assert [item["arm"] for item in second_pair] == ["hinted", "unhinted"]

    weakened = json.loads(json.dumps(first))
    weakened["configuration"]["max_turns"] = 1
    weakened_body = {key: value for key, value in weakened.items() if key != "plan_digest"}
    weakened["plan_digest"] = agy._digest(weakened_body)
    with pytest.raises(agy.ExperimentError, match="differ from its template"):
        agy.validate_plan(weakened)

    with pytest.raises(agy.ExperimentError, match="explicit agy"):
        agy.build_pilot_plan(
            experiment_id="agy-pilot",
            model="agy-subscription",
            run_output_root="/private/tmp/spade-agy-pilot",
        )


def test_plan_is_strict_canonical_and_evaluation_seeds_must_be_qualified(tmp_path) -> None:
    plan = _plan()
    path = _write_plan(tmp_path, plan)
    assert agy.load_plan(path) == plan

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["model"] = "other-model"
    path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(agy.ExperimentError, match="plan_digest"):
        agy.load_plan(path)

    with pytest.raises(agy.ExperimentError, match="subset"):
        agy.build_plan(
            experiment_id="bad-seeds",
            model="agy-model",
            skills=["logic"],
            difficulties=["hard"],
            qualification_seeds=[0],
            evaluation_seeds=[1],
            total_call_cap=10,
            source_revisions=REVISIONS,
            runtime_identity=RUNTIME,
            run_output_root="/private/tmp/spade-agy-bad-seeds",
        )


def test_cli_dry_run_and_bad_acknowledgement_make_zero_live_calls(
    monkeypatch, tmp_path
) -> None:
    output_root = tmp_path / "runs"
    path = _write_plan(tmp_path, _plan(cap=17, run_output_root=output_root))
    touched = False

    def forbidden(_plan):
        nonlocal touched
        touched = True
        raise AssertionError("live dependencies resolved")

    monkeypatch.setattr(agy, "_default_dependencies", forbidden)
    assert agy.main(["run", "--plan", str(path), "--output-root", str(output_root)]) == 0
    assert (
        agy.main(
            [
                "run",
                "--plan",
                str(path),
                "--output-root",
                str(tmp_path / "runs"),
                "--execute",
            ]
        )
        == 2
    )
    assert (
        agy.main(
            [
                "run",
                "--plan",
                str(path),
                "--output-root",
                str(tmp_path / "runs"),
                "--execute",
                "--acknowledge-call-cap",
                "16",
            ]
        )
        == 2
    )
    assert touched is False
    assert not (tmp_path / "runs").exists()


def test_complete_run_aggregates_once_and_resume_makes_no_calls(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    assay_calls: list[dict[str, Any]] = []
    first = asyncio.run(
        agy.run_experiment(
            plan_path,
            output_root,
            execute=True,
            acknowledged_call_cap=100,
            dependencies=_dependencies(run_dir, assay_calls=assay_calls),
        )
    )

    assert first.status == "complete"
    assert first.run_dir == run_dir
    assert first.call_count == 16
    assert len(assay_calls) == 1
    assay = assay_calls[0]
    assert len(assay["tasks"]) == len(assay["clusters"]) == 4
    assert assay["minimum_clusters"] == 4
    assert all(task["seed"] == 0 for task in assay["tasks"])
    assert all(json.loads(task["solution"]) == {"0": "42"} for task in assay["tasks"])
    ledger_roots = {
        task["metadata"]["experiment_ledger_root_digest"] for task in assay["tasks"]
    }
    assert len(ledger_roots) == 1
    assert all(task["metadata"]["backend_identity_attested"] is False for task in assay["tasks"])
    assert all(task["metadata"]["route_authority"] == "requested-route-only" for task in assay["tasks"])
    assert all("registered_plan_digest" not in task["metadata"] for task in assay["tasks"])
    result = json.loads((run_dir / "assay-result.json").read_text(encoding="utf-8"))
    assert result["release_authorized"] is False
    assert result["model_lock_path"] is None
    assert not list((run_dir / "assay").rglob("model.lock"))

    async def forbidden_call(*_args, **_kwargs):
        raise AssertionError("a completed provider outcome was replayed")

    resume_assay_calls: list[dict[str, Any]] = []
    resumed = asyncio.run(
        agy.run_experiment(
            plan_path,
            output_root,
            execute=True,
            acknowledged_call_cap=100,
            dependencies=_dependencies(
                run_dir,
                llm_call=forbidden_call,
                assay_calls=resume_assay_calls,
            ),
        )
    )
    assert resumed.call_count == first.call_count
    assert resume_assay_calls == []


def test_hard_call_cap_reserves_before_spawn_and_never_signals_assay(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(cap=1, run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    provider_calls = 0
    assay_calls: list[dict[str, Any]] = []

    async def counted_call(_client, _model, prompt, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        assert "Environment Designer" in prompt
        return "```python\n# only-call\nclass PuzzleEnv: pass\n```"

    with pytest.raises(agy.CallCapExceeded, match="call cap 1"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=1,
                dependencies=_dependencies(
                    run_dir,
                    llm_call=counted_call,
                    assay_calls=assay_calls,
                ),
            )
        )
    assert provider_calls == 1
    assert len(list((run_dir / "calls").glob("*/request.json"))) == 1
    assert assay_calls == []
    assert not (run_dir / "assay-request.json").exists()


def test_runtime_drift_fails_before_any_provider_call(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider called after runtime drift")

    dependencies = _dependencies(run_dir, llm_call=forbidden)
    dependencies = agy.RunnerDependencies(
        **{
            **vars(dependencies),
            "runtime_identity": {**RUNTIME, "agy_version": "changed"},
        }
    )
    with pytest.raises(agy.ExperimentIncomplete, match="runtime identity"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=dependencies,
            )
        )
    assert calls == 0


def test_tampered_completed_outcome_fails_closed_without_replay(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    asyncio.run(
        agy.run_experiment(
            plan_path,
            output_root,
            execute=True,
            acknowledged_call_cap=100,
            dependencies=_dependencies(run_dir),
        )
    )
    outcome_path = next((run_dir / "outcomes").glob("*/outcome.json"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["return"] = 0.25
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("tampered completed outcome was replayed")

    with pytest.raises(agy.ExperimentError, match="return does not match|digest mismatch"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(run_dir, llm_call=forbidden),
            )
        )
    assert calls == 0


def test_ambiguous_reserved_call_is_never_replayed(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(run_dir, llm_call=cancelled),
            )
        )
    assert len(list((run_dir / "calls").glob("*/request.json"))) == 1
    assert not list((run_dir / "calls").glob("*/result.json"))
    replay_calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("ambiguous provider request was replayed")

    with pytest.raises(agy.AmbiguousCall, match="unknown provider disposition"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(run_dir, llm_call=forbidden),
            )
        )
    assert replay_calls == 0


def test_actual_pilot_aggregates_18_by_three_and_balances_54_pairs(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(agy, "_source_revisions", lambda: REVISIONS)
    monkeypatch.setattr(agy, "_runtime_identity", lambda: RUNTIME)
    output_root = tmp_path / "runs"
    plan = agy.build_pilot_plan(
        experiment_id="agy-shape-pilot",
        model="agy-explicit-test-model",
        run_output_root=output_root,
        total_call_cap=450,
    )
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    assay_calls: list[dict[str, Any]] = []
    result = asyncio.run(
        agy.run_experiment(
            plan_path,
            output_root,
            execute=True,
            acknowledged_call_cap=450,
            dependencies=_dependencies(run_dir, assay_calls=assay_calls),
        )
    )

    assert result.call_count == 180
    assert result.run_dir.name.endswith(plan["plan_digest"].removeprefix("sha256:"))
    assert len(assay_calls) == 1
    assay = assay_calls[0]
    assert len(assay["tasks"]) == len(assay["clusters"]) == 18
    assert all(len(cluster["candidate_returns"]) == 3 for cluster in assay["clusters"])
    assert all(len(cluster["base_returns"]) == 3 for cluster in assay["clusters"])
    assert assay["minimum_clusters"] == 18
    assert assay["non_inferiority_margin"] == 0.10
    pairs = [plan["outcome_schedule"][index : index + 2] for index in range(0, 108, 2)]
    assert sum(pair[0]["arm"] == "hinted" for pair in pairs) == 27
    assert sum(pair[0]["arm"] == "unhinted" for pair in pairs) == 27
    assert all(
        set(task["metadata"]["solution_digests_by_seed"]) == {"0", "1", "42"}
        for task in assay["tasks"]
    )


def test_tampered_rejected_disposition_fails_before_resume_calls(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)

    async def failed_call(*_args, **_kwargs):
        raise RuntimeError("offline failure")

    with pytest.raises(agy.ExperimentIncomplete, match="exhausted"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(run_dir, llm_call=failed_call),
            )
        )
    disposition_path = next((run_dir / "candidates").glob("*/*/disposition.json"))
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["category"] = "outcome_selected_replacement"
    body = {key: value for key, value in disposition.items() if key != "disposition_digest"}
    disposition["disposition_digest"] = agy._digest(body)
    disposition_path.write_text(
        json.dumps(disposition, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider called after tampered rejection")

    with pytest.raises(agy.ExperimentError, match="invalid rejection category"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(run_dir, llm_call=forbidden),
            )
        )
    assert calls == 0


def test_single_writer_lock_refuses_parallel_execution(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider called under writer contention")

    with agy._single_writer(run_dir):
        with pytest.raises(agy.ExperimentIncomplete, match="another writer"):
            asyncio.run(
                agy.run_experiment(
                    plan_path,
                    output_root,
                    execute=True,
                    acknowledged_call_cap=100,
                    dependencies=_dependencies(run_dir, llm_call=forbidden),
                )
            )
    assert calls == 0


def test_resume_rejects_missing_assay_bundle_file_without_any_calls(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    asyncio.run(
        agy.run_experiment(
            plan_path,
            output_root,
            execute=True,
            acknowledged_call_cap=100,
            dependencies=_dependencies(run_dir),
        )
    )
    assay_result = json.loads(
        (run_dir / "assay-result.json").read_text(encoding="utf-8")
    )
    missing = run_dir / "assay" / assay_result["evaluation_relative_path"]
    missing.unlink()
    provider_calls = 0
    assay_calls: list[dict[str, Any]] = []

    async def forbidden(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider called while validating a completed Assay bundle")

    with pytest.raises(agy.ExperimentError, match="inventory or bytes changed"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(
                    run_dir,
                    llm_call=forbidden,
                    assay_calls=assay_calls,
                ),
            )
        )
    assert provider_calls == 0
    assert assay_calls == []


def test_alternate_output_root_is_refused_before_dependencies_or_calls(
    monkeypatch, tmp_path
) -> None:
    sealed_root = tmp_path / "sealed-runs"
    alternate_root = tmp_path / "alternate-runs"
    plan = _plan(run_output_root=sealed_root)
    plan_path = _write_plan(tmp_path, plan)
    touched = False

    def forbidden(_plan, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("dependencies resolved for an alternate output root")

    monkeypatch.setattr(agy, "_default_dependencies", forbidden)
    with pytest.raises(agy.ExperimentError, match="sealed run_output_root"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                alternate_root,
                execute=True,
                acknowledged_call_cap=100,
            )
        )
    assert touched is False
    assert not alternate_root.exists()


def test_nested_candidate_symlink_is_rejected_before_outside_write_or_call(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    outside = tmp_path / "outside"
    outside.mkdir()
    candidates = run_dir / "candidates"
    candidates.mkdir(parents=True)
    (candidates / plan["cluster_schedule"][0]["cluster_id"]).symlink_to(
        outside, target_is_directory=True
    )
    provider_calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider called with an unsafe artifact tree")

    with pytest.raises(agy.ExperimentError, match="symlinks are forbidden"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(run_dir, llm_call=forbidden),
            )
        )
    assert provider_calls == 0
    assert list(outside.iterdir()) == []


def test_recomputed_turn_reward_tamper_fails_deterministic_replay_without_calls(
    tmp_path,
) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    asyncio.run(
        agy.run_experiment(
            plan_path,
            output_root,
            execute=True,
            acknowledged_call_cap=100,
            dependencies=_dependencies(run_dir),
        )
    )
    outcome_path = next((run_dir / "outcomes").glob("*/outcome.json"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    turn_path = outcome_path.parent / "turn-01.json"
    turn = json.loads(turn_path.read_text(encoding="utf-8"))
    turn["reward"] = 0.25
    turn_body = {key: value for key, value in turn.items() if key != "turn_digest"}
    turn["turn_digest"] = agy._digest(turn_body)
    turn_path.write_text(json.dumps(turn, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outcome["trajectory"][0] = turn
    outcome["return"] = 0.25
    outcome_body = {key: value for key, value in outcome.items() if key != "outcome_digest"}
    outcome["outcome_digest"] = agy._digest(outcome_body)
    outcome_path.write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provider_calls = 0
    assay_calls: list[dict[str, Any]] = []

    async def forbidden(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider called while replay-validating completed outcomes")

    with pytest.raises(agy.ExperimentError, match="deterministic replay diverged"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=_dependencies(
                    run_dir,
                    llm_call=forbidden,
                    assay_calls=assay_calls,
                ),
            )
        )
    assert provider_calls == 0
    assert assay_calls == []


def test_corrupt_persisted_probe_cannot_trigger_reserve_substitution(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(cap=1, reserve_count=1, run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    with pytest.raises(agy.CallCapExceeded, match="call cap 1"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=1,
                dependencies=_dependencies(run_dir),
            )
        )
    probe_path = next((run_dir / "candidates").glob("*/*/probes/seed-0.json"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["observation_digest"] = "sha256:" + "0" * 64
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calls = 0

    async def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider called after persisted probe corruption")

    with pytest.raises(agy.ExperimentError, match="observation digest mismatch"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=1,
                dependencies=_dependencies(run_dir, llm_call=forbidden),
            )
        )
    first_cluster = plan["cluster_schedule"][0]
    reserve_id = first_cluster["candidate_slots"][1]["candidate_id"]
    reserve_dir = run_dir / "candidates" / first_cluster["cluster_id"] / reserve_id
    assert calls == 0
    assert not reserve_dir.exists()


def test_future_partial_turn_is_rejected_before_an_actor_call(tmp_path) -> None:
    output_root = tmp_path / "runs"
    plan = _plan(run_output_root=output_root)
    plan_path = _write_plan(tmp_path, plan)
    run_dir = agy.derive_run_dir(output_root, plan)
    actor_calls = 0
    designer_calls = 0
    assay_calls: list[dict[str, Any]] = []

    async def counted_call(_client, _model, prompt, **_kwargs):
        nonlocal actor_calls, designer_calls
        if "Environment Designer" in prompt:
            designer_calls += 1
            return (
                f"```python\n# partial-turn-test-{designer_calls}\n"
                "class PuzzleEnv: pass\n```"
            )
        if "expert tutor" in prompt:
            return "Use a general elimination strategy."
        actor_calls += 1
        return r"\boxed{ok}"

    dependencies = _dependencies(
        run_dir,
        llm_call=counted_call,
        assay_calls=assay_calls,
    )
    engine = agy._Engine(plan, agy._pretty_json(plan), run_dir, dependencies)
    with agy._single_writer(run_dir):
        engine.initialize()
        asyncio.run(engine.prepare_cohort())

    first_outcome = plan["outcome_schedule"][0]["outcome_id"]
    future_turn = run_dir / "outcomes" / first_outcome / "turn-02.json"
    future_turn.parent.mkdir(parents=True)
    future_turn.write_text("{}\n", encoding="utf-8")
    with pytest.raises(agy.ExperimentError, match="contiguous prefix"):
        asyncio.run(
            agy.run_experiment(
                plan_path,
                output_root,
                execute=True,
                acknowledged_call_cap=100,
                dependencies=dependencies,
            )
        )
    assert actor_calls == 0
    assert assay_calls == []
