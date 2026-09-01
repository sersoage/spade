"""Adversarial, provider-free tests for the counterfactual-witness runner."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from spade.core.counterfactual_witness import SourceVariant, WitnessProbe, source_digest
from tools import run_spade_agy_experiment as base
from tools import run_counterfactual_witness_experiment as witness


_PROBES = (
    WitnessProbe.create(seed=0, actions=(), role="reset"),
    WitnessProbe.create(seed=0, actions=("DECISIVE",), role="base_oracle"),
    WitnessProbe.create(seed=0, actions=("DECOY",), role="well_formed_wrong"),
)

_VARIANT_SPECS = (
    (
        "semantic_mutant",
        "train",
        "fixture_train_semantics",
        "fixture_train_mutant",
    ),
    (
        "semantic_mutant",
        "heldout",
        "fixture_heldout_semantics",
        "fixture_heldout_mutant",
    ),
    (
        "equivalent_control",
        "train",
        "fixture_equivalence",
        "fixture_train_control",
    ),
    (
        "equivalent_control",
        "heldout",
        "fixture_equivalence",
        "fixture_heldout_control",
    ),
)


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fake_variants(game_code: str) -> tuple[SourceVariant, ...]:
    variants: list[SourceVariant] = []
    for kind, partition, family, operator in _VARIANT_SPECS:
        variant_source = f"{game_code.rstrip()}\n# fixture-variant:{operator}\n"
        digest = source_digest(variant_source)
        variants.append(
            SourceVariant(
                variant_id=_digest_bytes(
                    f"{source_digest(game_code)}:{operator}:{digest}".encode("utf-8")
                ),
                kind=kind,
                partition=partition,
                family=family,
                operator=operator,
                source=variant_source,
                source_digest=digest,
                entrypoint="OfflineWitnessEnv",
                expected_effect=f"offline fixture {operator}",
            )
        )
    return tuple(variants)


def _fake_probes(**_kwargs: Any) -> tuple[WitnessProbe, ...]:
    return _PROBES


def _result(source: str, seed: int, actions: Sequence[str]) -> dict[str, Any]:
    operator_match = re.search(r"# fixture-variant:([^\n]+)", source)
    operator = operator_match.group(1) if operator_match else "base"
    semantic = operator in {"fixture_train_mutant", "fixture_heldout_mutant"}
    decisive = tuple(actions) == ("DECISIVE",)
    success = decisive
    reward = 1.0 if success else 0.0
    trajectory: list[dict[str, Any]] = [
        {"role": "environment", "observation": f"reset seed={seed}"}
    ]
    if actions:
        observation = "base transition"
        if decisive and semantic:
            observation = f"mutated transition {operator}"
        trajectory.append(
            {
                "action": actions[0],
                "observation": observation,
                "reward": reward,
                "terminated": success,
                "truncated": False,
                "info": {"fixture": True},
            }
        )
    return {
        "success": success,
        "reward": reward,
        "turn_count": len(actions),
        "terminated": success,
        "truncated": False,
        "trajectory": trajectory,
        "error": None,
    }


@dataclass
class _Ledger:
    target_factories: int = 0
    sandbox_operations: int = 0
    provider_calls: int = 0


class _FakeTarget:
    def __init__(self, source: str, ledger: _Ledger) -> None:
        self.source = source
        self.ledger = ledger

    def inspect(self, seed: int, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        del timeout_seconds
        self.ledger.sandbox_operations += 1
        return _result(self.source, seed, ())

    def run_actions(
        self,
        seed: int | None,
        actions: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds
        self.ledger.sandbox_operations += 1
        return _result(self.source, 0 if seed is None else seed, actions)

    def run_oracle(self, seed: int = 0, timeout_seconds: float | None = None) -> dict[str, Any]:
        del timeout_seconds
        self.ledger.sandbox_operations += 1
        return _result(self.source, seed, ("DECISIVE",))


@dataclass
class _Harness:
    source_dir: Path
    output_root: Path
    runtime_file: Path
    snapshot_ref: dict[str, Any]
    dependencies: witness.RunnerDependencies
    ledger: _Ledger

    @property
    def snapshot(self) -> Any:
        return self.snapshot_ref["value"]

    @snapshot.setter
    def snapshot(self, value: Any) -> None:
        self.snapshot_ref["value"] = value

    def seal(self) -> tuple[dict[str, Any], Path]:
        plan = witness.build_plan(
            source_run_dir=self.source_dir,
            output_root=self.output_root,
            witness_budget=1,
            random_baseline_draws=128,
            operation_timeout_seconds=0.25,
            dependencies=self.dependencies,
        )
        plan_path = self.source_dir.parent / "witness-plan.json"
        witness.write_plan(plan_path, plan)
        return plan, plan_path


def _snapshot() -> Any:
    schedule: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    for ordinal in range(1, witness.AUTHORIZED_CLUSTERS + 1):
        cluster_id = f"c{ordinal:03d}"
        schedule.append({"cluster_id": cluster_id})
        code = f"# offline-witness-cluster:{ordinal}\nclass OfflineWitnessEnv:\n    pass\n"
        selections[cluster_id] = {
            "cluster_id": cluster_id,
            "candidate_id": f"candidate-{ordinal:03d}",
            "skill": f"skill-{(ordinal - 1) // 2:02d}",
            "difficulty": "medium" if ordinal % 2 else "hard",
            "environment_name": "OfflineWitnessEnv",
            "code": code,
            "code_digest": witness._digest(code),
            "environment_digest": source_digest(code),
            "qualification_digest": _digest_bytes(f"qualification:{ordinal}".encode()),
            "probes": {str(seed): {"solution": "DECISIVE"} for seed in witness.AUTHORIZED_SEEDS},
        }
    manifest = [
        {
            "path": f"authorized-leaf-{index:03d}.json",
            "digest": _digest_bytes(f"authorized-leaf:{index}".encode()),
        }
        for index in range(321)
    ]
    return SimpleNamespace(
        plan={
            "plan_digest": witness.AUTHORIZED_SOURCE_PLAN_DIGEST,
            "cluster_schedule": schedule,
        },
        cohort={"cohort_digest": witness.AUTHORIZED_SOURCE_COHORT_DIGEST},
        selections=selections,
        manifest=manifest,
    )


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    monkeypatch.setattr(witness, "generate_source_variants", _fake_variants)
    monkeypatch.setattr(witness, "build_candidate_probes", _fake_probes)

    ledger = _Ledger()

    async def provider_bomb(*_args: Any, **_kwargs: Any) -> str:
        ledger.provider_calls += 1
        raise AssertionError("provider access is forbidden in the offline witness runner")

    def default_dependency_bomb(*_args: Any, **_kwargs: Any) -> Any:
        ledger.provider_calls += 1
        raise AssertionError("injected dependencies must prevent ambient resolution")

    monkeypatch.setattr(base.live, "call_llm", provider_bomb)
    monkeypatch.setattr(witness, "_default_dependencies", default_dependency_bomb)

    source_dir = (tmp_path / "authorized-source").resolve()
    source_dir.mkdir()
    output_root = (tmp_path / "witness-output").resolve()
    runtime_file = (tmp_path / "sealed-runtime.txt").resolve()
    runtime_file.write_text("sealed offline runtime\n", encoding="utf-8")
    python_path = Path(sys.executable).resolve()
    runtime_identity = {
        "python_executable": str(python_path),
        "python_executable_digest": witness._file_digest(python_path),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "spade_revision": "1" * 40,
        "proofpack_revision": "2" * 40,
        "files": {
            "offline_fixture": {
                "path": str(runtime_file),
                "digest": witness._file_digest(runtime_file),
            }
        },
        "operation_runtime_files": ["offline_fixture"],
        "execution_boundary": "macos-sandbox-exec-worker/v1",
        "network_or_provider_calls": False,
    }
    snapshot_ref = {"value": _snapshot()}

    def load_source_snapshot(path: Path) -> Any:
        assert path == source_dir
        return snapshot_ref["value"]

    def target_factory(
        source: str,
        *,
        action_format: str,
        max_turns: int,
        operation_timeout_seconds: float,
    ) -> _FakeTarget:
        assert isinstance(source, str)
        assert action_format == "boxed"
        assert max_turns == witness.MAX_TURNS
        assert operation_timeout_seconds == 0.25
        ledger.target_factories += 1
        return _FakeTarget(source, ledger)

    dependencies = witness.RunnerDependencies(
        load_source_snapshot=load_source_snapshot,
        target_factory=target_factory,
        runtime_identity=runtime_identity,
    )
    return _Harness(
        source_dir=source_dir,
        output_root=output_root,
        runtime_file=runtime_file,
        snapshot_ref=snapshot_ref,
        dependencies=dependencies,
        ledger=ledger,
    )


def _rewrite_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(base._pretty_json(dict(value)))


def _first_operation(
    plan: Mapping[str, Any], snapshot: Any
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], WitnessProbe]:
    cluster = plan["clusters"][0]
    source, _variants, probes = witness._regenerate_cluster(snapshot, cluster)
    return cluster, source, witness._base_variant_record(cluster), probes[0]


def _seed_complete_leaf(
    harness: _Harness, plan: Mapping[str, Any]
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], WitnessProbe]:
    run_dir = witness.derive_run_dir(plan)
    cluster, source, variant, probe = _first_operation(plan, harness.snapshot)
    witness._execute_trace_leaf(
        plan=plan,
        run_dir=run_dir,
        cluster=cluster,
        source=source,
        variant=variant,
        probe=probe,
        dependencies=harness.dependencies,
    )
    return run_dir, cluster, variant, probe


def test_plan_is_response_free_sealed_and_dry_run_has_no_side_effects(
    harness: _Harness,
) -> None:
    plan, plan_path = harness.seal()

    assert plan["provider_calls"] == 0
    assert plan["learner_updates"] == 0
    assert plan["configuration"]["sandbox_operation_ceiling"] == 540
    assert len(plan["clusters"]) == 18
    assert plan["plan_digest"] == witness._digest(
        {key: value for key, value in plan.items() if key != "plan_digest"}
    )

    result = witness.run_experiment(
        plan_path,
        execute=False,
        dependencies=harness.dependencies,
    )

    assert result.status == "validated"
    assert result.plan_digest == plan["plan_digest"]
    assert not result.run_dir.exists()
    assert harness.ledger == _Ledger()


def test_full_fake_execution_aggregates_zero_provider_calls_and_resumes_exactly(
    harness: _Harness,
) -> None:
    plan, plan_path = harness.seal()

    first = witness.run_experiment(
        plan_path,
        execute=True,
        dependencies=harness.dependencies,
    )
    calls_after_first = harness.ledger.sandbox_operations
    factories_after_first = harness.ledger.target_factories

    assert first.status == "pass"
    assert first.aggregate_path is not None
    aggregate = base._read_json(first.aggregate_path)
    assert aggregate["status"] == "pass"
    assert aggregate["provider_calls"] == 0
    assert aggregate["learner_updates"] == 0
    assert aggregate["sandbox_operations_completed"] == 540
    assert aggregate["metrics"]["training_mutant_recall_macro"] == 1.0
    assert aggregate["metrics"]["heldout_mutant_recall_macro"] == 1.0
    assert aggregate["metrics"]["heldout_equivalent_false_rejection_rate"] == 0.0
    assert aggregate["metrics"]["witness_minus_random_recall"] > 0.15
    assert all(aggregate["gates"].values())
    assert calls_after_first == plan["configuration"]["sandbox_operation_ceiling"]
    assert factories_after_first == calls_after_first // plan["configuration"]["repetitions"]
    assert harness.ledger.provider_calls == 0

    manifest = base._read_json(first.run_dir / "run-manifest.json")
    clusters = [
        base._read_json(first.run_dir / "clusters" / f"c{index:03d}.json") for index in range(1, 19)
    ]
    assert manifest["provider_calls"] == 0
    assert all(cluster["provider_calls"] == 0 for cluster in clusters)
    assert all(cluster["learner_updates"] == 0 for cluster in clusters)

    resumed = witness.run_experiment(
        plan_path,
        execute=True,
        dependencies=harness.dependencies,
    )

    assert resumed == first
    assert harness.ledger.sandbox_operations == calls_after_first
    assert harness.ledger.target_factories == factories_after_first
    assert harness.ledger.provider_calls == 0


def test_tampered_plan_is_rejected_before_execution(harness: _Harness) -> None:
    _plan, plan_path = harness.seal()
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["configuration"]["witness_budget"] = 2
    _rewrite_json(plan_path, tampered)

    with pytest.raises(witness.WitnessExperimentError, match="plan self-digest mismatch"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == 0


def test_source_drift_after_sealing_is_rejected_before_execution(harness: _Harness) -> None:
    _plan, plan_path = harness.seal()
    drifted = copy.deepcopy(harness.snapshot)
    drifted.manifest[0]["digest"] = _digest_bytes(b"drifted authorized leaf")
    harness.snapshot = drifted

    with pytest.raises(witness.WitnessExperimentError, match="source snapshot drifted"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == 0


def test_runtime_bytes_drift_after_sealing_is_rejected_before_execution(
    harness: _Harness,
) -> None:
    _plan, plan_path = harness.seal()
    harness.runtime_file.write_text("tampered runtime\n", encoding="utf-8")

    with pytest.raises(witness.WitnessExperimentError, match="runtime file bytes drifted"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == 0


def test_operation_runtime_drift_fails_before_reservation_or_sandbox_call(
    harness: _Harness,
) -> None:
    plan, _plan_path = harness.seal()
    run_dir = witness.derive_run_dir(plan)
    cluster, source, variant, probe = _first_operation(plan, harness.snapshot)
    request_path, _result_path = witness._operation_paths(
        run_dir,
        str(cluster["cluster_id"]),
        str(variant["variant_id"]),
        probe.probe_id,
        1,
    )
    harness.runtime_file.write_text("drifted after target construction\n", encoding="utf-8")

    with pytest.raises(witness.WitnessExperimentError, match="operation runtime file drifted"):
        witness._execute_trace_leaf(
            plan=plan,
            run_dir=run_dir,
            cluster=cluster,
            source=source,
            variant=variant,
            probe=probe,
            dependencies=harness.dependencies,
        )

    assert not request_path.exists()
    assert harness.ledger.sandbox_operations == 0


def test_resealed_trace_tamper_is_rejected_on_resume_without_another_call(
    harness: _Harness,
) -> None:
    plan, plan_path = harness.seal()
    run_dir, cluster, variant, probe = _seed_complete_leaf(harness, plan)
    calls_before_resume = harness.ledger.sandbox_operations
    trace_path = witness._trace_leaf_path(
        run_dir,
        str(cluster["cluster_id"]),
        str(variant["variant_id"]),
        probe.probe_id,
    )
    leaf = base._read_json(trace_path)
    leaf["result"]["trajectory"][0]["observation"] = "forged trace observation"
    leaf_body = {key: value for key, value in leaf.items() if key != "leaf_digest"}
    leaf["leaf_digest"] = witness._digest(leaf_body)
    _rewrite_json(trace_path, leaf)

    with pytest.raises(
        witness.WitnessExperimentError,
        match="persisted trace leaf differs from deterministic replay",
    ):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == calls_before_resume


def test_resealed_operation_result_tamper_is_caught_by_repetition(
    harness: _Harness,
) -> None:
    plan, plan_path = harness.seal()
    run_dir, cluster, variant, probe = _seed_complete_leaf(harness, plan)
    calls_before_resume = harness.ledger.sandbox_operations
    _request_path, result_path = witness._operation_paths(
        run_dir,
        str(cluster["cluster_id"]),
        str(variant["variant_id"]),
        probe.probe_id,
        1,
    )
    operation = base._read_json(result_path)
    operation["result"]["trajectory"][0]["observation"] = "forged operation result"
    operation_body = {key: value for key, value in operation.items() if key != "result_digest"}
    operation["result_digest"] = witness._digest(operation_body)
    _rewrite_json(result_path, operation)

    with pytest.raises(witness.WitnessExperimentError, match="nondeterministic witness response"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == calls_before_resume


def test_malformed_resealed_operation_result_is_wrapped_as_protocol_error(
    harness: _Harness,
) -> None:
    plan, plan_path = harness.seal()
    run_dir, cluster, variant, probe = _seed_complete_leaf(harness, plan)
    calls_before_resume = harness.ledger.sandbox_operations
    _request_path, result_path = witness._operation_paths(
        run_dir,
        str(cluster["cluster_id"]),
        str(variant["variant_id"]),
        probe.probe_id,
        1,
    )
    operation = base._read_json(result_path)
    operation["result"]["turn_count"] = True
    operation_body = {key: value for key, value in operation.items() if key != "result_digest"}
    operation["result_digest"] = witness._digest(operation_body)
    _rewrite_json(result_path, operation)

    with pytest.raises(witness.WitnessExperimentError, match="invalid trace shape"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == calls_before_resume


def test_fixed_baseline_and_fourth_gate_are_sealed_deterministically(
    harness: _Harness,
) -> None:
    plan, _plan_path = harness.seal()
    ids = witness._fixed_baseline_ids(
        _PROBES,
        [probe.probe_id for probe in _PROBES],
        1,
    )

    assert ids == (_PROBES[1].probe_id,)
    assert (
        plan["configuration"]["minimum_safe_bank_heldout_recall"]
        == witness.MIN_SAFE_BANK_HELDOUT_RECALL
    )


def test_request_without_result_is_ambiguous_and_never_replayed(harness: _Harness) -> None:
    plan, plan_path = harness.seal()
    run_dir = witness.derive_run_dir(plan)
    cluster, _source, variant, probe = _first_operation(plan, harness.snapshot)
    request = witness._operation_request(
        plan=plan,
        cluster=cluster,
        variant=variant,
        probe=probe,
        repetition=1,
    )
    request_path, _result_path = witness._operation_paths(
        run_dir,
        str(cluster["cluster_id"]),
        str(variant["variant_id"]),
        probe.probe_id,
        1,
    )
    base._write_json(request_path, request)

    with pytest.raises(witness.WitnessExperimentError, match="ambiguous sandbox operation"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == 0


def test_unknown_run_inventory_is_rejected_before_execution(harness: _Harness) -> None:
    plan, plan_path = harness.seal()
    run_dir = witness.derive_run_dir(plan)
    run_dir.mkdir(parents=True)
    (run_dir / "unrecognized.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(witness.WitnessExperimentError, match="unrecognized artifacts"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == 0


def test_symlinked_run_inventory_is_rejected_before_execution(harness: _Harness) -> None:
    plan, plan_path = harness.seal()
    run_dir = witness.derive_run_dir(plan)
    run_dir.mkdir(parents=True)
    (run_dir / "unsafe-link").symlink_to(harness.runtime_file)

    with pytest.raises(base.ExperimentError, match="symlinked artifact paths are forbidden"):
        witness.run_experiment(
            plan_path,
            execute=True,
            dependencies=harness.dependencies,
        )

    assert harness.ledger.sandbox_operations == 0
