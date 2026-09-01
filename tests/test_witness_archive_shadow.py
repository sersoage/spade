"""Adversarial tests for the opt-in counterfactual-witness shadow archive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from spade.core.counterfactual_witness import (
    WITNESS_SCHEMA_VERSION,
    WitnessProbe,
    mutation_catalog_digest,
)
from spade.core.witness_archive_shadow import (
    AGGREGATE_SCHEMA,
    AUTHORIZED_CLUSTER_COUNT,
    AUTHORIZED_SOURCE_COHORT_DIGEST,
    AUTHORIZED_SOURCE_PLAN_DIGEST,
    CLUSTER_RESULT_SCHEMA,
    PLAN_SCHEMA,
    PROTOCOL_ID,
    ShadowArchiveLedger,
    WitnessArchiveShadowError,
    _digest,
    _pretty_json,
    load_validated_cwa_run,
)


QUALITY_POLICY = "constant-zero-integration-smoke/v1"
LINEAGE_POLICY = "singleton-environment-digest/v1"


@dataclass(frozen=True)
class _Fixture:
    run_dir: Path
    aggregate_digest: str
    game_files: tuple[Path, ...]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pretty_json(value))


def _descriptor(index: int, *, same_cell: bool) -> dict[str, Any]:
    return {
        "action_format": "boxed",
        "oracle_depth_bin": 1 if same_cell else index % 4,
        "reset_seed_diversity_bin": 1,
        "invalid_reward_bin": 0,
        "invalid_end_state": "continues",
        "recovery_success": "observed_true",
        "trace_order_divergent": "observed_false",
    }


def _selection(probe_id: str) -> dict[str, Any]:
    return {
        "selected_probe_ids": [probe_id],
        "safe_probe_ids": [probe_id],
        "rejected_control_breakers": [],
        "killed_train_mutants": ["mutant"],
        "uncovered_train_mutants": [],
        "family_coverage": {"synthetic": {"killed": 1, "total": 1}},
        "budget": 1,
        "algorithm": "safe-inverse-family-greedy-set-cover/v1",
    }


def _make_fixture(tmp_path: Path, *, same_cell: bool = False) -> _Fixture:
    run_dir = tmp_path / "sealed-run"
    game_root = tmp_path / "games"
    clusters = []
    game_files = []
    probes_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for ordinal in range(1, AUTHORIZED_CLUSTER_COUNT + 1):
        cluster_id = f"c{ordinal:03d}-synthetic"
        source = f"class Env:\n    marker = {ordinal}\n".encode()
        game_file = game_root / f"{cluster_id}.py"
        game_file.parent.mkdir(parents=True, exist_ok=True)
        game_file.write_bytes(source)
        game_files.append(game_file)
        probe = WitnessProbe.create(
            seed=0,
            actions=(f"\\boxed{{answer-{ordinal}}}",),
            role="base_oracle",
        )
        probes = [probe.to_dict()]
        probes_by_cluster[cluster_id] = probes
        skill = "one-skill" if same_cell else f"skill-{(ordinal - 1) // 2:02d}"
        difficulty = "medium" if same_cell else ("medium" if ordinal % 2 else "hard")
        clusters.append(
            {
                "cluster_id": cluster_id,
                "candidate_id": f"{cluster_id}-primary",
                "skill": skill,
                "difficulty": difficulty,
                "environment_name": f"SyntheticEnv{ordinal}",
                "code_digest": _digest(source.decode()),
                "environment_digest": "sha256:" + hashlib.sha256(source).hexdigest(),
                "qualification_digest": _digest({"qualified": ordinal}),
                "variants": [
                    {
                        "variant_id": _digest({"variant": ordinal}),
                        "kind": "semantic_mutant",
                        "partition": "train",
                        "family": "synthetic",
                        "operator": "synthetic",
                        "source_digest": _digest({"source": ordinal}),
                        "entrypoint": f"SyntheticEnv{ordinal}",
                        "expected_effect": "synthetic fixture",
                    }
                ],
                "probes": probes,
                "schedule_ordinal": ordinal,
            }
        )
    plan_body = {
        "schema_version": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "spade-counterfactual-witness-v1",
        "source": {
            "run_dir": str(tmp_path / "source"),
            "plan_digest": AUTHORIZED_SOURCE_PLAN_DIGEST,
            "cohort_digest": AUTHORIZED_SOURCE_COHORT_DIGEST,
            "import_manifest_digest": _digest({"manifest": "synthetic"}),
            "import_leaf_count": 321,
        },
        "output_root": str(tmp_path),
        "runtime_identity": {"fixture": True},
        "configuration": {
            "action_format": "boxed",
            "seeds": [0, 1, 42],
            "max_turns": 5,
            "operation_timeout_seconds": 5.0,
            "repetitions": 2,
            "witness_budget": 1,
            "random_baseline_draws": 1,
            "sandbox_operation_ceiling": 18,
            "primary_recall_margin": 0.15,
            "max_equivalent_false_rejection_rate": 0.05,
            "minimum_training_recall": 0.9,
            "minimum_safe_bank_heldout_recall": 0.9,
        },
        "family_split": {
            "training_semantic": ["synthetic"],
            "heldout_semantic": ["synthetic-heldout"],
            "training_controls": ["control"],
            "heldout_controls": ["control-heldout"],
        },
        "clusters": clusters,
        "analysis_role": "offline-representation-falsification-only",
        "provider_calls": 0,
        "learner_updates": 0,
    }
    plan = {**plan_body, "plan_digest": _digest(plan_body)}
    _write_json(run_dir / "plan.json", plan)

    cluster_results = []
    for index, cluster in enumerate(clusters):
        cluster_id = cluster["cluster_id"]
        probe = probes_by_cluster[cluster_id][0]
        selection = _selection(probe["probe_id"])
        certificate_body = {
            "schema_version": WITNESS_SCHEMA_VERSION,
            "environment_digest": cluster["environment_digest"],
            "mutation_catalog_digest": mutation_catalog_digest(),
            "candidate_pool_digest": _digest(cluster["probes"]),
            "selection": selection,
            "probes": [probe],
            "expected_signatures": {probe["probe_id"]: _digest({"signature": index})},
            "metadata": {
                "plan_digest": plan["plan_digest"],
                "source_plan_digest": AUTHORIZED_SOURCE_PLAN_DIGEST,
                "source_cohort_digest": AUTHORIZED_SOURCE_COHORT_DIGEST,
                "cluster_id": cluster_id,
                "candidate_id": cluster["candidate_id"],
                "qualification_digest": cluster["qualification_digest"],
                "analysis_role": "offline-representation-falsification-only",
            },
        }
        certificate = {
            **certificate_body,
            "certificate_digest": _digest(certificate_body),
        }
        _write_json(run_dir / "certificates" / f"{cluster_id}.json", certificate)
        result_body = {
            "schema_version": CLUSTER_RESULT_SCHEMA,
            "plan_digest": plan["plan_digest"],
            "cluster": cluster,
            "selection": selection,
            "scores": {
                "training": {},
                "heldout": {},
                "safe_bank_heldout": {},
                "proofpack_fixed_heldout": {},
            },
            "applicability": {"training": {}, "heldout": {}},
            "proofpack_fixed_probe_ids": [probe["probe_id"]],
            "random_baseline": {},
            "behavior_descriptor": _descriptor(index, same_cell=same_cell),
            "certificate": {
                "path": f"certificates/{cluster_id}.json",
                "digest": certificate["certificate_digest"],
            },
            "trace_inventory": [],
            "provider_calls": 0,
            "learner_updates": 0,
        }
        result = {**result_body, "cluster_result_digest": _digest(result_body)}
        _write_json(run_dir / "clusters" / f"{cluster_id}.json", result)
        cluster_results.append(result)
    aggregate_body = {
        "schema_version": AGGREGATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan["plan_digest"],
        "source_plan_digest": AUTHORIZED_SOURCE_PLAN_DIGEST,
        "source_cohort_digest": AUTHORIZED_SOURCE_COHORT_DIGEST,
        "status": "pass",
        "metrics": {"fixture": 1.0},
        "gates": {"fixture": True},
        "thresholds": {
            "primary_recall_margin": 0.15,
            "max_equivalent_false_rejection_rate": 0.05,
            "minimum_training_recall": 0.9,
            "minimum_safe_bank_heldout_recall": 0.9,
        },
        "cluster_result_digests": [result["cluster_result_digest"] for result in cluster_results],
        "sandbox_operations_completed": 18,
        "provider_calls": 0,
        "learner_updates": 0,
        "claim_boundary": "synthetic fixture",
    }
    aggregate = {**aggregate_body, "aggregate_digest": _digest(aggregate_body)}
    _write_json(run_dir / "aggregate.json", aggregate)
    return _Fixture(run_dir, aggregate["aggregate_digest"], tuple(game_files))


def _load(fixture: _Fixture):
    return load_validated_cwa_run(fixture.run_dir, fixture.aggregate_digest)


def _event_files(ledger_dir: Path) -> list[Path]:
    return sorted((ledger_dir / "events").iterdir())


def test_loader_binds_external_aggregate_certificate_and_verified_game_bytes(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    cohort = _load(fixture)

    assert len(cohort.evidence) == 18
    assert cohort.aggregate_digest == fixture.aggregate_digest
    entry = cohort.evidence[0].bind(
        fixture.game_files[0],
        0.0,
        (cohort.evidence[0].environment_digest,),
        quality_policy_id=QUALITY_POLICY,
        lineage_policy_id=LINEAGE_POLICY,
    )
    assert entry.skill == "skill-00"
    assert entry.difficulty == "medium"
    assert entry.game_file == f"verified-game@{entry.environment_digest}"
    assert str(fixture.game_files[0]) not in json.dumps(entry.to_dict())

    with pytest.raises(TypeError):
        cohort.evidence[0].bind(
            fixture.game_files[0],
            0.0,
            ("root",),
            quality_policy_id=QUALITY_POLICY,
            lineage_policy_id=LINEAGE_POLICY,
            skill="caller-override",  # type: ignore[call-arg]
        )
    fixture.game_files[0].write_bytes(b"tampered")
    with pytest.raises(WitnessArchiveShadowError, match="game bytes"):
        cohort.evidence[0].bind(
            fixture.game_files[0],
            0.0,
            ("root",),
            quality_policy_id=QUALITY_POLICY,
            lineage_policy_id=LINEAGE_POLICY,
        )


@pytest.mark.parametrize("leaf", ["aggregate", "cluster", "certificate"])
def test_loader_rejects_tampered_evidence_leaves(tmp_path: Path, leaf: str) -> None:
    fixture = _make_fixture(tmp_path)
    if leaf == "aggregate":
        path = fixture.run_dir / "aggregate.json"
        value = json.loads(path.read_text())
        value["status"] = "fail"
    elif leaf == "cluster":
        path = sorted((fixture.run_dir / "clusters").iterdir())[0]
        value = json.loads(path.read_text())
        value["behavior_descriptor"]["invalid_reward_bin"] = 1
    else:
        path = sorted((fixture.run_dir / "certificates").iterdir())[0]
        value = json.loads(path.read_text())
        value["metadata"]["candidate_id"] = "tampered"
    _write_json(path, value)

    with pytest.raises(WitnessArchiveShadowError):
        _load(fixture)


def test_loader_rejects_invalid_resealed_descriptor_and_unsafe_inventory(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    cluster_path = sorted((fixture.run_dir / "clusters").iterdir())[0]
    result = json.loads(cluster_path.read_text())
    result["behavior_descriptor"]["recovery_success"] = "unknown"
    result_body = {key: value for key, value in result.items() if key != "cluster_result_digest"}
    result["cluster_result_digest"] = _digest(result_body)
    _write_json(cluster_path, result)
    aggregate_path = fixture.run_dir / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["cluster_result_digests"][0] = result["cluster_result_digest"]
    aggregate_body = {key: value for key, value in aggregate.items() if key != "aggregate_digest"}
    aggregate["aggregate_digest"] = _digest(aggregate_body)
    _write_json(aggregate_path, aggregate)
    with pytest.raises(WitnessArchiveShadowError, match="descriptor"):
        load_validated_cwa_run(fixture.run_dir, aggregate["aggregate_digest"])

    fixture = _make_fixture(tmp_path / "unknown")
    _write_json(fixture.run_dir / "clusters" / "extra.json", {"extra": True})
    with pytest.raises(WitnessArchiveShadowError, match="inventory"):
        _load(fixture)

    fixture = _make_fixture(tmp_path / "symlink")
    certificate = sorted((fixture.run_dir / "certificates").iterdir())[0]
    content = certificate.read_bytes()
    outside = tmp_path / "outside-certificate.json"
    outside.write_bytes(content)
    certificate.unlink()
    certificate.symlink_to(outside)
    with pytest.raises(WitnessArchiveShadowError, match="artifact"):
        _load(fixture)


def test_real_shape_constant_zero_singleton_smoke_is_18_insertions(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    cohort = _load(fixture)
    ledger_dir = tmp_path / "ledger"
    ledger = ShadowArchiveLedger(
        ledger_dir,
        cohort=cohort,
        quality_policy_id=QUALITY_POLICY,
        lineage_policy_id=LINEAGE_POLICY,
    )

    receipts = [
        ledger.consider(
            event_id=f"shadow:{evidence.cluster_id}",
            evidence=evidence,
            game_file=fixture.game_files[index],
            quality_score=0.0,
            lineage=(evidence.environment_digest,),
        )
        for index, evidence in enumerate(cohort.evidence)
    ]
    summary = ledger.summary()

    assert [receipt.action for receipt in receipts] == ["champion_inserted"] * 18
    assert summary.event_count == 18
    assert summary.cell_count == 18
    assert summary.action_counts == {"champion_inserted": 18}
    assert not hasattr(ledger, "get")
    assert not hasattr(ledger, "select")


def test_ledger_exactly_once_resume_and_conflicting_duplicate(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    cohort = _load(fixture)
    evidence = cohort.evidence[0]
    ledger_dir = tmp_path / "ledger"
    ledger = ShadowArchiveLedger(
        ledger_dir,
        cohort=cohort,
        quality_policy_id=QUALITY_POLICY,
        lineage_policy_id=LINEAGE_POLICY,
    )
    arguments = {
        "event_id": "event-one",
        "evidence": evidence,
        "game_file": fixture.game_files[0],
        "quality_score": 0.0,
        "lineage": (evidence.environment_digest,),
    }

    first = ledger.consider(**arguments)
    resumed = ledger.consider(**arguments)
    reopened = ShadowArchiveLedger(
        ledger_dir,
        cohort=cohort,
        quality_policy_id=QUALITY_POLICY,
        lineage_policy_id=LINEAGE_POLICY,
    ).consider(**arguments)

    assert not first.resumed
    assert resumed.resumed and reopened.resumed
    assert first.event_digest == resumed.event_digest == reopened.event_digest
    assert len(_event_files(ledger_dir)) == 1
    with pytest.raises(WitnessArchiveShadowError, match="conflicting"):
        ledger.consider(**{**arguments, "quality_score": 0.1})


def test_synthetic_same_cell_ledger_covers_every_archive_action_and_eviction(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path, same_cell=True)
    cohort = _load(fixture)
    ledger_dir = tmp_path / "ledger"
    ledger = ShadowArchiveLedger(
        ledger_dir,
        cohort=cohort,
        quality_policy_id="synthetic-quality/v1",
        lineage_policy_id="synthetic-lineage/v1",
    )
    cases = [
        (0, 0.4, ("line-a",)),
        (1, 0.3, ("line-b",)),
        (2, 0.8, ("line-c",)),
        (3, 0.5, ("line-d",)),
        (2, 0.8, ("line-c",)),
        (4, 0.7, ("line-c", "child-e")),
        (5, 0.2, ("line-f",)),
    ]
    actions = []
    for event_index, (evidence_index, score, lineage) in enumerate(cases, start=1):
        receipt = ledger.consider(
            event_id=f"synthetic-{event_index}",
            evidence=cohort.evidence[evidence_index],
            game_file=fixture.game_files[evidence_index],
            quality_score=score,
            lineage=lineage,
        )
        actions.append(receipt.action)

    assert actions == [
        "champion_inserted",
        "challenger_inserted",
        "champion_replaced",
        "challenger_replaced",
        "rejected_duplicate",
        "rejected_same_lineage",
        "rejected_lower_quality",
    ]
    replacement = json.loads(_event_files(ledger_dir)[2].read_text())
    assert replacement["decision"]["demoted_digest"] == cohort.evidence[0].environment_digest
    assert replacement["decision"]["evicted_digests"] == [cohort.evidence[1].environment_digest]
    assert replacement["actual_baseline_selection"] == {
        "cell_key": replacement["decision"]["cell_key"],
        "champion_environment_digest": cohort.evidence[2].environment_digest,
        "challenger_environment_digest": cohort.evidence[0].environment_digest,
    }
    assert ledger.summary().cell_count == 1


@pytest.mark.parametrize("attack", ["tamper", "unknown", "symlink"])
def test_ledger_rejects_tamper_unknown_files_and_symlinks(tmp_path: Path, attack: str) -> None:
    fixture = _make_fixture(tmp_path)
    cohort = _load(fixture)
    ledger_dir = tmp_path / "ledger"
    ledger = ShadowArchiveLedger(
        ledger_dir,
        cohort=cohort,
        quality_policy_id=QUALITY_POLICY,
        lineage_policy_id=LINEAGE_POLICY,
    )
    evidence = cohort.evidence[0]
    ledger.consider(
        event_id="event-one",
        evidence=evidence,
        game_file=fixture.game_files[0],
        quality_score=0.0,
        lineage=(evidence.environment_digest,),
    )
    event = _event_files(ledger_dir)[0]
    if attack == "tamper":
        value = json.loads(event.read_text())
        value["post_state_digest"] = _digest({"tampered": True})
        _write_json(event, value)
    elif attack == "unknown":
        (ledger_dir / "events" / "notes.txt").write_text("unexpected")
    else:
        symlink = ledger_dir / "events" / f"000002-{'0' * 64}.json"
        symlink.symlink_to(event)

    with pytest.raises(WitnessArchiveShadowError):
        ShadowArchiveLedger(
            ledger_dir,
            cohort=cohort,
            quality_policy_id=QUALITY_POLICY,
            lineage_policy_id=LINEAGE_POLICY,
        )
