import os
from pathlib import Path

import pytest

from spade.core.learner_branch_pools import (
    HELDOUT_V4_STRATA,
    TRAINING_STRATA,
    LearnerPoolError,
    load_learner_pool_manifest,
    materialize_learner_pools,
)


_DEFAULT_ACTOR_PLAN = Path(
    "/Users/sergio.soage/code/spade-baseline-stack-20260901/spade/.assay/"
    "spade-coverage-forced-proxy-v2/"
    "spade-google-coverage-forced-matched-swap-v2-"
    "df1a06c7fb854d5267ec4d1e41cd44c1ffd229018bf0f5a0dcde512fa47c4c09/"
    "actor-plan.json"
)
_ACTOR_PLAN = Path(os.environ.get("SPADE_SEALED_ACTOR_PLAN", _DEFAULT_ACTOR_PLAN))


@pytest.fixture()
def materialized(tmp_path: Path):
    if not _ACTOR_PLAN.is_file():
        pytest.skip("sealed df1a/fc798 actor-plan fixture is not available")
    return materialize_learner_pools(_ACTOR_PLAN, tmp_path / "bundle")


def test_materializes_aligned_write_once_training_and_disjoint_heldout_pools(
    materialized,
) -> None:
    bundle = materialized
    manifest = bundle.manifest
    treatment = manifest["pools"]["coverage_forced"]["entries"]
    control = manifest["pools"]["redundant_historical"]["entries"]
    heldout = manifest["pools"]["heldout_v4"]["entries"]

    assert len(treatment) == len(control) == len(heldout) == 12
    assert [item["basename"] for item in treatment] == [item["basename"] for item in control]
    assert tuple(manifest["training_strata"]) == TRAINING_STRATA
    assert tuple(manifest["heldout_strata"]) == HELDOUT_V4_STRATA
    assert set(manifest["training_strata"]).isdisjoint(manifest["heldout_strata"])
    training_environment_digests = {item["environment_digest"] for item in treatment + control}
    heldout_environment_digests = {item["environment_digest"] for item in heldout}
    assert len(heldout_environment_digests) == 12
    assert training_environment_digests.isdisjoint(heldout_environment_digests)

    for index, (left, right) in enumerate(zip(treatment, control)):
        if index % 2:
            assert left["slot_role"] == right["slot_role"] == "retained"
            assert left["environment_digest"] == right["environment_digest"]
        else:
            assert left["slot_role"] == right["slot_role"] == "swap"
            assert left["environment_digest"] != right["environment_digest"]

    assert manifest["execution_contract"]["external_actor_substitution"] == "forbidden"
    assert manifest["execution_contract"]["heldout_evaluation_model"] == (
        "final-branch-trained-checkpoint-via-selected-backend"
    )
    assert load_learner_pool_manifest(bundle.manifest_path).manifest == manifest
    with pytest.raises(LearnerPoolError, match="refusing overwrite"):
        materialize_learner_pools(_ACTOR_PLAN, bundle.root)


def test_manifest_verification_rejects_materialized_game_tampering(materialized) -> None:
    game = materialized.coverage_forced_dir / "game_000_swap.py"
    game.chmod(0o644)
    game.write_bytes(game.read_bytes() + b"\n# tampered\n")
    with pytest.raises(LearnerPoolError, match="entry digest differs"):
        load_learner_pool_manifest(materialized.root)


def test_manifest_contains_no_actor_outcome_import(materialized) -> None:
    manifest = materialized.manifest
    serialized = str(manifest).lower()
    assert "actor-outcome" not in serialized
    assert "outcomes/" not in serialized
    heldout_source = manifest["source_artifacts"]["heldout_v4"]
    assert set(heldout_source) == {
        "plan_digest",
        "cohort_digest",
        "source_manifest_digest",
        "source_manifest_leaf_count",
        "selection_digests",
    }
