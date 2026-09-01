"""Offline tests for paired and controlled marginal teachability evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from spade.core import paired_teachability as pt


GAME_SHA = pt.sha256_hex("sealed game")
HINT_SHA = pt.sha256_hex("one locked hint")
OTHER_HINT_SHA = pt.sha256_hex("a different locked hint")
RAW_SHA = pt.sha256_hex("raw starting observation")
SUCCESS_RULE_SHA = pt.sha256_hex("success iff terminal reward is positive")
POLICY_CHECKPOINT_SHA = pt.sha256_hex("paired policy checkpoint/base content")
OPTIMIZER_SHA = pt.sha256_hex("optimizer and scheduler configuration")
INITIAL_CHECKPOINT_SHA = pt.sha256_hex("checkpoint/base content")
TREATMENT_OUTPUT_SHA = pt.sha256_hex("checkpoint/treatment-post-update content")
CONTROL_OUTPUT_SHA = pt.sha256_hex("checkpoint/control-post-update content")
TRAINER_SHA = pt.sha256_hex("trainer executable and configuration")
RUNTIME_SHA = pt.sha256_hex("runtime lock and hardware contract")
TRAINING_RNG_SHA = pt.sha256_hex("training RNG algorithm and state contract")
TREATMENT_MANIFEST_SHA = pt.sha256_hex("treatment training manifest")
CONTROL_MANIFEST_SHA = pt.sha256_hex("control training manifest")


def _paired_artifact(
    patterns: list[tuple[bool, bool]],
    *,
    run_seed: int = 17,
    schedule: pt.PairSchedule | None = None,
    locked_hint_sha256: str = HINT_SHA,
    policy_checkpoint: str = "checkpoint/base",
    policy_checkpoint_sha256: str = POLICY_CHECKPOINT_SHA,
) -> pt.PairedRunArtifact:
    if schedule is None:
        schedule = pt.make_pair_schedule(
            game_sha256=GAME_SHA,
            run_seed=run_seed,
            rollout_id=3,
            pair_count=len(patterns),
        )
    assert len(patterns) == len(schedule.pairs)

    outcomes = []
    for spec, (hint_success, no_hint_success) in zip(schedule.pairs, patterns):
        observation_sha = pt.sha256_hex(f"observation:{spec.pair_id}")
        hint = pt.ArmOutcome.observed(
            arm=pt.Arm.HINT,
            environment_seed=spec.environment_seed,
            sampling_seed=spec.sampling_seed,
            policy_checkpoint=policy_checkpoint,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
            raw_observation_sha256=observation_sha,
            reward=float(hint_success),
            success=hint_success,
            turns=2,
        )
        no_hint = pt.ArmOutcome.observed(
            arm=pt.Arm.NO_HINT,
            environment_seed=spec.environment_seed,
            sampling_seed=spec.sampling_seed,
            policy_checkpoint=policy_checkpoint,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
            raw_observation_sha256=observation_sha,
            reward=float(no_hint_success),
            success=no_hint_success,
            turns=2,
        )
        outcomes.append(
            pt.PairOutcome(
                pair_id=spec.pair_id,
                pair_index=spec.pair_index,
                observed_order=spec.order.arms,
                hint=hint,
                no_hint=no_hint,
            )
        )
    return pt.make_paired_run_artifact(
        schedule=schedule,
        locked_hint_sha256=locked_hint_sha256,
        success_rule_sha256=SUCCESS_RULE_SHA,
        policy_checkpoint=policy_checkpoint,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
        pairs=outcomes,
    )


def _rebuild_paired(
    artifact: pt.PairedRunArtifact,
    pairs: list[pt.PairOutcome],
) -> pt.PairedRunArtifact:
    return pt.make_paired_run_artifact(
        schedule=artifact.schedule,
        locked_hint_sha256=artifact.locked_hint_sha256,
        success_rule_sha256=artifact.success_rule_sha256,
        policy_checkpoint=artifact.policy_checkpoint,
        policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        pairs=pairs,
    )


def _paired_family_plan(
    artifacts_by_candidate: dict[str, pt.PairedRunArtifact],
    *,
    family_alpha: float = 0.05,
    point_effect_floor: float = 0.5,
) -> pt.PairedFamilyPlan:
    pair_counts = {len(artifact.schedule.pairs) for artifact in artifacts_by_candidate.values()}
    assert len(pair_counts) == 1
    members = tuple(
        pt.make_paired_family_member(
            candidate_id=candidate_id,
            schedule=artifact.schedule,
            locked_hint_sha256=artifact.locked_hint_sha256,
            policy_checkpoint=artifact.policy_checkpoint,
            policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        )
        for candidate_id, artifact in artifacts_by_candidate.items()
    )
    return pt.make_paired_family_plan(
        family_id="sealed-family",
        members=members,
        success_rule_sha256=SUCCESS_RULE_SHA,
        pair_count=pair_counts.pop(),
        family_alpha=family_alpha,
        point_effect_floor=point_effect_floor,
    )


def _training_branch(
    branch: pt.ControlledBranch,
    *,
    initial_checkpoint: str = "checkpoint/base",
    train_token_budget: int = 10_000,
    training_example_budget: int = 64,
    optimizer_step_budget: int = 16,
    optimizer_config_sha256: str = OPTIMIZER_SHA,
    training_rng_seed: int = 91,
    training_seeds: tuple[int, ...] = (101, 102, 103, 104),
) -> pt.TrainingBranchPlan:
    return pt.TrainingBranchPlan(
        branch=branch,
        initial_checkpoint=initial_checkpoint,
        initial_checkpoint_sha256=INITIAL_CHECKPOINT_SHA,
        train_token_budget=train_token_budget,
        training_example_budget=training_example_budget,
        optimizer_step_budget=optimizer_step_budget,
        optimizer_config_sha256=optimizer_config_sha256,
        trainer_sha256=TRAINER_SHA,
        runtime_sha256=RUNTIME_SHA,
        training_rng_seed=training_rng_seed,
        training_seeds=training_seeds,
        training_rng_sha256=TRAINING_RNG_SHA,
        training_manifest_sha256=(
            TREATMENT_MANIFEST_SHA
            if branch is pt.ControlledBranch.TREATMENT
            else CONTROL_MANIFEST_SHA
        ),
    )


def _default_strata() -> tuple[pt.HoldoutStratumPlan, ...]:
    return (
        pt.HoldoutStratumPlan(
            stratum_id="same-family",
            kind=pt.HoldoutKind.SAME_FAMILY,
            family_id="candidate-family",
            sibling_of_family_id=None,
            units=(
                pt.EvaluationUnit("same-0", "same-game", 1_001, 2_001),
                pt.EvaluationUnit("same-1", "same-game", 1_002, 2_002),
            ),
        ),
        pt.HoldoutStratumPlan(
            stratum_id="sibling-family",
            kind=pt.HoldoutKind.SIBLING,
            family_id="related-family",
            sibling_of_family_id="candidate-family",
            units=(
                pt.EvaluationUnit("sibling-0", "sibling-game", 1_003, 2_003),
                pt.EvaluationUnit("sibling-1", "sibling-game", 1_004, 2_004),
            ),
        ),
    )


def _controlled_protocol(
    *,
    treatment: pt.TrainingBranchPlan | None = None,
    control: pt.TrainingBranchPlan | None = None,
    holdout_strata: tuple[pt.HoldoutStratumPlan, ...] | None = None,
) -> pt.ControlledMarginalProtocol:
    return pt.make_controlled_marginal_protocol(
        protocol_id="candidate-17-replicate-2",
        candidate_family_id="candidate-family",
        treatment=treatment or _training_branch(pt.ControlledBranch.TREATMENT),
        control=control or _training_branch(pt.ControlledBranch.CONTROL),
        holdout_strata=holdout_strata if holdout_strata is not None else _default_strata(),
    )


def _branch_receipt(
    protocol: pt.ControlledMarginalProtocol,
    branch: pt.ControlledBranch,
    output_checkpoint_sha256: str,
    **realized_overrides: object,
) -> pt.BranchExecutionReceipt:
    plan = protocol.treatment if branch is pt.ControlledBranch.TREATMENT else protocol.control
    values: dict[str, object] = {
        "realized_train_token_budget": plan.train_token_budget,
        "realized_training_example_budget": plan.training_example_budget,
        "realized_optimizer_step_budget": plan.optimizer_step_budget,
        "optimizer_config_sha256": plan.optimizer_config_sha256,
        "trainer_sha256": plan.trainer_sha256,
        "runtime_sha256": plan.runtime_sha256,
        "training_rng_sha256": plan.training_rng_sha256,
        "training_manifest_sha256": plan.training_manifest_sha256,
    }
    values.update(realized_overrides)
    return pt.make_branch_execution_receipt(
        protocol=protocol,
        branch=branch,
        output_checkpoint_sha256=output_checkpoint_sha256,
        **values,  # type: ignore[arg-type]
    )


def _controlled_artifact(
    *,
    protocol: pt.ControlledMarginalProtocol | None = None,
    patterns: dict[str, tuple[bool, bool]] | None = None,
) -> pt.ControlledMarginalArtifact:
    protocol = protocol or _controlled_protocol()
    patterns = patterns or {
        "same-0": (True, False),
        "same-1": (True, False),
        "sibling-0": (False, True),
        "sibling-1": (False, False),
    }
    treatment_receipt = _branch_receipt(
        protocol,
        pt.ControlledBranch.TREATMENT,
        TREATMENT_OUTPUT_SHA,
    )
    control_receipt = _branch_receipt(
        protocol,
        pt.ControlledBranch.CONTROL,
        CONTROL_OUTPUT_SHA,
    )
    pairs = []
    for stratum in protocol.holdout_strata:
        for unit in stratum.units:
            treatment_success, control_success = patterns[unit.eval_id]
            pairs.append(
                pt.ControlledOutcomePair(
                    eval_id=unit.eval_id,
                    treatment=pt.BranchOutcome.observed(
                        branch=pt.ControlledBranch.TREATMENT,
                        unit=unit,
                        output_checkpoint_sha256=TREATMENT_OUTPUT_SHA,
                        reward=float(treatment_success),
                        success=treatment_success,
                    ),
                    control=pt.BranchOutcome.observed(
                        branch=pt.ControlledBranch.CONTROL,
                        unit=unit,
                        output_checkpoint_sha256=CONTROL_OUTPUT_SHA,
                        reward=float(control_success),
                        success=control_success,
                    ),
                )
            )
    return pt.make_controlled_marginal_artifact(
        protocol=protocol,
        treatment_receipt=treatment_receipt,
        control_receipt=control_receipt,
        pairs=pairs,
    )


def _rebuild_controlled(
    artifact: pt.ControlledMarginalArtifact,
    pairs: list[pt.ControlledOutcomePair],
) -> pt.ControlledMarginalArtifact:
    return pt.make_controlled_marginal_artifact(
        protocol=artifact.protocol,
        treatment_receipt=artifact.treatment_receipt,
        control_receipt=artifact.control_receipt,
        pairs=pairs,
    )


def test_pair_schedule_is_deterministic_balanced_and_content_addressed() -> None:
    first = pt.make_pair_schedule(
        game_sha256=pt.sha256_hex("game"),
        run_seed=17,
        rollout_id=3,
        pair_count=4,
    )
    second = pt.make_pair_schedule(
        game_sha256=pt.sha256_hex("game"),
        run_seed=17,
        rollout_id=3,
        pair_count=4,
    )

    assert first == second
    assert first.digest == "ddf83ecf85b8289fd6bbfab8bb56e834fe0123f402c42f39c7fa5eff2a8fdaaa"
    assert [pair.order for pair in first.pairs].count(pt.ArmOrder.HINT_FIRST) == 2
    assert [pair.order for pair in first.pairs].count(pt.ArmOrder.NO_HINT_FIRST) == 2
    assert len({pair.environment_seed for pair in first.pairs}) == 4
    assert len({pair.sampling_seed for pair in first.pairs}) == 4
    assert [
        (
            pair.pair_id,
            pair.environment_seed,
            pair.sampling_seed,
            pair.order.value,
        )
        for pair in first.pairs
    ] == [
        ("8fb7be5259009a55df313285", 192765089, 1846103963, "hint_first"),
        ("b7e3b5216f97a9e6975feb96", 629285362, 1270293191, "no_hint_first"),
        ("2e16362ece8eb4b6c6dfbac3", 1318438521, 1068035737, "hint_first"),
        ("061b87a959ed291e0c160209", 717575439, 1728551844, "no_hint_first"),
    ]
    pt.validate_pair_schedule(first)


def test_pair_schedule_changes_with_every_sealed_identity_component() -> None:
    base = pt.make_pair_schedule(
        game_sha256=GAME_SHA,
        run_seed=17,
        rollout_id=3,
        pair_count=4,
    )
    changed_game = pt.make_pair_schedule(
        game_sha256=pt.sha256_hex("another game"),
        run_seed=17,
        rollout_id=3,
        pair_count=4,
    )
    changed_run = pt.make_pair_schedule(
        game_sha256=GAME_SHA,
        run_seed=18,
        rollout_id=3,
        pair_count=4,
    )
    changed_rollout = pt.make_pair_schedule(
        game_sha256=GAME_SHA,
        run_seed=17,
        rollout_id=4,
        pair_count=4,
    )

    assert len({base.digest, changed_game.digest, changed_run.digest, changed_rollout.digest}) == 4


@pytest.mark.parametrize("pair_count", [0, 1, 3, 5])
def test_pair_schedule_rejects_counts_that_cannot_be_balanced(pair_count: int) -> None:
    with pytest.raises(ValueError, match="pair_count"):
        pt.make_pair_schedule(
            game_sha256=GAME_SHA,
            run_seed=1,
            rollout_id=1,
            pair_count=pair_count,
        )


def test_pair_schedule_validation_detects_tampering() -> None:
    schedule = pt.make_pair_schedule(
        game_sha256=GAME_SHA,
        run_seed=17,
        rollout_id=3,
        pair_count=4,
    )
    first = schedule.pairs[0]
    wrong_order = (
        pt.ArmOrder.NO_HINT_FIRST
        if first.order is pt.ArmOrder.HINT_FIRST
        else pt.ArmOrder.HINT_FIRST
    )
    tampered = replace(
        schedule,
        pairs=(replace(first, order=wrong_order), *schedule.pairs[1:]),
    )

    with pytest.raises(pt.ArtifactIntegrityError, match="canonical SHA-256 schedule"):
        pt.validate_pair_schedule(tampered)


@pytest.mark.parametrize(
    ("favorable", "unfavorable", "expected"),
    [
        (0, 0, 1.0),
        (5, 0, 1 / 32),
        (4, 0, 1 / 16),
        (3, 2, 0.5),
        (0, 5, 1.0),
    ],
)
def test_exact_one_sided_mcnemar(
    favorable: int,
    unfavorable: int,
    expected: float,
) -> None:
    assert pt.exact_mcnemar_one_sided(favorable, unfavorable) == pytest.approx(expected)


@pytest.mark.parametrize(("favorable", "unfavorable"), [(-1, 0), (0, -1), (True, 0)])
def test_exact_one_sided_mcnemar_rejects_invalid_counts(
    favorable: int,
    unfavorable: int,
) -> None:
    with pytest.raises(ValueError):
        pt.exact_mcnemar_one_sided(favorable, unfavorable)


def test_exact_one_sided_mcnemar_rejects_unrepresentable_pair_counts() -> None:
    with pytest.raises(ValueError, match="representable p-value"):
        pt.exact_mcnemar_one_sided(1_075, 0)


def test_complete_pair_artifact_estimates_only_within_seeded_pairs() -> None:
    artifact = _paired_artifact(
        [
            (True, False),
            (True, False),
            (True, False),
            (True, False),
            (True, False),
            (False, False),
        ]
    )

    pt.validate_complete_pair_artifact(artifact)
    estimate = pt.assess_paired_artifact(artifact, effect_floor=0.5)

    assert estimate.complete_pairs == 6
    assert estimate.hint_only_success == 5
    assert estimate.no_hint_only_success == 0
    assert estimate.both_success == 0
    assert estimate.both_failure == 1
    assert estimate.ate_success == pytest.approx(5 / 6)
    assert estimate.mean_reward_delta == pytest.approx(5 / 6)
    assert estimate.p_hint_better == pytest.approx(1 / 32)
    assert estimate.p_no_hint_better == 1.0
    assert estimate.decision is pt.TeachabilityDecision.PENDING_FAMILY
    assert estimate.credit == 0.0
    assert estimate.locked_hint_sha256 == HINT_SHA


def test_estimate_identity_distinguishes_locked_hints_on_the_same_schedule() -> None:
    patterns = [(True, False), (True, False), (False, False), (False, False)]
    first = _paired_artifact(patterns)
    second = _paired_artifact(
        patterns,
        schedule=first.schedule,
        locked_hint_sha256=OTHER_HINT_SHA,
    )

    first_estimate = pt.assess_paired_artifact(first)
    second_estimate = pt.assess_paired_artifact(second)

    assert first_estimate.schedule_digest == second_estimate.schedule_digest
    assert first_estimate.estimate_id != second_estimate.estimate_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("order", "execution order"),
        ("environment_seed", "environment seed"),
        ("sampling_seed", "sampling seed"),
        ("checkpoint", "policy checkpoint"),
        ("checkpoint_digest", "policy checkpoint digest"),
        ("observation", "same raw observation"),
        ("arm", "canonical serialization failed"),
    ],
)
def test_pair_artifact_rejects_broken_pair_identity(mutation: str, message: str) -> None:
    artifact = _paired_artifact([(True, False), (False, False)])
    pairs = list(artifact.pairs)
    first = pairs[0]
    if mutation == "order":
        first = replace(
            first,
            observed_order=tuple(reversed(first.observed_order)),  # type: ignore[arg-type]
        )
    elif mutation == "environment_seed":
        first = replace(
            first,
            hint=replace(first.hint, environment_seed=first.hint.environment_seed + 1),
        )
    elif mutation == "sampling_seed":
        first = replace(
            first,
            no_hint=replace(first.no_hint, sampling_seed=first.no_hint.sampling_seed + 1),
        )
    elif mutation == "checkpoint":
        first = replace(first, hint=replace(first.hint, policy_checkpoint="checkpoint/other"))
    elif mutation == "checkpoint_digest":
        first = replace(
            first,
            hint=replace(
                first.hint,
                policy_checkpoint_sha256=pt.sha256_hex("different checkpoint bytes"),
            ),
        )
    elif mutation == "observation":
        first = replace(
            first,
            no_hint=replace(first.no_hint, raw_observation_sha256=RAW_SHA),
        )
    else:
        first = replace(first, hint=replace(first.hint, arm="hint"))  # type: ignore[arg-type]
    pairs[0] = first
    malformed = (
        _rebuild_paired(artifact, pairs)
        if mutation != "arm"
        else replace(artifact, pairs=tuple(pairs))
    )

    with pytest.raises(pt.ArtifactIntegrityError, match=message):
        pt.validate_complete_pair_artifact(malformed)
    assert pt.assess_paired_artifact(malformed).decision is pt.TeachabilityDecision.INVALID_ARTIFACT


def test_pair_artifact_missing_or_unavailable_arm_abstains_without_imputation() -> None:
    artifact = _paired_artifact([(True, False), (False, False)])
    stale_missing = replace(artifact, pairs=artifact.pairs[:-1])
    assert (
        pt.assess_paired_artifact(stale_missing).decision
        is pt.TeachabilityDecision.INVALID_ARTIFACT
    )
    missing = _rebuild_paired(artifact, list(artifact.pairs[:-1]))
    assert pt.assess_paired_artifact(missing).decision is pt.TeachabilityDecision.ABSTAIN_INCOMPLETE

    pairs = list(artifact.pairs)
    first = pairs[0]
    unavailable = pt.ArmOutcome.unavailable(
        arm=pt.Arm.HINT,
        environment_seed=first.hint.environment_seed,
        sampling_seed=first.hint.sampling_seed,
        policy_checkpoint=first.hint.policy_checkpoint,
        policy_checkpoint_sha256=first.hint.policy_checkpoint_sha256,
        error="worker timeout",
        failure_classification=pt.FailureClassification.EXOGENOUS_INFRASTRUCTURE,
    )
    pairs[0] = replace(first, hint=unavailable)
    stale_unavailable = replace(artifact, pairs=tuple(pairs))
    assert (
        pt.assess_paired_artifact(stale_unavailable).decision
        is pt.TeachabilityDecision.INVALID_ARTIFACT
    )
    incomplete = _rebuild_paired(artifact, pairs)
    estimate = pt.assess_paired_artifact(incomplete)

    assert estimate.decision is pt.TeachabilityDecision.ABSTAIN_INCOMPLETE
    assert estimate.complete_pairs == 0
    assert estimate.ate_success is None
    assert estimate.credit == 0.0


def test_unavailable_pair_arm_with_imputed_result_is_invalid() -> None:
    artifact = _paired_artifact([(True, False), (False, False)])
    pairs = list(artifact.pairs)
    first = pairs[0]
    unavailable = pt.ArmOutcome.unavailable(
        arm=pt.Arm.HINT,
        environment_seed=first.hint.environment_seed,
        sampling_seed=first.hint.sampling_seed,
        policy_checkpoint=first.hint.policy_checkpoint,
        policy_checkpoint_sha256=first.hint.policy_checkpoint_sha256,
        error="worker timeout",
        failure_classification=pt.FailureClassification.EXOGENOUS_INFRASTRUCTURE,
    )
    pairs[0] = replace(first, hint=replace(unavailable, reward=0.0))
    malformed = _rebuild_paired(artifact, pairs)

    assert pt.assess_paired_artifact(malformed).decision is pt.TeachabilityDecision.INVALID_ARTIFACT


def test_pair_artifact_digest_and_locked_hint_are_verified() -> None:
    artifact = _paired_artifact([(True, False), (False, False)])
    tampered_digest = replace(artifact, digest=pt.sha256_hex("wrong artifact"))
    bad_hint = replace(artifact, locked_hint_sha256="not-a-digest")

    assert (
        pt.assess_paired_artifact(tampered_digest).decision
        is pt.TeachabilityDecision.INVALID_ARTIFACT
    )
    assert pt.assess_paired_artifact(bad_hint).decision is pt.TeachabilityDecision.INVALID_ARTIFACT


def test_holm_correction_counts_abstentions_and_stops_step_down() -> None:
    strong = pt.assess_paired_artifact(
        _paired_artifact([(True, False)] * 8, run_seed=1),
        effect_floor=0.5,
    )
    medium = pt.assess_paired_artifact(
        _paired_artifact([(True, False)] * 6 + [(False, False)] * 2, run_seed=2),
        effect_floor=0.5,
    )
    incomplete_artifact = _paired_artifact([(True, False), (False, False)], run_seed=3)
    incomplete = pt.assess_paired_artifact(
        _rebuild_paired(incomplete_artifact, list(incomplete_artifact.pairs[:-1]))
    )

    corrected = pt._apply_holm_family_unsealed(
        [strong, medium, incomplete],
        family_id="sealed-candidate-family",
        family_alpha=0.05,
    )

    assert corrected[0].decision is pt.TeachabilityDecision.SCREEN_CREDIT
    assert corrected[0].credit == 1.0
    assert corrected[0].directional_alpha == pytest.approx(0.025)
    assert corrected[0].holm_threshold == pytest.approx(0.025 / 3)
    assert corrected[1].p_hint_better == pytest.approx(1 / 64)
    assert corrected[1].holm_threshold == pytest.approx(0.025 / 2)
    assert corrected[1].holm_rejected is False
    assert corrected[1].decision is pt.TeachabilityDecision.ABSTAIN_UNCERTAIN
    assert corrected[2].decision is pt.TeachabilityDecision.ABSTAIN_INCOMPLETE
    assert corrected[2].family_size == 3
    assert corrected[2].holm_rank == 3

    without_abstention = pt._apply_holm_family_unsealed(
        [strong, medium],
        family_id="post-hoc-smaller-family",
        family_alpha=0.05,
    )
    assert without_abstention[1].holm_rejected is True
    assert without_abstention[1].decision is pt.TeachabilityDecision.SCREEN_CREDIT


def test_negative_direction_uses_multiplicity_correction_and_quality_rejects() -> None:
    strong_harm = pt.assess_paired_artifact(_paired_artifact([(False, True)] * 8, run_seed=11))
    medium_harm = pt.assess_paired_artifact(
        _paired_artifact([(False, True)] * 6 + [(False, False)] * 2, run_seed=12)
    )
    incomplete_artifact = _paired_artifact([(False, True), (False, False)], run_seed=13)
    incomplete = pt.assess_paired_artifact(
        _rebuild_paired(incomplete_artifact, list(incomplete_artifact.pairs[:-1]))
    )

    corrected = pt._apply_holm_family_unsealed(
        [strong_harm, medium_harm, incomplete],
        family_id="sealed-harm-family",
        family_alpha=0.05,
    )

    assert corrected[0].decision is pt.TeachabilityDecision.HARMFUL
    assert corrected[0].quality_reject is True
    assert corrected[0].credit == 0.0
    assert corrected[0].holm_rejected is False
    assert corrected[0].harm_holm_rejected is True
    assert corrected[0].harm_holm_threshold == pytest.approx(0.025 / 3)
    assert corrected[1].p_no_hint_better == pytest.approx(1 / 64)
    assert corrected[1].harm_holm_threshold == pytest.approx(0.025 / 2)
    assert corrected[1].harm_holm_rejected is False
    assert corrected[1].decision is pt.TeachabilityDecision.ABSTAIN_UNCERTAIN
    assert corrected[1].quality_reject is False
    assert corrected[2].decision is pt.TeachabilityDecision.ABSTAIN_INCOMPLETE
    assert corrected[2].harm_holm_rank == 3

    without_abstention = pt._apply_holm_family_unsealed(
        [strong_harm, medium_harm],
        family_id="post-hoc-smaller-harm-family",
        family_alpha=0.05,
    )
    assert without_abstention[1].harm_holm_rejected is True
    assert without_abstention[1].decision is pt.TeachabilityDecision.HARMFUL
    assert without_abstention[1].quality_reject is True


def test_holm_requires_significance_and_precommitted_effect_floor_for_credit() -> None:
    estimate = pt.assess_paired_artifact(
        _paired_artifact([(True, False)] * 7 + [(False, False)]),
        effect_floor=0.9,
    )
    (corrected,) = pt._apply_holm_family_unsealed([estimate], family_id="one-candidate")

    assert corrected.holm_rejected is True
    assert corrected.ate_success == pytest.approx(0.875)
    assert corrected.decision is pt.TeachabilityDecision.ABSTAIN_BELOW_EFFECT
    assert corrected.credit == 0.0
    assert "screening floor" in corrected.reason


def test_holm_rejects_duplicate_estimate_ids() -> None:
    estimate = pt.assess_paired_artifact(_paired_artifact([(True, False)] * 4))
    with pytest.raises(ValueError, match="unique estimate_id"):
        pt._apply_holm_family_unsealed([estimate, estimate], family_id="bad-family")


def test_paired_metadata_is_json_serializable_and_carries_decision_audit_fields() -> None:
    estimate = pt.assess_paired_artifact(_paired_artifact([(True, False)] * 6))
    (corrected,) = pt._apply_holm_family_unsealed([estimate], family_id="family-1")
    metadata = corrected.to_metadata()

    json.dumps(metadata, allow_nan=False)
    assert metadata["decision"] == "screen_credit"
    assert metadata["locked_hint_sha256"] == HINT_SHA
    assert metadata["family_id"] == "family-1"
    assert metadata["holm_rejected"] is True
    assert metadata["harm_holm_rejected"] is False
    assert metadata["quality_reject"] is False
    assert "screening credit" in metadata["reason"]


def test_sealed_family_plan_drives_membership_floor_alpha_and_holm_decisions() -> None:
    schedules = {
        "candidate-a": pt.make_pair_schedule(
            game_sha256=GAME_SHA,
            run_seed=21,
            rollout_id=3,
            pair_count=8,
        ),
        "candidate-b": pt.make_pair_schedule(
            game_sha256=GAME_SHA,
            run_seed=22,
            rollout_id=3,
            pair_count=8,
        ),
    }
    members = tuple(
        pt.make_paired_family_member(
            candidate_id=candidate_id,
            schedule=schedule,
            locked_hint_sha256=HINT_SHA,
            policy_checkpoint="checkpoint/base",
            policy_checkpoint_sha256=POLICY_CHECKPOINT_SHA,
        )
        for candidate_id, schedule in schedules.items()
    )
    plan = pt.make_paired_family_plan(
        family_id="sealed-family",
        members=members,
        success_rule_sha256=SUCCESS_RULE_SHA,
        pair_count=8,
        point_effect_floor=0.5,
    )
    # Outcomes are constructed only after the plan bytes and digest are fixed.
    artifacts = {
        "candidate-a": _paired_artifact(
            [(True, False)] * 8,
            schedule=schedules["candidate-a"],
        ),
        "candidate-b": _paired_artifact(
            [(True, False)] * 6 + [(False, False)] * 2,
            schedule=schedules["candidate-b"],
        ),
    }

    pt.validate_paired_family_plan(plan)
    results = pt.assess_paired_family(plan, artifacts)

    assert plan.correction_name == pt.HOLM_DIRECTIONAL_BONFERRONI
    assert plan.pair_count == 8
    assert [result.decision for result in results] == [
        pt.TeachabilityDecision.SCREEN_CREDIT,
        pt.TeachabilityDecision.SCREEN_CREDIT,
    ]
    assert all(result.effect_floor == 0.5 for result in results)
    assert all(result.family_alpha == 0.05 for result in results)
    assert all(result.family_plan_digest == plan.digest for result in results)
    assert results[0].holm_threshold == pytest.approx(0.025 / 2)
    assert results[1].holm_threshold == pytest.approx(0.025)
    assert results[0].to_metadata()["decision"] == "screen_credit"


def test_unsealed_holm_correction_is_not_a_public_api() -> None:
    assert not hasattr(pt, "apply_holm_family")
    assert "apply_holm_family" not in pt.__all__
    assert "assess_paired_family" in pt.__all__


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family_alpha", 0.10, "digest mismatch"),
        ("point_effect_floor", 0.75, "digest mismatch"),
        ("success_rule_sha256", pt.sha256_hex("post-hoc success rule"), "digest mismatch"),
        ("correction_name", "post-hoc-correction", "unsupported correction_name"),
    ],
)
def test_family_plan_rejects_posthoc_decision_rule_changes(
    field: str,
    value: object,
    message: str,
) -> None:
    artifacts = {"candidate-a": _paired_artifact([(True, False)] * 8)}
    plan = _paired_family_plan(artifacts)
    tampered = replace(plan, **{field: value})  # type: ignore[arg-type]

    with pytest.raises(pt.ArtifactIntegrityError, match=message):
        pt.assess_paired_family(tampered, artifacts)


def test_family_assessment_rejects_posthoc_subsets_extras_and_member_plan_tampering() -> None:
    artifacts = {
        "candidate-a": _paired_artifact([(True, False)] * 8, run_seed=31),
        "candidate-b": _paired_artifact([(False, False)] * 8, run_seed=32),
    }
    plan = _paired_family_plan(artifacts)

    with pytest.raises(pt.ArtifactIntegrityError, match="missing members"):
        pt.assess_paired_family(plan, {"candidate-a": artifacts["candidate-a"]})
    with pytest.raises(pt.ArtifactIntegrityError, match="extra members"):
        pt.assess_paired_family(
            plan,
            {**artifacts, "post-hoc-extra": artifacts["candidate-a"]},
        )
    with pytest.raises(pt.ArtifactIntegrityError, match="digest mismatch"):
        pt.assess_paired_family(replace(plan, members=plan.members[:-1]), artifacts)


def test_family_plan_rejects_k_that_cannot_pass_split_first_holm_threshold() -> None:
    impossible_schedule = pt.make_pair_schedule(
        game_sha256=GAME_SHA,
        run_seed=41,
        rollout_id=1,
        pair_count=4,
    )
    impossible_member = pt.make_paired_family_member(
        candidate_id="candidate-a",
        schedule=impossible_schedule,
        locked_hint_sha256=HINT_SHA,
        policy_checkpoint="checkpoint/base",
        policy_checkpoint_sha256=POLICY_CHECKPOINT_SHA,
    )

    with pytest.raises(pt.ArtifactIntegrityError, match=r"2\^-K"):
        pt.make_paired_family_plan(
            family_id="underpowered-family",
            members=(impossible_member,),
            success_rule_sha256=SUCCESS_RULE_SHA,
            pair_count=4,
        )

    feasible_schedule = pt.make_pair_schedule(
        game_sha256=GAME_SHA,
        run_seed=42,
        rollout_id=1,
        pair_count=6,
    )
    feasible_member = pt.make_paired_family_member(
        candidate_id="candidate-a",
        schedule=feasible_schedule,
        locked_hint_sha256=HINT_SHA,
        policy_checkpoint="checkpoint/base",
        policy_checkpoint_sha256=POLICY_CHECKPOINT_SHA,
    )
    feasible = pt.make_paired_family_plan(
        family_id="just-powered-family",
        members=(feasible_member,),
        success_rule_sha256=SUCCESS_RULE_SHA,
        pair_count=6,
    )
    assert 2**-feasible.pair_count <= feasible.family_alpha / 2


def test_family_plan_requires_one_policy_checkpoint_across_all_candidates() -> None:
    first = _paired_artifact([(False, False)] * 8, run_seed=51)
    second = _paired_artifact(
        [(False, False)] * 8,
        run_seed=52,
        policy_checkpoint="checkpoint/later",
    )
    members = tuple(
        pt.make_paired_family_member(
            candidate_id=candidate_id,
            schedule=artifact.schedule,
            locked_hint_sha256=artifact.locked_hint_sha256,
            policy_checkpoint=artifact.policy_checkpoint,
            policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        )
        for candidate_id, artifact in {"first": first, "second": second}.items()
    )

    with pytest.raises(pt.ArtifactIntegrityError, match="one identical policy checkpoint"):
        pt.make_paired_family_plan(
            family_id="confounded-family",
            members=members,
            success_rule_sha256=SUCCESS_RULE_SHA,
            pair_count=8,
        )


def test_family_plan_requires_one_policy_checkpoint_digest_across_candidates() -> None:
    first = _paired_artifact([(False, False)] * 8, run_seed=61)
    second = _paired_artifact(
        [(False, False)] * 8,
        run_seed=62,
        policy_checkpoint_sha256=pt.sha256_hex("different bytes under the same label"),
    )
    members = tuple(
        pt.make_paired_family_member(
            candidate_id=candidate_id,
            schedule=artifact.schedule,
            locked_hint_sha256=artifact.locked_hint_sha256,
            policy_checkpoint=artifact.policy_checkpoint,
            policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        )
        for candidate_id, artifact in {"first": first, "second": second}.items()
    )

    with pytest.raises(pt.ArtifactIntegrityError, match="policy checkpoint digest"):
        pt.make_paired_family_plan(
            family_id="byte-confounded-family",
            members=members,
            success_rule_sha256=SUCCESS_RULE_SHA,
            pair_count=8,
        )


def test_holm_feasibility_rejects_subnormal_threshold_underflow() -> None:
    members = tuple(
        pt.PairedFamilyMember(
            candidate_id=f"candidate-{index}",
            game_sha256=pt.sha256_hex(f"game-{index}"),
            schedule_digest=pt.sha256_hex(f"schedule-{index}"),
            locked_hint_sha256=HINT_SHA,
            policy_checkpoint="checkpoint/base",
            policy_checkpoint_sha256=POLICY_CHECKPOINT_SHA,
            pair_count=1_074,
        )
        for index in range(3)
    )

    with pytest.raises(pt.ArtifactIntegrityError, match="floating-point resolution"):
        pt.make_paired_family_plan(
            family_id="subnormal-threshold-family",
            members=members,
            success_rule_sha256=SUCCESS_RULE_SHA,
            pair_count=1_074,
            family_alpha=5e-324,
        )


def test_family_assessment_rejects_artifact_identity_changes() -> None:
    artifact = _paired_artifact([(True, False)] * 6)
    plan = _paired_family_plan({"candidate-a": artifact})
    changed_rule = replace(artifact, success_rule_sha256=pt.sha256_hex("changed after outcomes"))

    with pytest.raises(pt.ArtifactIntegrityError, match="success-rule digest"):
        pt.assess_paired_family(plan, {"candidate-a": changed_rule})


def test_candidate_caused_paired_failure_is_quality_reject_not_missingness() -> None:
    artifact = _paired_artifact([(False, False)] * 6)
    pairs = list(artifact.pairs)
    first = pairs[0]
    pairs[0] = replace(
        first,
        hint=pt.ArmOutcome.unavailable(
            arm=pt.Arm.HINT,
            environment_seed=first.hint.environment_seed,
            sampling_seed=first.hint.sampling_seed,
            policy_checkpoint=first.hint.policy_checkpoint,
            policy_checkpoint_sha256=first.hint.policy_checkpoint_sha256,
            error="hint payload crashed the candidate environment",
            failure_classification=pt.FailureClassification.CANDIDATE_OR_INTERVENTION,
        ),
    )
    stale_digest_failure = replace(artifact, pairs=tuple(pairs))
    failed = _rebuild_paired(artifact, pairs)
    plan = _paired_family_plan({"candidate-a": artifact})

    assert (
        pt.assess_paired_artifact(stale_digest_failure).decision
        is pt.TeachabilityDecision.INVALID_ARTIFACT
    )
    direct = pt.assess_paired_artifact(failed)
    (family_result,) = pt.assess_paired_family(plan, {"candidate-a": failed})

    assert direct.decision is pt.TeachabilityDecision.QUALITY_REJECT
    assert direct.quality_reject is True
    assert family_result.decision is pt.TeachabilityDecision.QUALITY_REJECT
    assert family_result.quality_reject is True
    assert family_result.credit == 0.0


def test_unavailable_paired_outcome_without_failure_classification_is_invalid() -> None:
    artifact = _paired_artifact([(False, False)] * 2)
    pairs = list(artifact.pairs)
    first = pairs[0]
    unavailable = pt.ArmOutcome.unavailable(
        arm=pt.Arm.HINT,
        environment_seed=first.hint.environment_seed,
        sampling_seed=first.hint.sampling_seed,
        policy_checkpoint=first.hint.policy_checkpoint,
        policy_checkpoint_sha256=first.hint.policy_checkpoint_sha256,
        error="unclassified failure",
        failure_classification=pt.FailureClassification.EXOGENOUS_INFRASTRUCTURE,
    )
    pairs[0] = replace(first, hint=replace(unavailable, failure_classification=None))

    assert (
        pt.assess_paired_artifact(_rebuild_paired(artifact, pairs)).decision
        is pt.TeachabilityDecision.INVALID_ARTIFACT
    )


def test_unavailable_paired_outcome_with_malformed_enum_is_invalid_not_quality_reject() -> None:
    artifact = _paired_artifact([(False, False)] * 2)
    pairs = list(artifact.pairs)
    first = pairs[0]
    unavailable = pt.ArmOutcome.unavailable(
        arm=pt.Arm.HINT,
        environment_seed=first.hint.environment_seed,
        sampling_seed=first.hint.sampling_seed,
        policy_checkpoint=first.hint.policy_checkpoint,
        policy_checkpoint_sha256=first.hint.policy_checkpoint_sha256,
        error="candidate failure with malformed arm",
        failure_classification=pt.FailureClassification.CANDIDATE_OR_INTERVENTION,
    )
    pairs[0] = replace(
        first,
        hint=replace(unavailable, arm="hint"),  # type: ignore[arg-type]
    )
    assessment = pt.assess_paired_artifact(replace(artifact, pairs=tuple(pairs)))

    assert assessment.decision is pt.TeachabilityDecision.INVALID_ARTIFACT
    assert "canonical serialization failed" in assessment.reason


def test_controlled_protocol_accepts_only_a_fully_locked_causal_comparison() -> None:
    protocol = _controlled_protocol()
    validation = pt.validate_controlled_marginal_protocol(protocol)

    assert validation.decision is pt.ProtocolDecision.READY
    assert validation.valid is True
    assert validation.train_seed_count == 4
    assert validation.eval_unit_count == 4
    assert validation.holdout_kinds == ("same_family", "sibling")
    json.dumps(validation.to_metadata(), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_checkpoint", "checkpoint/other", "same checkpoint"),
        (
            "initial_checkpoint_sha256",
            pt.sha256_hex("other initial checkpoint"),
            "initial checkpoint digests",
        ),
        ("train_token_budget", 9_999, "token budgets"),
        ("training_example_budget", 63, "example budgets"),
        ("optimizer_step_budget", 15, "optimizer step budgets"),
        ("optimizer_config_sha256", pt.sha256_hex("other optimizer"), "configurations"),
        ("trainer_sha256", pt.sha256_hex("other trainer"), "trainer digests"),
        ("runtime_sha256", pt.sha256_hex("other runtime"), "runtime digests"),
        ("training_rng_seed", 92, "training RNG seeds"),
        ("training_seeds", (101, 102, 103, 105), "training seed schedules"),
        ("training_rng_sha256", pt.sha256_hex("other RNG contract"), "RNG digests"),
    ],
)
def test_controlled_protocol_rejects_branch_confounds(
    field: str,
    value: object,
    message: str,
) -> None:
    control = replace(
        _training_branch(pt.ControlledBranch.CONTROL),
        **{field: value},  # type: ignore[arg-type]
    )
    protocol = _controlled_protocol(control=control)
    validation = pt.validate_controlled_marginal_protocol(protocol)

    assert validation.decision is pt.ProtocolDecision.INVALID
    assert any(message in error for error in validation.errors)


def test_controlled_protocol_requires_distinct_treatment_and_control_manifests() -> None:
    control = replace(
        _training_branch(pt.ControlledBranch.CONTROL),
        training_manifest_sha256=TREATMENT_MANIFEST_SHA,
    )
    validation = pt.validate_controlled_marginal_protocol(_controlled_protocol(control=control))

    assert validation.decision is pt.ProtocolDecision.INVALID
    assert any("manifest digests must differ" in error for error in validation.errors)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("environment_seed", "environment seeds overlap"),
        ("sampling_seed", "sampling seeds overlap"),
    ],
)
def test_controlled_protocol_requires_disjoint_train_and_eval_seeds(
    field: str,
    message: str,
) -> None:
    strata = list(_default_strata())
    units = list(strata[0].units)
    units[0] = replace(units[0], **{field: 101})  # type: ignore[arg-type]
    strata[0] = replace(strata[0], units=tuple(units))
    protocol = _controlled_protocol(holdout_strata=tuple(strata))
    validation = pt.validate_controlled_marginal_protocol(protocol)

    assert validation.decision is pt.ProtocolDecision.INVALID
    assert any(message in error for error in validation.errors)


def test_controlled_protocol_requires_both_holdout_kinds_and_valid_family_relations() -> None:
    same, sibling = _default_strata()
    missing_sibling = _controlled_protocol(holdout_strata=(same,))
    wrong_same_family = _controlled_protocol(
        holdout_strata=(replace(same, family_id="other-family"), sibling)
    )
    wrong_sibling_relation = _controlled_protocol(
        holdout_strata=(same, replace(sibling, sibling_of_family_id="unrelated-family"))
    )

    assert any(
        "requires a sibling-family" in error
        for error in pt.validate_controlled_marginal_protocol(missing_sibling).errors
    )
    assert any(
        "must use candidate family" in error
        for error in pt.validate_controlled_marginal_protocol(wrong_same_family).errors
    )
    assert any(
        "must name the candidate family" in error
        for error in pt.validate_controlled_marginal_protocol(wrong_sibling_relation).errors
    )


def test_controlled_protocol_rejects_duplicate_holdout_randomization_units() -> None:
    same, sibling = _default_strata()
    duplicated = replace(
        sibling.units[0],
        eval_id="duplicate-under-another-id",
        environment_id=same.units[0].environment_id,
        environment_seed=same.units[0].environment_seed,
        sampling_seed=same.units[0].sampling_seed,
    )
    sibling = replace(sibling, units=(duplicated, *sibling.units[1:]))
    protocol = _controlled_protocol(holdout_strata=(same, sibling))
    validation = pt.validate_controlled_marginal_protocol(protocol)

    assert validation.decision is pt.ProtocolDecision.INVALID
    assert any("environment/seed/sampling tuples" in error for error in validation.errors)


def test_controlled_protocol_rejects_zero_budgets_and_digest_tampering() -> None:
    zero_examples = replace(
        _training_branch(pt.ControlledBranch.TREATMENT),
        training_example_budget=0,
    )
    invalid_budget = _controlled_protocol(treatment=zero_examples)
    valid = _controlled_protocol()
    tampered = replace(valid, digest=pt.sha256_hex("tampered protocol"))

    assert any(
        "training_example_budget" in error
        for error in pt.validate_controlled_marginal_protocol(invalid_budget).errors
    )
    assert any(
        "digest mismatch" in error
        for error in pt.validate_controlled_marginal_protocol(tampered).errors
    )


def test_malformed_protocol_enums_fail_closed_without_attribute_error() -> None:
    protocol = _controlled_protocol()
    malformed_branch = replace(
        protocol,
        treatment=replace(
            protocol.treatment,
            branch="treatment",  # type: ignore[arg-type]
        ),
    )
    same, sibling = protocol.holdout_strata
    malformed_kind = replace(
        protocol,
        holdout_strata=(
            replace(same, kind="same_family"),  # type: ignore[arg-type]
            sibling,
        ),
    )

    branch_validation = pt.validate_controlled_marginal_protocol(malformed_branch)
    kind_validation = pt.validate_controlled_marginal_protocol(malformed_kind)

    assert branch_validation.decision is pt.ProtocolDecision.INVALID
    assert kind_validation.decision is pt.ProtocolDecision.INVALID
    assert branch_validation.errors
    assert kind_validation.errors


def test_branch_execution_receipts_are_content_addressed_and_match_the_sealed_plan() -> None:
    protocol = _controlled_protocol()
    treatment = _branch_receipt(
        protocol,
        pt.ControlledBranch.TREATMENT,
        TREATMENT_OUTPUT_SHA,
    )
    control = _branch_receipt(
        protocol,
        pt.ControlledBranch.CONTROL,
        CONTROL_OUTPUT_SHA,
    )

    pt.validate_branch_execution_receipt(
        treatment,
        protocol=protocol,
        expected_branch=pt.ControlledBranch.TREATMENT,
    )
    pt.validate_branch_execution_receipt(
        control,
        protocol=protocol,
        expected_branch=pt.ControlledBranch.CONTROL,
    )
    assert treatment.initial_checkpoint_sha256 == INITIAL_CHECKPOINT_SHA
    assert treatment.output_checkpoint_sha256 == TREATMENT_OUTPUT_SHA
    assert treatment.planned_train_token_budget == treatment.realized_train_token_budget
    assert treatment.planned_training_example_budget == treatment.realized_training_example_budget
    assert treatment.planned_optimizer_step_budget == treatment.realized_optimizer_step_budget
    assert treatment.digest != control.digest
    json.dumps(treatment.to_payload(), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("realized_train_token_budget", 9_999, "realized token budget"),
        ("realized_training_example_budget", 63, "realized example budget"),
        ("realized_optimizer_step_budget", 15, "realized optimizer-step budget"),
    ],
)
def test_forged_realized_branch_budgets_are_rejected_even_with_a_valid_receipt_digest(
    field: str,
    value: int,
    message: str,
) -> None:
    base = _controlled_artifact()
    forged = _branch_receipt(
        base.protocol,
        pt.ControlledBranch.TREATMENT,
        TREATMENT_OUTPUT_SHA,
        **{field: value},
    )
    assert len(forged.digest) == 64
    malformed = pt.make_controlled_marginal_artifact(
        protocol=base.protocol,
        treatment_receipt=forged,
        control_receipt=base.control_receipt,
        pairs=base.pairs,
    )

    with pytest.raises(pt.ArtifactIntegrityError, match=message):
        pt.validate_complete_controlled_artifact(malformed)
    assert (
        pt.assess_controlled_marginal_artifact(malformed).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("optimizer_config_sha256", "optimizer configuration digest"),
        ("trainer_sha256", "trainer digest"),
        ("runtime_sha256", "runtime digest"),
        ("training_rng_sha256", "training RNG digest"),
        ("training_manifest_sha256", "training manifest digest"),
    ],
)
def test_branch_receipt_rejects_actual_execution_digest_mismatches(
    field: str,
    message: str,
) -> None:
    protocol = _controlled_protocol()
    receipt = _branch_receipt(
        protocol,
        pt.ControlledBranch.TREATMENT,
        TREATMENT_OUTPUT_SHA,
        **{field: pt.sha256_hex(f"forged {field}")},
    )

    with pytest.raises(pt.ArtifactIntegrityError, match=message):
        pt.validate_branch_execution_receipt(
            receipt,
            protocol=protocol,
            expected_branch=pt.ControlledBranch.TREATMENT,
        )


def test_branch_receipt_content_tampering_is_rejected() -> None:
    protocol = _controlled_protocol()
    receipt = _branch_receipt(
        protocol,
        pt.ControlledBranch.TREATMENT,
        TREATMENT_OUTPUT_SHA,
    )
    tampered = replace(receipt, digest=pt.sha256_hex("forged receipt digest"))

    with pytest.raises(pt.ArtifactIntegrityError, match="receipt digest mismatch"):
        pt.validate_branch_execution_receipt(
            tampered,
            protocol=protocol,
            expected_branch=pt.ControlledBranch.TREATMENT,
        )


def test_complete_controlled_artifact_reports_overall_and_stratified_effects() -> None:
    artifact = _controlled_artifact()

    pt.validate_complete_controlled_artifact(artifact)
    assessment = pt.assess_controlled_marginal_artifact(artifact)

    assert assessment.decision is pt.ControlledEvidenceDecision.ESTIMATED
    assert assessment.planned_pairs == 4
    assert assessment.complete_pairs == 4
    assert assessment.ate_success == pytest.approx(0.25)
    assert assessment.mean_reward_delta == pytest.approx(0.25)
    assert assessment.p_treatment_better == pytest.approx(0.5)
    assert assessment.p_control_better == pytest.approx(0.875)

    by_kind = {stratum.kind: stratum for stratum in assessment.strata}
    assert by_kind[pt.HoldoutKind.SAME_FAMILY].ate_success == 1.0
    assert by_kind[pt.HoldoutKind.SAME_FAMILY].p_treatment_better == pytest.approx(0.25)
    assert by_kind[pt.HoldoutKind.SIBLING].ate_success == pytest.approx(-0.5)
    assert by_kind[pt.HoldoutKind.SIBLING].p_control_better == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("environment_seed", "environment seed mismatch"),
        ("sampling_seed", "sampling seed mismatch"),
        ("branch", "canonical serialization failed"),
    ],
)
def test_controlled_artifact_rejects_broken_crn_pair_identity(
    mutation: str,
    message: str,
) -> None:
    artifact = _controlled_artifact()
    pairs = list(artifact.pairs)
    first = pairs[0]
    treatment = first.treatment
    if mutation == "environment_seed":
        treatment = replace(treatment, environment_seed=treatment.environment_seed + 1)
    elif mutation == "sampling_seed":
        treatment = replace(treatment, sampling_seed=treatment.sampling_seed + 1)
    else:
        treatment = replace(treatment, branch="treatment")  # type: ignore[arg-type]
    pairs[0] = replace(first, treatment=treatment)
    malformed = (
        _rebuild_controlled(artifact, pairs)
        if mutation != "branch"
        else replace(artifact, pairs=tuple(pairs))
    )

    with pytest.raises(pt.ArtifactIntegrityError, match=message):
        pt.validate_complete_controlled_artifact(malformed)
    assert (
        pt.assess_controlled_marginal_artifact(malformed).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )


@pytest.mark.parametrize("branch", [pt.ControlledBranch.TREATMENT, pt.ControlledBranch.CONTROL])
def test_controlled_artifact_binds_every_outcome_to_receipt_output_checkpoint(
    branch: pt.ControlledBranch,
) -> None:
    artifact = _controlled_artifact()
    pairs = list(artifact.pairs)
    first = pairs[0]
    if branch is pt.ControlledBranch.TREATMENT:
        first = replace(
            first,
            treatment=replace(
                first.treatment,
                output_checkpoint_sha256=pt.sha256_hex("different-treatment-update"),
            ),
        )
    else:
        first = replace(
            first,
            control=replace(
                first.control,
                output_checkpoint_sha256=pt.sha256_hex("different-control-update"),
            ),
        )
    pairs[0] = first
    malformed = _rebuild_controlled(artifact, pairs)

    with pytest.raises(pt.ArtifactIntegrityError, match="execution receipt"):
        pt.validate_complete_controlled_artifact(malformed)


def test_identical_output_checkpoint_digests_cannot_claim_contradictory_crn_outcomes() -> None:
    artifact = _controlled_artifact()
    control_receipt = _branch_receipt(
        artifact.protocol,
        pt.ControlledBranch.CONTROL,
        TREATMENT_OUTPUT_SHA,
    )
    pairs = [
        replace(
            pair,
            control=replace(
                pair.control,
                output_checkpoint_sha256=TREATMENT_OUTPUT_SHA,
            ),
        )
        for pair in artifact.pairs
    ]
    contradictory = pt.make_controlled_marginal_artifact(
        protocol=artifact.protocol,
        treatment_receipt=artifact.treatment_receipt,
        control_receipt=control_receipt,
        pairs=pairs,
    )

    with pytest.raises(pt.ArtifactIntegrityError, match="identical.*contradictory"):
        pt.validate_complete_controlled_artifact(contradictory)
    assert (
        pt.assess_controlled_marginal_artifact(contradictory).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )


def test_controlled_artifact_missing_or_unavailable_pair_abstains_without_imputation() -> None:
    artifact = _controlled_artifact()
    stale_missing = replace(artifact, pairs=artifact.pairs[:-1])
    assert (
        pt.assess_controlled_marginal_artifact(stale_missing).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )
    missing = _rebuild_controlled(artifact, list(artifact.pairs[:-1]))
    assert (
        pt.assess_controlled_marginal_artifact(missing).decision
        is pt.ControlledEvidenceDecision.ABSTAIN_INCOMPLETE
    )

    pairs = list(artifact.pairs)
    first = pairs[0]
    unavailable = pt.BranchOutcome.unavailable(
        branch=pt.ControlledBranch.TREATMENT,
        unit=pt.EvaluationUnit(
            eval_id=first.treatment.eval_id,
            environment_id=first.treatment.environment_id,
            environment_seed=first.treatment.environment_seed,
            sampling_seed=first.treatment.sampling_seed,
        ),
        output_checkpoint_sha256=first.treatment.output_checkpoint_sha256,
        error="evaluation worker timeout",
        failure_classification=pt.FailureClassification.EXOGENOUS_INFRASTRUCTURE,
    )
    pairs[0] = replace(first, treatment=unavailable)
    stale_unavailable = replace(artifact, pairs=tuple(pairs))
    assert (
        pt.assess_controlled_marginal_artifact(stale_unavailable).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )
    incomplete = _rebuild_controlled(artifact, pairs)
    assessment = pt.assess_controlled_marginal_artifact(incomplete)

    assert assessment.decision is pt.ControlledEvidenceDecision.ABSTAIN_INCOMPLETE
    assert assessment.complete_pairs == 0
    assert assessment.ate_success is None
    assert assessment.strata == ()


def test_unavailable_controlled_branch_with_imputed_result_is_invalid() -> None:
    artifact = _controlled_artifact()
    pairs = list(artifact.pairs)
    first = pairs[0]
    unit = pt.EvaluationUnit(
        eval_id=first.treatment.eval_id,
        environment_id=first.treatment.environment_id,
        environment_seed=first.treatment.environment_seed,
        sampling_seed=first.treatment.sampling_seed,
    )
    unavailable = pt.BranchOutcome.unavailable(
        branch=pt.ControlledBranch.TREATMENT,
        unit=unit,
        output_checkpoint_sha256=first.treatment.output_checkpoint_sha256,
        error="evaluation worker timeout",
        failure_classification=pt.FailureClassification.EXOGENOUS_INFRASTRUCTURE,
    )
    pairs[0] = replace(first, treatment=replace(unavailable, reward=0.0))
    malformed = _rebuild_controlled(artifact, pairs)

    assert (
        pt.assess_controlled_marginal_artifact(malformed).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )


def test_intervention_caused_controlled_failure_is_a_quality_reject() -> None:
    artifact = _controlled_artifact()
    pairs = list(artifact.pairs)
    first = pairs[0]
    unit = pt.EvaluationUnit(
        eval_id=first.treatment.eval_id,
        environment_id=first.treatment.environment_id,
        environment_seed=first.treatment.environment_seed,
        sampling_seed=first.treatment.sampling_seed,
    )
    pairs[0] = replace(
        first,
        treatment=pt.BranchOutcome.unavailable(
            branch=pt.ControlledBranch.TREATMENT,
            unit=unit,
            output_checkpoint_sha256=first.treatment.output_checkpoint_sha256,
            error="treatment update corrupted the candidate runtime",
            failure_classification=pt.FailureClassification.CANDIDATE_OR_INTERVENTION,
        ),
    )
    stale_digest_failure = replace(artifact, pairs=tuple(pairs))
    assert (
        pt.assess_controlled_marginal_artifact(stale_digest_failure).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )
    assessment = pt.assess_controlled_marginal_artifact(_rebuild_controlled(artifact, pairs))

    assert assessment.decision is pt.ControlledEvidenceDecision.QUALITY_REJECT
    assert assessment.quality_reject is True
    assert assessment.complete_pairs == 0
    assert assessment.ate_success is None


def test_unavailable_controlled_outcome_without_failure_classification_is_invalid() -> None:
    artifact = _controlled_artifact()
    pairs = list(artifact.pairs)
    first = pairs[0]
    unit = pt.EvaluationUnit(
        eval_id=first.treatment.eval_id,
        environment_id=first.treatment.environment_id,
        environment_seed=first.treatment.environment_seed,
        sampling_seed=first.treatment.sampling_seed,
    )
    unavailable = pt.BranchOutcome.unavailable(
        branch=pt.ControlledBranch.TREATMENT,
        unit=unit,
        output_checkpoint_sha256=first.treatment.output_checkpoint_sha256,
        error="unclassified evaluation failure",
        failure_classification=pt.FailureClassification.EXOGENOUS_INFRASTRUCTURE,
    )
    pairs[0] = replace(
        first,
        treatment=replace(unavailable, failure_classification=None),
    )

    assert (
        pt.assess_controlled_marginal_artifact(_rebuild_controlled(artifact, pairs)).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )


def test_unavailable_controlled_outcome_with_malformed_enum_is_invalid_not_quality_reject() -> None:
    artifact = _controlled_artifact()
    pairs = list(artifact.pairs)
    first = pairs[0]
    unit = pt.EvaluationUnit(
        eval_id=first.treatment.eval_id,
        environment_id=first.treatment.environment_id,
        environment_seed=first.treatment.environment_seed,
        sampling_seed=first.treatment.sampling_seed,
    )
    unavailable = pt.BranchOutcome.unavailable(
        branch=pt.ControlledBranch.TREATMENT,
        unit=unit,
        output_checkpoint_sha256=first.treatment.output_checkpoint_sha256,
        error="candidate failure with malformed branch",
        failure_classification=pt.FailureClassification.CANDIDATE_OR_INTERVENTION,
    )
    pairs[0] = replace(
        first,
        treatment=replace(unavailable, branch="treatment"),  # type: ignore[arg-type]
    )
    assessment = pt.assess_controlled_marginal_artifact(replace(artifact, pairs=tuple(pairs)))

    assert assessment.decision is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    assert "canonical serialization failed" in assessment.reason


def test_invalid_protocol_abstains_before_controlled_outcomes_are_interpreted() -> None:
    control = replace(
        _training_branch(pt.ControlledBranch.CONTROL),
        training_example_budget=63,
    )
    invalid_protocol = _controlled_protocol(control=control)
    artifact = _controlled_artifact(protocol=invalid_protocol)
    assessment = pt.assess_controlled_marginal_artifact(artifact)

    assert assessment.decision is pt.ControlledEvidenceDecision.ABSTAIN_PROTOCOL_INVALID
    assert assessment.complete_pairs == 0
    assert assessment.ate_success is None
    assert any("example budgets" in error for error in assessment.validation_errors)


def test_controlled_artifact_digest_is_verified() -> None:
    artifact = _controlled_artifact()
    tampered = replace(artifact, digest=pt.sha256_hex("wrong controlled artifact"))

    assert (
        pt.assess_controlled_marginal_artifact(tampered).decision
        is pt.ControlledEvidenceDecision.INVALID_ARTIFACT
    )


def test_controlled_metadata_is_json_serializable_and_preserves_holdout_kinds() -> None:
    assessment = pt.assess_controlled_marginal_artifact(_controlled_artifact())
    metadata = assessment.to_metadata()

    json.dumps(metadata, allow_nan=False)
    assert metadata["decision"] == "estimated"
    assert metadata["inference_scope"] == "descriptive-single-replicate"
    assert metadata["quality_reject"] is False
    assert "no across-replicate inference" in metadata["reason"]
    assert {stratum["kind"] for stratum in metadata["strata"]} == {
        "same_family",
        "sibling",
    }
