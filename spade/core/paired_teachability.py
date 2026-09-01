"""Pure, offline primitives for controlled teachability measurements.

This module deliberately has no backend, model, filesystem, or network dependencies.  It
defines two related evidence protocols:

``paired_teachability``
    Measures the effect of one locked hint against no hint at a fixed policy checkpoint.
    Every comparison uses the same environment and sampling seeds, and arm order is balanced.

``controlled_marginal_teachability``
    Measures the marginal effect of a training intervention by branching from one checkpoint,
    holding token and optimizer budgets equal, and evaluating treatment/control checkpoints on
    common-random-number holdouts in both same-family and sibling-family strata.

All validation is fail closed.  Missing observations produce abstention; contradictory or
tampered artifacts are invalid.  Statistical calculations use only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PAIR_SCHEDULE_SCHEMA = "spade.paired-teachability.schedule.v1"
PAIR_ARTIFACT_SCHEMA = "spade.paired-teachability.artifact.v1"
PAIRED_FAMILY_PLAN_SCHEMA = "spade.paired-teachability.family-plan.v1"
CONTROLLED_PROTOCOL_SCHEMA = "spade.controlled-marginal-teachability.protocol.v1"
BRANCH_RECEIPT_SCHEMA = "spade.controlled-marginal-teachability.branch-receipt.v1"
CONTROLLED_ARTIFACT_SCHEMA = "spade.controlled-marginal-teachability.artifact.v1"

HOLM_DIRECTIONAL_BONFERRONI = "holm-directional-bonferroni-v1"
DESCRIPTIVE_REPLICATE_SCOPE = "descriptive-single-replicate"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SEED = 2**31
_MAX_EXACT_BINARY_TRIALS = 1_074


class Arm(str, Enum):
    """The two arms in a locked-hint comparison."""

    HINT = "hint"
    NO_HINT = "no_hint"


class ArmOrder(str, Enum):
    """The prescribed within-pair execution order."""

    HINT_FIRST = "hint_first"
    NO_HINT_FIRST = "no_hint_first"

    @property
    def arms(self) -> Tuple[Arm, Arm]:
        if self is ArmOrder.HINT_FIRST:
            return (Arm.HINT, Arm.NO_HINT)
        return (Arm.NO_HINT, Arm.HINT)


class OutcomeStatus(str, Enum):
    """Whether an arm produced an interpretable policy outcome."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"


class FailureClassification(str, Enum):
    """Predeclared causal class for an unavailable outcome.

    Infrastructure loss is missing evidence and therefore causes abstention.  A failure caused
    by the candidate or evaluated intervention is adverse evidence and therefore causes a quality
    rejection; it must never be silently treated as missing-at-random or imputed as a reward.
    The enum enforces downstream semantics, not classifier authenticity; production runners must
    bind it to their trusted failure-evidence receipt.
    """

    EXOGENOUS_INFRASTRUCTURE = "exogenous_infrastructure"
    CANDIDATE_OR_INTERVENTION = "candidate_or_intervention"


class TeachabilityDecision(str, Enum):
    """Fail-closed decision attached to a paired estimate."""

    PENDING_FAMILY = "pending_family"
    SCREEN_CREDIT = "screen_credit"
    HARMFUL = "harmful"
    ABSTAIN_UNCERTAIN = "abstain_uncertain"
    ABSTAIN_BELOW_EFFECT = "abstain_below_effect"
    ABSTAIN_INCOMPLETE = "abstain_incomplete"
    QUALITY_REJECT = "quality_reject"
    INVALID_ARTIFACT = "invalid_artifact"


class ControlledBranch(str, Enum):
    """Branches in a controlled marginal-training comparison."""

    TREATMENT = "treatment"
    CONTROL = "control"


class HoldoutKind(str, Enum):
    """Required generalization strata for controlled marginal evidence."""

    SAME_FAMILY = "same_family"
    SIBLING = "sibling"


class ProtocolDecision(str, Enum):
    """Whether a controlled protocol is sufficiently specified to execute."""

    READY = "ready"
    INVALID = "invalid"


class ControlledEvidenceDecision(str, Enum):
    """Whether controlled marginal outcomes support an interpretable estimate."""

    ESTIMATED = "estimated"
    ABSTAIN_PROTOCOL_INVALID = "abstain_protocol_invalid"
    ABSTAIN_INCOMPLETE = "abstain_incomplete"
    QUALITY_REJECT = "quality_reject"
    INVALID_ARTIFACT = "invalid_artifact"


class ArtifactValidationError(ValueError):
    """Base class for fail-closed artifact validation errors."""


class IncompleteArtifactError(ArtifactValidationError):
    """A precommitted observation is absent or unavailable."""


class ArtifactIntegrityError(ArtifactValidationError):
    """An artifact contradicts its schedule, protocol, or own digest."""


class InterventionFailureError(ArtifactValidationError):
    """A candidate or intervention caused an unavailable outcome."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value canonically for content addressing."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON compatible: {exc}") from exc
    return text.encode("utf-8")


def sha256_hex(value: bytes | str) -> str:
    """Return a lowercase SHA-256 hex digest for bytes or UTF-8 text."""

    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, bytes):
        raise TypeError("sha256_hex expects bytes or str")
    return hashlib.sha256(value).hexdigest()


def _content_digest(value: Any) -> str:
    return sha256_hex(_canonical_json_bytes(value))


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 hex digest")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_int(value: int, field_name: str, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")


def _require_seed(value: int, field_name: str) -> None:
    _require_int(value, field_name)
    if value >= _MAX_SEED:
        raise ValueError(f"{field_name} must be < {_MAX_SEED}")


def _derive_digest(domain: str, *parts: Any) -> bytes:
    payload = {"domain": domain, "parts": parts}
    return hashlib.sha256(_canonical_json_bytes(payload)).digest()


def _derive_seed(domain: str, *parts: Any, nonce: int = 0) -> int:
    digest = _derive_digest(domain, *parts, nonce)
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % _MAX_SEED


@dataclass(frozen=True)
class PairSpec:
    """One precommitted same-instance, common-random-number comparison."""

    pair_id: str
    pair_index: int
    environment_seed: int
    sampling_seed: int
    order: ArmOrder

    def to_payload(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "pair_index": self.pair_index,
            "environment_seed": self.environment_seed,
            "sampling_seed": self.sampling_seed,
            "order": self.order.value,
        }


@dataclass(frozen=True)
class PairSchedule:
    """Content-addressed schedule sealed before any paired outcomes are observed."""

    schema_version: str
    game_sha256: str
    run_seed: int
    rollout_id: int
    pairs: Tuple[PairSpec, ...]
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "game_sha256": self.game_sha256,
            "run_seed": self.run_seed,
            "rollout_id": self.rollout_id,
            "pairs": [pair.to_payload() for pair in self.pairs],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _pair_schedule_payload(
    game_sha256: str,
    run_seed: int,
    rollout_id: int,
    pairs: Sequence[PairSpec],
) -> Dict[str, Any]:
    return {
        "schema_version": PAIR_SCHEDULE_SCHEMA,
        "game_sha256": game_sha256,
        "run_seed": run_seed,
        "rollout_id": rollout_id,
        "pairs": [pair.to_payload() for pair in pairs],
    }


def make_pair_schedule(
    *,
    game_sha256: str,
    run_seed: int,
    rollout_id: int,
    pair_count: int,
) -> PairSchedule:
    """Build an exactly balanced, process-stable schedule using SHA-256 only.

    ``pair_count`` is required to be even.  The environment and sampling seed namespaces are
    domain separated, and collision resolution is deterministic.  Arm order is assigned by a
    SHA-256 ranking, yielding exactly half hint-first and half no-hint-first pairs.
    """

    _require_sha256(game_sha256, "game_sha256")
    _require_int(run_seed, "run_seed")
    _require_int(rollout_id, "rollout_id")
    _require_int(pair_count, "pair_count", minimum=2)
    if pair_count % 2 != 0:
        raise ValueError("pair_count must be even for exact arm-order balance")

    rank = sorted(
        range(pair_count),
        key=lambda index: _derive_digest(
            "spade.paired-teachability.order.v1",
            game_sha256,
            run_seed,
            rollout_id,
            index,
        ),
    )
    hint_first = set(rank[: pair_count // 2])

    used_environment_seeds: set[int] = set()
    used_sampling_seeds: set[int] = set()
    pairs: List[PairSpec] = []
    for index in range(pair_count):
        nonce = 0
        while True:
            environment_seed = _derive_seed(
                "spade.paired-teachability.environment-seed.v1",
                game_sha256,
                run_seed,
                rollout_id,
                index,
                nonce=nonce,
            )
            if environment_seed not in used_environment_seeds:
                break
            nonce += 1
        used_environment_seeds.add(environment_seed)

        nonce = 0
        while True:
            sampling_seed = _derive_seed(
                "spade.paired-teachability.sampling-seed.v1",
                game_sha256,
                run_seed,
                rollout_id,
                index,
                nonce=nonce,
            )
            if sampling_seed not in used_sampling_seeds:
                break
            nonce += 1
        used_sampling_seeds.add(sampling_seed)

        pair_id = _derive_digest(
            "spade.paired-teachability.pair-id.v1",
            game_sha256,
            run_seed,
            rollout_id,
            index,
            environment_seed,
            sampling_seed,
        ).hex()[:24]
        pairs.append(
            PairSpec(
                pair_id=pair_id,
                pair_index=index,
                environment_seed=environment_seed,
                sampling_seed=sampling_seed,
                order=(ArmOrder.HINT_FIRST if index in hint_first else ArmOrder.NO_HINT_FIRST),
            )
        )

    payload = _pair_schedule_payload(game_sha256, run_seed, rollout_id, pairs)
    return PairSchedule(
        schema_version=PAIR_SCHEDULE_SCHEMA,
        game_sha256=game_sha256,
        run_seed=run_seed,
        rollout_id=rollout_id,
        pairs=tuple(pairs),
        digest=_content_digest(payload),
    )


def validate_pair_schedule(schedule: PairSchedule) -> None:
    """Reject malformed, unbalanced, noncanonical, or tampered schedules."""

    if schedule.schema_version != PAIR_SCHEDULE_SCHEMA:
        raise ArtifactIntegrityError(
            f"unsupported pair schedule schema: {schedule.schema_version!r}"
        )
    try:
        expected = make_pair_schedule(
            game_sha256=schedule.game_sha256,
            run_seed=schedule.run_seed,
            rollout_id=schedule.rollout_id,
            pair_count=len(schedule.pairs),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    if schedule != expected:
        raise ArtifactIntegrityError("pair schedule differs from its canonical SHA-256 schedule")


@dataclass(frozen=True)
class ArmOutcome:
    """One arm outcome; unavailable outcomes must not carry imputed rewards."""

    arm: Arm
    environment_seed: int
    sampling_seed: int
    policy_checkpoint: str
    policy_checkpoint_sha256: str
    raw_observation_sha256: Optional[str]
    status: OutcomeStatus
    reward: Optional[float]
    success: Optional[bool]
    turns: Optional[int]
    error: Optional[str] = None
    failure_classification: Optional[FailureClassification] = None

    @classmethod
    def observed(
        cls,
        *,
        arm: Arm,
        environment_seed: int,
        sampling_seed: int,
        policy_checkpoint: str,
        policy_checkpoint_sha256: str,
        raw_observation_sha256: str,
        reward: float,
        success: bool,
        turns: int,
    ) -> "ArmOutcome":
        return cls(
            arm=arm,
            environment_seed=environment_seed,
            sampling_seed=sampling_seed,
            policy_checkpoint=policy_checkpoint,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
            raw_observation_sha256=raw_observation_sha256,
            status=OutcomeStatus.OBSERVED,
            reward=reward,
            success=success,
            turns=turns,
            error=None,
            failure_classification=None,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        arm: Arm,
        environment_seed: int,
        sampling_seed: int,
        policy_checkpoint: str,
        policy_checkpoint_sha256: str,
        error: str,
        failure_classification: FailureClassification,
    ) -> "ArmOutcome":
        return cls(
            arm=arm,
            environment_seed=environment_seed,
            sampling_seed=sampling_seed,
            policy_checkpoint=policy_checkpoint,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
            raw_observation_sha256=None,
            status=OutcomeStatus.UNAVAILABLE,
            reward=None,
            success=None,
            turns=None,
            error=error,
            failure_classification=failure_classification,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "arm": self.arm.value,
            "environment_seed": self.environment_seed,
            "sampling_seed": self.sampling_seed,
            "policy_checkpoint": self.policy_checkpoint,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "raw_observation_sha256": self.raw_observation_sha256,
            "status": self.status.value,
            "reward": self.reward,
            "success": self.success,
            "turns": self.turns,
            "error": self.error,
            "failure_classification": (
                self.failure_classification.value
                if self.failure_classification is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PairOutcome:
    """Both outcomes for one scheduled pair, plus the actual execution order."""

    pair_id: str
    pair_index: int
    observed_order: Tuple[Arm, Arm]
    hint: ArmOutcome
    no_hint: ArmOutcome

    def to_payload(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "pair_index": self.pair_index,
            "observed_order": [arm.value for arm in self.observed_order],
            "hint": self.hint.to_payload(),
            "no_hint": self.no_hint.to_payload(),
        }


@dataclass(frozen=True)
class PairedRunArtifact:
    """A locked hint, one policy checkpoint, and a complete scheduled outcome family."""

    schema_version: str
    schedule: PairSchedule
    locked_hint_sha256: str
    success_rule_sha256: str
    policy_checkpoint: str
    policy_checkpoint_sha256: str
    pairs: Tuple[PairOutcome, ...]
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "schedule": self.schedule.to_payload(),
            "locked_hint_sha256": self.locked_hint_sha256,
            "success_rule_sha256": self.success_rule_sha256,
            "policy_checkpoint": self.policy_checkpoint,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "pairs": [pair.to_payload() for pair in self.pairs],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def make_paired_run_artifact(
    *,
    schedule: PairSchedule,
    locked_hint_sha256: str,
    success_rule_sha256: str,
    policy_checkpoint: str,
    policy_checkpoint_sha256: str,
    pairs: Sequence[PairOutcome],
) -> PairedRunArtifact:
    payload = {
        "schema_version": PAIR_ARTIFACT_SCHEMA,
        "schedule": schedule.to_payload(),
        "locked_hint_sha256": locked_hint_sha256,
        "success_rule_sha256": success_rule_sha256,
        "policy_checkpoint": policy_checkpoint,
        "policy_checkpoint_sha256": policy_checkpoint_sha256,
        "pairs": [pair.to_payload() for pair in pairs],
    }
    return PairedRunArtifact(
        schema_version=PAIR_ARTIFACT_SCHEMA,
        schedule=schedule,
        locked_hint_sha256=locked_hint_sha256,
        success_rule_sha256=success_rule_sha256,
        policy_checkpoint=policy_checkpoint,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
        pairs=tuple(pairs),
        digest=_content_digest(payload),
    )


def _validate_observed_arm(
    outcome: ArmOutcome,
    *,
    expected_arm: Arm,
    spec: PairSpec,
    policy_checkpoint: str,
    policy_checkpoint_sha256: str,
) -> None:
    if not isinstance(outcome.arm, Arm):
        raise ArtifactIntegrityError(
            f"pair {spec.pair_id} {expected_arm.value} outcome has an invalid arm label"
        )
    if outcome.arm is not expected_arm:
        raise ArtifactIntegrityError(
            f"pair {spec.pair_id} {expected_arm.value} outcome has arm={outcome.arm.value}"
        )
    if outcome.environment_seed != spec.environment_seed:
        raise ArtifactIntegrityError(f"pair {spec.pair_id} environment seed mismatch")
    if outcome.sampling_seed != spec.sampling_seed:
        raise ArtifactIntegrityError(f"pair {spec.pair_id} sampling seed mismatch")
    if outcome.policy_checkpoint != policy_checkpoint:
        raise ArtifactIntegrityError(f"pair {spec.pair_id} policy checkpoint mismatch")
    if outcome.policy_checkpoint_sha256 != policy_checkpoint_sha256:
        raise ArtifactIntegrityError(f"pair {spec.pair_id} policy checkpoint digest mismatch")
    if not isinstance(outcome.status, OutcomeStatus):
        raise ArtifactIntegrityError(f"pair {spec.pair_id} outcome has an invalid status")
    if outcome.status is not OutcomeStatus.OBSERVED:
        if outcome.reward is not None or outcome.success is not None or outcome.turns is not None:
            raise ArtifactIntegrityError(
                f"pair {spec.pair_id} unavailable outcome contains an imputed result"
            )
        if not isinstance(outcome.failure_classification, FailureClassification):
            raise ArtifactIntegrityError(
                f"pair {spec.pair_id} unavailable outcome lacks a valid failure classification"
            )
        try:
            _require_nonempty(outcome.error or "", "error")
        except ValueError as exc:
            raise ArtifactIntegrityError(f"pair {spec.pair_id}: {exc}") from exc
        if outcome.failure_classification is FailureClassification.CANDIDATE_OR_INTERVENTION:
            raise InterventionFailureError(
                f"pair {spec.pair_id} {expected_arm.value} failed because of the candidate or "
                "intervention; quality reject"
            )
        raise IncompleteArtifactError(
            f"pair {spec.pair_id} {expected_arm.value} outcome unavailable: "
            f"{outcome.error or 'unspecified error'}"
        )
    if outcome.error is not None:
        raise ArtifactIntegrityError(f"pair {spec.pair_id} observed outcome contains an error")
    if outcome.failure_classification is not None:
        raise ArtifactIntegrityError(
            f"pair {spec.pair_id} observed outcome contains a failure classification"
        )
    if (
        outcome.reward is None
        or isinstance(outcome.reward, bool)
        or not isinstance(outcome.reward, (int, float))
        or not math.isfinite(float(outcome.reward))
        or not -1.0 <= float(outcome.reward) <= 1.0
    ):
        raise ArtifactIntegrityError(
            f"pair {spec.pair_id} observed reward must be finite and within [-1, 1]"
        )
    if not isinstance(outcome.success, bool):
        raise ArtifactIntegrityError(f"pair {spec.pair_id} observed success must be bool")
    if not isinstance(outcome.turns, int) or isinstance(outcome.turns, bool) or outcome.turns < 0:
        raise ArtifactIntegrityError(f"pair {spec.pair_id} observed turns must be >= 0")
    try:
        _require_sha256(outcome.raw_observation_sha256 or "", "raw_observation_sha256")
    except ValueError as exc:
        raise ArtifactIntegrityError(f"pair {spec.pair_id}: {exc}") from exc


def _validate_paired_artifact_self_digest(artifact: PairedRunArtifact) -> None:
    """Verify serializable artifact bytes before assigning any evidence decision."""

    try:
        _require_sha256(artifact.digest, "paired artifact digest")
    except ValueError as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    try:
        expected_digest = _content_digest(artifact.to_payload(include_digest=False))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            f"paired artifact canonical serialization failed: {exc}"
        ) from exc
    if artifact.digest != expected_digest:
        raise ArtifactIntegrityError("paired artifact digest mismatch")


def validate_complete_pair_artifact(artifact: PairedRunArtifact) -> None:
    """Validate completeness, seed identity, arm order, checkpoint lock, and digests."""

    if artifact.schema_version != PAIR_ARTIFACT_SCHEMA:
        raise ArtifactIntegrityError(
            f"unsupported paired artifact schema: {artifact.schema_version!r}"
        )
    _validate_paired_artifact_self_digest(artifact)
    validate_pair_schedule(artifact.schedule)
    try:
        _require_sha256(artifact.locked_hint_sha256, "locked_hint_sha256")
        _require_sha256(artifact.success_rule_sha256, "success_rule_sha256")
        _require_nonempty(artifact.policy_checkpoint, "policy_checkpoint")
        _require_sha256(artifact.policy_checkpoint_sha256, "policy_checkpoint_sha256")
    except ValueError as exc:
        raise ArtifactIntegrityError(str(exc)) from exc

    scheduled_ids = {spec.pair_id for spec in artifact.schedule.pairs}
    observed_ids = [pair.pair_id for pair in artifact.pairs]
    if len(observed_ids) != len(set(observed_ids)):
        raise ArtifactIntegrityError("paired artifact contains duplicate pair IDs")
    missing = sorted(scheduled_ids - set(observed_ids))
    extras = sorted(set(observed_ids) - scheduled_ids)
    if extras:
        raise ArtifactIntegrityError(f"paired artifact contains unscheduled pairs: {extras}")
    if missing:
        raise IncompleteArtifactError(f"paired artifact is missing scheduled pairs: {missing}")

    by_id = {pair.pair_id: pair for pair in artifact.pairs}
    for pair in artifact.pairs:
        for outcome in (pair.hint, pair.no_hint):
            if (
                outcome.status is OutcomeStatus.UNAVAILABLE
                and outcome.failure_classification
                is FailureClassification.CANDIDATE_OR_INTERVENTION
            ):
                arm_value = outcome.arm.value if isinstance(outcome.arm, Arm) else repr(outcome.arm)
                raise InterventionFailureError(
                    f"pair {pair.pair_id} {arm_value} failed because of the candidate or "
                    "intervention; quality reject"
                )
    for spec in artifact.schedule.pairs:
        pair = by_id[spec.pair_id]
        if pair.pair_index != spec.pair_index:
            raise ArtifactIntegrityError(f"pair {spec.pair_id} index mismatch")
        if pair.observed_order != spec.order.arms:
            raise ArtifactIntegrityError(f"pair {spec.pair_id} execution order mismatch")
        _validate_observed_arm(
            pair.hint,
            expected_arm=Arm.HINT,
            spec=spec,
            policy_checkpoint=artifact.policy_checkpoint,
            policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        )
        _validate_observed_arm(
            pair.no_hint,
            expected_arm=Arm.NO_HINT,
            spec=spec,
            policy_checkpoint=artifact.policy_checkpoint,
            policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        )
        if pair.hint.raw_observation_sha256 != pair.no_hint.raw_observation_sha256:
            raise ArtifactIntegrityError(
                f"pair {spec.pair_id} arms did not receive the same raw observation"
            )

    expected_digest = _content_digest(artifact.to_payload(include_digest=False))
    if artifact.digest != expected_digest:
        raise ArtifactIntegrityError("paired artifact digest mismatch")


@dataclass(frozen=True)
class PairedFamilyMember:
    """One exact candidate identity sealed into a pre-outcome family plan."""

    candidate_id: str
    game_sha256: str
    schedule_digest: str
    locked_hint_sha256: str
    policy_checkpoint: str
    policy_checkpoint_sha256: str
    pair_count: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_sha256": self.game_sha256,
            "schedule_digest": self.schedule_digest,
            "locked_hint_sha256": self.locked_hint_sha256,
            "policy_checkpoint": self.policy_checkpoint,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "pair_count": self.pair_count,
        }


def make_paired_family_member(
    *,
    candidate_id: str,
    schedule: PairSchedule,
    locked_hint_sha256: str,
    policy_checkpoint: str,
    policy_checkpoint_sha256: str,
) -> PairedFamilyMember:
    """Seal one candidate's pre-outcome schedule, hint, and checkpoint identity."""

    validate_pair_schedule(schedule)
    _require_nonempty(candidate_id, "candidate_id")
    _require_sha256(locked_hint_sha256, "locked_hint_sha256")
    _require_nonempty(policy_checkpoint, "policy_checkpoint")
    _require_sha256(policy_checkpoint_sha256, "policy_checkpoint_sha256")
    return PairedFamilyMember(
        candidate_id=candidate_id,
        game_sha256=schedule.game_sha256,
        schedule_digest=schedule.digest,
        locked_hint_sha256=locked_hint_sha256,
        policy_checkpoint=policy_checkpoint,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
        pair_count=len(schedule.pairs),
    )


@dataclass(frozen=True)
class PairedFamilyPlan:
    """Content-addressed family definition that must be sealed before outcomes exist.

    A self-digest proves content integrity, not chronology.  Production protocols must publish
    this digest to a trusted append-only store before any member outcome is collected; assessment
    deliberately treats that temporal commitment as an external invariant.
    """

    schema_version: str
    family_id: str
    members: Tuple[PairedFamilyMember, ...]
    success_rule_sha256: str
    pair_count: int
    family_alpha: float
    point_effect_floor: float
    correction_name: str
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "members": [member.to_payload() for member in self.members],
            "success_rule_sha256": self.success_rule_sha256,
            "pair_count": self.pair_count,
            "family_alpha": self.family_alpha,
            "point_effect_floor": self.point_effect_floor,
            "correction_name": self.correction_name,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _validate_probability(value: float, field_name: str, *, allow_zero: bool) -> float:
    lower_ok = float(value) >= 0.0 if allow_zero else float(value) > 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lower_ok
        or float(value) > 1.0
    ):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{field_name} must be finite and within {interval}")
    return float(value)


def validate_paired_family_plan(plan: PairedFamilyPlan) -> None:
    """Reject a malformed, underpowered, or content-tampered family plan."""

    if not isinstance(plan, PairedFamilyPlan):
        raise ArtifactIntegrityError("plan must be a PairedFamilyPlan")
    if plan.schema_version != PAIRED_FAMILY_PLAN_SCHEMA:
        raise ArtifactIntegrityError(
            f"unsupported paired family plan schema: {plan.schema_version!r}"
        )
    try:
        _require_nonempty(plan.family_id, "family_id")
        _require_sha256(plan.success_rule_sha256, "success_rule_sha256")
        _require_int(plan.pair_count, "pair_count", minimum=2)
        family_alpha = _validate_probability(plan.family_alpha, "family_alpha", allow_zero=False)
        point_effect_floor = _validate_probability(
            plan.point_effect_floor,
            "point_effect_floor",
            allow_zero=True,
        )
        _require_nonempty(plan.correction_name, "correction_name")
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    if family_alpha >= 1.0:
        raise ArtifactIntegrityError("family_alpha must be finite and within (0, 1)")
    if point_effect_floor > 1.0:  # pragma: no cover - guarded above, documents the invariant
        raise ArtifactIntegrityError("point_effect_floor must be within [0, 1]")
    if plan.pair_count % 2:
        raise ArtifactIntegrityError("pair_count must be even for exact arm-order balance")
    if plan.pair_count > _MAX_EXACT_BINARY_TRIALS:
        raise ArtifactIntegrityError(
            f"pair_count must be <= {_MAX_EXACT_BINARY_TRIALS} for representable exact p-values"
        )
    if plan.correction_name != HOLM_DIRECTIONAL_BONFERRONI:
        raise ArtifactIntegrityError(f"unsupported correction_name: {plan.correction_name!r}")
    if not plan.members:
        raise ArtifactIntegrityError("paired family plan members must be non-empty")

    candidate_ids: List[str] = []
    identities: List[Tuple[str, str, str, str, str]] = []
    for member in plan.members:
        if not isinstance(member, PairedFamilyMember):
            raise ArtifactIntegrityError("paired family members must be PairedFamilyMember values")
        try:
            _require_nonempty(member.candidate_id, "candidate_id")
            _require_sha256(member.game_sha256, f"{member.candidate_id}.game_sha256")
            _require_sha256(member.schedule_digest, f"{member.candidate_id}.schedule_digest")
            _require_sha256(
                member.locked_hint_sha256,
                f"{member.candidate_id}.locked_hint_sha256",
            )
            _require_nonempty(
                member.policy_checkpoint,
                f"{member.candidate_id}.policy_checkpoint",
            )
            _require_sha256(
                member.policy_checkpoint_sha256,
                f"{member.candidate_id}.policy_checkpoint_sha256",
            )
            _require_int(member.pair_count, f"{member.candidate_id}.pair_count", minimum=2)
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(str(exc)) from exc
        if member.pair_count != plan.pair_count:
            raise ArtifactIntegrityError(
                f"candidate {member.candidate_id!r} pair_count differs from family K"
            )
        candidate_ids.append(member.candidate_id)
        identities.append(
            (
                member.game_sha256,
                member.schedule_digest,
                member.locked_hint_sha256,
                member.policy_checkpoint,
                member.policy_checkpoint_sha256,
            )
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ArtifactIntegrityError("paired family candidate IDs must be unique")
    if len(identities) != len(set(identities)):
        raise ArtifactIntegrityError("paired family candidate identities must be unique")
    policy_checkpoints = {member.policy_checkpoint for member in plan.members}
    if len(policy_checkpoints) != 1:
        raise ArtifactIntegrityError(
            "all paired family members must use one identical policy checkpoint"
        )
    policy_checkpoint_digests = {member.policy_checkpoint_sha256 for member in plan.members}
    if len(policy_checkpoint_digests) != 1:
        raise ArtifactIntegrityError(
            "all paired family members must use one identical policy checkpoint digest"
        )

    first_holm_threshold = family_alpha / (2.0 * len(plan.members))
    if first_holm_threshold == 0.0:
        raise ArtifactIntegrityError(
            "the split first Holm threshold is below binary64 floating-point resolution"
        )
    log_best_case_p = -plan.pair_count * math.log(2.0)
    log_first_holm_threshold = math.log(family_alpha) - math.log(2.0) - math.log(len(plan.members))
    if log_best_case_p > log_first_holm_threshold:
        raise ArtifactIntegrityError(
            "pair_count K is incapable of meeting the split first Holm threshold even when "
            "all K pairs favor one arm: 2^-K > family_alpha / (2 * family_size)"
        )

    try:
        _require_sha256(plan.digest, "family_plan.digest")
        expected_digest = _content_digest(plan.to_payload(include_digest=False))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    if plan.digest != expected_digest:
        raise ArtifactIntegrityError("paired family plan digest mismatch")


def make_paired_family_plan(
    *,
    family_id: str,
    members: Sequence[PairedFamilyMember],
    success_rule_sha256: str,
    pair_count: int,
    family_alpha: float = 0.05,
    point_effect_floor: float = 0.0,
    correction_name: str = HOLM_DIRECTIONAL_BONFERRONI,
) -> PairedFamilyPlan:
    """Seal and validate a complete paired family before collecting any outcomes."""

    payload = {
        "schema_version": PAIRED_FAMILY_PLAN_SCHEMA,
        "family_id": family_id,
        "members": [member.to_payload() for member in members],
        "success_rule_sha256": success_rule_sha256,
        "pair_count": pair_count,
        "family_alpha": family_alpha,
        "point_effect_floor": point_effect_floor,
        "correction_name": correction_name,
    }
    plan = PairedFamilyPlan(
        schema_version=PAIRED_FAMILY_PLAN_SCHEMA,
        family_id=family_id,
        members=tuple(members),
        success_rule_sha256=success_rule_sha256,
        pair_count=pair_count,
        family_alpha=family_alpha,
        point_effect_floor=point_effect_floor,
        correction_name=correction_name,
        digest=_content_digest(payload),
    )
    validate_paired_family_plan(plan)
    return plan


def exact_mcnemar_one_sided(favorable: int, unfavorable: int) -> float:
    """Return the exact conditional one-sided McNemar/binomial tail probability."""

    _require_int(favorable, "favorable")
    _require_int(unfavorable, "unfavorable")
    discordant = favorable + unfavorable
    if discordant == 0:
        return 1.0
    if discordant > _MAX_EXACT_BINARY_TRIALS:
        raise ValueError(
            f"discordant pair count must be <= {_MAX_EXACT_BINARY_TRIALS} for a representable p-value"
        )
    numerator = sum(math.comb(discordant, count) for count in range(favorable, discordant + 1))
    return float(numerator / (1 << discordant))


@dataclass(frozen=True)
class PairedTeachabilityEstimate:
    """Paired effect estimate before or after family-wise selection."""

    estimate_id: str
    game_sha256: str
    schedule_digest: str
    locked_hint_sha256: str
    success_rule_sha256: str
    policy_checkpoint: str
    policy_checkpoint_sha256: str
    planned_pairs: int
    complete_pairs: int
    hint_only_success: int
    no_hint_only_success: int
    both_success: int
    both_failure: int
    ate_success: Optional[float]
    mean_reward_delta: Optional[float]
    p_hint_better: float
    p_no_hint_better: float
    effect_floor: float
    decision: TeachabilityDecision
    credit: float
    quality_reject: bool
    reason: str
    validation_errors: Tuple[str, ...] = ()
    family_plan_digest: Optional[str] = None
    family_id: Optional[str] = None
    family_size: Optional[int] = None
    family_alpha: Optional[float] = None
    directional_alpha: Optional[float] = None
    holm_rank: Optional[int] = None
    holm_threshold: Optional[float] = None
    holm_adjusted_p: Optional[float] = None
    holm_rejected: Optional[bool] = None
    harm_holm_rank: Optional[int] = None
    harm_holm_threshold: Optional[float] = None
    harm_holm_adjusted_p: Optional[float] = None
    harm_holm_rejected: Optional[bool] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Return a stable, JSON-compatible audit record."""

        return {
            "estimate_id": self.estimate_id,
            "game_sha256": self.game_sha256,
            "schedule_digest": self.schedule_digest,
            "locked_hint_sha256": self.locked_hint_sha256,
            "success_rule_sha256": self.success_rule_sha256,
            "policy_checkpoint": self.policy_checkpoint,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "planned_pairs": self.planned_pairs,
            "complete_pairs": self.complete_pairs,
            "hint_only_success": self.hint_only_success,
            "no_hint_only_success": self.no_hint_only_success,
            "both_success": self.both_success,
            "both_failure": self.both_failure,
            "ate_success": self.ate_success,
            "mean_reward_delta": self.mean_reward_delta,
            "p_hint_better": self.p_hint_better,
            "p_no_hint_better": self.p_no_hint_better,
            "effect_floor": self.effect_floor,
            "decision": self.decision.value,
            "credit": self.credit,
            "quality_reject": self.quality_reject,
            "reason": self.reason,
            "validation_errors": list(self.validation_errors),
            "family_plan_digest": self.family_plan_digest,
            "family_id": self.family_id,
            "family_size": self.family_size,
            "family_alpha": self.family_alpha,
            "directional_alpha": self.directional_alpha,
            "holm_rank": self.holm_rank,
            "holm_threshold": self.holm_threshold,
            "holm_adjusted_p": self.holm_adjusted_p,
            "holm_rejected": self.holm_rejected,
            "harm_holm_rank": self.harm_holm_rank,
            "harm_holm_threshold": self.harm_holm_threshold,
            "harm_holm_adjusted_p": self.harm_holm_adjusted_p,
            "harm_holm_rejected": self.harm_holm_rejected,
        }


def _failed_paired_estimate(
    artifact: PairedRunArtifact,
    *,
    effect_floor: float,
    decision: TeachabilityDecision,
    reason: str,
    quality_reject: bool = False,
) -> PairedTeachabilityEstimate:
    estimate_id = _derive_digest(
        "spade.paired-teachability.estimate-id.v1",
        artifact.schedule.digest,
        artifact.locked_hint_sha256,
        artifact.success_rule_sha256,
        artifact.policy_checkpoint,
        artifact.policy_checkpoint_sha256,
    ).hex()
    return PairedTeachabilityEstimate(
        estimate_id=estimate_id,
        game_sha256=artifact.schedule.game_sha256,
        schedule_digest=artifact.schedule.digest,
        locked_hint_sha256=artifact.locked_hint_sha256,
        success_rule_sha256=artifact.success_rule_sha256,
        policy_checkpoint=artifact.policy_checkpoint,
        policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        planned_pairs=len(artifact.schedule.pairs),
        complete_pairs=0,
        hint_only_success=0,
        no_hint_only_success=0,
        both_success=0,
        both_failure=0,
        ate_success=None,
        mean_reward_delta=None,
        p_hint_better=1.0,
        p_no_hint_better=1.0,
        effect_floor=effect_floor,
        decision=decision,
        credit=0.0,
        quality_reject=quality_reject,
        reason=reason,
        validation_errors=(reason,),
    )


def assess_paired_artifact(
    artifact: PairedRunArtifact,
    *,
    effect_floor: float = 0.0,
) -> PairedTeachabilityEstimate:
    """Validate and estimate a paired artifact, leaving family selection pending."""

    if (
        isinstance(effect_floor, bool)
        or not isinstance(effect_floor, (int, float))
        or not math.isfinite(float(effect_floor))
        or not 0.0 <= float(effect_floor) <= 1.0
    ):
        raise ValueError("effect_floor must be finite and within [0, 1]")
    effect_floor = float(effect_floor)

    try:
        validate_complete_pair_artifact(artifact)
    except InterventionFailureError as exc:
        return _failed_paired_estimate(
            artifact,
            effect_floor=effect_floor,
            decision=TeachabilityDecision.QUALITY_REJECT,
            reason=str(exc),
            quality_reject=True,
        )
    except IncompleteArtifactError as exc:
        return _failed_paired_estimate(
            artifact,
            effect_floor=effect_floor,
            decision=TeachabilityDecision.ABSTAIN_INCOMPLETE,
            reason=str(exc),
        )
    except ArtifactValidationError as exc:
        return _failed_paired_estimate(
            artifact,
            effect_floor=effect_floor,
            decision=TeachabilityDecision.INVALID_ARTIFACT,
            reason=str(exc),
        )

    hint_only = 0
    no_hint_only = 0
    both_success = 0
    both_failure = 0
    reward_deltas: List[float] = []
    for pair in artifact.pairs:
        hint_success = bool(pair.hint.success)
        no_hint_success = bool(pair.no_hint.success)
        if hint_success and not no_hint_success:
            hint_only += 1
        elif no_hint_success and not hint_success:
            no_hint_only += 1
        elif hint_success:
            both_success += 1
        else:
            both_failure += 1
        assert pair.hint.reward is not None and pair.no_hint.reward is not None
        reward_deltas.append(float(pair.hint.reward) - float(pair.no_hint.reward))

    pair_count = len(artifact.pairs)
    ate = (hint_only - no_hint_only) / pair_count
    estimate_id = _derive_digest(
        "spade.paired-teachability.estimate-id.v1",
        artifact.schedule.digest,
        artifact.locked_hint_sha256,
        artifact.success_rule_sha256,
        artifact.policy_checkpoint,
        artifact.policy_checkpoint_sha256,
    ).hex()
    return PairedTeachabilityEstimate(
        estimate_id=estimate_id,
        game_sha256=artifact.schedule.game_sha256,
        schedule_digest=artifact.schedule.digest,
        locked_hint_sha256=artifact.locked_hint_sha256,
        success_rule_sha256=artifact.success_rule_sha256,
        policy_checkpoint=artifact.policy_checkpoint,
        policy_checkpoint_sha256=artifact.policy_checkpoint_sha256,
        planned_pairs=pair_count,
        complete_pairs=pair_count,
        hint_only_success=hint_only,
        no_hint_only_success=no_hint_only,
        both_success=both_success,
        both_failure=both_failure,
        ate_success=ate,
        mean_reward_delta=sum(reward_deltas) / pair_count,
        p_hint_better=exact_mcnemar_one_sided(hint_only, no_hint_only),
        p_no_hint_better=exact_mcnemar_one_sided(no_hint_only, hint_only),
        effect_floor=effect_floor,
        decision=TeachabilityDecision.PENDING_FAMILY,
        credit=0.0,
        quality_reject=False,
        reason="complete paired estimate awaiting precommitted family correction",
    )


def _holm_direction(
    estimates: Sequence[PairedTeachabilityEstimate],
    *,
    p_field: str,
    alpha: float,
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float], set[str]]:
    """Return ranks, thresholds, adjusted p-values, and rejections for one direction."""

    m = len(estimates)
    ordered = sorted(estimates, key=lambda item: (getattr(item, p_field), item.estimate_id))
    rejected_ids: set[str] = set()
    ranks: Dict[str, int] = {}
    thresholds: Dict[str, float] = {}
    adjusted: Dict[str, float] = {}
    continue_rejecting = True
    running_adjusted = 0.0

    for zero_rank, estimate in enumerate(ordered):
        p_value = float(getattr(estimate, p_field))
        multiplier = m - zero_rank
        threshold = alpha / multiplier
        ranks[estimate.estimate_id] = zero_rank + 1
        thresholds[estimate.estimate_id] = threshold
        running_adjusted = max(running_adjusted, min(1.0, multiplier * p_value))
        adjusted[estimate.estimate_id] = running_adjusted
        if continue_rejecting and p_value <= threshold:
            rejected_ids.add(estimate.estimate_id)
        else:
            continue_rejecting = False

    return ranks, thresholds, adjusted, rejected_ids


def _apply_holm_family_unsealed(
    estimates: Sequence[PairedTeachabilityEstimate],
    *,
    family_id: str,
    family_alpha: float = 0.05,
) -> Tuple[PairedTeachabilityEstimate, ...]:
    """Internal Holm mechanics for estimates already bound by a validated family plan.

    This function is intentionally private because it cannot prove that membership, alpha, or
    effect floors were chosen before outcomes.  Public callers must use
    :func:`assess_paired_family` with a content-addressed :class:`PairedFamilyPlan`.
    """

    _require_nonempty(family_id, "family_id")
    if (
        isinstance(family_alpha, bool)
        or not isinstance(family_alpha, (int, float))
        or not math.isfinite(float(family_alpha))
        or not 0.0 < float(family_alpha) < 1.0
    ):
        raise ValueError("family_alpha must be finite and within (0, 1)")
    if not estimates:
        raise ValueError("estimates must be a non-empty precommitted family")

    estimate_ids = [estimate.estimate_id for estimate in estimates]
    if len(estimate_ids) != len(set(estimate_ids)):
        raise ValueError("estimates must have unique estimate_id values")
    for estimate in estimates:
        if not 0.0 <= estimate.p_hint_better <= 1.0:
            raise ValueError("estimate p_hint_better values must be within [0, 1]")
        if not 0.0 <= estimate.p_no_hint_better <= 1.0:
            raise ValueError("estimate p_no_hint_better values must be within [0, 1]")

    m = len(estimates)
    family_alpha = float(family_alpha)
    directional_alpha = family_alpha / 2.0
    ranks, thresholds, adjusted, rejected_ids = _holm_direction(
        estimates,
        p_field="p_hint_better",
        alpha=directional_alpha,
    )
    harm_ranks, harm_thresholds, harm_adjusted, harm_rejected_ids = _holm_direction(
        estimates,
        p_field="p_no_hint_better",
        alpha=directional_alpha,
    )

    result: List[PairedTeachabilityEstimate] = []
    for estimate in estimates:
        rejected = estimate.estimate_id in rejected_ids
        harm_rejected = estimate.estimate_id in harm_rejected_ids
        if estimate.decision in {
            TeachabilityDecision.ABSTAIN_INCOMPLETE,
            TeachabilityDecision.QUALITY_REJECT,
            TeachabilityDecision.INVALID_ARTIFACT,
        }:
            decision = estimate.decision
            credit = 0.0
            quality_reject = estimate.quality_reject
            reason = estimate.reason
        elif rejected and harm_rejected:
            decision = TeachabilityDecision.INVALID_ARTIFACT
            credit = 0.0
            quality_reject = False
            reason = "contradictory beneficial and harmful directional rejections"
        elif harm_rejected:
            decision = TeachabilityDecision.HARMFUL
            credit = 0.0
            quality_reject = True
            reason = "negative paired effect passed multiplicity correction; quality reject"
        elif not rejected:
            decision = TeachabilityDecision.ABSTAIN_UNCERTAIN
            credit = 0.0
            quality_reject = False
            reason = "exact paired effect did not pass Holm family correction"
        elif estimate.ate_success is None or estimate.ate_success < estimate.effect_floor:
            decision = TeachabilityDecision.ABSTAIN_BELOW_EFFECT
            credit = 0.0
            quality_reject = False
            reason = "Holm-rejected positive effect was below the operational screening floor"
        else:
            decision = TeachabilityDecision.SCREEN_CREDIT
            credit = float(estimate.ate_success)
            quality_reject = False
            reason = (
                "screening credit: positive paired effect passed Holm correction and the observed "
                "point effect met the operational floor; controlled confirmation is still required"
            )

        result.append(
            replace(
                estimate,
                decision=decision,
                credit=credit,
                quality_reject=quality_reject,
                reason=reason,
                family_id=family_id,
                family_size=m,
                family_alpha=family_alpha,
                directional_alpha=directional_alpha,
                holm_rank=ranks[estimate.estimate_id],
                holm_threshold=thresholds[estimate.estimate_id],
                holm_adjusted_p=adjusted[estimate.estimate_id],
                holm_rejected=rejected,
                harm_holm_rank=harm_ranks[estimate.estimate_id],
                harm_holm_threshold=harm_thresholds[estimate.estimate_id],
                harm_holm_adjusted_p=harm_adjusted[estimate.estimate_id],
                harm_holm_rejected=harm_rejected,
            )
        )
    return tuple(result)


def _validate_member_artifact_identity(
    plan: PairedFamilyPlan,
    member: PairedFamilyMember,
    artifact: PairedRunArtifact,
) -> None:
    """Validate only pre-outcome identity so incomplete outcomes can still abstain."""

    validate_pair_schedule(artifact.schedule)
    mismatches: List[str] = []
    if artifact.schedule.game_sha256 != member.game_sha256:
        mismatches.append("game digest")
    if artifact.schedule.digest != member.schedule_digest:
        mismatches.append("schedule digest")
    if artifact.locked_hint_sha256 != member.locked_hint_sha256:
        mismatches.append("locked hint digest")
    if artifact.success_rule_sha256 != plan.success_rule_sha256:
        mismatches.append("success-rule digest")
    if artifact.policy_checkpoint != member.policy_checkpoint:
        mismatches.append("policy checkpoint")
    if artifact.policy_checkpoint_sha256 != member.policy_checkpoint_sha256:
        mismatches.append("policy checkpoint digest")
    if len(artifact.schedule.pairs) != plan.pair_count:
        mismatches.append("pair_count K")
    if mismatches:
        raise ArtifactIntegrityError(
            f"candidate {member.candidate_id!r} artifact differs from its sealed family member: "
            + ", ".join(mismatches)
        )


def assess_paired_family(
    plan: PairedFamilyPlan,
    artifacts_by_candidate: Mapping[str, PairedRunArtifact],
) -> Tuple[PairedTeachabilityEstimate, ...]:
    """Assess exactly the members and decision rules sealed in a pre-outcome plan.

    Missing or extra candidate keys are integrity errors rather than a smaller post-hoc family.
    Within a correctly identified member, exogenous missingness remains an abstention and
    candidate/intervention-caused failure remains a quality rejection; both continue to count in
    the Holm denominator.
    """

    validate_paired_family_plan(plan)
    if not isinstance(artifacts_by_candidate, Mapping):
        raise TypeError("artifacts_by_candidate must be a mapping keyed by candidate_id")
    expected_ids = {member.candidate_id for member in plan.members}
    supplied_ids = set(artifacts_by_candidate)
    if any(not isinstance(candidate_id, str) for candidate_id in supplied_ids):
        raise ArtifactIntegrityError("paired family artifact keys must be candidate_id strings")
    missing = sorted(expected_ids - supplied_ids)
    extras = sorted(supplied_ids - expected_ids)
    if missing or extras:
        details = []
        if missing:
            details.append(f"missing members: {missing}")
        if extras:
            details.append(f"extra members: {extras}")
        raise ArtifactIntegrityError(
            "paired family artifacts differ from the sealed membership: " + "; ".join(details)
        )

    estimates: List[PairedTeachabilityEstimate] = []
    for member in plan.members:
        artifact = artifacts_by_candidate[member.candidate_id]
        if not isinstance(artifact, PairedRunArtifact):
            raise ArtifactIntegrityError(
                f"candidate {member.candidate_id!r} must map to a PairedRunArtifact"
            )
        _validate_member_artifact_identity(plan, member, artifact)
        estimates.append(
            assess_paired_artifact(
                artifact,
                effect_floor=plan.point_effect_floor,
            )
        )

    corrected = _apply_holm_family_unsealed(
        estimates,
        family_id=plan.family_id,
        family_alpha=plan.family_alpha,
    )
    return tuple(replace(estimate, family_plan_digest=plan.digest) for estimate in corrected)


# ---------------------------------------------------------------------------
# Controlled marginal teachability protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingBranchPlan:
    """One branch of a token- and optimizer-budget-matched training intervention."""

    branch: ControlledBranch
    initial_checkpoint: str
    initial_checkpoint_sha256: str
    train_token_budget: int
    training_example_budget: int
    optimizer_step_budget: int
    optimizer_config_sha256: str
    trainer_sha256: str
    runtime_sha256: str
    training_rng_seed: int
    training_seeds: Tuple[int, ...]
    training_rng_sha256: str
    training_manifest_sha256: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "branch": self.branch.value,
            "initial_checkpoint": self.initial_checkpoint,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "train_token_budget": self.train_token_budget,
            "training_example_budget": self.training_example_budget,
            "optimizer_step_budget": self.optimizer_step_budget,
            "optimizer_config_sha256": self.optimizer_config_sha256,
            "trainer_sha256": self.trainer_sha256,
            "runtime_sha256": self.runtime_sha256,
            "training_rng_seed": self.training_rng_seed,
            "training_seeds": list(self.training_seeds),
            "training_rng_sha256": self.training_rng_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
        }


@dataclass(frozen=True)
class EvaluationUnit:
    """One precommitted holdout instance and policy-sampling draw."""

    eval_id: str
    environment_id: str
    environment_seed: int
    sampling_seed: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "environment_id": self.environment_id,
            "environment_seed": self.environment_seed,
            "sampling_seed": self.sampling_seed,
        }


@dataclass(frozen=True)
class HoldoutStratumPlan:
    """A same-family or explicitly related sibling-family holdout stratum."""

    stratum_id: str
    kind: HoldoutKind
    family_id: str
    sibling_of_family_id: Optional[str]
    units: Tuple[EvaluationUnit, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stratum_id": self.stratum_id,
            "kind": self.kind.value,
            "family_id": self.family_id,
            "sibling_of_family_id": self.sibling_of_family_id,
            "units": [unit.to_payload() for unit in self.units],
        }


@dataclass(frozen=True)
class ControlledMarginalProtocol:
    """Sealed plan for estimating a candidate's controlled marginal training effect."""

    schema_version: str
    protocol_id: str
    candidate_family_id: str
    treatment: TrainingBranchPlan
    control: TrainingBranchPlan
    holdout_strata: Tuple[HoldoutStratumPlan, ...]
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "candidate_family_id": self.candidate_family_id,
            "treatment": self.treatment.to_payload(),
            "control": self.control.to_payload(),
            "holdout_strata": [stratum.to_payload() for stratum in self.holdout_strata],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def make_controlled_marginal_protocol(
    *,
    protocol_id: str,
    candidate_family_id: str,
    treatment: TrainingBranchPlan,
    control: TrainingBranchPlan,
    holdout_strata: Sequence[HoldoutStratumPlan],
) -> ControlledMarginalProtocol:
    payload = {
        "schema_version": CONTROLLED_PROTOCOL_SCHEMA,
        "protocol_id": protocol_id,
        "candidate_family_id": candidate_family_id,
        "treatment": treatment.to_payload(),
        "control": control.to_payload(),
        "holdout_strata": [stratum.to_payload() for stratum in holdout_strata],
    }
    return ControlledMarginalProtocol(
        schema_version=CONTROLLED_PROTOCOL_SCHEMA,
        protocol_id=protocol_id,
        candidate_family_id=candidate_family_id,
        treatment=treatment,
        control=control,
        holdout_strata=tuple(holdout_strata),
        digest=_content_digest(payload),
    )


@dataclass(frozen=True)
class ProtocolValidation:
    """Fail-closed validation result for a controlled marginal protocol."""

    decision: ProtocolDecision
    protocol_id: str
    protocol_digest: str
    errors: Tuple[str, ...]
    train_seed_count: int
    eval_unit_count: int
    holdout_kinds: Tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.decision is ProtocolDecision.READY

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "protocol_id": self.protocol_id,
            "protocol_digest": self.protocol_digest,
            "errors": list(self.errors),
            "train_seed_count": self.train_seed_count,
            "eval_unit_count": self.eval_unit_count,
            "holdout_kinds": list(self.holdout_kinds),
        }


def _validate_branch(branch: TrainingBranchPlan, expected: ControlledBranch) -> List[str]:
    errors: List[str] = []
    if branch.branch is not expected:
        branch_value = (
            branch.branch.value
            if isinstance(branch.branch, ControlledBranch)
            else repr(branch.branch)
        )
        errors.append(f"{expected.value} plan has branch={branch_value}")
    try:
        _require_nonempty(branch.initial_checkpoint, f"{expected.value}.initial_checkpoint")
        _require_sha256(
            branch.initial_checkpoint_sha256,
            f"{expected.value}.initial_checkpoint_sha256",
        )
        _require_int(branch.train_token_budget, f"{expected.value}.train_token_budget", minimum=1)
        _require_int(
            branch.training_example_budget,
            f"{expected.value}.training_example_budget",
            minimum=1,
        )
        _require_int(
            branch.optimizer_step_budget,
            f"{expected.value}.optimizer_step_budget",
            minimum=1,
        )
        _require_sha256(
            branch.optimizer_config_sha256,
            f"{expected.value}.optimizer_config_sha256",
        )
        _require_sha256(branch.trainer_sha256, f"{expected.value}.trainer_sha256")
        _require_sha256(branch.runtime_sha256, f"{expected.value}.runtime_sha256")
        _require_seed(branch.training_rng_seed, f"{expected.value}.training_rng_seed")
        _require_sha256(
            branch.training_rng_sha256,
            f"{expected.value}.training_rng_sha256",
        )
        _require_sha256(
            branch.training_manifest_sha256,
            f"{expected.value}.training_manifest_sha256",
        )
    except ValueError as exc:
        errors.append(str(exc))
    if not branch.training_seeds:
        errors.append(f"{expected.value}.training_seeds must be non-empty")
    if len(branch.training_seeds) != len(set(branch.training_seeds)):
        errors.append(f"{expected.value}.training_seeds contains duplicates")
    for seed in branch.training_seeds:
        try:
            _require_seed(seed, f"{expected.value}.training_seed")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def validate_controlled_marginal_protocol(
    protocol: ControlledMarginalProtocol,
) -> ProtocolValidation:
    """Check every causal-control invariant without executing either branch."""

    errors: List[str] = []
    if protocol.schema_version != CONTROLLED_PROTOCOL_SCHEMA:
        errors.append(f"unsupported controlled protocol schema: {protocol.schema_version!r}")
    try:
        _require_nonempty(protocol.protocol_id, "protocol_id")
        _require_nonempty(protocol.candidate_family_id, "candidate_family_id")
    except ValueError as exc:
        errors.append(str(exc))

    errors.extend(_validate_branch(protocol.treatment, ControlledBranch.TREATMENT))
    errors.extend(_validate_branch(protocol.control, ControlledBranch.CONTROL))

    treatment = protocol.treatment
    control = protocol.control
    if treatment.initial_checkpoint != control.initial_checkpoint:
        errors.append("treatment and control must start from the same checkpoint")
    if treatment.initial_checkpoint_sha256 != control.initial_checkpoint_sha256:
        errors.append("treatment and control initial checkpoint digests must match")
    if treatment.train_token_budget != control.train_token_budget:
        errors.append("treatment and control train token budgets must be equal")
    if treatment.training_example_budget != control.training_example_budget:
        errors.append("treatment and control training example budgets must be equal")
    if treatment.optimizer_step_budget != control.optimizer_step_budget:
        errors.append("treatment and control optimizer step budgets must be equal")
    if treatment.optimizer_config_sha256 != control.optimizer_config_sha256:
        errors.append("treatment and control optimizer configurations must match")
    if treatment.trainer_sha256 != control.trainer_sha256:
        errors.append("treatment and control trainer digests must match")
    if treatment.runtime_sha256 != control.runtime_sha256:
        errors.append("treatment and control runtime digests must match")
    if treatment.training_rng_seed != control.training_rng_seed:
        errors.append("treatment and control training RNG seeds must match")
    if treatment.training_seeds != control.training_seeds:
        errors.append("treatment and control training seed schedules must match")
    if treatment.training_rng_sha256 != control.training_rng_sha256:
        errors.append("treatment and control training RNG digests must match")
    if treatment.training_manifest_sha256 == control.training_manifest_sha256:
        errors.append(
            "treatment and control training manifest digests must differ to identify the contrast"
        )

    kinds: List[HoldoutKind] = []
    stratum_ids: List[str] = []
    eval_ids: List[str] = []
    eval_environment_seeds: set[int] = set()
    eval_sampling_seeds: set[int] = set()
    eval_randomization_keys: List[Tuple[str, int, int]] = []
    eval_unit_count = 0
    for stratum in protocol.holdout_strata:
        if isinstance(stratum.kind, HoldoutKind):
            kinds.append(stratum.kind)
        stratum_ids.append(stratum.stratum_id)
        try:
            _require_nonempty(stratum.stratum_id, "stratum_id")
            _require_nonempty(stratum.family_id, f"{stratum.stratum_id}.family_id")
        except ValueError as exc:
            errors.append(str(exc))
        if not stratum.units:
            errors.append(f"holdout stratum {stratum.stratum_id!r} must be non-empty")
        if stratum.kind is HoldoutKind.SAME_FAMILY:
            if stratum.family_id != protocol.candidate_family_id:
                errors.append(
                    f"same-family stratum {stratum.stratum_id!r} must use candidate family"
                )
            if stratum.sibling_of_family_id is not None:
                errors.append(
                    f"same-family stratum {stratum.stratum_id!r} must not declare sibling_of"
                )
        elif stratum.kind is HoldoutKind.SIBLING:
            if stratum.family_id == protocol.candidate_family_id:
                errors.append(f"sibling stratum {stratum.stratum_id!r} must use a distinct family")
            if stratum.sibling_of_family_id != protocol.candidate_family_id:
                errors.append(
                    f"sibling stratum {stratum.stratum_id!r} must name the candidate family"
                )
        else:  # pragma: no cover - defensive against non-enum construction
            errors.append(f"unknown holdout kind for stratum {stratum.stratum_id!r}")

        for unit in stratum.units:
            eval_unit_count += 1
            eval_ids.append(unit.eval_id)
            try:
                _require_nonempty(unit.eval_id, "eval_id")
                _require_nonempty(unit.environment_id, f"{unit.eval_id}.environment_id")
                _require_seed(unit.environment_seed, f"{unit.eval_id}.environment_seed")
                _require_seed(unit.sampling_seed, f"{unit.eval_id}.sampling_seed")
            except ValueError as exc:
                errors.append(str(exc))
            eval_environment_seeds.add(unit.environment_seed)
            eval_sampling_seeds.add(unit.sampling_seed)
            eval_randomization_keys.append(
                (unit.environment_id, unit.environment_seed, unit.sampling_seed)
            )

    if len(stratum_ids) != len(set(stratum_ids)):
        errors.append("holdout stratum IDs must be unique")
    if len(eval_ids) != len(set(eval_ids)):
        errors.append("holdout eval IDs must be unique")
    if len(eval_randomization_keys) != len(set(eval_randomization_keys)):
        errors.append("holdout environment/seed/sampling tuples must be unique")
    if HoldoutKind.SAME_FAMILY not in kinds:
        errors.append("protocol requires a same-family holdout stratum")
    if HoldoutKind.SIBLING not in kinds:
        errors.append("protocol requires a sibling-family holdout stratum")

    train_seeds = set(treatment.training_seeds) | set(control.training_seeds)
    overlap_environment = sorted(train_seeds & eval_environment_seeds)
    overlap_sampling = sorted(train_seeds & eval_sampling_seeds)
    if overlap_environment:
        errors.append(f"training and holdout environment seeds overlap: {overlap_environment}")
    if overlap_sampling:
        errors.append(f"training and holdout sampling seeds overlap: {overlap_sampling}")

    try:
        expected_digest = _content_digest(protocol.to_payload(include_digest=False))
        _require_sha256(protocol.digest, "protocol.digest")
        if protocol.digest != expected_digest:
            errors.append("controlled protocol digest mismatch")
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    return ProtocolValidation(
        decision=ProtocolDecision.INVALID if errors else ProtocolDecision.READY,
        protocol_id=protocol.protocol_id,
        protocol_digest=protocol.digest,
        errors=tuple(errors),
        train_seed_count=len(train_seeds),
        eval_unit_count=eval_unit_count,
        holdout_kinds=tuple(sorted({kind.value for kind in kinds})),
    )


@dataclass(frozen=True)
class BranchExecutionReceipt:
    """Content-addressed claim that one planned training branch was executed exactly.

    Content addressing detects mutation and plan mismatch, but not who made the claim.  A
    production runner must authenticate the receipt through its trusted invocation/evidence layer.
    """

    schema_version: str
    protocol_digest: str
    branch: ControlledBranch
    initial_checkpoint_sha256: str
    output_checkpoint_sha256: str
    planned_train_token_budget: int
    realized_train_token_budget: int
    planned_training_example_budget: int
    realized_training_example_budget: int
    planned_optimizer_step_budget: int
    realized_optimizer_step_budget: int
    optimizer_config_sha256: str
    trainer_sha256: str
    runtime_sha256: str
    training_rng_sha256: str
    training_manifest_sha256: str
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "branch": self.branch.value,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "output_checkpoint_sha256": self.output_checkpoint_sha256,
            "planned_train_token_budget": self.planned_train_token_budget,
            "realized_train_token_budget": self.realized_train_token_budget,
            "planned_training_example_budget": self.planned_training_example_budget,
            "realized_training_example_budget": self.realized_training_example_budget,
            "planned_optimizer_step_budget": self.planned_optimizer_step_budget,
            "realized_optimizer_step_budget": self.realized_optimizer_step_budget,
            "optimizer_config_sha256": self.optimizer_config_sha256,
            "trainer_sha256": self.trainer_sha256,
            "runtime_sha256": self.runtime_sha256,
            "training_rng_sha256": self.training_rng_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _planned_branch(
    protocol: ControlledMarginalProtocol,
    branch: ControlledBranch,
) -> TrainingBranchPlan:
    if branch is ControlledBranch.TREATMENT:
        return protocol.treatment
    if branch is ControlledBranch.CONTROL:
        return protocol.control
    raise ValueError("branch must be a ControlledBranch")


def make_branch_execution_receipt(
    *,
    protocol: ControlledMarginalProtocol,
    branch: ControlledBranch,
    output_checkpoint_sha256: str,
    realized_train_token_budget: int,
    realized_training_example_budget: int,
    realized_optimizer_step_budget: int,
    optimizer_config_sha256: str,
    trainer_sha256: str,
    runtime_sha256: str,
    training_rng_sha256: str,
    training_manifest_sha256: str,
) -> BranchExecutionReceipt:
    """Record claimed branch execution; validation separately checks every claim against plan."""

    planned = _planned_branch(protocol, branch)
    payload = {
        "schema_version": BRANCH_RECEIPT_SCHEMA,
        "protocol_digest": protocol.digest,
        "branch": branch.value,
        "initial_checkpoint_sha256": planned.initial_checkpoint_sha256,
        "output_checkpoint_sha256": output_checkpoint_sha256,
        "planned_train_token_budget": planned.train_token_budget,
        "realized_train_token_budget": realized_train_token_budget,
        "planned_training_example_budget": planned.training_example_budget,
        "realized_training_example_budget": realized_training_example_budget,
        "planned_optimizer_step_budget": planned.optimizer_step_budget,
        "realized_optimizer_step_budget": realized_optimizer_step_budget,
        "optimizer_config_sha256": optimizer_config_sha256,
        "trainer_sha256": trainer_sha256,
        "runtime_sha256": runtime_sha256,
        "training_rng_sha256": training_rng_sha256,
        "training_manifest_sha256": training_manifest_sha256,
    }
    return BranchExecutionReceipt(
        schema_version=BRANCH_RECEIPT_SCHEMA,
        protocol_digest=protocol.digest,
        branch=branch,
        initial_checkpoint_sha256=planned.initial_checkpoint_sha256,
        output_checkpoint_sha256=output_checkpoint_sha256,
        planned_train_token_budget=planned.train_token_budget,
        realized_train_token_budget=realized_train_token_budget,
        planned_training_example_budget=planned.training_example_budget,
        realized_training_example_budget=realized_training_example_budget,
        planned_optimizer_step_budget=planned.optimizer_step_budget,
        realized_optimizer_step_budget=realized_optimizer_step_budget,
        optimizer_config_sha256=optimizer_config_sha256,
        trainer_sha256=trainer_sha256,
        runtime_sha256=runtime_sha256,
        training_rng_sha256=training_rng_sha256,
        training_manifest_sha256=training_manifest_sha256,
        digest=_content_digest(payload),
    )


def validate_branch_execution_receipt(
    receipt: BranchExecutionReceipt,
    *,
    protocol: ControlledMarginalProtocol,
    expected_branch: ControlledBranch,
) -> None:
    """Reject a forged, mismatched, or budget-deviating branch execution receipt."""

    if not isinstance(receipt, BranchExecutionReceipt):
        raise ArtifactIntegrityError("branch receipt must be a BranchExecutionReceipt")
    if not isinstance(protocol, ControlledMarginalProtocol):
        raise ArtifactIntegrityError("protocol must be a ControlledMarginalProtocol")
    protocol_validation = validate_controlled_marginal_protocol(protocol)
    if not protocol_validation.valid:
        raise ArtifactIntegrityError(
            "controlled protocol invalid: " + "; ".join(protocol_validation.errors)
        )
    if receipt.schema_version != BRANCH_RECEIPT_SCHEMA:
        raise ArtifactIntegrityError(
            f"unsupported branch receipt schema: {receipt.schema_version!r}"
        )
    if not isinstance(receipt.branch, ControlledBranch):
        raise ArtifactIntegrityError("branch receipt has an invalid branch label")
    if receipt.branch is not expected_branch:
        raise ArtifactIntegrityError(
            f"{expected_branch.value} branch receipt has branch={receipt.branch.value}"
        )
    plan = _planned_branch(protocol, expected_branch)
    try:
        _require_sha256(receipt.protocol_digest, "receipt.protocol_digest")
        _require_sha256(receipt.initial_checkpoint_sha256, "receipt.initial_checkpoint_sha256")
        _require_sha256(receipt.output_checkpoint_sha256, "receipt.output_checkpoint_sha256")
        _require_int(
            receipt.planned_train_token_budget,
            "receipt.planned_train_token_budget",
            minimum=1,
        )
        _require_int(
            receipt.realized_train_token_budget,
            "receipt.realized_train_token_budget",
        )
        _require_int(
            receipt.planned_training_example_budget,
            "receipt.planned_training_example_budget",
            minimum=1,
        )
        _require_int(
            receipt.realized_training_example_budget,
            "receipt.realized_training_example_budget",
        )
        _require_int(
            receipt.planned_optimizer_step_budget,
            "receipt.planned_optimizer_step_budget",
            minimum=1,
        )
        _require_int(
            receipt.realized_optimizer_step_budget,
            "receipt.realized_optimizer_step_budget",
        )
        _require_sha256(receipt.optimizer_config_sha256, "receipt.optimizer_config_sha256")
        _require_sha256(receipt.trainer_sha256, "receipt.trainer_sha256")
        _require_sha256(receipt.runtime_sha256, "receipt.runtime_sha256")
        _require_sha256(receipt.training_rng_sha256, "receipt.training_rng_sha256")
        _require_sha256(receipt.training_manifest_sha256, "receipt.training_manifest_sha256")
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc

    expected_values = {
        "protocol digest": (receipt.protocol_digest, protocol.digest),
        "initial checkpoint digest": (
            receipt.initial_checkpoint_sha256,
            plan.initial_checkpoint_sha256,
        ),
        "planned token budget": (
            receipt.planned_train_token_budget,
            plan.train_token_budget,
        ),
        "realized token budget": (
            receipt.realized_train_token_budget,
            plan.train_token_budget,
        ),
        "planned example budget": (
            receipt.planned_training_example_budget,
            plan.training_example_budget,
        ),
        "realized example budget": (
            receipt.realized_training_example_budget,
            plan.training_example_budget,
        ),
        "planned optimizer-step budget": (
            receipt.planned_optimizer_step_budget,
            plan.optimizer_step_budget,
        ),
        "realized optimizer-step budget": (
            receipt.realized_optimizer_step_budget,
            plan.optimizer_step_budget,
        ),
        "optimizer configuration digest": (
            receipt.optimizer_config_sha256,
            plan.optimizer_config_sha256,
        ),
        "trainer digest": (receipt.trainer_sha256, plan.trainer_sha256),
        "runtime digest": (receipt.runtime_sha256, plan.runtime_sha256),
        "training RNG digest": (receipt.training_rng_sha256, plan.training_rng_sha256),
        "training manifest digest": (
            receipt.training_manifest_sha256,
            plan.training_manifest_sha256,
        ),
    }
    mismatches = [
        name for name, (actual, expected) in expected_values.items() if actual != expected
    ]
    if mismatches:
        raise ArtifactIntegrityError(
            f"{expected_branch.value} branch receipt differs from its sealed plan: "
            + ", ".join(mismatches)
        )
    try:
        _require_sha256(receipt.digest, "receipt.digest")
        expected_digest = _content_digest(receipt.to_payload(include_digest=False))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    if receipt.digest != expected_digest:
        raise ArtifactIntegrityError(f"{expected_branch.value} branch receipt digest mismatch")


@dataclass(frozen=True)
class BranchOutcome:
    """One trained branch evaluated on one precommitted holdout unit."""

    branch: ControlledBranch
    eval_id: str
    environment_id: str
    environment_seed: int
    sampling_seed: int
    output_checkpoint_sha256: str
    status: OutcomeStatus
    reward: Optional[float]
    success: Optional[bool]
    error: Optional[str] = None
    failure_classification: Optional[FailureClassification] = None

    @classmethod
    def observed(
        cls,
        *,
        branch: ControlledBranch,
        unit: EvaluationUnit,
        output_checkpoint_sha256: str,
        reward: float,
        success: bool,
    ) -> "BranchOutcome":
        return cls(
            branch=branch,
            eval_id=unit.eval_id,
            environment_id=unit.environment_id,
            environment_seed=unit.environment_seed,
            sampling_seed=unit.sampling_seed,
            output_checkpoint_sha256=output_checkpoint_sha256,
            status=OutcomeStatus.OBSERVED,
            reward=reward,
            success=success,
            error=None,
            failure_classification=None,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        branch: ControlledBranch,
        unit: EvaluationUnit,
        output_checkpoint_sha256: str,
        error: str,
        failure_classification: FailureClassification,
    ) -> "BranchOutcome":
        return cls(
            branch=branch,
            eval_id=unit.eval_id,
            environment_id=unit.environment_id,
            environment_seed=unit.environment_seed,
            sampling_seed=unit.sampling_seed,
            output_checkpoint_sha256=output_checkpoint_sha256,
            status=OutcomeStatus.UNAVAILABLE,
            reward=None,
            success=None,
            error=error,
            failure_classification=failure_classification,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "branch": self.branch.value,
            "eval_id": self.eval_id,
            "environment_id": self.environment_id,
            "environment_seed": self.environment_seed,
            "sampling_seed": self.sampling_seed,
            "output_checkpoint_sha256": self.output_checkpoint_sha256,
            "status": self.status.value,
            "reward": self.reward,
            "success": self.success,
            "error": self.error,
            "failure_classification": (
                self.failure_classification.value
                if self.failure_classification is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ControlledOutcomePair:
    """Common-random-number treatment/control outcomes for one evaluation unit."""

    eval_id: str
    treatment: BranchOutcome
    control: BranchOutcome

    def to_payload(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "treatment": self.treatment.to_payload(),
            "control": self.control.to_payload(),
        }


@dataclass(frozen=True)
class ControlledMarginalArtifact:
    """Complete results for one sealed controlled marginal protocol."""

    schema_version: str
    protocol: ControlledMarginalProtocol
    treatment_receipt: BranchExecutionReceipt
    control_receipt: BranchExecutionReceipt
    pairs: Tuple[ControlledOutcomePair, ...]
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol": self.protocol.to_payload(),
            "treatment_receipt": self.treatment_receipt.to_payload(),
            "control_receipt": self.control_receipt.to_payload(),
            "pairs": [pair.to_payload() for pair in self.pairs],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def make_controlled_marginal_artifact(
    *,
    protocol: ControlledMarginalProtocol,
    treatment_receipt: BranchExecutionReceipt,
    control_receipt: BranchExecutionReceipt,
    pairs: Sequence[ControlledOutcomePair],
) -> ControlledMarginalArtifact:
    payload = {
        "schema_version": CONTROLLED_ARTIFACT_SCHEMA,
        "protocol": protocol.to_payload(),
        "treatment_receipt": treatment_receipt.to_payload(),
        "control_receipt": control_receipt.to_payload(),
        "pairs": [pair.to_payload() for pair in pairs],
    }
    return ControlledMarginalArtifact(
        schema_version=CONTROLLED_ARTIFACT_SCHEMA,
        protocol=protocol,
        treatment_receipt=treatment_receipt,
        control_receipt=control_receipt,
        pairs=tuple(pairs),
        digest=_content_digest(payload),
    )


def _protocol_units(
    protocol: ControlledMarginalProtocol,
) -> Tuple[Dict[str, EvaluationUnit], Dict[str, str]]:
    units: Dict[str, EvaluationUnit] = {}
    strata: Dict[str, str] = {}
    for stratum in protocol.holdout_strata:
        for unit in stratum.units:
            units[unit.eval_id] = unit
            strata[unit.eval_id] = stratum.stratum_id
    return units, strata


def _validate_branch_outcome(
    outcome: BranchOutcome,
    *,
    expected_branch: ControlledBranch,
    unit: EvaluationUnit,
    expected_output_checkpoint_sha256: str,
) -> None:
    if not isinstance(outcome.branch, ControlledBranch):
        raise ArtifactIntegrityError(
            f"eval {unit.eval_id} {expected_branch.value} has an invalid branch label"
        )
    if outcome.branch is not expected_branch:
        raise ArtifactIntegrityError(
            f"eval {unit.eval_id} {expected_branch.value} branch label mismatch"
        )
    if outcome.eval_id != unit.eval_id or outcome.environment_id != unit.environment_id:
        raise ArtifactIntegrityError(f"eval {unit.eval_id} unit identity mismatch")
    if outcome.environment_seed != unit.environment_seed:
        raise ArtifactIntegrityError(f"eval {unit.eval_id} environment seed mismatch")
    if outcome.sampling_seed != unit.sampling_seed:
        raise ArtifactIntegrityError(f"eval {unit.eval_id} sampling seed mismatch")
    try:
        _require_sha256(outcome.output_checkpoint_sha256, "output_checkpoint_sha256")
    except ValueError as exc:
        raise ArtifactIntegrityError(f"eval {unit.eval_id}: {exc}") from exc
    if outcome.output_checkpoint_sha256 != expected_output_checkpoint_sha256:
        raise ArtifactIntegrityError(
            f"eval {unit.eval_id} {expected_branch.value} output checkpoint does not match its "
            "branch execution receipt"
        )
    if not isinstance(outcome.status, OutcomeStatus):
        raise ArtifactIntegrityError(f"eval {unit.eval_id} branch has an invalid status")
    if outcome.status is not OutcomeStatus.OBSERVED:
        if outcome.reward is not None or outcome.success is not None:
            raise ArtifactIntegrityError(
                f"eval {unit.eval_id} unavailable branch contains an imputed outcome"
            )
        if not isinstance(outcome.failure_classification, FailureClassification):
            raise ArtifactIntegrityError(
                f"eval {unit.eval_id} unavailable branch lacks a valid failure classification"
            )
        try:
            _require_nonempty(outcome.error or "", "error")
        except ValueError as exc:
            raise ArtifactIntegrityError(f"eval {unit.eval_id}: {exc}") from exc
        if outcome.failure_classification is FailureClassification.CANDIDATE_OR_INTERVENTION:
            raise InterventionFailureError(
                f"eval {unit.eval_id} {expected_branch.value} failed because of the candidate or "
                "intervention; quality reject"
            )
        raise IncompleteArtifactError(
            f"eval {unit.eval_id} {expected_branch.value} unavailable: "
            f"{outcome.error or 'unspecified error'}"
        )
    if outcome.error is not None:
        raise ArtifactIntegrityError(f"eval {unit.eval_id} observed branch contains an error")
    if outcome.failure_classification is not None:
        raise ArtifactIntegrityError(
            f"eval {unit.eval_id} observed branch contains a failure classification"
        )
    if (
        outcome.reward is None
        or isinstance(outcome.reward, bool)
        or not isinstance(outcome.reward, (int, float))
        or not math.isfinite(float(outcome.reward))
        or not -1.0 <= float(outcome.reward) <= 1.0
    ):
        raise ArtifactIntegrityError(
            f"eval {unit.eval_id} observed reward must be finite and within [-1, 1]"
        )
    if not isinstance(outcome.success, bool):
        raise ArtifactIntegrityError(f"eval {unit.eval_id} observed success must be bool")


def _validate_controlled_artifact_self_digest(artifact: ControlledMarginalArtifact) -> None:
    """Verify serializable artifact bytes before assigning any evidence decision."""

    try:
        _require_sha256(artifact.digest, "controlled artifact digest")
    except ValueError as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    try:
        expected_digest = _content_digest(artifact.to_payload(include_digest=False))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            f"controlled artifact canonical serialization failed: {exc}"
        ) from exc
    if artifact.digest != expected_digest:
        raise ArtifactIntegrityError("controlled artifact digest mismatch")


def validate_complete_controlled_artifact(artifact: ControlledMarginalArtifact) -> None:
    """Validate a complete CRN-paired result set against its sealed protocol."""

    if artifact.schema_version != CONTROLLED_ARTIFACT_SCHEMA:
        raise ArtifactIntegrityError(
            f"unsupported controlled artifact schema: {artifact.schema_version!r}"
        )
    _validate_controlled_artifact_self_digest(artifact)
    protocol_validation = validate_controlled_marginal_protocol(artifact.protocol)
    if not protocol_validation.valid:
        raise ArtifactIntegrityError(
            "controlled protocol invalid: " + "; ".join(protocol_validation.errors)
        )
    validate_branch_execution_receipt(
        artifact.treatment_receipt,
        protocol=artifact.protocol,
        expected_branch=ControlledBranch.TREATMENT,
    )
    validate_branch_execution_receipt(
        artifact.control_receipt,
        protocol=artifact.protocol,
        expected_branch=ControlledBranch.CONTROL,
    )

    units, _ = _protocol_units(artifact.protocol)
    observed_ids = [pair.eval_id for pair in artifact.pairs]
    if len(observed_ids) != len(set(observed_ids)):
        raise ArtifactIntegrityError("controlled artifact contains duplicate eval IDs")
    extras = sorted(set(observed_ids) - set(units))
    missing = sorted(set(units) - set(observed_ids))
    if extras:
        raise ArtifactIntegrityError(f"controlled artifact contains unscheduled evals: {extras}")
    if missing:
        raise IncompleteArtifactError(f"controlled artifact is missing evals: {missing}")

    by_id = {pair.eval_id: pair for pair in artifact.pairs}
    for pair in artifact.pairs:
        for outcome in (pair.treatment, pair.control):
            if (
                outcome.status is OutcomeStatus.UNAVAILABLE
                and outcome.failure_classification
                is FailureClassification.CANDIDATE_OR_INTERVENTION
            ):
                branch_value = (
                    outcome.branch.value
                    if isinstance(outcome.branch, ControlledBranch)
                    else repr(outcome.branch)
                )
                raise InterventionFailureError(
                    f"eval {pair.eval_id} {branch_value} failed because of the candidate "
                    "or intervention; quality reject"
                )
    for eval_id, unit in units.items():
        pair = by_id[eval_id]
        _validate_branch_outcome(
            pair.treatment,
            expected_branch=ControlledBranch.TREATMENT,
            unit=unit,
            expected_output_checkpoint_sha256=(artifact.treatment_receipt.output_checkpoint_sha256),
        )
        _validate_branch_outcome(
            pair.control,
            expected_branch=ControlledBranch.CONTROL,
            unit=unit,
            expected_output_checkpoint_sha256=artifact.control_receipt.output_checkpoint_sha256,
        )
        # This repeats the plan checks intentionally: a malformed pair must fail even if both
        # arms agree with each other on the same wrong seed.
        if (
            pair.treatment.environment_seed != pair.control.environment_seed
            or pair.treatment.sampling_seed != pair.control.sampling_seed
        ):
            raise ArtifactIntegrityError(
                f"eval {eval_id} treatment/control outcomes do not use common random numbers"
            )

    if (
        artifact.treatment_receipt.output_checkpoint_sha256
        == artifact.control_receipt.output_checkpoint_sha256
    ):
        contradictory = [
            pair.eval_id
            for pair in artifact.pairs
            if pair.treatment.success != pair.control.success
            or pair.treatment.reward != pair.control.reward
        ]
        if contradictory:
            raise ArtifactIntegrityError(
                "identical treatment/control output checkpoint digests produced contradictory "
                f"common-random-number outcomes: {sorted(contradictory)}"
            )

    expected_digest = _content_digest(artifact.to_payload(include_digest=False))
    if artifact.digest != expected_digest:
        raise ArtifactIntegrityError("controlled artifact digest mismatch")


@dataclass(frozen=True)
class ControlledStratumEstimate:
    """Marginal treatment effect within one precommitted holdout stratum."""

    stratum_id: str
    kind: HoldoutKind
    pair_count: int
    treatment_only_success: int
    control_only_success: int
    both_success: int
    both_failure: int
    ate_success: float
    mean_reward_delta: float
    p_treatment_better: float
    p_control_better: float

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "stratum_id": self.stratum_id,
            "kind": self.kind.value,
            "pair_count": self.pair_count,
            "treatment_only_success": self.treatment_only_success,
            "control_only_success": self.control_only_success,
            "both_success": self.both_success,
            "both_failure": self.both_failure,
            "ate_success": self.ate_success,
            "mean_reward_delta": self.mean_reward_delta,
            "p_treatment_better": self.p_treatment_better,
            "p_control_better": self.p_control_better,
        }


@dataclass(frozen=True)
class ControlledMarginalAssessment:
    """Fail-closed descriptive evidence for exactly one controlled outer replicate."""

    decision: ControlledEvidenceDecision
    protocol_id: str
    protocol_digest: str
    planned_pairs: int
    complete_pairs: int
    ate_success: Optional[float]
    mean_reward_delta: Optional[float]
    p_treatment_better: float
    p_control_better: float
    strata: Tuple[ControlledStratumEstimate, ...]
    inference_scope: str
    quality_reject: bool
    reason: str
    validation_errors: Tuple[str, ...] = ()

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "protocol_id": self.protocol_id,
            "protocol_digest": self.protocol_digest,
            "planned_pairs": self.planned_pairs,
            "complete_pairs": self.complete_pairs,
            "ate_success": self.ate_success,
            "mean_reward_delta": self.mean_reward_delta,
            "p_treatment_better": self.p_treatment_better,
            "p_control_better": self.p_control_better,
            "strata": [stratum.to_metadata() for stratum in self.strata],
            "inference_scope": self.inference_scope,
            "quality_reject": self.quality_reject,
            "reason": self.reason,
            "validation_errors": list(self.validation_errors),
        }


def _controlled_failure(
    artifact: ControlledMarginalArtifact,
    decision: ControlledEvidenceDecision,
    reason: str,
    *,
    quality_reject: bool = False,
) -> ControlledMarginalAssessment:
    planned = sum(len(stratum.units) for stratum in artifact.protocol.holdout_strata)
    return ControlledMarginalAssessment(
        decision=decision,
        protocol_id=artifact.protocol.protocol_id,
        protocol_digest=artifact.protocol.digest,
        planned_pairs=planned,
        complete_pairs=0,
        ate_success=None,
        mean_reward_delta=None,
        p_treatment_better=1.0,
        p_control_better=1.0,
        strata=(),
        inference_scope=DESCRIPTIVE_REPLICATE_SCOPE,
        quality_reject=quality_reject,
        reason=reason,
        validation_errors=(reason,),
    )


def _summarize_controlled_pairs(
    stratum: HoldoutStratumPlan,
    pairs: Iterable[ControlledOutcomePair],
) -> ControlledStratumEstimate:
    treatment_only = 0
    control_only = 0
    both_success = 0
    both_failure = 0
    reward_deltas: List[float] = []
    pair_list = list(pairs)
    for pair in pair_list:
        treatment_success = bool(pair.treatment.success)
        control_success = bool(pair.control.success)
        if treatment_success and not control_success:
            treatment_only += 1
        elif control_success and not treatment_success:
            control_only += 1
        elif treatment_success:
            both_success += 1
        else:
            both_failure += 1
        assert pair.treatment.reward is not None and pair.control.reward is not None
        reward_deltas.append(float(pair.treatment.reward) - float(pair.control.reward))
    pair_count = len(pair_list)
    return ControlledStratumEstimate(
        stratum_id=stratum.stratum_id,
        kind=stratum.kind,
        pair_count=pair_count,
        treatment_only_success=treatment_only,
        control_only_success=control_only,
        both_success=both_success,
        both_failure=both_failure,
        ate_success=(treatment_only - control_only) / pair_count,
        mean_reward_delta=sum(reward_deltas) / pair_count,
        p_treatment_better=exact_mcnemar_one_sided(treatment_only, control_only),
        p_control_better=exact_mcnemar_one_sided(control_only, treatment_only),
    )


def assess_controlled_marginal_artifact(
    artifact: ControlledMarginalArtifact,
) -> ControlledMarginalAssessment:
    """Validate and descriptively summarize one CRN-controlled outer replicate.

    This function intentionally performs no inference across outer replicates.
    """

    protocol_validation = validate_controlled_marginal_protocol(artifact.protocol)
    if not protocol_validation.valid:
        return _controlled_failure(
            artifact,
            ControlledEvidenceDecision.ABSTAIN_PROTOCOL_INVALID,
            "; ".join(protocol_validation.errors),
        )
    try:
        validate_complete_controlled_artifact(artifact)
    except InterventionFailureError as exc:
        return _controlled_failure(
            artifact,
            ControlledEvidenceDecision.QUALITY_REJECT,
            str(exc),
            quality_reject=True,
        )
    except IncompleteArtifactError as exc:
        return _controlled_failure(
            artifact,
            ControlledEvidenceDecision.ABSTAIN_INCOMPLETE,
            str(exc),
        )
    except ArtifactValidationError as exc:
        return _controlled_failure(
            artifact,
            ControlledEvidenceDecision.INVALID_ARTIFACT,
            str(exc),
        )

    _, stratum_by_eval = _protocol_units(artifact.protocol)
    pairs_by_stratum: Dict[str, List[ControlledOutcomePair]] = {
        stratum.stratum_id: [] for stratum in artifact.protocol.holdout_strata
    }
    for pair in artifact.pairs:
        pairs_by_stratum[stratum_by_eval[pair.eval_id]].append(pair)

    strata = tuple(
        _summarize_controlled_pairs(stratum, pairs_by_stratum[stratum.stratum_id])
        for stratum in artifact.protocol.holdout_strata
    )
    total_pairs = sum(stratum.pair_count for stratum in strata)
    treatment_only = sum(stratum.treatment_only_success for stratum in strata)
    control_only = sum(stratum.control_only_success for stratum in strata)
    weighted_reward_delta = (
        sum(stratum.mean_reward_delta * stratum.pair_count for stratum in strata) / total_pairs
    )

    return ControlledMarginalAssessment(
        decision=ControlledEvidenceDecision.ESTIMATED,
        protocol_id=artifact.protocol.protocol_id,
        protocol_digest=artifact.protocol.digest,
        planned_pairs=total_pairs,
        complete_pairs=total_pairs,
        ate_success=(treatment_only - control_only) / total_pairs,
        mean_reward_delta=weighted_reward_delta,
        p_treatment_better=exact_mcnemar_one_sided(treatment_only, control_only),
        p_control_better=exact_mcnemar_one_sided(control_only, treatment_only),
        strata=strata,
        inference_scope=DESCRIPTIVE_REPLICATE_SCOPE,
        quality_reject=False,
        reason=(
            "complete descriptive controlled marginal evidence for one outer replicate across "
            "required holdout strata; no across-replicate inference is implied"
        ),
    )


__all__ = [
    "Arm",
    "ArmOrder",
    "ArmOutcome",
    "ArtifactIntegrityError",
    "ArtifactValidationError",
    "BranchExecutionReceipt",
    "BranchOutcome",
    "ControlledBranch",
    "ControlledEvidenceDecision",
    "ControlledMarginalArtifact",
    "ControlledMarginalAssessment",
    "ControlledMarginalProtocol",
    "ControlledOutcomePair",
    "ControlledStratumEstimate",
    "DESCRIPTIVE_REPLICATE_SCOPE",
    "EvaluationUnit",
    "FailureClassification",
    "HOLM_DIRECTIONAL_BONFERRONI",
    "HoldoutKind",
    "HoldoutStratumPlan",
    "IncompleteArtifactError",
    "InterventionFailureError",
    "OutcomeStatus",
    "PairOutcome",
    "PairSchedule",
    "PairSpec",
    "PairedRunArtifact",
    "PairedFamilyMember",
    "PairedFamilyPlan",
    "PairedTeachabilityEstimate",
    "ProtocolDecision",
    "ProtocolValidation",
    "TeachabilityDecision",
    "TrainingBranchPlan",
    "assess_controlled_marginal_artifact",
    "assess_paired_family",
    "assess_paired_artifact",
    "exact_mcnemar_one_sided",
    "make_branch_execution_receipt",
    "make_controlled_marginal_artifact",
    "make_controlled_marginal_protocol",
    "make_pair_schedule",
    "make_paired_family_member",
    "make_paired_family_plan",
    "make_paired_run_artifact",
    "sha256_hex",
    "validate_branch_execution_receipt",
    "validate_complete_controlled_artifact",
    "validate_complete_pair_artifact",
    "validate_controlled_marginal_protocol",
    "validate_pair_schedule",
    "validate_paired_family_plan",
]
