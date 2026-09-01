"""Pure portfolio locking and sensitivity analysis for the coverage pilot.

This module has no provider, environment, or filesystem boundary. It locks a
quality-matched coverage-forced swap before actor outcomes and analyzes the
realized hinted/unhinted association. Neither exact test is design-based: the
label-permutation test needs strong within-stratum label exchangeability, and
the sign-flip test needs symmetry of the six stratum contrasts.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PERMUTATIONS_PER_STRATUM = 6
PILOT_STRATA = 6
EXACT_LABEL_PERMUTATION_COUNT = PERMUTATIONS_PER_STRATUM**PILOT_STRATA
EXACT_SIGN_FLIP_COUNT = 2**PILOT_STRATA

STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR: Mapping[str, Any] = {
    "action_format": "boxed",
    "oracle_depth_bin": 1,
    "reset_seed_diversity_bin": 2,
    "invalid_reward_bin": 0,
    "invalid_end_state": "continues",
    "recovery_success": "observed_true",
    "trace_order_divergent": "unmeasured",
}

COVERAGE_CHALLENGER_DESCRIPTOR: Mapping[str, Any] = {
    "action_format": "boxed",
    "oracle_depth_bin": 1,
    "reset_seed_diversity_bin": 2,
    "invalid_reward_bin": 0,
    "invalid_end_state": "continues",
    "recovery_success": "observed_false",
    "trace_order_divergent": "unmeasured",
}


def _finite(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _descriptor_tuple(value: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    if not value or any(not isinstance(key, str) or not key for key in value):
        raise ValueError("descriptor must have non-empty string coordinates")
    normalized: list[tuple[str, str]] = []
    for key, item in sorted(value.items()):
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError("descriptor coordinates must be strings or integers")
        normalized.append((key, str(item)))
    return tuple(normalized)


_STANDARD_HISTORICAL_CELL = _descriptor_tuple(STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR)
_COVERAGE_CHALLENGER_CELL = _descriptor_tuple(COVERAGE_CHALLENGER_DESCRIPTOR)


@dataclass(frozen=True)
class ProxyCandidate:
    """One qualified, pre-outcome candidate with sealed CWA evidence."""

    candidate_id: str
    stratum_id: str
    source_arm: str
    quality_score: float
    descriptor: tuple[tuple[str, str], ...]
    environment_digest: str
    evidence_digest: str

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        stratum_id: str,
        source_arm: str,
        quality_score: float,
        descriptor: Mapping[str, Any],
        environment_digest: str,
        evidence_digest: str,
    ) -> "ProxyCandidate":
        for name, value in (
            ("candidate_id", candidate_id),
            ("stratum_id", stratum_id),
            ("source_arm", source_arm),
            ("environment_digest", environment_digest),
            ("evidence_digest", evidence_digest),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        quality = _finite(quality_score, "quality_score")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        return cls(
            candidate_id=candidate_id,
            stratum_id=stratum_id,
            source_arm=source_arm,
            quality_score=quality,
            descriptor=_descriptor_tuple(descriptor),
            environment_digest=environment_digest,
            evidence_digest=evidence_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stratum_id": self.stratum_id,
            "source_arm": self.source_arm,
            "quality_score": self.quality_score,
            "descriptor": dict(self.descriptor),
            "environment_digest": self.environment_digest,
            "evidence_digest": self.evidence_digest,
        }


def descriptor_distance(left: ProxyCandidate, right: ProxyCandidate) -> int:
    """Hamming distance over identical descriptor coordinates."""
    if tuple(key for key, _ in left.descriptor) != tuple(key for key, _ in right.descriptor):
        raise ValueError("candidate descriptors use different coordinates")
    return sum(a != b for (_, a), (_, b) in zip(left.descriptor, right.descriptor))


def _identity_order(candidate: ProxyCandidate) -> tuple[str, str]:
    return (candidate.environment_digest, candidate.evidence_digest)


@dataclass(frozen=True)
class LockedPortfolios:
    """One prospective redundant-control versus coverage-forced matched swap."""

    stratum_id: str
    candidate_ids: tuple[str, str, str]
    coverage_forced: tuple[str, str]
    redundant_historical: tuple[str, str]
    challenger_id: str
    retained_historical_id: str
    displaced_historical_id: str
    coverage_forced_quality: float
    redundant_historical_quality: float
    signed_quality_gap: float
    absolute_quality_gap: float

    @property
    def differs(self) -> bool:
        return set(self.coverage_forced) != set(self.redundant_historical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratum_id": self.stratum_id,
            "candidate_ids": list(self.candidate_ids),
            "coverage_forced": list(self.coverage_forced),
            "redundant_historical": list(self.redundant_historical),
            "challenger_id": self.challenger_id,
            "retained_historical_id": self.retained_historical_id,
            "displaced_historical_id": self.displaced_historical_id,
            "coverage_forced_quality": self.coverage_forced_quality,
            "redundant_historical_quality": self.redundant_historical_quality,
            "signed_quality_gap": self.signed_quality_gap,
            "absolute_quality_gap": self.absolute_quality_gap,
            "differs": self.differs,
        }


def lock_portfolios(candidates: Sequence[ProxyCandidate]) -> LockedPortfolios:
    """Lock the exact quality-matched coverage-forced portfolio swap.

    The redundant control is the two historical D2/recovery-success packages.
    Coverage is the target-cell challenger plus the historical package that was
    not displaced. The displaced historical is the one whose CWA quality is
    closest to the challenger; ties use only sealed environment/evidence
    digests, never source or candidate labels.
    """
    if len(candidates) != 3:
        raise ValueError("each sealed stratum requires exactly three candidates")
    ids = {item.candidate_id for item in candidates}
    strata = {item.stratum_id for item in candidates}
    sources = {item.source_arm for item in candidates}
    environment_digests = {item.environment_digest for item in candidates}
    evidence_digests = {item.evidence_digest for item in candidates}
    if len(ids) != 3 or len(strata) != 1:
        raise ValueError("candidate ids must be unique within one stratum")
    if sources != {"v3", "v4", "challenger"}:
        raise ValueError("each stratum requires exactly v3, v4, and challenger")
    if len(environment_digests) != 3 or len(evidence_digests) != 3:
        raise ValueError("environment and evidence digests must be unique")

    by_source = {item.source_arm: item for item in candidates}
    historical = (by_source["v3"], by_source["v4"])
    challenger = by_source["challenger"]
    if any(item.descriptor != _STANDARD_HISTORICAL_CELL for item in historical):
        raise ValueError("both historical controls must occupy the exact standard D2 cell")
    if challenger.descriptor != _COVERAGE_CHALLENGER_CELL:
        raise ValueError("challenger does not occupy the exact coverage target cell")

    displaced = min(
        historical,
        key=lambda item: (
            abs(item.quality_score - challenger.quality_score),
            *_identity_order(item),
        ),
    )
    retained = next(item for item in historical if item is not displaced)
    redundant = tuple(item.candidate_id for item in sorted(historical, key=_identity_order))
    coverage = (challenger.candidate_id, retained.candidate_id)
    redundant_quality = math.fsum(item.quality_score for item in historical) / 2.0
    coverage_quality = (challenger.quality_score + retained.quality_score) / 2.0
    signed_gap = coverage_quality - redundant_quality
    candidate_ids = tuple(item.candidate_id for item in sorted(candidates, key=_identity_order))
    return LockedPortfolios(
        stratum_id=challenger.stratum_id,
        candidate_ids=candidate_ids,
        coverage_forced=coverage,
        redundant_historical=redundant,
        challenger_id=challenger.candidate_id,
        retained_historical_id=retained.candidate_id,
        displaced_historical_id=displaced.candidate_id,
        coverage_forced_quality=coverage_quality,
        redundant_historical_quality=redundant_quality,
        signed_quality_gap=signed_gap,
        absolute_quality_gap=abs(signed_gap),
    )


def portfolio_quality_diagnostics(locks: Sequence[LockedPortfolios]) -> dict[str, Any]:
    if len(locks) != PILOT_STRATA:
        raise ValueError(f"quality diagnostics require exactly {PILOT_STRATA} strata")
    signed = tuple(item.signed_quality_gap for item in locks)
    absolute = tuple(item.absolute_quality_gap for item in locks)
    return {
        "signed_quality_gaps": list(signed),
        "absolute_quality_gaps": list(absolute),
        "maximum_absolute_quality_gap": max(absolute),
        "mean_absolute_quality_gap": math.fsum(absolute) / len(absolute),
    }


def lock_all_portfolios(
    candidates: Sequence[ProxyCandidate],
    *,
    maximum_stratum_absolute_quality_gap: float = 0.125,
    maximum_mean_absolute_quality_gap: float = 0.0625,
) -> tuple[LockedPortfolios, ...]:
    expected_candidates = PILOT_STRATA * 3
    if len(candidates) != expected_candidates:
        raise ValueError(f"the pilot requires exactly {expected_candidates} candidates")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("candidate ids must be globally unique")
    if len({item.environment_digest for item in candidates}) != len(candidates):
        raise ValueError("environment bytes must be globally unique across the panel")
    if len({item.evidence_digest for item in candidates}) != len(candidates):
        raise ValueError("candidate evidence digests must be globally unique")
    grouped: dict[str, list[ProxyCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.stratum_id, []).append(candidate)
    if len(grouped) != PILOT_STRATA:
        raise ValueError(f"the pilot requires exactly {PILOT_STRATA} strata")
    locks = tuple(lock_portfolios(grouped[key]) for key in sorted(grouped))
    if not all(item.differs for item in locks):
        raise ValueError("coverage and redundant portfolios must differ in every stratum")
    diagnostics = portfolio_quality_diagnostics(locks)
    maximum = _finite(maximum_stratum_absolute_quality_gap, "maximum_stratum_absolute_quality_gap")
    mean_maximum = _finite(maximum_mean_absolute_quality_gap, "maximum_mean_absolute_quality_gap")
    if diagnostics["maximum_absolute_quality_gap"] > maximum:
        raise ValueError("per-stratum portfolio quality-match gate failed")
    if diagnostics["mean_absolute_quality_gap"] > mean_maximum:
        raise ValueError("mean portfolio quality-match gate failed")
    return locks


@dataclass(frozen=True)
class PairedOutcome:
    candidate_id: str
    stratum_id: str
    seed: int
    unhinted: float
    hinted: float
    first_attempt_exogenous_failure: bool = False
    parser_failure: bool = False
    task_failure: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.stratum_id or type(self.seed) is not int:
            raise ValueError("outcome identity is invalid")
        _finite(self.unhinted, "unhinted")
        _finite(self.hinted, "hinted")
        for value in (
            self.first_attempt_exogenous_failure,
            self.parser_failure,
            self.task_failure,
        ):
            if type(value) is not bool:
                raise ValueError("outcome flags must be booleans")

    @property
    def gain(self) -> float:
        return float(self.hinted) - float(self.unhinted)

    @property
    def discordant(self) -> bool:
        return float(self.hinted) != float(self.unhinted)


def _candidate_gains(outcomes: Sequence[PairedOutcome]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in outcomes:
        grouped.setdefault(item.candidate_id, []).append(item.gain)
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("every candidate requires exactly two seed-paired outcomes")
    return {key: math.fsum(values) / len(values) for key, values in grouped.items()}


def _contrast(
    locks: Sequence[LockedPortfolios], candidate_gain: Mapping[str, float]
) -> tuple[float, tuple[float, ...]]:
    strata: list[float] = []
    for lock in locks:
        try:
            coverage = math.fsum(candidate_gain[key] for key in lock.coverage_forced) / 2.0
            redundant = math.fsum(candidate_gain[key] for key in lock.redundant_historical) / 2.0
        except KeyError as exc:
            raise ValueError(f"outcomes lack selected candidate {exc.args[0]}") from exc
        strata.append(coverage - redundant)
    return math.fsum(strata) / len(strata), tuple(strata)


def exact_stratified_label_permutation_pvalue(
    locks: Sequence[LockedPortfolios], candidate_gain: Mapping[str, float]
) -> tuple[float, int, int]:
    """Enumerate the exact ``(3!)**6`` label-permutation sensitivity space."""
    if len(locks) != PILOT_STRATA:
        raise ValueError(f"exact pilot analysis requires {PILOT_STRATA} strata")
    observed, _ = _contrast(locks, candidate_gain)
    stratum_values: list[tuple[float, ...]] = []
    for lock in locks:
        ids = lock.candidate_ids
        if len(ids) != 3:
            raise ValueError(f"portfolio lock lacks three candidates for {lock.stratum_id}")
        values = tuple(candidate_gain[key] for key in ids)
        effects: list[float] = []
        for assigned in itertools.permutations(values):
            permuted = dict(zip(ids, assigned))
            effect, _ = _contrast((lock,), permuted)
            effects.append(effect)
        stratum_values.append(tuple(effects))
    extreme = 0
    total = 0
    tolerance = 1e-12
    for assignment in itertools.product(*stratum_values):
        statistic = math.fsum(assignment) / len(assignment)
        total += 1
        if statistic + tolerance >= observed:
            extreme += 1
    if total != EXACT_LABEL_PERMUTATION_COUNT:
        raise AssertionError("exact label-permutation space has the wrong cardinality")
    return extreme / total, extreme, total


def exact_stratum_sign_flip_pvalue(
    stratum_contrasts: Sequence[float],
) -> tuple[float, int, int]:
    """Enumerate the exact one-sided ``2**6`` sign-flip sensitivity space."""
    if len(stratum_contrasts) != PILOT_STRATA:
        raise ValueError(f"sign-flip sensitivity requires {PILOT_STRATA} contrasts")
    contrasts = tuple(_finite(value, "stratum_contrast") for value in stratum_contrasts)
    observed = math.fsum(contrasts) / len(contrasts)
    extreme = 0
    total = 0
    tolerance = 1e-12
    for signs in itertools.product((-1.0, 1.0), repeat=PILOT_STRATA):
        statistic = math.fsum(sign * value for sign, value in zip(signs, contrasts)) / len(
            contrasts
        )
        total += 1
        if statistic + tolerance >= observed:
            extreme += 1
    if total != EXACT_SIGN_FLIP_COUNT:
        raise AssertionError("exact sign-flip space has the wrong cardinality")
    return extreme / total, extreme, total


@dataclass(frozen=True)
class ProxyAnalysis:
    passed: bool
    pooled_unhinted_rate: float
    discordant_pairs: int
    coverage_forced_delta: float
    label_permutation_p_value: float
    label_permutation_extreme: int
    label_permutation_total: int
    sign_flip_p_value: float
    sign_flip_extreme: int
    sign_flip_total: int
    stratum_contrasts: tuple[float, ...]
    leave_one_stratum_out: tuple[float, ...]
    first_attempt_exogenous_failure_rate: float
    parser_failures: int
    task_failures: int
    gates: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pooled_unhinted_rate": self.pooled_unhinted_rate,
            "discordant_pairs": self.discordant_pairs,
            "coverage_forced_delta": self.coverage_forced_delta,
            "label_permutation_p_value": self.label_permutation_p_value,
            "label_permutation_extreme": self.label_permutation_extreme,
            "label_permutation_total": self.label_permutation_total,
            "sign_flip_p_value": self.sign_flip_p_value,
            "sign_flip_extreme": self.sign_flip_extreme,
            "sign_flip_total": self.sign_flip_total,
            "stratum_contrasts": list(self.stratum_contrasts),
            "leave_one_stratum_out": list(self.leave_one_stratum_out),
            "first_attempt_exogenous_failure_rate": self.first_attempt_exogenous_failure_rate,
            "parser_failures": self.parser_failures,
            "task_failures": self.task_failures,
            "unit_interpretation": (
                "seeds 0 and 42 are averaged within each realized environment-plus-source-"
                "specific-hint package and are not independent environments"
            ),
            "inference_boundary": (
                "exploratory noncausal matched-swap association; label permutation and sign "
                "flip are sensitivity analyses, not design-based tests"
            ),
            "gates": dict(self.gates),
        }


def analyze_proxy_pilot(
    locks: Sequence[LockedPortfolios],
    outcomes: Sequence[PairedOutcome],
    *,
    baseline_min: float = 0.10,
    baseline_max: float = 0.90,
    minimum_discordant: int = 8,
    minimum_delta: float = 0.10,
    alpha: float = 0.05,
    maximum_first_attempt_failure_rate: float = 0.15,
) -> ProxyAnalysis:
    """Apply prospective gates to the complete 18-candidate paired panel."""
    if len(locks) != PILOT_STRATA or len(outcomes) != PILOT_STRATA * 3 * 2:
        raise ValueError("analysis requires six portfolio locks and 36 seed pairs")
    keys = {(item.candidate_id, item.seed) for item in outcomes}
    if len(keys) != len(outcomes):
        raise ValueError("outcomes contain duplicate candidate/seed pairs")
    gains = _candidate_gains(outcomes)
    if len(gains) != PILOT_STRATA * 3:
        raise ValueError("analysis requires exactly 18 candidates")
    outcome_topology = {(item.stratum_id, item.candidate_id, item.seed) for item in outcomes}
    expected_topology = {
        (lock.stratum_id, candidate_id, seed)
        for lock in locks
        for candidate_id in lock.candidate_ids
        for seed in (0, 42)
    }
    if outcome_topology != expected_topology:
        raise ValueError("outcomes differ from the locked 6 x 3 x {0,42} topology")

    delta, strata = _contrast(locks, gains)
    permutation_p, permutation_extreme, permutation_total = (
        exact_stratified_label_permutation_pvalue(locks, gains)
    )
    sign_p, sign_extreme, sign_total = exact_stratum_sign_flip_pvalue(strata)
    leave_one_out = tuple(
        math.fsum(value for index, value in enumerate(strata) if index != omitted)
        / (len(strata) - 1)
        for omitted in range(len(strata))
    )
    baseline = math.fsum(float(item.unhinted) for item in outcomes) / len(outcomes)
    discordant = sum(item.discordant for item in outcomes)
    exogenous = sum(item.first_attempt_exogenous_failure for item in outcomes) / len(outcomes)
    parser_failures = sum(item.parser_failure for item in outcomes)
    task_failures = sum(item.task_failure for item in outcomes)
    gates = {
        "pooled_baseline_within_bounds": baseline_min <= baseline <= baseline_max,
        "minimum_discordant_pairs": discordant >= minimum_discordant,
        "minimum_coverage_forced_delta": delta >= minimum_delta,
        "one_sided_exact_label_permutation_sensitivity": permutation_p <= alpha,
        "one_sided_exact_sign_flip_sensitivity": sign_p <= alpha,
        "every_leave_one_stratum_out_positive": all(value > 0 for value in leave_one_out),
        "first_attempt_exogenous_failure_rate": exogenous <= maximum_first_attempt_failure_rate,
    }
    return ProxyAnalysis(
        passed=all(gates.values()),
        pooled_unhinted_rate=baseline,
        discordant_pairs=discordant,
        coverage_forced_delta=delta,
        label_permutation_p_value=permutation_p,
        label_permutation_extreme=permutation_extreme,
        label_permutation_total=permutation_total,
        sign_flip_p_value=sign_p,
        sign_flip_extreme=sign_extreme,
        sign_flip_total=sign_total,
        stratum_contrasts=strata,
        leave_one_stratum_out=leave_one_out,
        first_attempt_exogenous_failure_rate=exogenous,
        parser_failures=parser_failures,
        task_failures=task_failures,
        gates=gates,
    )


def counterbalanced_pairs(
    candidate_ids_by_stratum: Mapping[str, Sequence[str]], seeds: Sequence[int]
) -> tuple[dict[str, Any], ...]:
    """Return the immutable 36-pair globally interleaved schedule."""
    if tuple(seeds) != (0, 42):
        raise ValueError("the sealed proxy pilot requires seeds [0, 42]")
    if len(candidate_ids_by_stratum) != PILOT_STRATA:
        raise ValueError(f"the sealed proxy pilot requires {PILOT_STRATA} strata")
    ordered_strata = sorted(candidate_ids_by_stratum)
    rotated: dict[str, tuple[str, ...]] = {}
    for stratum_index, stratum_id in enumerate(ordered_strata):
        candidates = tuple(candidate_ids_by_stratum[stratum_id])
        if len(candidates) != 3 or len(set(candidates)) != 3:
            raise ValueError("each stratum requires three unique candidate ids")
        offset = stratum_index % len(candidates)
        rotated[stratum_id] = candidates[offset:] + candidates[:offset]

    schedule: list[dict[str, Any]] = []
    ordinal = 0
    for candidate_position in range(3):
        for stratum_index, stratum_id in enumerate(ordered_strata):
            candidate_id = rotated[stratum_id][candidate_position]
            for seed_index, seed in enumerate(seeds):
                arm_order = (
                    ("unhinted", "hinted")
                    if (stratum_index + candidate_position + seed_index) % 2 == 0
                    else ("hinted", "unhinted")
                )
                schedule.append(
                    {
                        "pair_ordinal": ordinal,
                        "pair_id": f"pair-{ordinal:02d}",
                        "stratum_id": stratum_id,
                        "candidate_id": candidate_id,
                        "candidate_position_round": candidate_position,
                        "source_order_policy": "latin-rotated-global-position-rounds",
                        "seed": seed,
                        "arm_order": list(arm_order),
                        "horizon": 1,
                        "success_rule": "reward-positive-even-if-simultaneously-truncated",
                    }
                )
                ordinal += 1
    if len(schedule) != PILOT_STRATA * 3 * 2:
        raise AssertionError("sealed pair schedule cardinality drifted")
    return tuple(schedule)


def attempt_waves(
    schedule: Sequence[Mapping[str, Any]],
    unresolved_after_first: Iterable[str],
    unresolved_after_second: Iterable[str],
    *,
    third_wave_limit: int = 14,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the whole-pair retry policy and optional bounded third wave."""
    expected_pairs = PILOT_STRATA * 3 * 2
    all_ids = tuple(str(item["pair_id"]) for item in schedule)
    if len(all_ids) != expected_pairs or len(set(all_ids)) != expected_pairs:
        raise ValueError(f"attempt wave requires the exact {expected_pairs}-pair schedule")
    first_unresolved = tuple(unresolved_after_first)
    second_unresolved = tuple(unresolved_after_second)
    if len(set(first_unresolved)) != len(first_unresolved) or not set(first_unresolved).issubset(
        all_ids
    ):
        raise ValueError("first unresolved set is invalid")
    if len(set(second_unresolved)) != len(second_unresolved) or not set(second_unresolved).issubset(
        first_unresolved
    ):
        raise ValueError("second unresolved set must be a subset of the first")
    if len(second_unresolved) > third_wave_limit:
        raise ValueError(f"attempt 3 is forbidden with {len(second_unresolved)} unresolved pairs")
    order = {pair_id: index for index, pair_id in enumerate(all_ids)}
    return (
        all_ids,
        tuple(sorted(first_unresolved, key=order.__getitem__)),
        tuple(sorted(second_unresolved, key=order.__getitem__)),
    )


__all__ = [
    "COVERAGE_CHALLENGER_DESCRIPTOR",
    "EXACT_LABEL_PERMUTATION_COUNT",
    "EXACT_SIGN_FLIP_COUNT",
    "LockedPortfolios",
    "PILOT_STRATA",
    "PairedOutcome",
    "ProxyAnalysis",
    "ProxyCandidate",
    "STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR",
    "analyze_proxy_pilot",
    "attempt_waves",
    "counterbalanced_pairs",
    "descriptor_distance",
    "exact_stratified_label_permutation_pvalue",
    "exact_stratum_sign_flip_pvalue",
    "lock_all_portfolios",
    "lock_portfolios",
    "portfolio_quality_diagnostics",
]
