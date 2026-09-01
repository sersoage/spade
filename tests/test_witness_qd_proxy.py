from __future__ import annotations

from dataclasses import replace

import pytest

from spade.core.witness_qd_proxy import (
    COVERAGE_CHALLENGER_DESCRIPTOR,
    EXACT_LABEL_PERMUTATION_COUNT,
    EXACT_SIGN_FLIP_COUNT,
    STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR,
    PairedOutcome,
    ProxyCandidate,
    analyze_proxy_pilot,
    attempt_waves,
    counterbalanced_pairs,
    exact_stratum_sign_flip_pvalue,
    lock_all_portfolios,
    lock_portfolios,
    portfolio_quality_diagnostics,
)


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _candidate(
    stratum: int,
    arm: str,
    quality: float,
    digest_index: int,
) -> ProxyCandidate:
    descriptor = (
        COVERAGE_CHALLENGER_DESCRIPTOR
        if arm == "challenger"
        else STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR
    )
    return ProxyCandidate.create(
        candidate_id=f"c{stratum:03d}--{arm}",
        stratum_id=f"c{stratum:03d}",
        source_arm=arm,
        quality_score=quality,
        descriptor=descriptor,
        environment_digest=_digest(digest_index),
        evidence_digest=_digest(10_000 + digest_index),
    )


def _panel() -> list[ProxyCandidate]:
    candidates: list[ProxyCandidate] = []
    for stratum in (1, 3, 4, 5, 6, 7):
        base_index = stratum * 10
        candidates.extend(
            (
                _candidate(stratum, "v3", 0.75, base_index + 1),
                _candidate(stratum, "v4", 1.00, base_index + 2),
                _candidate(stratum, "challenger", 1.00, base_index + 3),
            )
        )
    return candidates


def test_exact_target_cells_and_quality_nearest_displacement() -> None:
    candidates = _panel()[:3]
    lock = lock_portfolios(candidates)
    assert lock.displaced_historical_id == "c001--v4"
    assert set(lock.coverage_forced) == {"c001--challenger", "c001--v3"}
    assert set(lock.redundant_historical) == {"c001--v3", "c001--v4"}
    assert lock.signed_quality_gap == 0.0
    assert lock.absolute_quality_gap == 0.0

    wrong_cell = replace(candidates[-1], descriptor=candidates[0].descriptor)
    with pytest.raises(ValueError, match="target cell"):
        lock_portfolios([*candidates[:-1], wrong_cell])
    wrong_historical = replace(candidates[0], descriptor=candidates[-1].descriptor)
    with pytest.raises(ValueError, match="standard D2"):
        lock_portfolios([wrong_historical, *candidates[1:]])


def test_displacement_tie_uses_digests_not_source_or_candidate_labels() -> None:
    candidates = [replace(item, quality_score=1.0) for item in _panel()[:3]]
    first = lock_portfolios(candidates)
    renamed = [
        replace(item, candidate_id=f"renamed-{index}") for index, item in enumerate(candidates)
    ]
    second = lock_portfolios(renamed)
    first_displaced_digest = next(
        item.environment_digest
        for item in candidates
        if item.candidate_id == first.displaced_historical_id
    )
    second_displaced_digest = next(
        item.environment_digest
        for item in renamed
        if item.candidate_id == second.displaced_historical_id
    )
    assert first_displaced_digest == second_displaced_digest


def test_panel_quality_match_gates_and_diagnostics() -> None:
    locks = lock_all_portfolios(_panel())
    diagnostics = portfolio_quality_diagnostics(locks)
    assert diagnostics["maximum_absolute_quality_gap"] == 0.0
    assert diagnostics["mean_absolute_quality_gap"] == 0.0
    assert all(lock.differs for lock in locks)

    bad = [*(_panel())]
    bad[-1] = replace(bad[-1], quality_score=0.0)
    with pytest.raises(ValueError, match="per-stratum"):
        lock_all_portfolios(bad)
    duplicate = replace(bad[-1], environment_digest=bad[0].environment_digest)
    with pytest.raises(ValueError, match="globally unique"):
        lock_all_portfolios([*bad[:-1], duplicate])


def test_schedule_is_latin_interleaved_counterbalanced_and_one_turn() -> None:
    grouped = {
        f"c{index:03d}": tuple(f"c{index:03d}--{arm}" for arm in ("v3", "v4", "challenger"))
        for index in (1, 3, 4, 5, 6, 7)
    }
    schedule = counterbalanced_pairs(grouped, (0, 42))
    assert len(schedule) == 36
    assert {item["horizon"] for item in schedule} == {1}
    assert len({item["stratum_id"] for item in schedule[:12]}) == 6
    for candidate_id in {item["candidate_id"] for item in schedule}:
        pairs = [item for item in schedule if item["candidate_id"] == candidate_id]
        assert {tuple(item["arm_order"]) for item in pairs} == {
            ("unhinted", "hinted"),
            ("hinted", "unhinted"),
        }


def test_whole_pair_attempt_waves_enforce_172_actor_ceiling() -> None:
    grouped = {
        f"c{index:03d}": tuple(f"c{index:03d}--{arm}" for arm in ("v3", "v4", "challenger"))
        for index in (1, 3, 4, 5, 6, 7)
    }
    schedule = counterbalanced_pairs(grouped, (0, 42))
    ids = [item["pair_id"] for item in schedule]
    waves = attempt_waves(schedule, ids[:20], ids[:14])
    assert [len(item) for item in waves] == [36, 20, 14]
    assert (36 + 36 + 14) * 2 == 172
    with pytest.raises(ValueError, match="attempt 3 is forbidden"):
        attempt_waves(schedule, ids[:20], ids[:15])


def _passing_outcomes() -> tuple[object, list[PairedOutcome]]:
    locks = lock_all_portfolios(_panel())
    outcomes: list[PairedOutcome] = []
    for lock in locks:
        coverage_only = lock.challenger_id
        control_only = lock.displaced_historical_id
        for candidate_id in lock.candidate_ids:
            for seed in (0, 42):
                if candidate_id == coverage_only:
                    unhinted, hinted = 0.0, 1.0
                elif candidate_id == control_only:
                    unhinted, hinted = 1.0, 0.0
                else:
                    unhinted = hinted = 0.0
                outcomes.append(
                    PairedOutcome(
                        candidate_id=candidate_id,
                        stratum_id=lock.stratum_id,
                        seed=seed,
                        unhinted=unhinted,
                        hinted=hinted,
                    )
                )
    return locks, outcomes


def test_exact_6_pow_6_and_2_pow_6_sensitivity_gates() -> None:
    locks, outcomes = _passing_outcomes()
    outcomes[0] = replace(outcomes[0], parser_failure=True)
    analysis = analyze_proxy_pilot(locks, outcomes)
    assert analysis.passed
    assert analysis.label_permutation_total == EXACT_LABEL_PERMUTATION_COUNT == 46_656
    assert analysis.sign_flip_total == EXACT_SIGN_FLIP_COUNT == 64
    assert analysis.label_permutation_p_value <= 0.05
    assert analysis.sign_flip_p_value == 1 / 64
    assert analysis.coverage_forced_delta == 1.0
    assert analysis.parser_failures == 1


def test_sign_flip_and_topology_failure_paths() -> None:
    p_value, extreme, total = exact_stratum_sign_flip_pvalue([1.0] * 6)
    assert (p_value, extreme, total) == (1 / 64, 1, 64)
    locks, outcomes = _passing_outcomes()
    bad = [*outcomes]
    bad[0] = replace(bad[0], seed=1)
    with pytest.raises(ValueError, match="topology"):
        analyze_proxy_pilot(locks, bad)
    flagged = [
        replace(item, first_attempt_exogenous_failure=index < 6)
        for index, item in enumerate(outcomes)
    ]
    assert not analyze_proxy_pilot(locks, flagged).passed
