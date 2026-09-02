import argparse
from pathlib import Path

import pytest

from spade.slime.static_pool import StaticPoolScheduleError, select_static_games
from spade.slime.arguments import add_spade_arguments


def _pool(root: str, size: int) -> list[Path]:
    return [Path(root, f"game_{index:03d}.py") for index in range(size)]


def _names(paths: list[Path]) -> list[str]:
    return [path.name for path in paths]


def test_selection_is_identical_for_fresh_and_resumed_callers() -> None:
    games = _pool("/treatment", 7)
    uninterrupted = [
        _names(
            select_static_games(
                games,
                3,
                seed=8128,
                schedule_id="sealed-pair-a",
                rollout_id=rollout_id,
            )
        )
        for rollout_id in range(10)
    ]

    # A resumed process has no prior module state; asking only for the resumed
    # rollout must reproduce the uninterrupted schedule exactly.
    for rollout_id in (0, 1, 4, 9):
        resumed = select_static_games(
            list(reversed(games)),
            3,
            seed=8128,
            schedule_id="sealed-pair-a",
            rollout_id=rollout_id,
        )
        assert _names(resumed) == uninterrupted[rollout_id]


def test_current_deck_is_exhausted_before_any_game_is_reused() -> None:
    games = _pool("/arm", 7)
    first_three = [
        _names(
            select_static_games(
                games,
                3,
                seed=42,
                schedule_id="exhaustion-proof",
                rollout_id=rollout_id,
            )
        )
        for rollout_id in range(3)
    ]

    # Two full batches consume six distinct slots.  The next batch must start
    # with the sole unconsumed slot before drawing from a new shuffled deck.
    assert len(set(first_three[0] + first_three[1])) == 6
    unseen = set(_names(games)) - set(first_three[0] + first_three[1])
    assert len(unseen) == 1
    assert first_three[2][0] in unseen
    assert all(len(set(batch)) == 3 for batch in first_three)


def test_paired_pool_roots_select_the_same_basename_slots() -> None:
    treatment = _pool("/experiment/coverage_forced", 12)
    control = _pool("/experiment/redundant_historical", 12)

    for rollout_id in range(8):
        treatment_batch = select_static_games(
            treatment,
            5,
            seed=1234,
            schedule_id="df1a-fc798-pair",
            rollout_id=rollout_id,
        )
        control_batch = select_static_games(
            list(reversed(control)),
            5,
            seed=1234,
            schedule_id="df1a-fc798-pair",
            rollout_id=rollout_id,
        )
        assert _names(treatment_batch) == _names(control_batch)
        assert all(
            left.parent != right.parent for left, right in zip(treatment_batch, control_batch)
        )


def test_num_games_larger_than_pool_fails_closed() -> None:
    with pytest.raises(StaticPoolScheduleError, match="exceeds unique static pool size"):
        select_static_games(
            _pool("/arm", 12),
            13,
            seed=42,
            schedule_id="sealed",
            rollout_id=0,
        )


def test_duplicate_basename_slots_fail_closed() -> None:
    with pytest.raises(StaticPoolScheduleError, match="duplicate slot"):
        select_static_games(
            [Path("/a/game_000.py"), Path("/b/game_000.py")],
            1,
            seed=42,
            schedule_id="sealed",
            rollout_id=0,
        )


def test_published_static_launcher_keeps_a_stable_default_schedule_id() -> None:
    parser = argparse.ArgumentParser()
    add_spade_arguments(parser)
    args = parser.parse_args(["--spade-static-game-pool", "--spade-no-replacement"])
    assert args.spade_static_pool_schedule_id == "spade-static-default-v1"
    assert args.spade_require_complete_rollout is False
