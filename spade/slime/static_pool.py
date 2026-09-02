"""Resume-stable scheduling for paired static SPADE game pools.

The scheduler is deliberately stateless.  A selection is a pure function of
the slot basenames, seed, schedule identifier, and rollout identifier, so a
resumed worker selects exactly the same games as an uninterrupted worker.
Using basenames (rather than absolute paths) also keeps corresponding slots in
paired treatment/control pool directories aligned.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterable


class StaticPoolScheduleError(ValueError):
    """The static pool cannot satisfy the sealed scheduling contract."""


SCHEDULE_SCHEMA = "spade-static-game-pool-schedule/v1"
DEFAULT_SCHEDULE_ID = "spade-static-default-v1"


def _validated_slots(game_files: Iterable[Path | str]) -> dict[str, Path]:
    slots: dict[str, Path] = {}
    for value in game_files:
        path = Path(value)
        basename = path.name
        if not basename.startswith("game_") or path.suffix != ".py":
            raise StaticPoolScheduleError(
                f"static pool entry must have a game_*.py basename: {path}"
            )
        if basename in slots:
            raise StaticPoolScheduleError(
                f"static pool basenames must be unique; duplicate slot: {basename}"
            )
        slots[basename] = path
    if not slots:
        raise StaticPoolScheduleError("static game pool is empty")
    return slots


def _cycle_deck(
    basenames: tuple[str, ...],
    *,
    seed: int,
    schedule_id: str,
    cycle_id: int,
) -> list[str]:
    material = f"{SCHEDULE_SCHEMA}\0{seed}\0{schedule_id}\0{cycle_id}".encode("utf-8")
    cycle_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    deck = list(basenames)
    random.Random(cycle_seed).shuffle(deck)
    return deck


def select_static_games(
    game_files: Iterable[Path | str],
    num_games: int,
    *,
    seed: int,
    schedule_id: str,
    rollout_id: int,
) -> list[Path]:
    """Select one deterministic, without-replacement rollout from a pool.

    Each cycle shuffles every basename exactly once.  A batch may cross a cycle
    boundary, but an entry already drawn into that batch is skipped in the new
    cycle.  This preserves the old no-duplicate-within-batch behavior while
    guaranteeing that the current deck is exhausted before any refill.

    Replaying decks from cycle zero is inexpensive for SPADE's small static
    curricula and removes all process-local state from checkpoint resumption.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StaticPoolScheduleError("static pool seed must be an integer")
    if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
        raise StaticPoolScheduleError("rollout_id must be a non-negative integer")
    if isinstance(num_games, bool) or not isinstance(num_games, int) or num_games <= 0:
        raise StaticPoolScheduleError("num_games must be a positive integer")
    if not isinstance(schedule_id, str) or not schedule_id.strip() or "\0" in schedule_id:
        raise StaticPoolScheduleError("schedule_id must be a non-empty string without NUL")

    slots = _validated_slots(game_files)
    if num_games > len(slots):
        raise StaticPoolScheduleError(
            f"num_games ({num_games}) exceeds unique static pool size ({len(slots)})"
        )

    basenames = tuple(sorted(slots))
    remaining: list[str] = []
    cycle_id = 0

    for current_rollout in range(rollout_id + 1):
        selected: list[str] = []
        while len(selected) < num_games:
            if not remaining:
                remaining = _cycle_deck(
                    basenames,
                    seed=seed,
                    schedule_id=schedule_id,
                    cycle_id=cycle_id,
                )
                cycle_id += 1
                if selected:
                    selected_set = set(selected)
                    remaining = [name for name in remaining if name not in selected_set]

            needed = num_games - len(selected)
            selected.extend(remaining[:needed])
            remaining = remaining[needed:]

        if current_rollout == rollout_id:
            return [slots[name] for name in selected]

    raise AssertionError("unreachable static-pool schedule state")
