"""Deterministic quality-diversity archive for counterfactual witnesses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

from spade.core.counterfactual_witness import TraceSignature, WitnessProbe


ArchiveAction = Literal[
    "champion_inserted",
    "champion_replaced",
    "challenger_inserted",
    "challenger_replaced",
    "rejected_duplicate",
    "rejected_lower_quality",
    "rejected_same_lineage",
]


def _reward_bin(value: float) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _depth_bin(depth: int) -> int:
    if depth <= 1:
        return depth
    if depth <= 3:
        return 2
    return 3


@dataclass(frozen=True, order=True)
class BehaviorDescriptor:
    """Fixed, comparable behavioral coordinates; response hashes are not distances."""

    action_format: str
    oracle_depth_bin: int
    reset_seed_diversity_bin: int
    invalid_reward_bin: int
    invalid_end_state: str
    recovery_success: Literal["observed_true", "observed_false", "unmeasured"]
    trace_order_divergent: Literal["observed_true", "observed_false", "unmeasured"]

    def __post_init__(self) -> None:
        if self.action_format not in {"boxed", "tool_call"}:
            raise ValueError("action_format must be 'boxed' or 'tool_call'")
        if self.oracle_depth_bin not in {0, 1, 2, 3}:
            raise ValueError("oracle_depth_bin must be within 0..3")
        if self.reset_seed_diversity_bin not in {0, 1, 2, 3}:
            raise ValueError("reset_seed_diversity_bin must be within 0..3")
        if self.invalid_reward_bin not in {-1, 0, 1}:
            raise ValueError("invalid_reward_bin must be -1, 0, or 1")
        if self.invalid_end_state not in {"continues", "terminated", "truncated", "error"}:
            raise ValueError("invalid_end_state is unsupported")
        for name, value in (
            ("recovery_success", self.recovery_success),
            ("trace_order_divergent", self.trace_order_divergent),
        ):
            if value not in {"observed_true", "observed_false", "unmeasured"}:
                raise ValueError(f"{name} has an unsupported observation state")

    @property
    def cell_key(self) -> Tuple[Any, ...]:
        return (
            self.action_format,
            self.oracle_depth_bin,
            self.reset_seed_diversity_bin,
            self.invalid_reward_bin,
            self.invalid_end_state,
            self.recovery_success,
            self.trace_order_divergent,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_format": self.action_format,
            "oracle_depth_bin": self.oracle_depth_bin,
            "reset_seed_diversity_bin": self.reset_seed_diversity_bin,
            "invalid_reward_bin": self.invalid_reward_bin,
            "invalid_end_state": self.invalid_end_state,
            "recovery_success": self.recovery_success,
            "trace_order_divergent": self.trace_order_divergent,
        }


def behavior_descriptor(
    *,
    action_format: str,
    probes: Iterable[WitnessProbe],
    signatures: Dict[str, TraceSignature],
) -> BehaviorDescriptor:
    """Derive a global archive coordinate from a base environment's probe outcomes."""
    probe_list = tuple(probes)
    if {probe.probe_id for probe in probe_list} != set(signatures):
        raise ValueError("descriptor signatures must exactly cover the supplied probes")

    def by_role(role: str) -> list[Tuple[WitnessProbe, TraceSignature]]:
        return [(probe, signatures[probe.probe_id]) for probe in probe_list if probe.role == role]

    oracle = by_role("base_oracle")
    resets = by_role("reset")
    invalid = by_role("well_formed_wrong")
    recoveries = by_role("wrong_then_oracle_recovery")
    reversed_runs = by_role("reversed_oracle")
    if not oracle or not resets or not invalid:
        raise ValueError("descriptor requires base_oracle, reset, and well_formed_wrong probes")

    oracle_depth = max(signature.turn_count for _probe, signature in oracle)
    reset_observations = {
        signature.observation_digests[0]
        for _probe, signature in resets
        if signature.observation_digests
    }
    invalid_rewards = [signature.final_reward for _probe, signature in invalid]
    maximum_invalid_reward = max(invalid_rewards)
    if any(signature.error is not None for _probe, signature in invalid):
        invalid_end = "error"
    elif any(signature.terminated for _probe, signature in invalid):
        invalid_end = "terminated"
    elif any(signature.truncated for _probe, signature in invalid):
        invalid_end = "truncated"
    else:
        invalid_end = "continues"
    recovery_success = (
        "observed_true"
        if recoveries and any(signature.success for _probe, signature in recoveries)
        else "observed_false"
        if recoveries
        else "unmeasured"
    )
    oracle_by_seed = {probe.seed: signature for probe, signature in oracle}
    trace_order_divergent = (
        "observed_true"
        if reversed_runs
        and any(
            probe.seed in oracle_by_seed and signature.digest != oracle_by_seed[probe.seed].digest
            for probe, signature in reversed_runs
        )
        else "observed_false"
        if reversed_runs
        else "unmeasured"
    )
    return BehaviorDescriptor(
        action_format=action_format,
        oracle_depth_bin=_depth_bin(oracle_depth),
        reset_seed_diversity_bin=min(len(reset_observations), 3),
        invalid_reward_bin=_reward_bin(maximum_invalid_reward),
        invalid_end_state=invalid_end,
        recovery_success=recovery_success,
        trace_order_divergent=trace_order_divergent,
    )


@dataclass(frozen=True)
class ArchiveEntry:
    """An evidence-bound environment eligible for a descriptor cell."""

    environment_digest: str
    witness_digest: str
    game_file: str
    skill: str
    difficulty: str
    descriptor: BehaviorDescriptor
    quality_score: float
    lineage: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment_digest or not self.witness_digest:
            raise ValueError("environment and witness digests must be non-empty")
        if not self.game_file or not self.skill or not self.difficulty:
            raise ValueError("game_file, skill, and difficulty must be non-empty")
        if not isinstance(self.quality_score, (int, float)) or isinstance(self.quality_score, bool):
            raise TypeError("quality_score must be numeric")
        if not math.isfinite(float(self.quality_score)):
            raise ValueError("quality_score must be finite")
        if not self.lineage or any(not isinstance(item, str) or not item for item in self.lineage):
            raise ValueError("lineage must contain non-empty identifiers")
        if len(set(self.lineage)) != len(self.lineage):
            raise ValueError("lineage identifiers must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment_digest": self.environment_digest,
            "witness_digest": self.witness_digest,
            "game_file": self.game_file,
            "skill": self.skill,
            "difficulty": self.difficulty,
            "descriptor": self.descriptor.to_dict(),
            "quality_score": float(self.quality_score),
            "lineage": list(self.lineage),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ArchiveCell:
    champion: ArchiveEntry
    challenger: Optional[ArchiveEntry] = None


@dataclass(frozen=True)
class ArchiveDecision:
    action: ArchiveAction
    cell_key: Tuple[Any, ...]
    accepted_digest: Optional[str]
    demoted_digest: Optional[str] = None
    evicted_digests: Tuple[str, ...] = ()


def _disjoint(left: ArchiveEntry, right: ArchiveEntry) -> bool:
    return set(left.lineage).isdisjoint(right.lineage)


def _better(left: ArchiveEntry, right: ArchiveEntry) -> bool:
    """Stable maximum quality with digest tie-breaking."""
    if float(left.quality_score) != float(right.quality_score):
        return float(left.quality_score) > float(right.quality_score)
    return left.environment_digest < right.environment_digest


class CounterfactualWitnessArchive:
    """One quality champion and one lineage-disjoint challenger per cell."""

    def __init__(self) -> None:
        self._cells: Dict[Tuple[Any, ...], ArchiveCell] = {}

    @property
    def cells(self) -> Dict[Tuple[Any, ...], ArchiveCell]:
        return dict(self._cells)

    def get(
        self, skill: str, difficulty: str, descriptor: BehaviorDescriptor
    ) -> Optional[ArchiveCell]:
        return self._cells.get((skill, difficulty, *descriptor.cell_key))

    def consider(self, entry: ArchiveEntry) -> ArchiveDecision:
        key = (entry.skill, entry.difficulty, *entry.descriptor.cell_key)
        cell = self._cells.get(key)
        if cell is None:
            self._cells[key] = ArchiveCell(champion=entry)
            return ArchiveDecision("champion_inserted", key, entry.environment_digest)
        existing = [cell.champion]
        if cell.challenger is not None:
            existing.append(cell.challenger)
        if any(item.environment_digest == entry.environment_digest for item in existing):
            return ArchiveDecision("rejected_duplicate", key, None)

        if _better(entry, cell.champion):
            challenger_candidates = [
                candidate for candidate in existing if _disjoint(entry, candidate)
            ]
            challenger = None
            for candidate in challenger_candidates:
                if challenger is None or _better(candidate, challenger):
                    challenger = candidate
            self._cells[key] = ArchiveCell(champion=entry, challenger=challenger)
            new_digests = {entry.environment_digest}
            if challenger is not None:
                new_digests.add(challenger.environment_digest)
            evicted = tuple(
                sorted(
                    candidate.environment_digest
                    for candidate in existing
                    if candidate.environment_digest not in new_digests
                )
            )
            return ArchiveDecision(
                "champion_replaced",
                key,
                entry.environment_digest,
                (
                    cell.champion.environment_digest
                    if challenger is not None
                    and challenger.environment_digest == cell.champion.environment_digest
                    else None
                ),
                evicted,
            )

        if not _disjoint(entry, cell.champion):
            return ArchiveDecision("rejected_same_lineage", key, None)
        if cell.challenger is None:
            self._cells[key] = ArchiveCell(champion=cell.champion, challenger=entry)
            return ArchiveDecision("challenger_inserted", key, entry.environment_digest)
        if _better(entry, cell.challenger):
            self._cells[key] = ArchiveCell(champion=cell.champion, challenger=entry)
            return ArchiveDecision(
                "challenger_replaced",
                key,
                entry.environment_digest,
                None,
                (cell.challenger.environment_digest,),
            )
        return ArchiveDecision("rejected_lower_quality", key, None)

    def to_dict(self) -> Dict[str, Any]:
        cells = []
        for key in sorted(self._cells, key=repr):
            cell = self._cells[key]
            cells.append(
                {
                    "cell_key": list(key),
                    "champion": cell.champion.to_dict(),
                    "challenger": (
                        cell.challenger.to_dict() if cell.challenger is not None else None
                    ),
                }
            )
        return {"schema_version": "spade-counterfactual-witness-archive/v1", "cells": cells}

    def __len__(self) -> int:
        return len(self._cells)


__all__ = [
    "ArchiveCell",
    "ArchiveDecision",
    "ArchiveEntry",
    "BehaviorDescriptor",
    "CounterfactualWitnessArchive",
    "behavior_descriptor",
]
