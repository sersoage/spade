"""Tests for the lineage-aware counterfactual witness archive."""

from __future__ import annotations

from types import SimpleNamespace

from spade.core.counterfactual_witness import WitnessProbe, trace_signature
from spade.core.witness_archive import (
    ArchiveEntry,
    BehaviorDescriptor,
    CounterfactualWitnessArchive,
    behavior_descriptor,
)


def _signature(
    tag: str,
    *,
    reward: float = 0.0,
    success: bool = False,
    terminated: bool = False,
    truncated: bool = False,
    turns: int = 1,
):
    trajectory = [{"role": "environment", "observation": f"reset:{tag}"}]
    for index in range(turns):
        trajectory.append(
            {
                "action": "ignored",
                "observation": f"step:{tag}:{index}",
                "reward": reward if index == turns - 1 else 0.0,
                "terminated": terminated if index == turns - 1 else False,
                "truncated": truncated if index == turns - 1 else False,
                "info": {},
            }
        )
    return trace_signature(
        SimpleNamespace(
            success=success,
            reward=reward,
            turn_count=turns,
            terminated=terminated,
            truncated=truncated,
            trajectory=trajectory,
            error=None,
        )
    )


def _probe(seed: int, role: str, *actions: str) -> WitnessProbe:
    return WitnessProbe.create(seed=seed, role=role, actions=actions)


def _descriptor() -> BehaviorDescriptor:
    return BehaviorDescriptor(
        action_format="boxed",
        oracle_depth_bin=2,
        reset_seed_diversity_bin=2,
        invalid_reward_bin=0,
        invalid_end_state="continues",
        recovery_success="observed_true",
        trace_order_divergent="observed_true",
    )


def _entry(
    digest: str,
    quality: float,
    lineage: tuple[str, ...],
    descriptor: BehaviorDescriptor | None = None,
) -> ArchiveEntry:
    return ArchiveEntry(
        environment_digest=digest,
        witness_digest=f"witness:{digest}",
        game_file=f"{digest}.py",
        skill="reasoning",
        difficulty="medium",
        descriptor=descriptor or _descriptor(),
        quality_score=quality,
        lineage=lineage,
    )


def test_behavior_descriptor_uses_fixed_relational_coordinates() -> None:
    probes = (
        _probe(0, "reset"),
        _probe(1, "reset"),
        _probe(0, "base_oracle", "one", "two"),
        _probe(0, "well_formed_wrong", "wrong"),
        _probe(0, "wrong_then_oracle_recovery", "wrong", "one", "two"),
        _probe(0, "reversed_oracle", "two", "one"),
    )
    signatures = {
        probes[0].probe_id: _signature("seed-0", turns=0),
        probes[1].probe_id: _signature("seed-1", turns=0),
        probes[2].probe_id: _signature(
            "oracle", reward=1.0, success=True, terminated=True, turns=2
        ),
        probes[3].probe_id: _signature("wrong"),
        probes[4].probe_id: _signature(
            "recovered", reward=1.0, success=True, terminated=True, turns=3
        ),
        probes[5].probe_id: _signature("reversed", turns=2),
    }

    descriptor = behavior_descriptor(
        action_format="boxed",
        probes=probes,
        signatures=signatures,
    )

    assert descriptor.oracle_depth_bin == 2
    assert descriptor.reset_seed_diversity_bin == 2
    assert descriptor.invalid_reward_bin == 0
    assert descriptor.invalid_end_state == "continues"
    assert descriptor.recovery_success == "observed_true"
    assert descriptor.trace_order_divergent == "observed_true"
    assert all(not str(value).startswith("sha256:") for value in descriptor.cell_key)


def test_archive_retains_champion_and_lineage_disjoint_challenger() -> None:
    archive = CounterfactualWitnessArchive()
    first = _entry("env-b", 0.4, ("root-b",))
    same_lineage = _entry("env-b-child", 0.5, ("root-b", "env-b"))
    challenger = _entry("env-c", 0.3, ("root-c",))

    assert archive.consider(first).action == "champion_inserted"
    assert archive.consider(same_lineage).action == "champion_replaced"
    # The former champion overlaps the new champion's ancestry and cannot survive as challenger.
    assert archive.get("reasoning", "medium", _descriptor()).challenger is None
    assert archive.consider(challenger).action == "challenger_inserted"
    cell = archive.get("reasoning", "medium", _descriptor())
    assert cell is not None
    assert cell.champion.environment_digest == "env-b-child"
    assert cell.challenger is not None
    assert cell.challenger.environment_digest == "env-c"


def test_champion_replacement_preserves_best_eligible_lineage() -> None:
    archive = CounterfactualWitnessArchive()
    archive.consider(_entry("old-champion", 0.7, ("line-a",)))
    archive.consider(_entry("old-challenger", 0.6, ("line-b",)))
    decision = archive.consider(_entry("new-champion", 0.9, ("line-c",)))

    assert decision.action == "champion_replaced"
    cell = archive.get("reasoning", "medium", _descriptor())
    assert cell is not None and cell.challenger is not None
    assert cell.champion.environment_digest == "new-champion"
    assert cell.challenger.environment_digest == "old-champion"
    assert decision.demoted_digest == "old-champion"
    assert decision.evicted_digests == ("old-challenger",)


def test_challenger_replacement_and_duplicate_rejection() -> None:
    archive = CounterfactualWitnessArchive()
    champion = _entry("champion", 1.0, ("line-a",))
    archive.consider(champion)
    archive.consider(_entry("weak", 0.1, ("line-b",)))

    assert archive.consider(_entry("strong", 0.5, ("line-c",))).action == "challenger_replaced"
    assert archive.consider(champion).action == "rejected_duplicate"
    assert archive.consider(_entry("same-line", 0.8, ("line-a", "child"))).action == (
        "rejected_same_lineage"
    )
    assert archive.consider(_entry("weaker", 0.2, ("line-d",))).action == ("rejected_lower_quality")


def test_quality_ties_are_digest_deterministic() -> None:
    left_first = CounterfactualWitnessArchive()
    right_first = CounterfactualWitnessArchive()
    a = _entry("a-env", 0.5, ("line-a",))
    b = _entry("b-env", 0.5, ("line-b",))

    left_first.consider(a)
    left_first.consider(b)
    right_first.consider(b)
    right_first.consider(a)

    left_cell = left_first.get("reasoning", "medium", _descriptor())
    right_cell = right_first.get("reasoning", "medium", _descriptor())
    assert left_cell is not None and right_cell is not None
    assert left_cell.champion.environment_digest == "a-env"
    assert right_cell.champion.environment_digest == "a-env"
    assert left_first.to_dict() == right_first.to_dict()


def test_archive_cells_are_separated_by_behavior_descriptor() -> None:
    archive = CounterfactualWitnessArchive()
    other = BehaviorDescriptor(
        action_format="boxed",
        oracle_depth_bin=1,
        reset_seed_diversity_bin=1,
        invalid_reward_bin=-1,
        invalid_end_state="truncated",
        recovery_success="observed_false",
        trace_order_divergent="observed_false",
    )
    archive.consider(_entry("one", 0.1, ("line-1",)))
    archive.consider(_entry("two", 0.1, ("line-2",), descriptor=other))

    assert len(archive) == 2
    assert len(archive.to_dict()["cells"]) == 2


def test_archive_cells_are_partitioned_by_skill_and_difficulty() -> None:
    archive = CounterfactualWitnessArchive()
    entries = []
    for skill in ("reasoning", "memory"):
        for difficulty in ("medium", "hard"):
            entry = _entry(f"{skill}-{difficulty}", 0.5, (f"{skill}-{difficulty}",))
            entry = ArchiveEntry(
                **{
                    **entry.to_dict(),
                    "descriptor": entry.descriptor,
                    "lineage": entry.lineage,
                    "difficulty": difficulty,
                    "skill": skill,
                }
            )
            entries.append(entry)
            archive.consider(entry)

    assert len(archive) == 4
    for entry in entries:
        cell = archive.get(entry.skill, entry.difficulty, entry.descriptor)
        assert cell is not None
        assert cell.champion.environment_digest == entry.environment_digest


def test_missing_optional_probes_are_unmeasured() -> None:
    probes = (
        _probe(0, "reset"),
        _probe(0, "base_oracle", "one"),
        _probe(0, "well_formed_wrong", "wrong"),
    )
    signatures = {
        probes[0].probe_id: _signature("reset", turns=0),
        probes[1].probe_id: _signature("oracle", reward=1.0, success=True, terminated=True),
        probes[2].probe_id: _signature("wrong"),
    }

    descriptor = behavior_descriptor(action_format="boxed", probes=probes, signatures=signatures)

    assert descriptor.recovery_success == "unmeasured"
    assert descriptor.trace_order_divergent == "unmeasured"
