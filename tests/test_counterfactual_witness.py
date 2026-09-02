"""Pure tests for counterfactual witness compilation and selection."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from spade.core.counterfactual_witness import (
    RESET_SIGNATURE_SELECTION_ALGORITHM,
    SELECTION_ALGORITHM,
    WitnessMatrix,
    WitnessProbe,
    build_candidate_probes,
    build_certificate,
    generate_source_variants,
    proofpack_compatible_entrypoints,
    score_probe_ids,
    select_witnesses,
    select_witnesses_with_reset_signature_coverage,
    source_digest,
    trace_signature,
    verify_witness_signatures,
)


BOXED_GAME = r"""
import re

class TinyPuzzleEnv:
    def __init__(self, max_turns=3, **kwargs):
        self.max_turns = max_turns
        self.turn = 0

    def reset(self, seed=None):
        self.turn = 0
        return f"Seed {seed}: submit ok.", {"seed": seed}

    def solution(self):
        return "ok"

    def step(self, action):
        self.turn += 1
        if action == "\\boxed{ok}":
            return "Correct", 1.0, True, False, {"turn": self.turn}
        return "Wrong", 0.0, False, self.turn >= self.max_turns, {"turn": self.turn}
"""


TOOL_GAME = r"""
class TinyToolEnv(ToolUseBaseEnv):
    def reset(self, seed=None):
        self.turn_count = 0
        self._expected_answer = "done"
        self._tools = {"mark": {"parameters": {"type": "object", "properties": {}}}}
        return "Call mark, then answer done.", {}

    def solution(self):
        return "1. mark() ANSWER: done"
"""


def _result(
    tag: str,
    *,
    reward: float = 0.0,
    success: bool = False,
    terminated: bool = False,
    truncated: bool = False,
    turns: int = 1,
    info: dict | None = None,
):
    trajectory = [{"role": "environment", "observation": f"reset:{tag}"}]
    for index in range(turns):
        trajectory.append(
            {
                "action": "ignored input",
                "observation": f"step:{tag}:{index}",
                "reward": reward if index == turns - 1 else 0.0,
                "terminated": terminated if index == turns - 1 else False,
                "truncated": truncated if index == turns - 1 else False,
                "info": info or {},
            }
        )
    return SimpleNamespace(
        success=success,
        reward=reward,
        turn_count=turns,
        terminated=terminated,
        truncated=truncated,
        trajectory=trajectory,
        error=None,
    )


def _signature(tag: str):
    return trace_signature(_result(tag))


def test_variant_panel_is_deterministic_partitioned_and_parseable() -> None:
    first = generate_source_variants(BOXED_GAME)
    second = generate_source_variants(BOXED_GAME)

    assert [variant.variant_id for variant in first] == [variant.variant_id for variant in second]
    assert len(first) == 12
    assert len({variant.operator for variant in first}) == len(first)
    assert {variant.partition for variant in first} == {"train", "heldout"}
    assert {variant.kind for variant in first} == {"semantic_mutant", "equivalent_control"}
    assert all(source_digest(variant.source) == variant.source_digest for variant in first)
    assert all(
        proofpack_compatible_entrypoints(variant.source) == ("TinyPuzzleEnv",) for variant in first
    )
    assert all(ast.parse(variant.source) for variant in first)


def test_wrapper_renames_original_to_non_env_base_and_keeps_one_entrypoint() -> None:
    variants = generate_source_variants(TOOL_GAME)
    wrapped = next(variant for variant in variants if variant.operator == "inert_subclass")
    tree = ast.parse(wrapped.source)
    class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

    assert class_names[-1] == "TinyToolEnv"
    assert class_names[-2].startswith("_CwaBase")
    assert not class_names[-2].endswith("Env")
    assert proofpack_compatible_entrypoints(tree) == ("TinyToolEnv",)


def test_generated_panel_passes_real_proofpack_static_target_gate_when_available() -> None:
    try:
        target_module = importlib.import_module("proofpack_env.spade_target")
    except (ImportError, SyntaxError):
        pytest.skip("a Python-compatible proofpack-env is not installed")
    for game_code, action_format in ((BOXED_GAME, "boxed"), (TOOL_GAME, "tool_call")):
        for variant in generate_source_variants(game_code):
            # Constructor validation is static only; no generated source is compiled or executed.
            target = target_module.SpadeEnvironmentTarget(
                variant.source,
                action_format=action_format,
                max_turns=3,
            )
            assert target.environment_name == variant.entrypoint


def test_equivalent_controls_include_ast_changing_wrappers() -> None:
    variants = generate_source_variants(BOXED_GAME)
    base_ast = ast.dump(ast.parse(BOXED_GAME), include_attributes=False)
    controls = [variant for variant in variants if variant.kind == "equivalent_control"]

    assert {variant.operator for variant in controls} == {
        "ast_roundtrip",
        "inert_subclass",
        "dead_branch_subclass",
    }
    assert ast.dump(ast.parse(controls[0].source), include_attributes=False) == base_ast
    assert all(
        ast.dump(ast.parse(control.source), include_attributes=False) != base_ast
        for control in controls[1:]
    )


def test_source_module_never_compiles_or_executes_generated_source() -> None:
    module_path = Path(__file__).resolve().parents[1] / "spade/core/counterfactual_witness.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    direct_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert direct_calls.isdisjoint({"compile", "eval", "exec"})


def test_candidate_probe_bank_is_deterministic_unique_and_bounded() -> None:
    kwargs = {
        "action_format": "boxed",
        "seeds": [0, 1],
        "base_oracle_actions": {0: [r"\boxed{one}", r"\boxed{two}"], 1: [r"\boxed{ok}"]},
        "mutant_oracle_actions": ({0: [r"\boxed{mutant}"], 1: [r"\boxed{other}"]},),
        "max_turns": 4,
        "max_candidates": 50,
    }
    first = build_candidate_probes(**kwargs)
    second = build_candidate_probes(**kwargs)

    assert first == second
    assert len({(probe.seed, probe.actions) for probe in first}) == len(first)
    assert all(len(probe.actions) <= 4 for probe in first)
    roles = {probe.role for probe in first}
    assert {
        "reset",
        "base_oracle",
        "whitespace_wrapped_oracle",
        "well_formed_wrong",
        "malformed",
        "blank_noop",
    } <= roles
    assert "wrong_then_oracle_recovery" in roles
    assert "reversed_oracle" in roles
    assert any(role.startswith("training_mutant_") for role in roles)


def test_candidate_probe_bank_supports_tool_actions_and_hard_cap() -> None:
    probes = build_candidate_probes(
        action_format="tool_call",
        seeds=[7],
        base_oracle_actions={7: ["<answer>done</answer>"]},
        max_turns=2,
        max_candidates=3,
    )
    assert len(probes) == 3
    assert probes[2].actions[0].strip().startswith("<answer>")


def test_trace_signature_is_canonical_and_behavior_sensitive() -> None:
    left = _result("same", info={"b": 2, "a": 1})
    right = _result("same", info={"a": 1, "b": 2})
    changed = _result("same", reward=0.25, info={"a": 1, "b": 2})

    assert trace_signature(left).digest == trace_signature(right).digest
    assert trace_signature(left).digest != trace_signature(changed).digest
    assert trace_signature(changed).reward_path == (0.25,)


def _selection_matrix() -> tuple[WitnessMatrix, dict[str, WitnessProbe]]:
    variants = generate_source_variants(BOXED_GAME)
    probes = {
        name: WitnessProbe.create(seed=0, actions=(name,), role=name)
        for name in ("broad", "focused", "control_breaker")
    }
    base = _signature("base")
    changed = _signature("changed")
    rows = {
        probe.probe_id: {"base": base, **{variant.variant_id: base for variant in variants}}
        for probe in probes.values()
    }
    train_mutants = [
        variant
        for variant in variants
        if variant.kind == "semantic_mutant" and variant.partition == "train"
    ]
    train_controls = [
        variant
        for variant in variants
        if variant.kind == "equivalent_control" and variant.partition == "train"
    ]
    # Broad kills one member from every family and therefore wins inverse-family weighting.
    for mutant in train_mutants[:-1]:
        rows[probes["broad"].probe_id][mutant.variant_id] = changed
    rows[probes["focused"].probe_id][train_mutants[-1].variant_id] = changed
    rows[probes["control_breaker"].probe_id][train_mutants[-1].variant_id] = changed
    rows[probes["control_breaker"].probe_id][train_controls[0].variant_id] = changed

    # Held-out variants differ everywhere but must not influence training selection.
    for variant in variants:
        if variant.partition == "heldout":
            rows[probes["focused"].probe_id][variant.variant_id] = changed
    matrix = WitnessMatrix(
        base_environment_digest=source_digest(BOXED_GAME),
        probes=tuple(probes.values()),
        variants=variants,
        signatures=rows,
    )
    return matrix, probes


def test_safe_set_cover_excludes_control_breakers_and_ignores_heldout() -> None:
    matrix, probes = _selection_matrix()
    selection = select_witnesses(matrix, budget=2)

    assert selection.selected_probe_ids == (
        probes["broad"].probe_id,
        probes["focused"].probe_id,
    )
    assert probes["control_breaker"].probe_id in selection.rejected_control_breakers
    assert not selection.uncovered_train_mutants

    heldout = score_probe_ids(matrix, selection.selected_probe_ids, partition="heldout")
    assert heldout.total_mutants > 0
    assert heldout.mutant_recall == 1.0
    assert heldout.equivalent_false_rejection_rate == 1.0


def _reset_coverage_matrix() -> tuple[WitnessMatrix, dict[str, WitnessProbe]]:
    probes = {
        "broad": WitnessProbe.create(seed=0, actions=("solve",), role="broad"),
        "reset_0": WitnessProbe.create(seed=0, actions=(), role="reset"),
        "reset_1": WitnessProbe.create(seed=1, actions=(), role="reset"),
        "reset_42": WitnessProbe.create(seed=42, actions=(), role="reset"),
    }
    variants = generate_source_variants(BOXED_GAME)
    train_mutants = [
        variant
        for variant in variants
        if variant.kind == "semantic_mutant" and variant.partition == "train"
    ]
    heldout_mutants = [
        variant
        for variant in variants
        if variant.kind == "semantic_mutant" and variant.partition == "heldout"
    ]
    base_by_seed = {0: _signature("reset-zero"), 1: _signature("reset-one"), 42: _signature("reset-42")}
    rows = {
        probe.probe_id: {
            "base": base_by_seed[probe.seed],
            **{variant.variant_id: base_by_seed[probe.seed] for variant in variants},
        }
        for probe in probes.values()
    }
    for mutant in train_mutants:
        rows[probes["broad"].probe_id][mutant.variant_id] = _signature(
            f"train-changed-{mutant.variant_id}"
        )
    rows[probes["reset_1"].probe_id][heldout_mutants[0].variant_id] = _signature(
        "heldout-seed-one-changed"
    )
    rows[probes["reset_42"].probe_id][heldout_mutants[1].variant_id] = _signature(
        "heldout-seed-42-changed"
    )
    return (
        WitnessMatrix(
            base_environment_digest=source_digest(BOXED_GAME),
            probes=tuple(probes.values()),
            variants=variants,
            signatures=rows,
        ),
        probes,
    )


def test_reset_signature_selector_preserves_training_and_covers_each_base_reset_state() -> None:
    matrix, probes = _reset_coverage_matrix()

    legacy = select_witnesses(matrix, budget=3)
    covered = select_witnesses_with_reset_signature_coverage(matrix, budget=3)

    assert legacy.selected_probe_ids == (probes["broad"].probe_id,)
    assert covered.selected_probe_ids == (
        probes["broad"].probe_id,
        probes["reset_1"].probe_id,
        probes["reset_42"].probe_id,
    )
    assert covered.killed_train_mutants == legacy.killed_train_mutants
    assert not covered.uncovered_train_mutants
    assert legacy.algorithm == SELECTION_ALGORITHM
    assert covered.algorithm == RESET_SIGNATURE_SELECTION_ALGORITHM
    assert score_probe_ids(matrix, covered.selected_probe_ids).killed_mutants == 2
    assert score_probe_ids(matrix, legacy.selected_probe_ids).killed_mutants == 0


def test_reset_signature_selector_respects_budget_before_all_seed_atoms_are_covered() -> None:
    matrix, probes = _reset_coverage_matrix()

    selection = select_witnesses_with_reset_signature_coverage(matrix, budget=2)

    assert selection.selected_probe_ids == (
        probes["broad"].probe_id,
        probes["reset_1"].probe_id,
    )
    assert not selection.uncovered_train_mutants


def test_reset_signature_selector_ignores_heldout_signatures() -> None:
    matrix, _probes = _reset_coverage_matrix()
    expected = select_witnesses_with_reset_signature_coverage(matrix, budget=3)

    for probe in matrix.probes:
        for variant in matrix.variants:
            if variant.partition == "heldout":
                matrix.signatures[probe.probe_id][variant.variant_id] = _signature(
                    f"arbitrary-heldout-change-{probe.probe_id}-{variant.variant_id}"
                )

    assert select_witnesses_with_reset_signature_coverage(matrix, budget=3) == expected


def test_reset_signature_selector_rejects_a_training_control_unsafe_reset() -> None:
    matrix, probes = _reset_coverage_matrix()
    training_control = next(
        variant
        for variant in matrix.variants
        if variant.kind == "equivalent_control" and variant.partition == "train"
    )
    matrix.signatures[probes["reset_1"].probe_id][training_control.variant_id] = _signature(
        "control-breaker"
    )

    with pytest.raises(ValueError, match="control-safe reset probes"):
        select_witnesses_with_reset_signature_coverage(matrix, budget=3)


def test_certificate_is_content_bound_and_verifiable() -> None:
    matrix, _probes = _selection_matrix()
    selection = select_witnesses(matrix, budget=2)
    certificate = build_certificate(matrix, selection, metadata={"qualification": "sealed"})
    observed = {
        probe.probe_id: matrix.signature(probe.probe_id, "base") for probe in certificate.probes
    }

    assert verify_witness_signatures(certificate, observed).passed is True
    first_id = certificate.probes[0].probe_id
    observed[first_id] = _signature("drift")
    verification = verify_witness_signatures(certificate, observed)
    assert verification.passed is False
    assert verification.mismatched_probe_ids == (first_id,)


def test_full_probe_panel_reports_observational_upper_bound() -> None:
    matrix, _probes = _selection_matrix()
    heldout_mutants = [
        variant
        for variant in matrix.variants
        if variant.kind == "semantic_mutant" and variant.partition == "heldout"
    ]
    observationally_equivalent = heldout_mutants[0]
    for probe in matrix.probes:
        matrix.signatures[probe.probe_id][observationally_equivalent.variant_id] = matrix.signature(
            probe.probe_id, "base"
        )

    upper_bound = score_probe_ids(
        matrix,
        [probe.probe_id for probe in matrix.probes],
        partition="heldout",
    )

    assert upper_bound.killed_mutants == upper_bound.total_mutants - 1
    assert upper_bound.mutant_recall < 1.0


@pytest.mark.parametrize("source", ["", "class A: pass", "class OneEnv: pass\nclass TwoEnv: pass"])
def test_variant_compiler_rejects_non_environment_sources(source: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        generate_source_variants(source)
