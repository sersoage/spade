"""Pure counterfactual-witness primitives for generated SPADE environments.

This module parses and rewrites generated source, but deliberately never compiles
or executes it.  Execution belongs behind SPADE's optional ProofPack boundary;
callers feed the resulting replay records back through :func:`trace_signature`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Tuple


WITNESS_SCHEMA_VERSION = "spade-counterfactual-witness/v1"
MUTATION_CATALOG_VERSION = "spade-counterfactual-mutations/v1"
SELECTION_ALGORITHM = "safe-inverse-family-greedy-set-cover/v1"

VariantKind = Literal["semantic_mutant", "equivalent_control"]
SelectionPartition = Literal["train", "heldout"]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_canonical_json(value))


def source_digest(source: str) -> str:
    """Return the canonical source binding used by witness artifacts."""
    if not isinstance(source, str):
        raise TypeError("source must be text")
    return _digest_bytes(source.encode("utf-8", errors="strict"))


@dataclass(frozen=True)
class WitnessProbe:
    """One deterministic replay input used as a behavioral witness."""

    probe_id: str
    seed: int
    actions: Tuple[str, ...]
    role: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("probe seed must be an integer")
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("probe role must be non-empty text")
        if not isinstance(self.actions, tuple) or any(
            not isinstance(action, str) for action in self.actions
        ):
            raise TypeError("probe actions must be a tuple of strings")
        expected = self.derive_id(seed=self.seed, actions=self.actions, role=self.role)
        if self.probe_id != expected:
            raise ValueError("probe_id is not bound to seed, actions, and role")

    @classmethod
    def create(cls, *, seed: int, actions: Sequence[str], role: str) -> "WitnessProbe":
        normalized = tuple(actions)
        return cls(
            probe_id=cls.derive_id(seed=seed, actions=normalized, role=role),
            seed=seed,
            actions=normalized,
            role=role,
        )

    @staticmethod
    def derive_id(*, seed: int, actions: Sequence[str], role: str) -> str:
        return _digest_json({"actions": list(actions), "role": role, "seed": seed})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "seed": self.seed,
            "actions": list(self.actions),
            "role": self.role,
        }


@dataclass(frozen=True)
class SourceVariant:
    """A deterministic source-level mutant or behavior-preserving control."""

    variant_id: str
    kind: VariantKind
    partition: SelectionPartition
    family: str
    operator: str
    source: str = field(repr=False)
    source_digest: str
    entrypoint: str
    expected_effect: str

    def __post_init__(self) -> None:
        if self.kind not in {"semantic_mutant", "equivalent_control"}:
            raise ValueError("unsupported variant kind")
        if self.partition not in {"train", "heldout"}:
            raise ValueError("unsupported variant partition")
        if not self.family or not self.operator or not self.entrypoint:
            raise ValueError("variant identifiers must be non-empty")
        if source_digest(self.source) != self.source_digest:
            raise ValueError("source_digest does not bind the variant source")

    def to_dict(self, *, include_source: bool = False) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "variant_id": self.variant_id,
            "kind": self.kind,
            "partition": self.partition,
            "family": self.family,
            "operator": self.operator,
            "source_digest": self.source_digest,
            "entrypoint": self.entrypoint,
            "expected_effect": self.expected_effect,
        }
        if include_source:
            value["source"] = self.source
        return value


@dataclass(frozen=True)
class TraceSignature:
    """Canonical digest plus useful, non-textual replay features."""

    digest: str
    success: bool
    final_reward: float
    turn_count: int
    terminated: bool
    truncated: bool
    error: Optional[str]
    observation_digests: Tuple[str, ...]
    reward_path: Tuple[float, ...]
    termination_path: Tuple[bool, ...]
    truncation_path: Tuple[bool, ...]
    info_digests: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "digest": self.digest,
            "success": self.success,
            "final_reward": self.final_reward,
            "turn_count": self.turn_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "error": self.error,
            "observation_digests": list(self.observation_digests),
            "reward_path": list(self.reward_path),
            "termination_path": list(self.termination_path),
            "truncation_path": list(self.truncation_path),
            "info_digests": list(self.info_digests),
        }


def _result_field(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise ValueError(f"execution result lacks {name!r}")
        return result[name]
    if not hasattr(result, name):
        raise ValueError(f"execution result lacks {name!r}")
    return getattr(result, name)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def trace_signature(result: Any) -> TraceSignature:
    """Canonicalize a ProofPack-like isolated replay result.

    Only externally visible transition fields are retained.  Input actions are
    already bound by :class:`WitnessProbe`, so echoed action text is excluded.
    Mapping key order and integer-versus-float reward representation cannot
    perturb the signature.
    """

    success = _result_field(result, "success")
    terminated = _result_field(result, "terminated")
    truncated = _result_field(result, "truncated")
    turn_count = _result_field(result, "turn_count")
    error = _result_field(result, "error")
    trajectory = _result_field(result, "trajectory")
    if type(success) is not bool or type(terminated) is not bool or type(truncated) is not bool:
        raise TypeError("execution status fields must be booleans")
    if type(turn_count) is not int or turn_count < 0:
        raise ValueError("turn_count must be a non-negative integer")
    if error is not None and not isinstance(error, str):
        raise TypeError("execution error must be text or None")
    if not isinstance(trajectory, list) or any(
        not isinstance(item, Mapping) for item in trajectory
    ):
        raise TypeError("trajectory must be a list of mappings")

    final_reward = _finite_number(_result_field(result, "reward"), "final reward")
    projected: list[Dict[str, Any]] = []
    observation_digests: list[str] = []
    reward_path: list[float] = []
    termination_path: list[bool] = []
    truncation_path: list[bool] = []
    info_digests: list[str] = []
    for index, item in enumerate(trajectory):
        observation = item.get("observation")
        if not isinstance(observation, str):
            raise TypeError("trajectory observations must be text")
        observation_digest = _digest_bytes(observation.encode("utf-8"))
        observation_digests.append(observation_digest)
        step: Dict[str, Any] = {"observation": observation}
        if index:
            reward = _finite_number(item.get("reward"), "trajectory reward")
            step_terminated = item.get("terminated")
            step_truncated = item.get("truncated")
            info = item.get("info")
            if type(step_terminated) is not bool or type(step_truncated) is not bool:
                raise TypeError("trajectory termination flags must be booleans")
            if not isinstance(info, Mapping):
                raise TypeError("trajectory info must be a mapping")
            info_value = dict(info)
            # Validate serializability and normalize mapping order now.
            info_digest = _digest_json(info_value)
            step.update(
                {
                    "reward": reward,
                    "terminated": step_terminated,
                    "truncated": step_truncated,
                    "info": info_value,
                }
            )
            reward_path.append(reward)
            termination_path.append(step_terminated)
            truncation_path.append(step_truncated)
            info_digests.append(info_digest)
        projected.append(step)

    payload = {
        "success": success,
        "reward": final_reward,
        "turn_count": turn_count,
        "terminated": terminated,
        "truncated": truncated,
        "trajectory": projected,
        "error": error,
    }
    return TraceSignature(
        digest=_digest_json(payload),
        success=success,
        final_reward=final_reward,
        turn_count=turn_count,
        terminated=terminated,
        truncated=truncated,
        error=error,
        observation_digests=tuple(observation_digests),
        reward_path=tuple(reward_path),
        termination_path=tuple(termination_path),
        truncation_path=tuple(truncation_path),
        info_digests=tuple(info_digests),
    )


def _method_names(node: ast.ClassDef) -> set[str]:
    return {
        item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def proofpack_compatible_entrypoints(tree_or_source: Any) -> Tuple[str, ...]:
    """Statically mirror ProofPack's concrete ``*Env`` discovery semantics."""
    tree = ast.parse(tree_or_source) if isinstance(tree_or_source, str) else tree_or_source
    if not isinstance(tree, ast.Module):
        raise TypeError("expected source text or ast.Module")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def base_names(node: ast.ClassDef) -> Tuple[str, ...]:
        return tuple(base.id for base in node.bases if isinstance(base, ast.Name))

    def supplies(name: str, method: str, seen: frozenset[str] = frozenset()) -> bool:
        if name == "ToolUseBaseEnv":
            return method in {"reset", "step", "solution"}
        if name in seen or name not in classes:
            return False
        node = classes[name]
        return method in _method_names(node) or any(
            supplies(base, method, seen | {name}) for base in base_names(node)
        )

    return tuple(
        name
        for name in classes
        if name.endswith("Env") and supplies(name, "reset") and supplies(name, "step")
    )


_VARIANT_SPECS: Tuple[Tuple[VariantKind, SelectionPartition, str, str, str], ...] = (
    (
        "equivalent_control",
        "train",
        "equivalence",
        "ast_roundtrip",
        "Formatting-only AST round trip",
    ),
    (
        "equivalent_control",
        "train",
        "equivalence",
        "inert_subclass",
        "Rename the implementation class and add an inert public subclass",
    ),
    (
        "equivalent_control",
        "heldout",
        "equivalence",
        "dead_branch_subclass",
        "Add a statically dead class-body branch",
    ),
    (
        "semantic_mutant",
        "train",
        "reward",
        "suppress_positive_reward",
        "Suppress positive transition rewards",
    ),
    (
        "semantic_mutant",
        "train",
        "termination",
        "swap_terminal_flags",
        "Swap terminated and truncated flags",
    ),
    (
        "semantic_mutant",
        "train",
        "action_acceptance",
        "sentinel_success_backdoor",
        "Accept a planted sentinel action as success",
    ),
    (
        "semantic_mutant",
        "train",
        "transition_observation",
        "tag_step_observation",
        "Alter every post-action observation",
    ),
    (
        "semantic_mutant",
        "train",
        "answer_leakage",
        "leak_solution_on_reset",
        "Expose solution data in the reset observation",
    ),
    (
        "semantic_mutant",
        "heldout",
        "seed_semantics",
        "collapse_reset_seed",
        "Force every reset onto seed zero",
    ),
    (
        "semantic_mutant",
        "heldout",
        "reward_shaping",
        "bonus_nonpositive_reward",
        "Award positive shaping reward after non-positive transitions",
    ),
    (
        "semantic_mutant",
        "heldout",
        "horizon",
        "truncate_nonterminal_step",
        "Prematurely truncate every continuing transition",
    ),
    (
        "semantic_mutant",
        "heldout",
        "parser",
        "corrupt_first_closing_brace",
        "Corrupt the first closing brace before the environment parses an action",
    ),
)


def mutation_catalog_digest() -> str:
    return _digest_json(
        {
            "schema_version": MUTATION_CATALOG_VERSION,
            "variants": [list(spec) for spec in _VARIANT_SPECS],
        }
    )


def _wrapper_body(operator: str, entrypoint: str, base_name: str) -> ast.ClassDef:
    bodies = {
        "inert_subclass": "pass",
        "dead_branch_subclass": "if False:\n        pass",
        "suppress_positive_reward": """def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if reward > 0:
            reward = 0.0
        return observation, reward, terminated, truncated, info""",
        "swap_terminal_flags": """def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, truncated, terminated, info""",
        "sentinel_success_backdoor": """def step(self, action):
        if "__cwa_semantic_trigger__" in action:
            return "CWA planted success", 1.0, True, False, {}
        return super().step(action)""",
        "tag_step_observation": """def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        observation = str(observation) + "\\n[CWA MUTATED TRANSITION]"
        return observation, reward, terminated, truncated, info""",
        "leak_solution_on_reset": """def reset(self, seed=None):
        observation, info = super().reset(seed=seed)
        observation = str(observation) + "\\n[CWA LEAK: " + str(self.solution()) + "]"
        return observation, info""",
        "collapse_reset_seed": """def reset(self, seed=None):
        return super().reset(seed=0)""",
        "bonus_nonpositive_reward": """def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if reward <= 0:
            reward = 0.25
        return observation, reward, terminated, truncated, info""",
        "truncate_nonterminal_step": """def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if not terminated and not truncated:
            truncated = True
        return observation, reward, terminated, truncated, info""",
        "corrupt_first_closing_brace": """def step(self, action):
        action = action.replace("}", "__cwa_parser_mutant__}", 1)
        return super().step(action)""",
    }
    body = bodies.get(operator)
    if body is None:
        raise ValueError(f"operator {operator!r} has no wrapper")
    source = f"class {entrypoint}({base_name}):\n    " + body.replace("\n", "\n    ") + "\n"
    parsed = ast.parse(source)
    wrapper = parsed.body[0]
    assert isinstance(wrapper, ast.ClassDef)
    return wrapper


def _wrapped_source(game_code: str, *, entrypoint: str, operator: str) -> str:
    tree = ast.parse(game_code)
    base_name = "_CwaBase" + source_digest(game_code).split(":", 1)[1][:12]
    if base_name in {node.name for node in tree.body if isinstance(node, ast.ClassDef)}:
        raise ValueError("generated source collides with reserved CWA base name")
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.ClassDef) and node.name == entrypoint:
            node.name = base_name
            wrapper = _wrapper_body(operator, entrypoint, base_name)
            tree.body.insert(index + 1, wrapper)
            ast.fix_missing_locations(tree)
            source = ast.unparse(tree) + "\n"
            discovered = proofpack_compatible_entrypoints(source)
            if discovered != (entrypoint,):
                raise ValueError(
                    "variant wrapper does not preserve exactly one ProofPack-compatible entrypoint"
                )
            return source
    raise ValueError(f"environment entrypoint {entrypoint!r} disappeared during rewriting")


def generate_source_variants(game_code: str) -> Tuple[SourceVariant, ...]:
    """Build a deterministic, source-only train/held-out mutation panel.

    The same operator families apply to explicit ``step`` implementations and
    ToolUseBaseEnv subclasses.  The original environment class is renamed to a
    non-``Env`` base before a public wrapper is inserted, keeping ProofPack's
    exactly-one-entrypoint invariant intact.
    """
    if not isinstance(game_code, str) or not game_code.strip():
        raise ValueError("game_code must be non-empty text")
    tree = ast.parse(game_code)
    entrypoints = proofpack_compatible_entrypoints(tree)
    if len(entrypoints) != 1:
        raise ValueError(
            "source must contain exactly one ProofPack-compatible environment; "
            f"found {list(entrypoints)}"
        )
    entrypoint = entrypoints[0]
    base_digest = source_digest(game_code)
    variants: list[SourceVariant] = []
    for kind, partition, family, operator, expected_effect in _VARIANT_SPECS:
        if operator == "ast_roundtrip":
            variant_source = ast.unparse(tree) + "\n"
            while variant_source == game_code:
                variant_source += "\n"
        else:
            variant_source = _wrapped_source(
                game_code,
                entrypoint=entrypoint,
                operator=operator,
            )
        digest = source_digest(variant_source)
        variant_id = _digest_json(
            {
                "base_digest": base_digest,
                "catalog": MUTATION_CATALOG_VERSION,
                "operator": operator,
                "source_digest": digest,
            }
        )
        variants.append(
            SourceVariant(
                variant_id=variant_id,
                kind=kind,
                partition=partition,
                family=family,
                operator=operator,
                source=variant_source,
                source_digest=digest,
                entrypoint=entrypoint,
                expected_effect=expected_effect,
            )
        )
    return tuple(variants)


def _probe_actions(action_format: str) -> Tuple[str, str, str]:
    if action_format == "boxed":
        return (r"\boxed{__cwa_semantic_trigger__}", r"\boxed{", "")
    if action_format == "tool_call":
        return (
            '<tool_call>{"arguments":{},"name":"__cwa_semantic_trigger__"}</tool_call>',
            "<tool_call>{</tool_call>",
            "",
        )
    raise ValueError("action_format must be 'boxed' or 'tool_call'")


def build_candidate_probes(
    *,
    action_format: str,
    seeds: Sequence[int],
    base_oracle_actions: Mapping[int, Sequence[str]],
    mutant_oracle_actions: Iterable[Mapping[int, Sequence[str]]] = (),
    max_turns: int = 20,
    max_candidates: int = 256,
) -> Tuple[WitnessProbe, ...]:
    """Construct a bounded deterministic branch-probe bank.

    Mutant oracle actions may enrich discovery, but callers should pass training
    mutants only.  Probes are deduplicated by exact ``(seed, actions)`` so role
    aliases never consume additional execution budget.
    """
    normalized_seeds = tuple(seeds)
    if not normalized_seeds or any(type(seed) is not int for seed in normalized_seeds):
        raise ValueError("seeds must be a non-empty sequence of integers")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if type(max_candidates) is not int or max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    wrong, malformed, blank = _probe_actions(action_format)
    mutant_maps = tuple(mutant_oracle_actions)
    candidates: list[WitnessProbe] = []
    seen: set[Tuple[int, Tuple[str, ...]]] = set()

    def add(seed: int, actions: Sequence[str], role: str) -> None:
        normalized = tuple(actions)
        key = (seed, normalized)
        if len(normalized) > max_turns or key in seen or len(candidates) >= max_candidates:
            return
        seen.add(key)
        candidates.append(WitnessProbe.create(seed=seed, actions=normalized, role=role))

    for seed in normalized_seeds:
        if seed not in base_oracle_actions:
            raise ValueError(f"base_oracle_actions lacks seed {seed}")
        oracle = tuple(base_oracle_actions[seed])
        if not oracle or any(not isinstance(action, str) for action in oracle):
            raise ValueError(f"oracle actions for seed {seed} must be non-empty text")
        if len(oracle) > max_turns:
            raise ValueError(f"oracle actions for seed {seed} exceed max_turns")
        add(seed, (), "reset")
        add(seed, oracle, "base_oracle")
        add(
            seed,
            tuple(f" \t{action}\n" for action in oracle),
            "whitespace_wrapped_oracle",
        )
        add(seed, (wrong,), "well_formed_wrong")
        add(seed, (malformed,), "malformed")
        add(seed, (blank,), "blank_noop")
        if len(oracle) + 1 <= max_turns:
            add(seed, (wrong, *oracle), "wrong_then_oracle_recovery")
        if len(oracle) >= 2:
            add(seed, tuple(reversed(oracle)), "reversed_oracle")
        if max_turns >= 2:
            add(seed, (oracle[0], oracle[0]), "repeat_first_oracle_action")
        for index in range(1, len(oracle)):
            add(seed, (*oracle[:index], wrong), f"oracle_prefix_{index}_then_wrong")
            if len(oracle) + 1 <= max_turns:
                add(
                    seed,
                    (*oracle[:index], wrong, *oracle[index:]),
                    f"oracle_prefix_{index}_wrong_then_recover",
                )
        for mutant_index, mutant_map in enumerate(mutant_maps):
            actions = mutant_map.get(seed)
            if actions:
                add(seed, tuple(actions), f"training_mutant_{mutant_index}_oracle")
    return tuple(candidates)


@dataclass
class WitnessMatrix:
    """Replay signatures for one base environment, its variants, and probes."""

    base_environment_digest: str
    probes: Tuple[WitnessProbe, ...]
    variants: Tuple[SourceVariant, ...]
    signatures: Dict[str, Dict[str, TraceSignature]]
    base_key: str = "base"

    def signature(self, probe_id: str, variant_key: str) -> TraceSignature:
        try:
            return self.signatures[probe_id][variant_key]
        except KeyError as error:
            raise ValueError(
                f"witness matrix lacks signature probe={probe_id!r}, variant={variant_key!r}"
            ) from error

    def validate(self) -> None:
        probe_ids = {probe.probe_id for probe in self.probes}
        if len(probe_ids) != len(self.probes):
            raise ValueError("witness matrix contains duplicate probe IDs")
        variant_ids = {variant.variant_id for variant in self.variants}
        if len(variant_ids) != len(self.variants):
            raise ValueError("witness matrix contains duplicate variant IDs")
        expected = variant_ids | {self.base_key}
        if set(self.signatures) != probe_ids:
            raise ValueError("witness matrix signature probes differ from probe bank")
        for probe_id, row in self.signatures.items():
            if set(row) != expected:
                raise ValueError(f"signature row {probe_id!r} differs from variant panel")


@dataclass(frozen=True)
class WitnessSelection:
    selected_probe_ids: Tuple[str, ...]
    safe_probe_ids: Tuple[str, ...]
    rejected_control_breakers: Tuple[str, ...]
    killed_train_mutants: Tuple[str, ...]
    uncovered_train_mutants: Tuple[str, ...]
    family_coverage: Dict[str, Tuple[int, int]]
    budget: int
    algorithm: str = SELECTION_ALGORITHM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_probe_ids": list(self.selected_probe_ids),
            "safe_probe_ids": list(self.safe_probe_ids),
            "rejected_control_breakers": list(self.rejected_control_breakers),
            "killed_train_mutants": list(self.killed_train_mutants),
            "uncovered_train_mutants": list(self.uncovered_train_mutants),
            "family_coverage": {
                family: {"killed": counts[0], "total": counts[1]}
                for family, counts in sorted(self.family_coverage.items())
            },
            "budget": self.budget,
            "algorithm": self.algorithm,
        }


def select_witnesses(matrix: WitnessMatrix, *, budget: int = 16) -> WitnessSelection:
    """Select probes with a strict equivalent-control safety constraint."""
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be positive")
    matrix.validate()
    controls = tuple(
        variant
        for variant in matrix.variants
        if variant.kind == "equivalent_control" and variant.partition == "train"
    )
    mutants = tuple(
        variant
        for variant in matrix.variants
        if variant.kind == "semantic_mutant" and variant.partition == "train"
    )
    family_sizes = Counter(variant.family for variant in mutants)
    safe: list[WitnessProbe] = []
    breakers: list[str] = []
    kills_by_probe: Dict[str, set[str]] = {}
    for probe in matrix.probes:
        base = matrix.signature(probe.probe_id, matrix.base_key).digest
        if any(
            matrix.signature(probe.probe_id, control.variant_id).digest != base
            for control in controls
        ):
            breakers.append(probe.probe_id)
            continue
        safe.append(probe)
        kills_by_probe[probe.probe_id] = {
            mutant.variant_id
            for mutant in mutants
            if matrix.signature(probe.probe_id, mutant.variant_id).digest != base
        }

    mutant_by_id = {variant.variant_id: variant for variant in mutants}
    uncovered = set(mutant_by_id)
    selected: list[WitnessProbe] = []
    available = list(safe)
    while uncovered and available and len(selected) < budget:
        scored: list[Tuple[float, int, str, WitnessProbe]] = []
        for probe in available:
            newly_killed = kills_by_probe[probe.probe_id] & uncovered
            gain = sum(
                1.0 / family_sizes[mutant_by_id[variant_id].family] for variant_id in newly_killed
            )
            scored.append((-gain, len(probe.actions), probe.probe_id, probe))
        scored.sort(key=lambda value: (value[0], value[1], value[2]))
        negative_gain, _length, _probe_id, winner = scored[0]
        if negative_gain == 0:
            break
        selected.append(winner)
        uncovered.difference_update(kills_by_probe[winner.probe_id])
        available = [probe for probe in available if probe.probe_id != winner.probe_id]

    killed = set(mutant_by_id) - uncovered
    coverage: Dict[str, Tuple[int, int]] = {}
    for family, total in sorted(family_sizes.items()):
        family_ids = {variant.variant_id for variant in mutants if variant.family == family}
        coverage[family] = (len(family_ids & killed), total)
    return WitnessSelection(
        selected_probe_ids=tuple(probe.probe_id for probe in selected),
        safe_probe_ids=tuple(probe.probe_id for probe in safe),
        rejected_control_breakers=tuple(breakers),
        killed_train_mutants=tuple(sorted(killed)),
        uncovered_train_mutants=tuple(sorted(uncovered)),
        family_coverage=coverage,
        budget=budget,
    )


@dataclass(frozen=True)
class SelectionScore:
    mutant_recall: float
    family_macro_recall: float
    equivalent_false_rejection_rate: float
    killed_mutants: int
    total_mutants: int
    rejected_controls: int
    total_controls: int


def score_probe_ids(
    matrix: WitnessMatrix,
    probe_ids: Sequence[str],
    *,
    partition: SelectionPartition = "heldout",
) -> SelectionScore:
    """Score a fixed probe set on a disjoint variant partition."""
    matrix.validate()
    selected = tuple(probe_ids)
    known = {probe.probe_id for probe in matrix.probes}
    if len(set(selected)) != len(selected) or any(probe_id not in known for probe_id in selected):
        raise ValueError("probe_ids must be unique members of the matrix")
    mutants = [
        variant
        for variant in matrix.variants
        if variant.kind == "semantic_mutant" and variant.partition == partition
    ]
    controls = [
        variant
        for variant in matrix.variants
        if variant.kind == "equivalent_control" and variant.partition == partition
    ]

    def detected(variant: SourceVariant) -> bool:
        return any(
            matrix.signature(probe_id, variant.variant_id).digest
            != matrix.signature(probe_id, matrix.base_key).digest
            for probe_id in selected
        )

    killed = [variant for variant in mutants if detected(variant)]
    rejected = [variant for variant in controls if detected(variant)]
    family_totals = Counter(variant.family for variant in mutants)
    family_kills = Counter(variant.family for variant in killed)
    family_macro = (
        sum(family_kills[family] / total for family, total in family_totals.items())
        / len(family_totals)
        if family_totals
        else 0.0
    )
    return SelectionScore(
        mutant_recall=len(killed) / len(mutants) if mutants else 0.0,
        family_macro_recall=family_macro,
        equivalent_false_rejection_rate=(len(rejected) / len(controls) if controls else 0.0),
        killed_mutants=len(killed),
        total_mutants=len(mutants),
        rejected_controls=len(rejected),
        total_controls=len(controls),
    )


@dataclass(frozen=True)
class WitnessCertificate:
    schema_version: str
    environment_digest: str
    mutation_catalog_digest: str
    candidate_pool_digest: str
    selection: WitnessSelection
    probes: Tuple[WitnessProbe, ...]
    expected_signatures: Dict[str, str]
    metadata: Dict[str, Any]
    certificate_digest: str

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment_digest": self.environment_digest,
            "mutation_catalog_digest": self.mutation_catalog_digest,
            "candidate_pool_digest": self.candidate_pool_digest,
            "selection": self.selection.to_dict(),
            "probes": [probe.to_dict() for probe in self.probes],
            "expected_signatures": dict(sorted(self.expected_signatures.items())),
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        value = self.unsigned_dict()
        value["certificate_digest"] = self.certificate_digest
        return value


def build_certificate(
    matrix: WitnessMatrix,
    selection: WitnessSelection,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WitnessCertificate:
    matrix.validate()
    by_id = {probe.probe_id: probe for probe in matrix.probes}
    probes = tuple(by_id[probe_id] for probe_id in selection.selected_probe_ids)
    expected = {
        probe.probe_id: matrix.signature(probe.probe_id, matrix.base_key).digest for probe in probes
    }
    candidate_digest = _digest_json([probe.to_dict() for probe in matrix.probes])
    unsigned = {
        "schema_version": WITNESS_SCHEMA_VERSION,
        "environment_digest": matrix.base_environment_digest,
        "mutation_catalog_digest": mutation_catalog_digest(),
        "candidate_pool_digest": candidate_digest,
        "selection": selection.to_dict(),
        "probes": [probe.to_dict() for probe in probes],
        "expected_signatures": dict(sorted(expected.items())),
        "metadata": dict(metadata or {}),
    }
    return WitnessCertificate(
        schema_version=WITNESS_SCHEMA_VERSION,
        environment_digest=matrix.base_environment_digest,
        mutation_catalog_digest=mutation_catalog_digest(),
        candidate_pool_digest=candidate_digest,
        selection=selection,
        probes=probes,
        expected_signatures=expected,
        metadata=dict(metadata or {}),
        certificate_digest=_digest_json(unsigned),
    )


@dataclass(frozen=True)
class WitnessVerification:
    passed: bool
    matched_probe_ids: Tuple[str, ...]
    mismatched_probe_ids: Tuple[str, ...]
    missing_probe_ids: Tuple[str, ...]


def verify_witness_signatures(
    certificate: WitnessCertificate,
    observed: Mapping[str, TraceSignature],
) -> WitnessVerification:
    """Compare externally gathered isolated replays with a sealed certificate."""
    if _digest_json(certificate.unsigned_dict()) != certificate.certificate_digest:
        raise ValueError("certificate digest does not bind its contents")
    matched: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    for probe in certificate.probes:
        signature = observed.get(probe.probe_id)
        if signature is None:
            missing.append(probe.probe_id)
        elif signature.digest == certificate.expected_signatures[probe.probe_id]:
            matched.append(probe.probe_id)
        else:
            mismatched.append(probe.probe_id)
    return WitnessVerification(
        passed=not mismatched and not missing,
        matched_probe_ids=tuple(matched),
        mismatched_probe_ids=tuple(mismatched),
        missing_probe_ids=tuple(missing),
    )


__all__ = [
    "MUTATION_CATALOG_VERSION",
    "SELECTION_ALGORITHM",
    "WITNESS_SCHEMA_VERSION",
    "SelectionScore",
    "SourceVariant",
    "TraceSignature",
    "WitnessCertificate",
    "WitnessMatrix",
    "WitnessProbe",
    "WitnessSelection",
    "WitnessVerification",
    "build_candidate_probes",
    "build_certificate",
    "generate_source_variants",
    "mutation_catalog_digest",
    "proofpack_compatible_entrypoints",
    "score_probe_ids",
    "select_witnesses",
    "source_digest",
    "trace_signature",
    "verify_witness_signatures",
]
