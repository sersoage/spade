#!/usr/bin/env python3
"""Run the sealed Google-only coverage-forced matched-swap pilot.

This is an exploratory requested-route experiment, not a learner update and not
a causal estimate of portfolio or environment quality. It compares a redundant
historical portfolio with a quality-matched coverage-forced swap, evaluates
every realized environment-plus-source-specific-hint package, and reports two
exact sensitivity analyses under explicit strong assumptions.

The protocol has two chronological seals:

* a generation-intent plan is written before any new AGY request;
* a concrete actor plan is written after all 18 candidates, qualifications,
  one-turn viability receipts, witnesses, and hints exist, but before any actor
  request.

Every provider request is durably reserved in both its run and the shared
global ledger before spawn.  A surviving request without a result is ambiguous
and is never replayed.  Only exact empty-response and provider-timeout failures
open a whole-pair actor retry.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spade.core.counterfactual_witness import (  # noqa: E402
    WitnessMatrix,
    build_candidate_probes,
    build_certificate,
    generate_source_variants,
    score_probe_ids,
    select_witnesses,
    trace_signature,
)
from spade.core.proofpack_bridge import validate_positive_proofpack_receipt  # noqa: E402
from spade.core.witness_archive import behavior_descriptor  # noqa: E402
from spade.core.witness_qd_proxy import (  # noqa: E402
    COVERAGE_CHALLENGER_DESCRIPTOR,
    EXACT_LABEL_PERMUTATION_COUNT,
    EXACT_SIGN_FLIP_COUNT,
    STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR,
    LockedPortfolios,
    PairedOutcome,
    ProxyCandidate,
    analyze_proxy_pilot,
    counterbalanced_pairs,
    lock_all_portfolios,
    lock_portfolios,
    portfolio_quality_diagnostics,
)
from tools import run_counterfactual_witness_experiment as cwa  # noqa: E402
from tools import run_live_spade_eval as live  # noqa: E402
from tools import run_spade_agy_experiment as base  # noqa: E402
from tools import run_spade_agy_outcome_replay as replay  # noqa: E402


INTENT_SCHEMA = "spade-coverage-forced-generation-intent/v1"
ACTOR_PLAN_SCHEMA = "spade-coverage-forced-actor-plan/v1"
RUN_SCHEMA = "spade-coverage-forced-run/v1"
CALL_REQUEST_SCHEMA = "spade-coverage-forced-call-request/v1"
CALL_RESULT_SCHEMA = "spade-coverage-forced-call-result/v1"
LEDGER_HEADER_SCHEMA = "spade-shared-agy-ledger/v1"
LEDGER_ENTRY_SCHEMA = "spade-shared-agy-ledger-entry/v1"
CANDIDATE_SCHEMA = "spade-coverage-forced-candidate-evidence/v1"
CWA_SCHEMA = "spade-coverage-forced-cwa-evidence/v1"
PAIR_ATTEMPT_SCHEMA = "spade-coverage-forced-pair-attempt/v1"
PAIR_RESOLUTION_SCHEMA = "spade-coverage-forced-pair-resolution/v1"
AGGREGATE_SCHEMA = "spade-coverage-forced-aggregate/v1"
PROTOCOL_ID = "spade-google-coverage-forced-matched-swap-pilot/v1"

V3_PLAN_DIGEST = "sha256:1ac5e27ddad0f68baaa54bd9eb67a6950773226cd936afd1a890a475822b2746"
V4_PLAN_DIGEST = "sha256:8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc"
V4_COHORT_DIGEST = "sha256:161353ebd4454516e3379414444323dd13aeab95640eb130ec7414f23876b84b"
DESIGN_MODEL = "gemini-3.1-pro-high"
ACTOR_MODEL = "gemini-3.7-flash-high"
SEEDS = (0, 42)
SHARED_STRATA = 6
SHARED_STRATUM_PREFIXES = ("c001", "c003", "c004", "c005", "c006", "c007")
CANDIDATES_PER_STRATUM = 3
DESIGN_ATTEMPTS = 2
HINT_ATTEMPTS = 2
ACTOR_HORIZON = 1
MAX_CHALLENGER_NONBLANK_LINES = 120
MAX_CHALLENGER_CHARACTERS = 8_000
PRIOR_CHARGED_CALLS = 205
NEW_CALL_CAP = 208
AUTHORIZED_GLOBAL_CALL_CAP = 450
THIRD_WAVE_LIMIT = 14
PAIR_COUNT = SHARED_STRATA * CANDIDATES_PER_STRATUM * len(SEEDS)
FIRST_TWO_ACTOR_WAVES = PAIR_COUNT * 2 * 2
THIRD_ACTOR_WAVE_CEILING = THIRD_WAVE_LIMIT * 2
PRE_ACTOR_CALL_CEILING = (
    SHARED_STRATA * DESIGN_ATTEMPTS + SHARED_STRATA * len(SEEDS) * HINT_ATTEMPTS
)
ACTOR_CALL_CEILING = FIRST_TWO_ACTOR_WAVES + THIRD_ACTOR_WAVE_CEILING
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,191}$")

if PRE_ACTOR_CALL_CEILING != 36 or ACTOR_CALL_CEILING != 172:
    raise AssertionError("sealed phase call ceilings drifted")
if PRE_ACTOR_CALL_CEILING + ACTOR_CALL_CEILING != NEW_CALL_CAP:
    raise AssertionError("sealed 208-call budget arithmetic drifted")


class ProxyExperimentError(RuntimeError):
    """Fail-closed plan, evidence, or execution error."""


class ProxyExperimentIncomplete(ProxyExperimentError):
    """The full locked panel did not complete."""


class AmbiguousProviderCall(ProxyExperimentIncomplete):
    """A durable request exists without a durable provider disposition."""


class CallCapExceeded(ProxyExperimentIncomplete):
    """The shared or local hard call cap would be exceeded."""


class _Target(Protocol):
    def inspect(self, seed: int, *, timeout_seconds: float | None = None) -> Any: ...
    def run_actions(
        self, seed: int | None, actions: Sequence[str], timeout_seconds: float | None = None
    ) -> Any: ...
    def run_oracle(self, seed: int = 0, timeout_seconds: float | None = None) -> Any: ...
    def instantiate(self, **kwargs: Any) -> Any: ...


LlmCall = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class RunnerDependencies:
    """Injected live boundaries.  Offline tests replace every boundary."""

    llm_call: LlmCall
    qualify: Callable[..., Any]
    target_factory: Callable[..., _Target]
    client_or_bin: Any
    runtime_identity: Mapping[str, Any]
    cwa_evaluator: (
        Callable[[Mapping[str, Any], Callable[..., _Target]], Mapping[str, Any]] | None
    ) = None


@dataclass(frozen=True)
class RunResult:
    status: str
    intent_digest: str
    run_dir: Path
    call_count: int
    actor_plan_path: Path | None = None
    aggregate_path: Path | None = None


@dataclass(frozen=True)
class HistoricalPanel:
    label: str
    run_dir: Path
    plan: Mapping[str, Any]
    selections: Mapping[str, Mapping[str, Any]]
    manifest: tuple[Mapping[str, Any], ...]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ProxyExperimentError(f"non-canonical evidence value: {type(value).__name__}")


def _safe_id(value: object, where: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ProxyExperimentError(f"{where} is not a safe identifier")
    return value


def _canonical_dir(value: Path | str, where: str, *, exists: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ProxyExperimentError(f"{where} must be absolute")
    base._reject_symlink_ancestors(path)
    resolved = path.resolve(strict=False)
    if path != resolved or path.is_symlink():
        raise ProxyExperimentError(f"{where} must be canonical and non-symlinked")
    if exists and not path.is_dir():
        raise ProxyExperimentError(f"{where} does not exist")
    if not exists and path.exists() and not path.is_dir():
        raise ProxyExperimentError(f"{where} is not a directory")
    return path


def _safe_tree(root: Path) -> tuple[Path, ...]:
    base._reject_symlink_ancestors(root)
    if root.is_symlink() or not root.is_dir():
        raise ProxyExperimentError(f"unsafe evidence tree: {root}")
    leaves: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            if (current_path / name).is_symlink():
                raise ProxyExperimentError(f"symlink in evidence tree: {current_path / name}")
        for name in files:
            path = current_path / name
            if path.suffix != ".json" or not path.is_file():
                raise ProxyExperimentError(f"unexpected evidence leaf: {path}")
            leaves.append(path)
    return tuple(sorted(leaves))


def _selected_candidate_inventory(root: Path, plan: Mapping[str, Any]) -> tuple[Path, ...]:
    """Require the exact selected-candidate pre-actor directory grammar."""
    leaves = _safe_tree(root)
    root_files = {path.name for path in root.iterdir() if path.is_file()}
    root_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    design_names = sorted(
        name for name in root_files if re.fullmatch(r"design-attempt-[0-9]{2}\.json", name)
    )
    if (
        root_dirs != {"probes", "hints"}
        or root_files != {"disposition.json", *design_names}
        or not design_names
        or len(design_names) > int(plan["configuration"]["design_attempts_per_slot"])
        or design_names
        != [f"design-attempt-{index:02d}.json" for index in range(1, len(design_names) + 1)]
    ):
        raise ProxyExperimentError(f"selected candidate root inventory is invalid: {root}")
    expected_seeds = {str(seed) for seed in plan["evaluation_seeds"]}
    probes = root / "probes"
    if {path.name for path in probes.iterdir()} != {f"seed-{seed}.json" for seed in expected_seeds}:
        raise ProxyExperimentError("selected candidate probe inventory differs")
    hints = root / "hints"
    if {path.name for path in hints.iterdir()} != expected_seeds:
        raise ProxyExperimentError("selected candidate hint seed inventory differs")
    for seed in expected_seeds:
        seed_root = hints / seed
        if seed_root.is_symlink() or not seed_root.is_dir():
            raise ProxyExperimentError("selected candidate hint seed path is unsafe")
        names = sorted(path.name for path in seed_root.iterdir())
        if (
            not names
            or len(names) > int(plan["configuration"]["hint_attempts"])
            or names != [f"attempt-{index:02d}.json" for index in range(1, len(names) + 1)]
        ):
            raise ProxyExperimentError("selected candidate hint attempts are not contiguous")
    return leaves


def _manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "digest": _bytes_digest(content),
        "size_bytes": len(content),
    }


def _dummy_dependencies(plan: Mapping[str, Any]) -> base.RunnerDependencies:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise ProxyExperimentError("historical validation crossed a live boundary")

    async def forbidden_async(*_args: Any, **_kwargs: Any) -> str:
        raise ProxyExperimentError("historical validation crossed a provider boundary")

    return base.RunnerDependencies(
        llm_call=forbidden_async,
        qualify=forbidden,
        target_factory=forbidden,
        assay_writer=forbidden,
        task_factory=forbidden,
        cluster_factory=forbidden,
        run_metadata_factory=forbidden,
        client_or_bin="historical-validation-only",
        source_revisions=dict(plan["source_revisions"]),
        runtime_identity=dict(plan["runtime_identity"]),
    )


def _shared_clusters(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    schedule = plan.get("cluster_schedule")
    if not isinstance(schedule, list):
        raise ProxyExperimentError("historical plan has no cluster schedule")
    by_prefix: dict[str, Mapping[str, Any]] = {}
    for item in schedule:
        if not isinstance(item, Mapping):
            raise ProxyExperimentError("historical cluster record is invalid")
        prefix = str(item.get("cluster_id", ""))[:4]
        if prefix in SHARED_STRATUM_PREFIXES:
            if prefix in by_prefix:
                raise ProxyExperimentError("historical plan duplicates a sealed stratum")
            by_prefix[prefix] = item
    if tuple(by_prefix) != SHARED_STRATUM_PREFIXES:
        # Dict insertion order is source chronology; require the exact prospective order.
        raise ProxyExperimentError("historical sealed stratum order differs")
    return [by_prefix[prefix] for prefix in SHARED_STRATUM_PREFIXES]


def _historical_panel(
    label: str, run_dir: Path | str, expected_plan_digest: str
) -> HistoricalPanel:
    root = _canonical_dir(run_dir, f"{label}_run", exists=True)
    plan = base.load_plan(root / "plan.json")
    if (
        plan["plan_digest"] != expected_plan_digest
        or plan["provider"] != "agy"
        or plan["model"] != DESIGN_MODEL
        or plan["backend_identity_attested"] is not False
        or plan["route_authority"] != "requested-route-only"
    ):
        raise ProxyExperimentError(f"{label} source plan identity differs")
    if base.derive_run_dir(plan["run_output_root"], plan) != root:
        raise ProxyExperimentError(f"{label} source run path is not canonical for its plan")
    engine = base._Engine(dict(plan), base._pretty_json(plan), root, _dummy_dependencies(plan))
    if label == "v4":
        cohort = engine._load_cohort()
        if cohort["cohort_digest"] != V4_COHORT_DIGEST:
            raise ProxyExperimentError("v4 cohort digest differs from the authorized source")
    shared_clusters = _shared_clusters(plan)
    selections: dict[str, Mapping[str, Any]] = {}
    manifest_paths: set[Path] = {root / "plan.json", root / "run-manifest.json"}
    if label == "v4":
        manifest_paths.add(root / "cohort-lock.json")
    referenced_calls: set[str] = set()
    for cluster in shared_clusters:
        cluster_id = str(cluster["cluster_id"])
        selection = dict(engine._load_selection(cluster_id))
        selections[cluster_id] = selection
        selection_path = root / "selections" / f"{cluster_id}.json"
        manifest_paths.add(selection_path)
        candidate_root = root / "candidates" / cluster_id / str(selection["candidate_id"])
        for path in _selected_candidate_inventory(candidate_root, plan):
            value = base._read_json(path)
            call_id = value.get("call_id")
            if isinstance(call_id, str):
                referenced_calls.add(call_id)
            manifest_paths.add(path)
    for call_id in referenced_calls:
        call_root = root / "calls" / call_id
        request_path = call_root / "request.json"
        result_path = call_root / "result.json"
        request = engine._validate_call_request(base._read_json(request_path))
        result = engine._validate_call_result(base._read_json(result_path), request)
        if request["purpose"].get("phase") not in {"designer", "hint"}:
            raise ProxyExperimentError("historical import closure contains actor evidence")
        if result["status"] not in {"success", "error"}:
            raise ProxyExperimentError("historical call has an invalid disposition")
        manifest_paths.update({request_path, result_path})
    manifest = tuple(_manifest_entry(path, root) for path in sorted(manifest_paths))
    if not manifest:
        raise ProxyExperimentError(f"{label} import manifest is empty")
    return HistoricalPanel(label, root, plan, selections, manifest)


def _runtime_identity() -> dict[str, Any]:
    runtime = dict(base._runtime_identity())
    proofpack_target = Path(sys.modules[live.SpadeEnvironmentTarget.__module__].__file__).resolve()
    proofpack_sandbox = Path(
        getattr(sys.modules[live.SpadeEnvironmentTarget.__module__], "SANDBOX_EXECUTABLE")
    ).resolve()
    runner_files = {
        "coverage_forced_runner": Path(__file__).resolve(),
        "coverage_forced_core": Path(sys.modules[ProxyCandidate.__module__].__file__).resolve(),
        "counterfactual_witness_core": Path(
            sys.modules[WitnessMatrix.__module__].__file__
        ).resolve(),
        "witness_archive_core": Path(
            sys.modules[behavior_descriptor.__module__].__file__
        ).resolve(),
        "counterfactual_witness_runner": Path(cwa.__file__).resolve(),
        "outcome_replay_runner": Path(replay.__file__).resolve(),
        "proofpack_spade_target": proofpack_target,
        "proofpack_spade_launcher": proofpack_target.with_name("spade_launcher.py"),
        "proofpack_spade_worker": proofpack_target.with_name("spade_worker.py"),
        "proofpack_sandbox_executable": proofpack_sandbox,
    }
    runtime["coverage_forced_files"] = {
        name: {"path": str(path), "digest": _bytes_digest(path.read_bytes())}
        for name, path in sorted(runner_files.items())
    }
    return runtime


def _validate_runtime_identity(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProxyExperimentError("runtime_identity must be an object")
    base_identity = dict(value)
    files = base_identity.pop("coverage_forced_files", None)
    base._validate_runtime_identity(base_identity)
    if not isinstance(files, Mapping) or set(files) != {
        "coverage_forced_runner",
        "coverage_forced_core",
        "counterfactual_witness_core",
        "witness_archive_core",
        "counterfactual_witness_runner",
        "outcome_replay_runner",
        "proofpack_spade_target",
        "proofpack_spade_launcher",
        "proofpack_spade_worker",
        "proofpack_sandbox_executable",
    }:
        raise ProxyExperimentError("runtime identity lacks the exact coverage-pilot files")
    for record in files.values():
        if not isinstance(record, Mapping) or set(record) != {"path", "digest"}:
            raise ProxyExperimentError("runtime file identity has invalid fields")
        path = Path(str(record["path"]))
        base._reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file() or path != path.resolve():
            raise ProxyExperimentError(f"unsafe sealed runtime file: {path}")
        if _bytes_digest(path.read_bytes()) != record["digest"]:
            raise ProxyExperimentError(f"sealed runtime file bytes drifted: {path}")
    return value


def _source_record(panel: HistoricalPanel) -> dict[str, Any]:
    return {
        "label": panel.label,
        "run_dir": str(panel.run_dir),
        "plan_digest": panel.plan["plan_digest"],
        "cohort_digest": (V4_COHORT_DIGEST if panel.label == "v4" else None),
        "manifest": list(panel.manifest),
        "manifest_digest": _digest(panel.manifest),
        "manifest_leaf_count": len(panel.manifest),
    }


def _candidate_id(stratum_id: str, source_arm: str) -> str:
    return _safe_id(f"{stratum_id}--{source_arm}", "candidate_id")


def build_generation_intent(
    *,
    experiment_id: str,
    v3_run: Path | str,
    v4_run: Path | str,
    output_root: Path | str,
    shared_ledger_root: Path | str,
    runtime_identity: Mapping[str, Any] | None = None,
    llm_timeout_seconds: float = 180.0,
    qualification_timeout_seconds: float = 5.0,
    cwa_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Validate historical evidence and seal every response-independent choice."""
    _safe_id(experiment_id, "experiment_id")
    output = _canonical_dir(output_root, "output_root", exists=False)
    ledger_root = _canonical_dir(shared_ledger_root, "shared_ledger_root", exists=False)
    if ledger_root.parent != output:
        raise ProxyExperimentError("shared ledger must be a direct child of output_root")
    for name, value in (
        ("llm_timeout_seconds", llm_timeout_seconds),
        ("qualification_timeout_seconds", qualification_timeout_seconds),
        ("cwa_timeout_seconds", cwa_timeout_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 3600:
            raise ProxyExperimentError(f"{name} must be finite in (0,3600]")
    v3 = _historical_panel("v3", v3_run, V3_PLAN_DIGEST)
    v4 = _historical_panel("v4", v4_run, V4_PLAN_DIGEST)
    strata: list[dict[str, Any]] = []
    for ordinal, (v3_item, v4_item) in enumerate(
        zip(_shared_clusters(v3.plan), _shared_clusters(v4.plan))
    ):
        fields = ("cluster_id", "skill", "difficulty")
        if any(v3_item[field] != v4_item[field] for field in fields):
            raise ProxyExperimentError("v3/v4 shared stratum definitions differ")
        stratum_id = str(v3_item["cluster_id"])
        historical = []
        for source_arm, panel in (("v3", v3), ("v4", v4)):
            selection = panel.selections[stratum_id]
            historical.append(
                {
                    "candidate_id": _candidate_id(stratum_id, source_arm),
                    "source_arm": source_arm,
                    "source_selection_digest": selection["selection_digest"],
                    "source_environment_digest": selection["environment_digest"],
                }
            )
        strata.append(
            {
                "stratum_ordinal": ordinal,
                "stratum_id": stratum_id,
                "skill": v3_item["skill"],
                "difficulty": v3_item["difficulty"],
                "historical_candidates": historical,
                "challenger_candidate_id": _candidate_id(stratum_id, "challenger"),
            }
        )
    runtime = dict(runtime_identity or _runtime_identity())
    _validate_runtime_identity(runtime)
    budget = {
        "governance_scope": (
            "externally-authorized-context-bound-to-this-canonical-ledger;"
            "not-a-cross-output-root-cryptographic-registry"
        ),
        "prior_charged_calls": PRIOR_CHARGED_CALLS,
        "new_call_cap": NEW_CALL_CAP,
        "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
        "planned_max_global_calls": PRIOR_CHARGED_CALLS + NEW_CALL_CAP,
        "headroom_calls": AUTHORIZED_GLOBAL_CALL_CAP - PRIOR_CHARGED_CALLS - NEW_CALL_CAP,
        "breakdown": {
            "challenger_design": SHARED_STRATA * DESIGN_ATTEMPTS,
            "challenger_hints": SHARED_STRATA * len(SEEDS) * HINT_ATTEMPTS,
            "actor_first_two_whole_pair_waves": FIRST_TWO_ACTOR_WAVES,
            "actor_optional_third_wave": THIRD_ACTOR_WAVE_CEILING,
        },
    }
    body = {
        "schema_version": INTENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": experiment_id,
        "analysis_role": (
            "exploratory-requested-route-quality-matched-coverage-forced-portfolio-swap-association"
        ),
        "claim_exclusions": [
            "no-backend-identity-attestation",
            "no-design-based-randomization-inference",
            "no-causal-portfolio-or-environment-effect",
            "no-learner-update-or-learner-improvement",
        ],
        "output_root": str(output),
        "shared_ledger_root": str(ledger_root),
        "provider": "agy",
        "models": {"designer_and_hint": DESIGN_MODEL, "actor": ACTOR_MODEL},
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "runtime_identity": runtime,
        "sources": [_source_record(v3), _source_record(v4)],
        "strata": strata,
        "configuration": {
            "action_format": "boxed",
            "seeds": list(SEEDS),
            "actor_horizon": ACTOR_HORIZON,
            "actor_success_rule": "reward-positive-even-if-simultaneously-truncated",
            "design_attempts_per_stratum": DESIGN_ATTEMPTS,
            "hint_attempts_per_seed": HINT_ATTEMPTS,
            "qualification_max_turns": 5,
            "llm_timeout_seconds": float(llm_timeout_seconds),
            "qualification_timeout_seconds": float(qualification_timeout_seconds),
            "cwa_timeout_seconds": float(cwa_timeout_seconds),
            "cwa_repetitions": 2,
            "witness_budget": 16,
            "cwa_minimum_training_recall": 0.90,
            "cwa_maximum_heldout_control_fpr": 0.05,
            "portfolio_quality": "heldout_applicable_recall_selected_over_safe-bank-observable",
            "portfolio_capacity": 2,
            "historical_control_descriptor": dict(STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR),
            "coverage_challenger_descriptor": dict(COVERAGE_CHALLENGER_DESCRIPTOR),
            "maximum_stratum_absolute_quality_gap": 0.125,
            "maximum_mean_absolute_quality_gap": 0.0625,
            "actor_pair_attempts": 3,
            "third_wave_unresolved_limit": THIRD_WAVE_LIMIT,
        },
        "budget": budget,
        "analysis_gates": {
            "pooled_unhinted_min": 0.10,
            "pooled_unhinted_max": 0.90,
            "minimum_discordant_pairs": 8,
            "minimum_coverage_forced_delta": 0.10,
            "one_sided_label_permutation_alpha": 0.05,
            "one_sided_sign_flip_alpha": 0.05,
            "every_leave_one_stratum_out_positive": True,
            "maximum_first_attempt_exogenous_failure_rate": 0.15,
            "label_permutation_space": EXACT_LABEL_PERMUTATION_COUNT,
            "label_permutation_assumption": (
                "strong-within-stratum-candidate-label-exchangeability"
            ),
            "sign_flip_space": EXACT_SIGN_FLIP_COUNT,
            "sign_flip_assumption": "six-stratum-contrast-sign-symmetry",
        },
    }
    intent = {**body, "intent_digest": _digest(body)}
    validate_generation_intent(intent)
    return intent


def validate_generation_intent(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProxyExperimentError("generation intent must be an object")
    required = {
        "schema_version",
        "protocol_id",
        "experiment_id",
        "analysis_role",
        "claim_exclusions",
        "output_root",
        "shared_ledger_root",
        "provider",
        "models",
        "backend_identity_attested",
        "route_authority",
        "runtime_identity",
        "sources",
        "strata",
        "configuration",
        "budget",
        "analysis_gates",
        "intent_digest",
    }
    if set(value) != required:
        raise ProxyExperimentError("generation intent fields differ from schema")
    if (
        value["schema_version"] != INTENT_SCHEMA
        or value["protocol_id"] != PROTOCOL_ID
        or value["provider"] != "agy"
        or value["models"] != {"designer_and_hint": DESIGN_MODEL, "actor": ACTOR_MODEL}
        or value["backend_identity_attested"] is not False
        or value["route_authority"] != "requested-route-only"
    ):
        raise ProxyExperimentError("generation intent provider identity differs")
    if value["analysis_role"] != (
        "exploratory-requested-route-quality-matched-coverage-forced-portfolio-swap-association"
    ) or value["claim_exclusions"] != [
        "no-backend-identity-attestation",
        "no-design-based-randomization-inference",
        "no-causal-portfolio-or-environment-effect",
        "no-learner-update-or-learner-improvement",
    ]:
        raise ProxyExperimentError("analysis role or claim boundary differs")
    _safe_id(value["experiment_id"], "experiment_id")
    output = _canonical_dir(str(value["output_root"]), "output_root", exists=False)
    ledger = _canonical_dir(str(value["shared_ledger_root"]), "shared_ledger_root", exists=False)
    if ledger.parent != output:
        raise ProxyExperimentError("shared ledger escapes output root")
    _validate_runtime_identity(value["runtime_identity"])
    sources = value["sources"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise ProxyExperimentError("intent must bind v3 and v4 sources")
    for source, label, digest in zip(sources, ("v3", "v4"), (V3_PLAN_DIGEST, V4_PLAN_DIGEST)):
        if (
            not isinstance(source, dict)
            or source.get("label") != label
            or source.get("plan_digest") != digest
        ):
            raise ProxyExperimentError("historical source identity differs")
        if source.get("cohort_digest") != (V4_COHORT_DIGEST if label == "v4" else None):
            raise ProxyExperimentError("historical source cohort binding differs")
        manifest = source.get("manifest")
        if (
            not isinstance(manifest, list)
            or len(manifest) != source.get("manifest_leaf_count")
            or _digest(manifest) != source.get("manifest_digest")
        ):
            raise ProxyExperimentError("historical source manifest is not sealed")
        paths = [item.get("path") for item in manifest if isinstance(item, dict)]
        if len(paths) != len(manifest) or len(set(paths)) != len(paths):
            raise ProxyExperimentError("historical source manifest paths are invalid")
        for item in manifest:
            if set(item) != {"path", "digest", "size_bytes"}:
                raise ProxyExperimentError("historical source manifest fields differ")
            if (
                not isinstance(item["digest"], str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", item["digest"]) is None
                or isinstance(item["size_bytes"], bool)
                or not isinstance(item["size_bytes"], int)
                or item["size_bytes"] < 1
            ):
                raise ProxyExperimentError("historical source manifest digest/size is invalid")
        if any("outcome" in str(path).lower() or "assay" in str(path).lower() for path in paths):
            raise ProxyExperimentError("historical manifest includes outcome/Assay evidence")
    strata = value["strata"]
    if not isinstance(strata, list) or len(strata) != SHARED_STRATA:
        raise ProxyExperimentError("intent requires exactly six strata")
    if [item.get("stratum_ordinal") for item in strata] != list(range(SHARED_STRATA)):
        raise ProxyExperimentError("stratum ordinals are not contiguous")
    ids: set[str] = set()
    if [str(item.get("stratum_id", ""))[:4] for item in strata] != list(SHARED_STRATUM_PREFIXES):
        raise ProxyExperimentError("intent strata differ from the sealed six-stratum panel")
    for item in strata:
        stratum_id = _safe_id(item.get("stratum_id"), "stratum_id")
        historical = item.get("historical_candidates")
        if not isinstance(historical, list) or [x.get("source_arm") for x in historical] != [
            "v3",
            "v4",
        ]:
            raise ProxyExperimentError("historical candidate topology differs")
        candidate_ids = [x.get("candidate_id") for x in historical] + [
            item.get("challenger_candidate_id")
        ]
        if candidate_ids != [_candidate_id(stratum_id, arm) for arm in ("v3", "v4", "challenger")]:
            raise ProxyExperimentError("candidate ids differ from the fixed topology")
        if ids.intersection(candidate_ids):
            raise ProxyExperimentError("candidate ids are duplicated")
        ids.update(candidate_ids)
    config = value["configuration"]
    expected_config = {
        "action_format": "boxed",
        "seeds": list(SEEDS),
        "actor_horizon": 1,
        "actor_success_rule": "reward-positive-even-if-simultaneously-truncated",
        "design_attempts_per_stratum": 2,
        "hint_attempts_per_seed": 2,
        "qualification_max_turns": 5,
        "cwa_repetitions": 2,
        "witness_budget": 16,
        "cwa_minimum_training_recall": 0.90,
        "cwa_maximum_heldout_control_fpr": 0.05,
        "portfolio_quality": "heldout_applicable_recall_selected_over_safe-bank-observable",
        "portfolio_capacity": 2,
        "historical_control_descriptor": dict(STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR),
        "coverage_challenger_descriptor": dict(COVERAGE_CHALLENGER_DESCRIPTOR),
        "maximum_stratum_absolute_quality_gap": 0.125,
        "maximum_mean_absolute_quality_gap": 0.0625,
        "actor_pair_attempts": 3,
        "third_wave_unresolved_limit": 14,
    }
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            *expected_config,
            "llm_timeout_seconds",
            "qualification_timeout_seconds",
            "cwa_timeout_seconds",
        }
        or any(config.get(key) != item for key, item in expected_config.items())
    ):
        raise ProxyExperimentError("sealed protocol configuration differs")
    for name in ("llm_timeout_seconds", "qualification_timeout_seconds", "cwa_timeout_seconds"):
        timeout = config[name]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 0 < float(timeout) <= 3600
        ):
            raise ProxyExperimentError(f"{name} is outside the sealed finite range")
    budget = value["budget"]
    expected_budget = {
        "governance_scope": (
            "externally-authorized-context-bound-to-this-canonical-ledger;"
            "not-a-cross-output-root-cryptographic-registry"
        ),
        "prior_charged_calls": 205,
        "new_call_cap": 208,
        "authorized_global_call_cap": 450,
        "planned_max_global_calls": 413,
        "headroom_calls": 37,
        "breakdown": {
            "challenger_design": 12,
            "challenger_hints": 24,
            "actor_first_two_whole_pair_waves": 144,
            "actor_optional_third_wave": 28,
        },
    }
    if budget != expected_budget:
        raise ProxyExperimentError("sealed 205 + 208 = 413 budget differs")
    gates = value["analysis_gates"]
    expected_gates = {
        "pooled_unhinted_min": 0.10,
        "pooled_unhinted_max": 0.90,
        "minimum_discordant_pairs": 8,
        "minimum_coverage_forced_delta": 0.10,
        "one_sided_label_permutation_alpha": 0.05,
        "one_sided_sign_flip_alpha": 0.05,
        "every_leave_one_stratum_out_positive": True,
        "maximum_first_attempt_exogenous_failure_rate": 0.15,
        "label_permutation_space": EXACT_LABEL_PERMUTATION_COUNT,
        "label_permutation_assumption": ("strong-within-stratum-candidate-label-exchangeability"),
        "sign_flip_space": EXACT_SIGN_FLIP_COUNT,
        "sign_flip_assumption": "six-stratum-contrast-sign-symmetry",
    }
    if gates != expected_gates:
        raise ProxyExperimentError("analysis gates differ from exact 6^6/2^6 protocol")
    body = {key: item for key, item in value.items() if key != "intent_digest"}
    if value["intent_digest"] != _digest(body):
        raise ProxyExperimentError("intent digest mismatch")
    return value


def load_generation_intent(path: Path | str) -> dict[str, Any]:
    return validate_generation_intent(base._read_json(Path(path)))


def derive_run_dir(intent: Mapping[str, Any]) -> Path:
    value = validate_generation_intent(dict(intent))
    return Path(str(value["output_root"])) / (
        f"{value['experiment_id']}-{str(value['intent_digest']).removeprefix('sha256:')}"
    )


def _default_dependencies(intent: Mapping[str, Any]) -> RunnerDependencies:
    live._require_integrations()
    client_or_bin, resolved = live.get_llm_client(
        "agy",
        DESIGN_MODEL,
        timeout_seconds=float(intent["configuration"]["llm_timeout_seconds"]),
    )
    if resolved != DESIGN_MODEL:
        raise ProxyExperimentError("AGY resolved a different designer route")
    runtime = _runtime_identity()
    return RunnerDependencies(
        llm_call=live.call_llm,
        qualify=live.qualify_spade_environment,
        target_factory=live.SpadeEnvironmentTarget,
        client_or_bin=client_or_bin,
        runtime_identity=runtime,
    )


@dataclass(frozen=True)
class _RecordedCallFailure(Exception):
    call_id: str
    category: str
    error: str

    @property
    def retryable(self) -> bool:
        return self.category in {"empty_response", "provider_timeout"}

    def __str__(self) -> str:
        return f"{self.call_id}: {self.category}: {self.error}"


def _failure_category(exc: Exception, timeout_seconds: float) -> str:
    message = str(exc)
    if isinstance(exc, live.LiveEvalError):
        if message == "agy returned an empty response":
            return "empty_response"
        if message == f"agy call timed out after {timeout_seconds:.0f}s":
            return "provider_timeout"
        if message == "agy failed with exit 1: Error: timeout waiting for response":
            return "provider_timeout"
    return "fatal_transport"


def _validate_manifest_bytes(source: Mapping[str, Any]) -> None:
    root = _canonical_dir(str(source["run_dir"]), f"{source['label']}_run", exists=True)
    expected_paths: set[str] = set()
    for item in source["manifest"]:
        if not isinstance(item, dict) or set(item) != {"path", "digest", "size_bytes"}:
            raise ProxyExperimentError("source manifest entry has invalid fields")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != item["path"]:
            raise ProxyExperimentError("source manifest path is unsafe")
        path = root / relative
        base._reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file():
            raise ProxyExperimentError(f"source manifest leaf is missing: {path}")
        content = path.read_bytes()
        if len(content) != item["size_bytes"] or _bytes_digest(content) != item["digest"]:
            raise ProxyExperimentError(f"source evidence bytes changed: {path}")
        expected_paths.add(relative.as_posix())
    if len(expected_paths) != len(source["manifest"]):
        raise ProxyExperimentError("source manifest has duplicate paths")


def _copy_imports(intent: Mapping[str, Any], run_dir: Path) -> None:
    for source in intent["sources"]:
        _validate_manifest_bytes(source)
        source_root = Path(str(source["run_dir"]))
        import_root = run_dir / "imports" / str(source["label"])
        for item in source["manifest"]:
            relative = Path(str(item["path"]))
            content = (source_root / relative).read_bytes()
            base._write_immutable_bytes(import_root / relative, content)
        observed = sorted(
            (_manifest_entry(path, import_root) for path in _safe_tree(import_root)),
            key=lambda item: item["path"],
        )
        if observed != sorted(source["manifest"], key=lambda item: item["path"]):
            raise ProxyExperimentError("copied historical import differs from its seal")


def _validate_runtime_now(sealed: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    _validate_runtime_identity(sealed)
    _validate_runtime_identity(current)
    if dict(sealed) != dict(current):
        raise ProxyExperimentIncomplete("current runtime identity differs from the intent seal")


class _Engine:
    def __init__(
        self,
        intent: dict[str, Any],
        intent_bytes: bytes,
        run_dir: Path,
        dependencies: RunnerDependencies,
    ) -> None:
        self.intent = intent
        self.intent_bytes = intent_bytes
        self.run_dir = run_dir
        self.dependencies = dependencies
        self.config = intent["configuration"]

    @property
    def call_count(self) -> int:
        root = self.run_dir / "calls"
        return len(list(root.glob("*/request.json"))) if root.is_dir() else 0

    @property
    def ledger_root(self) -> Path:
        return Path(str(self.intent["shared_ledger_root"]))

    def _validate_tree(self) -> None:
        for root in (self.run_dir, self.ledger_root):
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                raise ProxyExperimentError(f"unsafe artifact root: {root}")
            for current, directories, files in os.walk(root, topdown=True, followlinks=False):
                current_path = Path(current)
                for name in (*directories, *files):
                    if (current_path / name).is_symlink():
                        raise ProxyExperimentError(
                            f"symlink in experiment artifacts: {current_path / name}"
                        )

    def _ledger_header(self) -> dict[str, Any]:
        body = {
            "schema_version": LEDGER_HEADER_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "prior_charged_calls": PRIOR_CHARGED_CALLS,
            "new_call_cap": NEW_CALL_CAP,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "first_new_global_ordinal": PRIOR_CHARGED_CALLS + 1,
            "last_permitted_global_ordinal": PRIOR_CHARGED_CALLS + NEW_CALL_CAP,
        }
        return {**body, "header_digest": _digest(body)}

    def initialize(self) -> None:
        _validate_runtime_now(self.intent["runtime_identity"], self.dependencies.runtime_identity)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_root.mkdir(parents=True, exist_ok=True)
        self._validate_tree()
        base._write_immutable_bytes(self.run_dir / "generation-intent.json", self.intent_bytes)
        _copy_imports(self.intent, self.run_dir)
        base._write_json(self.ledger_root / "header.json", self._ledger_header())
        manifest = {
            "schema_version": RUN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "provider": "agy",
            "models": self.intent["models"],
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "new_call_cap": NEW_CALL_CAP,
            "shared_ledger_root": str(self.ledger_root),
            "runtime_identity": self.intent["runtime_identity"],
        }
        base._write_json(self.run_dir / "run-manifest.json", manifest)
        self._validate_existing_calls()

    def _verify_provider_executable(self) -> None:
        executable = Path(str(self.dependencies.client_or_bin)).resolve()
        runtime = self.intent["runtime_identity"]
        if str(executable) != runtime["agy_executable"]:
            raise ProxyExperimentIncomplete("AGY executable path drifted")
        if (
            not executable.is_file()
            or _bytes_digest(executable.read_bytes()) != runtime["agy_executable_digest"]
        ):
            raise ProxyExperimentIncomplete("AGY executable bytes drifted")

    def _model_for_phase(self, phase: str) -> str:
        if phase in {"challenger-design", "challenger-hint"}:
            return DESIGN_MODEL
        if phase == "actor":
            return ACTOR_MODEL
        raise ProxyExperimentError(f"unknown provider phase: {phase}")

    def _proofpack_call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Bracket every sandbox boundary with sealed-byte validation."""
        _validate_runtime_identity(self.intent["runtime_identity"])
        try:
            return operation(*args, **kwargs)
        finally:
            _validate_runtime_identity(self.intent["runtime_identity"])

    def _validate_call_request(
        self, value: Mapping[str, Any], expected: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        keys = {
            "schema_version",
            "intent_digest",
            "actor_plan_digest",
            "call_id",
            "local_ordinal",
            "global_ordinal",
            "purpose",
            "provider",
            "model",
            "backend_identity_attested",
            "route_authority",
            "runtime_identity_digest",
            "agy_executable_digest",
            "prompt",
            "prompt_digest",
            "system",
            "system_digest",
            "timeout_seconds",
            "workdir_policy",
            "reservation_status",
            "reserved_at_utc",
            "request_digest",
        }
        if set(value) != keys or value.get("schema_version") != CALL_REQUEST_SCHEMA:
            raise ProxyExperimentError("provider request fields differ from schema")
        body = {key: item for key, item in value.items() if key != "request_digest"}
        if value["request_digest"] != _digest(body):
            raise ProxyExperimentError("provider request digest mismatch")
        purpose = value["purpose"]
        if not isinstance(purpose, dict) or "phase" not in purpose:
            raise ProxyExperimentError("provider request purpose is invalid")
        phase = str(purpose["phase"])
        model = self._model_for_phase(phase)
        # Pre-actor requests are intentionally sealed before actor-plan.json exists.
        # They remain bound to a null actor digest after the second seal is written.
        actor_digest = self._actor_plan_digest(required=True) if phase == "actor" else None
        fixed = {
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": actor_digest,
            "provider": "agy",
            "model": model,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "runtime_identity_digest": _digest(self.intent["runtime_identity"]),
            "agy_executable_digest": self.intent["runtime_identity"]["agy_executable_digest"],
            "timeout_seconds": float(self.config["llm_timeout_seconds"]),
            "workdir_policy": "fresh-temporary-directory-per-call",
            "reservation_status": "reserved-before-spawn",
        }
        if any(value.get(key) != item for key, item in fixed.items()):
            raise ProxyExperimentError("provider request differs from sealed identity")
        if expected and any(value.get(key) != item for key, item in expected.items()):
            raise ProxyExperimentError("provider request differs from expected call")
        if value["call_id"] != base._call_id(purpose):
            raise ProxyExperimentError("provider call id does not bind its purpose")
        if value["global_ordinal"] != PRIOR_CHARGED_CALLS + value["local_ordinal"]:
            raise ProxyExperimentError("provider call global ordinal is invalid")
        if value["prompt_digest"] != _digest(value["prompt"]) or value["system_digest"] != _digest(
            value["system"]
        ):
            raise ProxyExperimentError("provider request prompt/system digest mismatch")
        base._validate_timestamp(value["reserved_at_utc"], "reserved_at_utc")
        return value

    def _validate_call_result(
        self, value: Mapping[str, Any], request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        keys = {
            "schema_version",
            "intent_digest",
            "call_id",
            "local_ordinal",
            "global_ordinal",
            "request_digest",
            "status",
            "failure_category",
            "exception_type",
            "error",
            "exit_status",
            "response",
            "response_digest",
            "started_at_utc",
            "finished_at_utc",
            "duration_seconds",
            "result_digest",
        }
        if set(value) != keys or value.get("schema_version") != CALL_RESULT_SCHEMA:
            raise ProxyExperimentError("provider result fields differ from schema")
        body = {key: item for key, item in value.items() if key != "result_digest"}
        if value["result_digest"] != _digest(body):
            raise ProxyExperimentError("provider result digest mismatch")
        fixed = {
            "intent_digest": self.intent["intent_digest"],
            "call_id": request["call_id"],
            "local_ordinal": request["local_ordinal"],
            "global_ordinal": request["global_ordinal"],
            "request_digest": request["request_digest"],
        }
        if any(value.get(key) != item for key, item in fixed.items()):
            raise ProxyExperimentError("provider result is not bound to its request")
        reserved = base._validate_timestamp(request["reserved_at_utc"], "reserved_at_utc")
        started = base._validate_timestamp(value["started_at_utc"], "started_at_utc")
        finished = base._validate_timestamp(value["finished_at_utc"], "finished_at_utc")
        duration = value["duration_seconds"]
        if (
            not reserved <= started <= finished
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ProxyExperimentError("provider result timing is invalid")
        if value["status"] == "success":
            if (
                any(
                    value[key] is not None
                    for key in ("failure_category", "exception_type", "error")
                )
                or value["exit_status"] != 0
            ):
                raise ProxyExperimentError("successful provider result is contradictory")
            if not isinstance(value["response"], str) or not value["response"].strip():
                raise ProxyExperimentError("successful provider response is empty")
            if value["response_digest"] != _digest(value["response"]):
                raise ProxyExperimentError("provider response digest mismatch")
        elif value["status"] == "error":
            if value["response"] is not None or value["response_digest"] is not None:
                raise ProxyExperimentError("failed provider result contains a response")
            if value["failure_category"] not in {
                "empty_response",
                "provider_timeout",
                "fatal_transport",
            }:
                raise ProxyExperimentError("provider failure category is invalid")
            if not isinstance(value["error"], str) or not value["error"]:
                raise ProxyExperimentError("provider failure lacks an error")
            exception_type = value["exception_type"]
            if not isinstance(exception_type, str) or not exception_type:
                raise ProxyExperimentError("provider failure lacks exception type")
            synthetic: Exception = (
                live.LiveEvalError(str(value["error"]), 4)
                if exception_type == "LiveEvalError"
                else RuntimeError(str(value["error"]))
            )
            expected_category = _failure_category(
                synthetic, float(self.config["llm_timeout_seconds"])
            )
            if value["failure_category"] != expected_category:
                raise ProxyExperimentError("provider failure category is not derivable")
            match = re.fullmatch(
                r"agy failed with exit (-?[0-9]+):.*", str(value["error"]), re.DOTALL
            )
            expected_exit = int(match.group(1)) if match else None
            if value["exit_status"] != expected_exit:
                raise ProxyExperimentError("provider failure exit status is not derivable")
        else:
            raise ProxyExperimentError("provider result status is invalid")
        return value

    def _ledger_entry(self, request: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema_version": LEDGER_ENTRY_SCHEMA,
            "header_digest": self._ledger_header()["header_digest"],
            "intent_digest": self.intent["intent_digest"],
            "global_ordinal": request["global_ordinal"],
            "local_ordinal": request["local_ordinal"],
            "call_id": request["call_id"],
            "request_digest": request["request_digest"],
            "request_path": str(
                (self.run_dir / "calls" / str(request["call_id"]) / "request.json").relative_to(
                    Path(str(self.intent["output_root"]))
                )
            ),
            "model": request["model"],
            "reserved_at_utc": request["reserved_at_utc"],
        }
        return {**body, "entry_digest": _digest(body)}

    def _validate_ledger_entry(self, value: Mapping[str, Any], request: Mapping[str, Any]) -> None:
        expected = self._ledger_entry(request)
        if dict(value) != expected:
            raise ProxyExperimentError("shared ledger entry differs from call reservation")

    def _actor_plan_digest(self, *, required: bool) -> str | None:
        path = self.run_dir / "actor-plan.json"
        if not path.is_file():
            if required:
                raise ProxyExperimentError("actor call is forbidden before the actor plan seal")
            return None
        return validate_actor_plan(base._read_json(path), self.intent, run_dir=self.run_dir)[
            "actor_plan_digest"
        ]

    def _validate_existing_calls(self) -> None:
        calls = self.run_dir / "calls"
        requests = sorted(calls.glob("*/request.json")) if calls.is_dir() else []
        results = sorted(calls.glob("*/result.json")) if calls.is_dir() else []
        if {item.parent for item in results} - {item.parent for item in requests}:
            raise ProxyExperimentError("provider result exists without a request")
        ordinals: list[int] = []
        for path in requests:
            request = self._validate_call_request(base._read_json(path))
            if path.parent.name != request["call_id"]:
                raise ProxyExperimentError("provider request directory is noncanonical")
            ordinals.append(int(request["local_ordinal"]))
            ledger_path = (
                self.ledger_root / "entries" / f"global-{request['global_ordinal']:04d}.json"
            )
            self._validate_ledger_entry(base._read_json(ledger_path), request)
            result_path = path.parent / "result.json"
            if not result_path.is_file():
                raise AmbiguousProviderCall(
                    f"call {request['call_id']} has unknown provider disposition"
                )
            self._validate_call_result(base._read_json(result_path), request)
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            raise ProxyExperimentError("provider call ordinals are not contiguous")
        if len(ordinals) > NEW_CALL_CAP:
            raise CallCapExceeded("persisted calls exceed the 208-call cap")
        entry_paths = (
            sorted((self.ledger_root / "entries").glob("global-*.json"))
            if (self.ledger_root / "entries").is_dir()
            else []
        )
        expected_names = {
            f"global-{PRIOR_CHARGED_CALLS + ordinal:04d}.json" for ordinal in ordinals
        }
        if {path.name for path in entry_paths} != expected_names:
            raise ProxyExperimentError("shared ledger inventory differs from run reservations")

    async def call(
        self,
        *,
        purpose: Mapping[str, Any],
        prompt: str,
        system: str = "",
    ) -> tuple[str, str]:
        phase = str(purpose.get("phase"))
        model = self._model_for_phase(phase)
        call_id = base._call_id(purpose)
        directory = self.run_dir / "calls" / call_id
        request_path = directory / "request.json"
        result_path = directory / "result.json"
        expected = {
            "call_id": call_id,
            "purpose": dict(purpose),
            "prompt": prompt,
            "prompt_digest": _digest(prompt),
            "system": system,
            "system_digest": _digest(system),
            "model": model,
        }
        if request_path.is_file():
            request = self._validate_call_request(base._read_json(request_path), expected)
            if not result_path.is_file():
                raise AmbiguousProviderCall(f"call {call_id} is ambiguous and will not replay")
            result = self._validate_call_result(base._read_json(result_path), request)
            if result["status"] == "error":
                raise _RecordedCallFailure(call_id, result["failure_category"], result["error"])
            return str(result["response"]), call_id
        if self.call_count >= NEW_CALL_CAP:
            raise CallCapExceeded("sealed new-call cap 208 reached")
        local_ordinal = self.call_count + 1
        global_ordinal = PRIOR_CHARGED_CALLS + local_ordinal
        if global_ordinal > AUTHORIZED_GLOBAL_CALL_CAP:
            raise CallCapExceeded("authorized global cap 450 would be exceeded")
        _validate_runtime_identity(self.intent["runtime_identity"])
        self._verify_provider_executable()
        self._validate_tree()
        request_body = {
            "schema_version": CALL_REQUEST_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self._actor_plan_digest(required=phase == "actor"),
            "call_id": call_id,
            "local_ordinal": local_ordinal,
            "global_ordinal": global_ordinal,
            "purpose": dict(purpose),
            "provider": "agy",
            "model": model,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "runtime_identity_digest": _digest(self.intent["runtime_identity"]),
            "agy_executable_digest": self.intent["runtime_identity"]["agy_executable_digest"],
            "prompt": prompt,
            "prompt_digest": _digest(prompt),
            "system": system,
            "system_digest": _digest(system),
            "timeout_seconds": float(self.config["llm_timeout_seconds"]),
            "workdir_policy": "fresh-temporary-directory-per-call",
            "reservation_status": "reserved-before-spawn",
            "reserved_at_utc": base._utc_now(),
        }
        request = {**request_body, "request_digest": _digest(request_body)}
        self._validate_call_request(request, expected)
        base._write_json(request_path, request)
        ledger_path = self.ledger_root / "entries" / f"global-{global_ordinal:04d}.json"
        base._write_json(ledger_path, self._ledger_entry(request))
        started_at = base._utc_now()
        started_clock = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="spade-coverage-forced-agy-") as workdir:
                response = await self.dependencies.llm_call(
                    self.dependencies.client_or_bin,
                    model,
                    prompt,
                    system=system,
                    provider="agy",
                    workdir=Path(workdir),
                    timeout_seconds=float(self.config["llm_timeout_seconds"]),
                )
            if not isinstance(response, str) or not response.strip():
                raise live.LiveEvalError("agy returned an empty response", 4)
            response = response.strip()
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            match = re.fullmatch(r"agy failed with exit (-?[0-9]+):.*", error, re.DOTALL)
            result_body = {
                "schema_version": CALL_RESULT_SCHEMA,
                "intent_digest": self.intent["intent_digest"],
                "call_id": call_id,
                "local_ordinal": local_ordinal,
                "global_ordinal": global_ordinal,
                "request_digest": request["request_digest"],
                "status": "error",
                "failure_category": _failure_category(
                    exc, float(self.config["llm_timeout_seconds"])
                ),
                "exception_type": type(exc).__name__,
                "error": error,
                "exit_status": int(match.group(1)) if match else None,
                "response": None,
                "response_digest": None,
                "started_at_utc": started_at,
                "finished_at_utc": base._utc_now(),
                "duration_seconds": time.monotonic() - started_clock,
            }
            result = {**result_body, "result_digest": _digest(result_body)}
            self._validate_call_result(result, request)
            base._write_json(result_path, result)
            raise _RecordedCallFailure(call_id, result["failure_category"], error) from exc
        result_body = {
            "schema_version": CALL_RESULT_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "call_id": call_id,
            "local_ordinal": local_ordinal,
            "global_ordinal": global_ordinal,
            "request_digest": request["request_digest"],
            "status": "success",
            "failure_category": None,
            "exception_type": None,
            "error": None,
            "exit_status": 0,
            "response": response,
            "response_digest": _digest(response),
            "started_at_utc": started_at,
            "finished_at_utc": base._utc_now(),
            "duration_seconds": time.monotonic() - started_clock,
        }
        result = {**result_body, "result_digest": _digest(result_body)}
        self._validate_call_result(result, request)
        base._write_json(result_path, result)
        return response, call_id

    def _candidate_path(self, candidate_id: str) -> Path:
        return (
            self.run_dir / "candidate-evidence" / f"{_safe_id(candidate_id, 'candidate_id')}.json"
        )

    def _source_selection(self, label: str, stratum_id: str) -> dict[str, Any]:
        path = self.run_dir / "imports" / label / "selections" / f"{stratum_id}.json"
        value = base._read_json(path)
        if value.get("cluster_id") != stratum_id:
            raise ProxyExperimentError("imported selection stratum mismatch")
        return value

    def _qualification(self, code: str, candidate_id: str) -> dict[str, Any]:
        _validate_runtime_identity(self.intent["runtime_identity"])
        try:
            report = self._proofpack_call(
                self.dependencies.qualify,
                code,
                seeds=list(SEEDS),
                timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                max_turns=int(self.config["qualification_max_turns"]),
            )
            if isinstance(report, Mapping):
                receipt = _plain(report)
                passed = receipt.get("passed") is True
                environment_name = receipt.get("environment_name")
                environment_digest = receipt.get("environment_digest")
                reason = str(receipt.get("reason", "mapping qualification receipt"))
            else:
                receipt = base._decode_json(
                    (report.to_json() + "\n").encode(),
                    f"qualification receipt for {candidate_id}",
                )
                passed, reason = validate_positive_proofpack_receipt(
                    report,
                    game_code=code,
                    action_format="boxed",
                    seeds=SEEDS,
                    timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                    max_turns=int(self.config["qualification_max_turns"]),
                )
                environment_name = getattr(report, "environment_name", None)
                environment_digest = getattr(report, "environment_digest", None)
        except Exception as exc:
            raise ProxyExperimentIncomplete(
                f"qualification failed for {candidate_id}: {type(exc).__name__}: {exc}"
            ) from exc
        raw_digest = _bytes_digest(code.encode())
        if (
            not passed
            or not isinstance(environment_name, str)
            or not environment_name
            or environment_digest != raw_digest
        ):
            raise ProxyExperimentIncomplete(
                f"candidate {candidate_id} failed requalification: {reason}"
            )
        body = {
            "candidate_id": candidate_id,
            "seeds": list(SEEDS),
            "max_turns": int(self.config["qualification_max_turns"]),
            "timeout_seconds": float(self.config["qualification_timeout_seconds"]),
            "passed": True,
            "reason": reason,
            "environment_name": environment_name,
            "environment_digest": environment_digest,
            "receipt": receipt,
            "receipt_digest": _digest(receipt),
        }
        return {**body, "qualification_digest": _digest(body)}

    def _probe_candidate(self, code: str, candidate_id: str) -> dict[str, Any]:
        _validate_runtime_identity(self.intent["runtime_identity"])
        target = self._proofpack_call(
            self.dependencies.target_factory,
            code,
            action_format="boxed",
            max_turns=int(self.config["qualification_max_turns"]),
            operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
        )
        probes: dict[str, Any] = {}
        for seed in SEEDS:
            try:
                inspection = self._proofpack_call(
                    target.inspect,
                    seed=seed,
                    timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                )
                oracle = self._proofpack_call(
                    target.run_oracle,
                    seed=seed,
                    timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                )
                if getattr(oracle, "error", None) is not None:
                    raise ProxyExperimentError(str(oracle.error))
                session = self._proofpack_call(target.instantiate)
                observation, _ = self._proofpack_call(session.reset, seed=seed)
                solution = base._normalize_solution(self._proofpack_call(session.solution))
            except Exception as exc:
                raise ProxyExperimentIncomplete(
                    f"candidate probe failed for {candidate_id}/{seed}: {exc}"
                ) from exc
            observed = getattr(inspection, "observation", observation)
            if str(observed) != str(observation):
                raise ProxyExperimentIncomplete("candidate inspection/session reset diverged")
            probes[str(seed)] = {
                "seed": seed,
                "observation": str(observation),
                "observation_digest": _digest(str(observation)),
                "solution": solution,
                "solution_digest": _digest(solution),
                "qualification_oracle": _plain(oracle),
                "qualification_oracle_digest": _digest(_plain(oracle)),
            }
        return probes

    def _one_turn_viability(
        self, code: str, candidate_id: str, probes: Mapping[str, Any]
    ) -> dict[str, Any]:
        _validate_runtime_identity(self.intent["runtime_identity"])
        target = self._proofpack_call(
            self.dependencies.target_factory,
            code,
            action_format="boxed",
            max_turns=ACTOR_HORIZON,
            operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
        )
        seeds: dict[str, Any] = {}
        for seed in SEEDS:
            traces = [
                _plain(
                    self._proofpack_call(
                        target.run_oracle,
                        seed=seed,
                        timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                    )
                )
                for _ in range(2)
            ]
            if traces[0] != traces[1]:
                raise ProxyExperimentIncomplete(
                    f"one-turn oracle is nondeterministic for {candidate_id}/{seed}"
                )
            trace = traces[0]
            if not isinstance(trace, dict):
                raise ProxyExperimentIncomplete("one-turn oracle trace is not an object")
            viable = (
                trace.get("error") is None
                and float(trace.get("reward", 0.0)) > 0.0
                and trace.get("turn_count") == 1
            )
            if not viable:
                raise ProxyExperimentIncomplete(
                    f"candidate {candidate_id}/{seed} is not reward-positive at horizon 1"
                )
            seeds[str(seed)] = {
                "probe_digest": _digest(probes[str(seed)]),
                "repetitions": 2,
                "reward_positive_success": True,
                "terminated": trace.get("terminated"),
                "truncated": trace.get("truncated"),
                "proofpack_composite_success": trace.get("success"),
                "trace": trace,
                "trace_digest": _digest(trace),
            }
        body = {
            "candidate_id": candidate_id,
            "horizon": 1,
            "success_rule": "reward-positive-even-if-simultaneously-truncated",
            "termination_required": False,
            "simultaneous_truncation_is_recorded_not_rejected": True,
            "seeds": seeds,
        }
        return {**body, "viability_digest": _digest(body)}

    def _cwa_evidence(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        _validate_runtime_identity(self.intent["runtime_identity"])
        if self.dependencies.cwa_evaluator is not None:
            raw = _plain(
                self._proofpack_call(
                    self.dependencies.cwa_evaluator,
                    candidate,
                    self.dependencies.target_factory,
                )
            )
            if not isinstance(raw, dict):
                raise ProxyExperimentError("injected CWA evaluator did not return an object")
            body = {"schema_version": CWA_SCHEMA, **raw}
            if "evidence_digest" in body:
                supplied = body.pop("evidence_digest")
                if supplied != _digest(body):
                    raise ProxyExperimentError("injected CWA evidence digest mismatch")
            result = {**body, "evidence_digest": _digest(body)}
            self._validate_cwa_evidence(result)
            return result
        return self._evaluate_cwa_live(candidate)

    def _evaluate_cwa_live(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        code = str(candidate["code"])
        variants = tuple(generate_source_variants(code))
        base_oracles = {
            seed: replay._boxed_oracle_actions(candidate["probes"][str(seed)]["solution"])
            for seed in SEEDS
        }
        probes = tuple(
            build_candidate_probes(
                action_format="boxed",
                seeds=SEEDS,
                base_oracle_actions=base_oracles,
                max_turns=int(self.config["qualification_max_turns"]),
            )
        )
        sources: list[tuple[str, str]] = [("base", code)] + [
            (str(variant.variant_id), str(variant.source)) for variant in variants
        ]
        signatures: dict[str, dict[str, Any]] = {probe.probe_id: {} for probe in probes}
        trace_digests: dict[str, dict[str, str]] = {probe.probe_id: {} for probe in probes}
        for matrix_key, source in sources:
            target = self._proofpack_call(
                self.dependencies.target_factory,
                source,
                action_format="boxed",
                max_turns=int(self.config["qualification_max_turns"]),
                operation_timeout_seconds=float(self.config["cwa_timeout_seconds"]),
            )
            for probe in probes:
                results = [
                    self._proofpack_call(
                        cwa._execute_probe,
                        target,
                        probe,
                        timeout_seconds=float(self.config["cwa_timeout_seconds"]),
                    )
                    for _ in range(int(self.config["cwa_repetitions"]))
                ]
                for result in results:
                    cwa._validate_execution_outcome(
                        result,
                        variant_id=matrix_key,
                        probe_id=probe.probe_id,
                    )
                plain_results = [_plain(result) for result in results]
                if any(item != plain_results[0] for item in plain_results[1:]):
                    raise ProxyExperimentIncomplete(
                        f"CWA replay is nondeterministic for {candidate['candidate_id']}"
                    )
                signature = trace_signature(results[0])
                signatures[probe.probe_id][matrix_key] = signature
                trace_digests[probe.probe_id][matrix_key] = _digest(plain_results[0])
        matrix = WitnessMatrix(
            base_environment_digest=str(candidate["environment_digest"]),
            probes=probes,
            variants=variants,
            signatures=signatures,
        )
        matrix.validate()
        selection = select_witnesses(matrix, budget=int(self.config["witness_budget"]))
        train = score_probe_ids(matrix, selection.selected_probe_ids, partition="train")
        heldout = score_probe_ids(matrix, selection.selected_probe_ids, partition="heldout")
        safe = score_probe_ids(matrix, selection.safe_probe_ids, partition="heldout")
        applicability = cwa._detection_profile(
            matrix,
            selected_probe_ids=selection.selected_probe_ids,
            safe_probe_ids=selection.safe_probe_ids,
            partition="heldout",
        )
        base_signatures = {probe.probe_id: signatures[probe.probe_id]["base"] for probe in probes}
        descriptor = behavior_descriptor(
            action_format="boxed",
            probes=probes,
            signatures=base_signatures,
        )
        certificate = build_certificate(
            matrix,
            selection,
            metadata={
                "intent_digest": self.intent["intent_digest"],
                "candidate_id": candidate["candidate_id"],
                "qualification_digest": candidate["qualification"]["qualification_digest"],
            },
        )
        eligibility_gates = {
            "training_mutant_recall_at_least_0_90": float(train.mutant_recall)
            >= float(self.config["cwa_minimum_training_recall"]),
            "heldout_admitted_control_fpr_at_most_0_05": (
                float(heldout.equivalent_false_rejection_rate)
                <= float(self.config["cwa_maximum_heldout_control_fpr"])
            ),
        }
        if not all(eligibility_gates.values()):
            raise ProxyExperimentIncomplete(
                f"candidate {candidate['candidate_id']} failed the pre-portfolio CWA gates"
            )
        quality_score = float(applicability["applicable_recall"])
        body = {
            "schema_version": CWA_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "environment_digest": candidate["environment_digest"],
            "repetitions": int(self.config["cwa_repetitions"]),
            "variant_count": len(variants),
            "probe_count": len(probes),
            "variants": [_plain(item) for item in variants],
            "probes": [_plain(item) for item in probes],
            "trace_digests": trace_digests,
            "signature_matrix": {
                probe_id: {key: _plain(item) for key, item in row.items()}
                for probe_id, row in signatures.items()
            },
            "selection": _plain(selection),
            "scores": {
                "training": _plain(train),
                "heldout": _plain(heldout),
                "safe_bank_heldout": _plain(safe),
            },
            "heldout_applicability": applicability,
            "eligibility_gates": eligibility_gates,
            "eligible": True,
            "descriptor": descriptor.to_dict(),
            "quality_score": quality_score,
            "quality_formula": "heldout_applicable_recall_selected_over_safe-bank-observable",
            "certificate": _plain(certificate),
        }
        result = {**body, "evidence_digest": _digest(body)}
        self._validate_cwa_evidence(result)
        return result

    @staticmethod
    def _validate_cwa_evidence(value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != CWA_SCHEMA:
            raise ProxyExperimentError("CWA evidence schema differs")
        body = {key: item for key, item in value.items() if key != "evidence_digest"}
        if value.get("evidence_digest") != _digest(body):
            raise ProxyExperimentError("CWA evidence digest mismatch")
        descriptor = value.get("descriptor")
        if not isinstance(descriptor, dict) or not descriptor:
            raise ProxyExperimentError("CWA evidence lacks a descriptor")
        quality = value.get("quality_score")
        if (
            isinstance(quality, bool)
            or not isinstance(quality, (int, float))
            or not math.isfinite(quality)
            or not 0 <= float(quality) <= 1
        ):
            raise ProxyExperimentError("CWA quality score is invalid")
        applicability = value.get("heldout_applicability")
        if not isinstance(applicability, dict) or applicability.get("applicable_recall") != quality:
            raise ProxyExperimentError("CWA quality is not heldout applicable recall")
        if (
            not isinstance(value.get("candidate_id"), str)
            or not isinstance(value.get("environment_digest"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["environment_digest"]) is None
        ):
            raise ProxyExperimentError("CWA candidate/environment identity is invalid")
        if (
            value.get("quality_formula")
            != "heldout_applicable_recall_selected_over_safe-bank-observable"
            or value.get("eligible") is not True
            or not isinstance(value.get("eligibility_gates"), dict)
            or set(value["eligibility_gates"])
            != {
                "training_mutant_recall_at_least_0_90",
                "heldout_admitted_control_fpr_at_most_0_05",
            }
            or not all(item is True for item in value["eligibility_gates"].values())
        ):
            raise ProxyExperimentError("CWA eligibility/quality policy differs")

    @staticmethod
    def _design_prompt(stratum: Mapping[str, Any], feedback: str) -> str:
        difficulty = str(stratum["difficulty"])
        clause = (
            "\nMedium difficulty: require 2-3 dependent reasoning steps and one unique boxed answer."
            if difficulty == "medium"
            else "\nHard difficulty: require 4-6 dependent reasoning steps, plausible distractors, "
            "and one unique boxed answer."
            if difficulty == "hard"
            else f"\nRequested difficulty: {difficulty}."
        )
        return (
            live.DESIGNER_PROMPT.format(skill=stratum["skill"])
            + f"\nSealed coverage-forced challenger for {stratum['stratum_id']}."
            + clause
            + " It must have a reward-positive correct oracle within exactly one actor turn; "
            "simultaneous termination and truncation are permitted but will be recorded. "
            "Seeds 0 and 42 must produce genuinely distinct reset observations and correct "
            "solutions. A well-formed wrong boxed action must return reward 0, remain both "
            "nonterminal and nontruncated, and irreversibly move the episode into a decoy or "
            "corrupted state where the original fresh-reset oracle answer no longer succeeds. "
            "The fresh-reset oracle must still solve in one turn." + feedback
        )

    @staticmethod
    def _hint_prompt(observation: str, feedback: str) -> str:
        return live.HINT_PROMPT.format(observation=observation) + feedback

    def _historical_proxy_candidates(self, stratum_id: str) -> tuple[ProxyCandidate, ...]:
        result: list[ProxyCandidate] = []
        for source_arm in ("v3", "v4"):
            path = self._candidate_path(_candidate_id(stratum_id, source_arm))
            if not path.is_file():
                raise ProxyExperimentError(
                    "historical scientific evidence must precede challenger assessment"
                )
            evidence = self._validate_candidate_evidence(base._read_json(path))
            result.append(self._proxy_candidate(evidence))
        return tuple(result)

    def _assess_design_response(
        self, raw: str, candidate_id: str, stratum_id: str
    ) -> dict[str, Any]:
        code: str | None = None
        character_count: int | None = None
        nonblank_line_count: int | None = None
        preview: Mapping[str, Any] | None = None
        scientific_preview: Mapping[str, Any] | None = None
        status = "rejected"
        reason = "candidate could not be parsed"
        try:
            code = live.extract_python_code(raw)
            character_count = len(code)
            nonblank_line_count = sum(1 for line in code.splitlines() if line.strip())
            if (
                character_count > MAX_CHALLENGER_CHARACTERS
                or nonblank_line_count > MAX_CHALLENGER_NONBLANK_LINES
            ):
                raise ProxyExperimentError(
                    "challenger source exceeds 8,000 characters or 120 nonblank lines"
                )
            preview = self._qualification(code, candidate_id)
            probes = self._probe_candidate(code, candidate_id)
            viability = self._one_turn_viability(code, candidate_id, probes)
            candidate_for_cwa = {
                "candidate_id": candidate_id,
                "stratum_id": stratum_id,
                "source_arm": "challenger",
                "code": code,
                "environment_digest": preview["environment_digest"],
                "qualification": preview,
                "probes": probes,
            }
            cwa_evidence = self._cwa_evidence(candidate_for_cwa)
            challenger = ProxyCandidate.create(
                candidate_id=candidate_id,
                stratum_id=stratum_id,
                source_arm="challenger",
                quality_score=float(cwa_evidence["quality_score"]),
                descriptor=cwa_evidence["descriptor"],
                environment_digest=str(preview["environment_digest"]),
                evidence_digest=str(cwa_evidence["evidence_digest"]),
            )
            lock = lock_portfolios((*self._historical_proxy_candidates(stratum_id), challenger))
            maximum_gap = float(self.config["maximum_stratum_absolute_quality_gap"])
            if lock.absolute_quality_gap > maximum_gap:
                raise ProxyExperimentIncomplete(
                    "coverage portfolio absolute CWA quality gap "
                    f"{lock.absolute_quality_gap:.6f} exceeds {maximum_gap:.6f}"
                )
            scientific_preview = {
                "environment_digest": preview["environment_digest"],
                "qualification_digest": preview["qualification_digest"],
                "probes_digest": _digest(probes),
                "viability_digest": viability["viability_digest"],
                "cwa_evidence_digest": cwa_evidence["evidence_digest"],
                "descriptor": cwa_evidence["descriptor"],
                "quality_score": cwa_evidence["quality_score"],
                "coverage_forced": list(lock.coverage_forced),
                "redundant_historical": list(lock.redundant_historical),
                "signed_quality_gap": lock.signed_quality_gap,
                "absolute_quality_gap": lock.absolute_quality_gap,
            }
            status = "coverage_eligible"
            reason = (
                "qualification, one-turn viability, exact recovery-false target cell, CWA "
                "eligibility, and per-stratum quality match passed"
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        feedback = (
            ""
            if status == "coverage_eligible"
            else "\nThe prior candidate failed the sealed qualification, one-turn, CWA, exact "
            "recovery-false descriptor, or quality-match screen: "
            f"{reason}\nReturn complete corrected source, not a patch. Preserve genuinely "
            "distinct seeds and irreversible wrong-action corruption."
        )
        return {
            "status": status,
            "reason": reason,
            "code": code,
            "code_digest": _digest(code) if code else None,
            "source_character_count": character_count,
            "source_nonblank_line_count": nonblank_line_count,
            "qualification_preview": preview,
            "scientific_preview": scientific_preview,
            "feedback_for_next_attempt": feedback,
        }

    def _validate_design_attempt(
        self,
        value: Mapping[str, Any],
        *,
        stratum: Mapping[str, Any],
        attempt: int,
        feedback: str,
    ) -> Mapping[str, Any]:
        stratum_id = str(stratum["stratum_id"])
        candidate_id = str(stratum["challenger_candidate_id"])
        keys = {
            "schema_version",
            "intent_digest",
            "stratum_id",
            "candidate_id",
            "attempt",
            "call_id",
            "status",
            "reason",
            "code",
            "code_digest",
            "source_character_count",
            "source_nonblank_line_count",
            "qualification_preview",
            "scientific_preview",
            "feedback_for_next_attempt",
            "attempt_digest",
        }
        if (
            set(value) != keys
            or value.get("schema_version") != "spade-coverage-forced-design-attempt/v1"
        ):
            raise ProxyExperimentError("challenger design attempt fields differ")
        _sealed_artifact(value, "attempt_digest")
        purpose = {
            "phase": "challenger-design",
            "stratum_id": stratum_id,
            "candidate_id": candidate_id,
            "attempt": attempt,
        }
        if (
            value["intent_digest"] != self.intent["intent_digest"]
            or value["stratum_id"] != stratum_id
            or value["candidate_id"] != candidate_id
            or value["attempt"] != attempt
            or value["call_id"] != base._call_id(purpose)
        ):
            raise ProxyExperimentError("challenger design attempt identity differs")
        prompt = self._design_prompt(stratum, feedback)
        request_path = self.run_dir / "calls" / str(value["call_id"]) / "request.json"
        request = self._validate_call_request(
            base._read_json(request_path),
            {
                "purpose": purpose,
                "prompt": prompt,
                "prompt_digest": _digest(prompt),
                "system": "Return secure, deterministic environment source code only.",
                "system_digest": _digest(
                    "Return secure, deterministic environment source code only."
                ),
            },
        )
        result = self._validate_call_result(
            base._read_json(request_path.parent / "result.json"), request
        )
        if result["status"] == "error":
            if result["failure_category"] not in {"empty_response", "provider_timeout"}:
                raise ProxyExperimentError("fatal design transport cannot open an attempt leaf")
            expected = {
                "status": "retryable_transport",
                "reason": str(
                    _RecordedCallFailure(
                        str(value["call_id"]), str(result["failure_category"]), str(result["error"])
                    )
                ),
                "code": None,
                "code_digest": None,
                "source_character_count": None,
                "source_nonblank_line_count": None,
                "qualification_preview": None,
                "scientific_preview": None,
                "feedback_for_next_attempt": "\nThe provider attempt failed; return complete source.",
            }
        else:
            expected = self._assess_design_response(
                str(result["response"]), candidate_id, stratum_id
            )
        if any(value.get(key) != item for key, item in expected.items()):
            raise ProxyExperimentError("challenger design attempt differs from its request/result")
        return value

    def _validate_hint_attempt(
        self,
        value: Mapping[str, Any],
        *,
        candidate_id: str,
        seed: int,
        observation: str,
        solution: Any,
        attempt: int,
        feedback: str,
    ) -> Mapping[str, Any]:
        keys = {
            "schema_version",
            "intent_digest",
            "candidate_id",
            "seed",
            "attempt",
            "call_id",
            "status",
            "reason",
            "hint",
            "hint_digest",
            "observation_digest",
            "solution_digest",
            "feedback_for_next_attempt",
            "attempt_digest",
        }
        if (
            set(value) != keys
            or value.get("schema_version") != "spade-coverage-forced-hint-attempt/v1"
        ):
            raise ProxyExperimentError("challenger hint attempt fields differ")
        _sealed_artifact(value, "attempt_digest")
        purpose = {
            "phase": "challenger-hint",
            "candidate_id": candidate_id,
            "seed": seed,
            "attempt": attempt,
        }
        if (
            value["intent_digest"] != self.intent["intent_digest"]
            or value["candidate_id"] != candidate_id
            or value["seed"] != seed
            or value["attempt"] != attempt
            or value["call_id"] != base._call_id(purpose)
            or value["observation_digest"] != _digest(observation)
            or value["solution_digest"] != _digest(solution)
        ):
            raise ProxyExperimentError("challenger hint attempt identity/probe binding differs")
        prompt = self._hint_prompt(observation, feedback)
        system = "Provide strategy only; never solve the puzzle for the player."
        request_path = self.run_dir / "calls" / str(value["call_id"]) / "request.json"
        request = self._validate_call_request(
            base._read_json(request_path),
            {
                "purpose": purpose,
                "prompt": prompt,
                "prompt_digest": _digest(prompt),
                "system": system,
                "system_digest": _digest(system),
            },
        )
        result = self._validate_call_result(
            base._read_json(request_path.parent / "result.json"), request
        )
        retry_feedback = "\nThe prior hint was unusable. Return only general strategy and no exact answer values."
        if result["status"] == "error":
            if result["failure_category"] not in {"empty_response", "provider_timeout"}:
                raise ProxyExperimentError("fatal hint transport cannot open an attempt leaf")
            expected = {
                "status": "retryable_transport",
                "reason": str(
                    _RecordedCallFailure(
                        str(value["call_id"]), str(result["failure_category"]), str(result["error"])
                    )
                ),
                "hint": None,
                "hint_digest": None,
                "feedback_for_next_attempt": retry_feedback,
            }
        else:
            hint = str(result["response"])
            leaked = live.hint_reveals_solution(hint, solution, observation)
            expected = {
                "status": "leaked" if leaked else "accepted",
                "reason": "exact solution leakage" if leaked else "nonleaking",
                "hint": hint,
                "hint_digest": _digest(hint),
                "feedback_for_next_attempt": retry_feedback if leaked else "",
            }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ProxyExperimentError("challenger hint attempt differs from its request/result")
        return value

    async def _generate_challenger(
        self, stratum: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any]]:
        stratum_id = str(stratum["stratum_id"])
        candidate_id = str(stratum["challenger_candidate_id"])
        directory = self.run_dir / "challenger-generation" / candidate_id
        feedback = ""
        for attempt in range(1, DESIGN_ATTEMPTS + 1):
            path = directory / f"attempt-{attempt:02d}.json"
            if path.is_file():
                leaf = self._validate_design_attempt(
                    base._read_json(path), stratum=stratum, attempt=attempt, feedback=feedback
                )
                if leaf["status"] == "coverage_eligible":
                    if any(
                        (directory / f"attempt-{later:02d}.json").exists()
                        for later in range(attempt + 1, DESIGN_ATTEMPTS + 1)
                    ):
                        raise ProxyExperimentError(
                            "design attempt exists after terminal eligibility"
                        )
                    return str(leaf["code"]), leaf
                feedback = str(leaf["feedback_for_next_attempt"])
                continue
            prompt = self._design_prompt(stratum, feedback)
            purpose = {
                "phase": "challenger-design",
                "stratum_id": stratum_id,
                "candidate_id": candidate_id,
                "attempt": attempt,
            }
            try:
                raw, call_id = await self.call(
                    purpose=purpose,
                    prompt=prompt,
                    system="Return secure, deterministic environment source code only.",
                )
                assessment = self._assess_design_response(raw, candidate_id, stratum_id)
            except _RecordedCallFailure as exc:
                if not exc.retryable:
                    raise ProxyExperimentIncomplete(f"fatal challenger transport: {exc}") from exc
                call_id = exc.call_id
                assessment = {
                    "status": "retryable_transport",
                    "reason": str(exc),
                    "code": None,
                    "code_digest": None,
                    "source_character_count": None,
                    "source_nonblank_line_count": None,
                    "qualification_preview": None,
                    "scientific_preview": None,
                    "feedback_for_next_attempt": "\nThe provider attempt failed; return complete source.",
                }
            body = {
                "schema_version": "spade-coverage-forced-design-attempt/v1",
                "intent_digest": self.intent["intent_digest"],
                "stratum_id": stratum_id,
                "candidate_id": candidate_id,
                "attempt": attempt,
                "call_id": call_id,
                **assessment,
            }
            leaf = {**body, "attempt_digest": _digest(body)}
            base._write_json(path, leaf)
            if leaf["status"] == "coverage_eligible":
                return str(leaf["code"]), leaf
            feedback = str(leaf["feedback_for_next_attempt"])
        raise ProxyExperimentIncomplete(f"challenger design exhausted for {stratum_id}")

    async def _lock_hint(
        self,
        *,
        candidate_id: str,
        seed: int,
        observation: str,
        solution: Any,
    ) -> Mapping[str, Any]:
        directory = self.run_dir / "challenger-hints" / candidate_id / str(seed)
        feedback = ""
        for attempt in range(1, HINT_ATTEMPTS + 1):
            path = directory / f"attempt-{attempt:02d}.json"
            if path.is_file():
                leaf = self._validate_hint_attempt(
                    base._read_json(path),
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=observation,
                    solution=solution,
                    attempt=attempt,
                    feedback=feedback,
                )
                if leaf["status"] == "accepted":
                    if any(
                        (directory / f"attempt-{later:02d}.json").exists()
                        for later in range(attempt + 1, HINT_ATTEMPTS + 1)
                    ):
                        raise ProxyExperimentError("hint attempt exists after terminal acceptance")
                    return leaf
                feedback = str(leaf["feedback_for_next_attempt"])
                continue
            prompt = self._hint_prompt(observation, feedback)
            purpose = {
                "phase": "challenger-hint",
                "candidate_id": candidate_id,
                "seed": seed,
                "attempt": attempt,
            }
            try:
                hint, call_id = await self.call(
                    purpose=purpose,
                    prompt=prompt,
                    system="Provide strategy only; never solve the puzzle for the player.",
                )
                leaked = live.hint_reveals_solution(hint, solution, observation)
                status = "leaked" if leaked else "accepted"
                reason = "exact solution leakage" if leaked else "nonleaking"
            except _RecordedCallFailure as exc:
                if not exc.retryable:
                    raise ProxyExperimentIncomplete(f"fatal hint transport: {exc}") from exc
                hint, call_id, status, reason = None, exc.call_id, "retryable_transport", str(exc)
            retry_feedback = "\nThe prior hint was unusable. Return only general strategy and no exact answer values."
            body = {
                "schema_version": "spade-coverage-forced-hint-attempt/v1",
                "intent_digest": self.intent["intent_digest"],
                "candidate_id": candidate_id,
                "seed": seed,
                "attempt": attempt,
                "call_id": call_id,
                "status": status,
                "reason": reason,
                "hint": hint,
                "hint_digest": _digest(hint) if hint is not None else None,
                "observation_digest": _digest(observation),
                "solution_digest": _digest(solution),
                "feedback_for_next_attempt": "" if status == "accepted" else retry_feedback,
            }
            leaf = {**body, "attempt_digest": _digest(body)}
            base._write_json(path, leaf)
            if status == "accepted":
                return leaf
            feedback = str(leaf["feedback_for_next_attempt"])
        raise ProxyExperimentIncomplete(f"hint attempts exhausted for {candidate_id}/{seed}")

    @staticmethod
    def _validate_candidate_hint(
        hint: object,
        *,
        probe: Mapping[str, Any],
        seed_text: str,
        source_arm: str,
    ) -> None:
        if not isinstance(hint, dict) or hint.get("status") != "accepted":
            raise ProxyExperimentError("candidate hint is not accepted")
        hint_text = hint.get("hint")
        invalid = (
            not isinstance(hint_text, str)
            or not hint_text
            or hint.get("hint_digest") != _digest(hint_text)
            or hint.get("attempt_digest")
            != _digest({key: item for key, item in hint.items() if key != "attempt_digest"})
            or live.hint_reveals_solution(
                hint_text, probe.get("solution"), str(probe.get("observation"))
            )
        )
        if source_arm in {"v3", "v4"}:
            invalid = invalid or (
                hint.get("schema_version") != "spade-agy-hint-attempt/v1"
                or hint.get("seed") != int(seed_text)
                or "observation_digest" in hint
                or "solution_digest" in hint
            )
        else:
            invalid = invalid or (
                hint.get("schema_version") != "spade-coverage-forced-hint-attempt/v1"
                or hint.get("seed") != int(seed_text)
                or hint.get("observation_digest") != probe.get("observation_digest")
                or hint.get("solution_digest") != probe.get("solution_digest")
            )
        if invalid:
            raise ProxyExperimentError("candidate hint receipt leaks or has invalid digests")

    @staticmethod
    def _validate_candidate_evidence(value: Mapping[str, Any]) -> Mapping[str, Any]:
        required = {
            "schema_version",
            "intent_digest",
            "stratum_id",
            "candidate_id",
            "source_arm",
            "source_provenance",
            "code",
            "code_digest",
            "environment_name",
            "environment_digest",
            "qualification",
            "probes",
            "hints",
            "one_turn_viability",
            "cwa",
            "evidence_digest",
        }
        if set(value) != required or value.get("schema_version") != CANDIDATE_SCHEMA:
            raise ProxyExperimentError("candidate evidence fields differ from schema")
        body = {key: item for key, item in value.items() if key != "evidence_digest"}
        if value["evidence_digest"] != _digest(body):
            raise ProxyExperimentError("candidate evidence digest mismatch")
        candidate_id = _safe_id(value["candidate_id"], "candidate_id")
        stratum_id = _safe_id(value["stratum_id"], "stratum_id")
        if (
            not isinstance(value["intent_digest"], str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["intent_digest"]) is None
        ):
            raise ProxyExperimentError("candidate intent binding is invalid")
        source_arm = value["source_arm"]
        if candidate_id != _candidate_id(stratum_id, str(source_arm)) or source_arm not in {
            "v3",
            "v4",
            "challenger",
        }:
            raise ProxyExperimentError("candidate evidence identity differs")
        code = value["code"]
        if not isinstance(code, str) or not code:
            raise ProxyExperimentError("candidate source is empty")
        if value["code_digest"] != _digest(code) or value["environment_digest"] != _bytes_digest(
            code.encode()
        ):
            raise ProxyExperimentError("candidate source digest mismatch")
        qualification = value["qualification"]
        if not isinstance(qualification, dict) or qualification.get("passed") is not True:
            raise ProxyExperimentError("candidate lacks positive requalification")
        if qualification.get("environment_digest") != value["environment_digest"]:
            raise ProxyExperimentError("candidate qualification source digest differs")
        qualification_body = {
            key: item for key, item in qualification.items() if key != "qualification_digest"
        }
        if qualification.get("qualification_digest") != _digest(qualification_body):
            raise ProxyExperimentError("candidate qualification digest mismatch")
        if qualification.get("candidate_id") != candidate_id or qualification.get("seeds") != [
            0,
            42,
        ]:
            raise ProxyExperimentError("candidate qualification identity differs")
        if qualification.get("receipt_digest") != _digest(qualification.get("receipt")):
            raise ProxyExperimentError("candidate qualification receipt digest mismatch")
        probes = value["probes"]
        hints = value["hints"]
        if not isinstance(probes, dict) or set(probes) != {"0", "42"}:
            raise ProxyExperimentError("candidate probes differ from seeds [0,42]")
        if not isinstance(hints, dict) or set(hints) != {"0", "42"}:
            raise ProxyExperimentError("candidate hints differ from seeds [0,42]")
        for seed_text in ("0", "42"):
            probe = probes[seed_text]
            if (
                not isinstance(probe, dict)
                or probe.get("seed") != int(seed_text)
                or probe.get("observation_digest") != _digest(probe.get("observation"))
                or probe.get("solution_digest") != _digest(probe.get("solution"))
                or probe.get("qualification_oracle_digest")
                != _digest(probe.get("qualification_oracle"))
            ):
                raise ProxyExperimentError("candidate probe receipt differs")
            _Engine._validate_candidate_hint(
                hints[seed_text],
                probe=probe,
                seed_text=seed_text,
                source_arm=str(source_arm),
            )
        viability = value["one_turn_viability"]
        if (
            not isinstance(viability, dict)
            or viability.get("horizon") != 1
            or viability.get("success_rule") != "reward-positive-even-if-simultaneously-truncated"
            or any(
                viability.get("seeds", {}).get(seed, {}).get("reward_positive_success") is not True
                for seed in ("0", "42")
            )
        ):
            raise ProxyExperimentError("candidate one-turn viability differs")
        viability_body = {key: item for key, item in viability.items() if key != "viability_digest"}
        if viability.get("viability_digest") != _digest(viability_body):
            raise ProxyExperimentError("candidate viability digest mismatch")
        if (
            viability.get("termination_required") is not False
            or viability.get("simultaneous_truncation_is_recorded_not_rejected") is not True
        ):
            raise ProxyExperimentError("candidate viability policy differs")
        for seed_text in ("0", "42"):
            record = viability["seeds"][seed_text]
            trace = record.get("trace")
            if (
                not isinstance(trace, dict)
                or record.get("probe_digest") != _digest(probes[seed_text])
                or record.get("repetitions") != 2
                or record.get("trace_digest") != _digest(trace)
                or record.get("terminated") is not trace.get("terminated")
                or record.get("truncated") is not trace.get("truncated")
                or record.get("proofpack_composite_success") is not trace.get("success")
                or trace.get("error") is not None
                or trace.get("turn_count") != 1
                or isinstance(trace.get("reward"), bool)
                or not isinstance(trace.get("reward"), (int, float))
                or float(trace["reward"]) <= 0
            ):
                raise ProxyExperimentError("candidate viability trace differs from reward rule")
        cwa_value = value["cwa"]
        if not isinstance(cwa_value, dict):
            raise ProxyExperimentError("candidate CWA evidence is invalid")
        _Engine._validate_cwa_evidence(cwa_value)
        if (
            cwa_value.get("candidate_id") != candidate_id
            or cwa_value.get("environment_digest") != value["environment_digest"]
        ):
            raise ProxyExperimentError("candidate CWA identity differs")
        expected_descriptor = (
            dict(COVERAGE_CHALLENGER_DESCRIPTOR)
            if source_arm == "challenger"
            else dict(STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR)
        )
        if cwa_value.get("descriptor") != expected_descriptor:
            raise ProxyExperimentError(
                "candidate does not occupy its exact sealed coverage/control descriptor cell"
            )
        return value

    async def _prepare_candidate(
        self,
        stratum: Mapping[str, Any],
        source_arm: str,
        *,
        defer_challenger_hints: bool = False,
    ) -> Mapping[str, Any]:
        stratum_id = str(stratum["stratum_id"])
        candidate_id = _candidate_id(stratum_id, source_arm)
        path = self._candidate_path(candidate_id)
        if path.is_file():
            value = self._validate_candidate_evidence(base._read_json(path))
            if value["intent_digest"] != self.intent["intent_digest"]:
                raise ProxyExperimentError("candidate evidence belongs to a different intent")
            self._validate_candidate_lineage(value)
            self._reaudit_candidate(value)
            return value
        if source_arm in {"v3", "v4"}:
            selection = self._source_selection(source_arm, stratum_id)
            code = str(selection["code"])
            source_provenance: Mapping[str, Any] = {
                "kind": "historical-selection",
                "source_arm": source_arm,
                "source_plan_digest": V3_PLAN_DIGEST if source_arm == "v3" else V4_PLAN_DIGEST,
                "source_selection_digest": selection["selection_digest"],
            }
            imported_probes = {seed: selection["probes"][seed] for seed in ("0", "42")}
            imported_hints = {seed: selection["hints"][seed] for seed in ("0", "42")}
        elif source_arm == "challenger":
            code, generation = await self._generate_challenger(stratum)
            source_provenance = {
                "kind": "new-challenger",
                "generation_attempt_digest": generation["attempt_digest"],
                "generation_call_id": generation["call_id"],
            }
            imported_probes = None
            imported_hints = None
        else:
            raise ProxyExperimentError(f"unknown candidate source arm: {source_arm}")
        qualification = self._qualification(code, candidate_id)
        probes = self._probe_candidate(code, candidate_id)
        if imported_probes is not None:
            for seed in ("0", "42"):
                if (
                    probes[seed]["observation"] != imported_probes[seed]["observation"]
                    or probes[seed]["solution"] != imported_probes[seed]["solution"]
                ):
                    raise ProxyExperimentIncomplete(
                        f"historical probe drifted during requalification: {candidate_id}/{seed}"
                    )
        viability = self._one_turn_viability(code, candidate_id, probes)
        candidate_for_cwa = {
            "candidate_id": candidate_id,
            "stratum_id": stratum_id,
            "source_arm": source_arm,
            "code": code,
            "environment_digest": qualification["environment_digest"],
            "qualification": qualification,
            "probes": probes,
        }
        cwa_evidence = self._cwa_evidence(candidate_for_cwa)
        if source_arm == "challenger":
            scientific_preview = generation.get("scientific_preview")
            expected_preview = {
                "environment_digest": qualification["environment_digest"],
                "qualification_digest": qualification["qualification_digest"],
                "probes_digest": _digest(probes),
                "viability_digest": viability["viability_digest"],
                "cwa_evidence_digest": cwa_evidence["evidence_digest"],
                "descriptor": cwa_evidence["descriptor"],
                "quality_score": cwa_evidence["quality_score"],
            }
            if not isinstance(scientific_preview, Mapping) or any(
                scientific_preview.get(key) != item for key, item in expected_preview.items()
            ):
                raise ProxyExperimentError(
                    "challenger scientific evidence differs from its terminal design attempt"
                )
        if imported_hints is not None:
            hints = imported_hints
        elif defer_challenger_hints:
            hints = {}
        else:
            # Hints are not an input to CWA eligibility.  Spend their provider
            # calls only after qualification, h1 viability, and CWA pass.
            hints = {
                str(seed): await self._lock_hint(
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=str(probes[str(seed)]["observation"]),
                    solution=probes[str(seed)]["solution"],
                )
                for seed in SEEDS
            }
        body = {
            "schema_version": CANDIDATE_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "stratum_id": stratum_id,
            "candidate_id": candidate_id,
            "source_arm": source_arm,
            "source_provenance": source_provenance,
            "code": code,
            "code_digest": _digest(code),
            "environment_name": qualification["environment_name"],
            "environment_digest": qualification["environment_digest"],
            "qualification": qualification,
            "probes": probes,
            "hints": hints,
            "one_turn_viability": viability,
            "cwa": cwa_evidence,
        }
        value = {**body, "evidence_digest": _digest(body)}
        if defer_challenger_hints:
            if source_arm != "challenger" or hints:
                raise ProxyExperimentError("only a new challenger can defer hint locking")
            return value
        self._validate_candidate_evidence(value)
        base._write_json(path, value)
        self._validate_candidate_lineage(value)
        return value

    async def _finalize_deferred_challenger(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        if evidence.get("source_arm") != "challenger" or evidence.get("hints") != {}:
            raise ProxyExperimentError("deferred challenger state is invalid")
        candidate_id = str(evidence["candidate_id"])
        probes = evidence["probes"]
        hints = {
            str(seed): await self._lock_hint(
                candidate_id=candidate_id,
                seed=seed,
                observation=str(probes[str(seed)]["observation"]),
                solution=probes[str(seed)]["solution"],
            )
            for seed in SEEDS
        }
        body = {
            key: item for key, item in evidence.items() if key not in {"hints", "evidence_digest"}
        }
        body["hints"] = hints
        value = {**body, "evidence_digest": _digest(body)}
        self._validate_candidate_evidence(value)
        base._write_json(self._candidate_path(candidate_id), value)
        self._validate_candidate_lineage(value)
        return value

    def _stratum(self, stratum_id: str) -> Mapping[str, Any]:
        matches = [item for item in self.intent["strata"] if item["stratum_id"] == stratum_id]
        if len(matches) != 1:
            raise ProxyExperimentError(f"unknown candidate stratum: {stratum_id}")
        return matches[0]

    def _validate_candidate_lineage(self, evidence: Mapping[str, Any]) -> None:
        candidate_id = str(evidence["candidate_id"])
        stratum_id = str(evidence["stratum_id"])
        source_arm = str(evidence["source_arm"])
        provenance = evidence["source_provenance"]
        if source_arm in {"v3", "v4"}:
            selection = self._source_selection(source_arm, stratum_id)
            expected = {
                "kind": "historical-selection",
                "source_arm": source_arm,
                "source_plan_digest": V3_PLAN_DIGEST if source_arm == "v3" else V4_PLAN_DIGEST,
                "source_selection_digest": selection["selection_digest"],
            }
            if provenance != expected or evidence["code"] != selection["code"]:
                raise ProxyExperimentError("historical candidate lineage differs from import")
            for seed in ("0", "42"):
                if (
                    evidence["probes"][seed]["observation"]
                    != selection["probes"][seed]["observation"]
                    or evidence["probes"][seed]["solution"] != selection["probes"][seed]["solution"]
                    or evidence["hints"][seed] != selection["hints"][seed]
                ):
                    raise ProxyExperimentError("historical probe/hint lineage differs")
            return
        if (
            source_arm != "challenger"
            or not isinstance(provenance, dict)
            or set(provenance)
            != {
                "kind",
                "generation_attempt_digest",
                "generation_call_id",
            }
            or provenance["kind"] != "new-challenger"
        ):
            raise ProxyExperimentError("challenger provenance fields differ")
        stratum = self._stratum(stratum_id)
        generation_root = self.run_dir / "challenger-generation" / candidate_id
        generation: Mapping[str, Any] | None = None
        feedback = ""
        for attempt in range(1, DESIGN_ATTEMPTS + 1):
            attempt_path = generation_root / f"attempt-{attempt:02d}.json"
            if not attempt_path.is_file():
                if any(
                    (generation_root / f"attempt-{later:02d}.json").exists()
                    for later in range(attempt + 1, DESIGN_ATTEMPTS + 1)
                ):
                    raise ProxyExperimentError("challenger generation attempts are not contiguous")
                break
            leaf = self._validate_design_attempt(
                base._read_json(attempt_path),
                stratum=stratum,
                attempt=attempt,
                feedback=feedback,
            )
            if leaf["status"] == "coverage_eligible":
                if any(
                    (generation_root / f"attempt-{later:02d}.json").exists()
                    for later in range(attempt + 1, DESIGN_ATTEMPTS + 1)
                ):
                    raise ProxyExperimentError("design attempt exists after terminal eligibility")
                generation = leaf
                break
            feedback = str(leaf["feedback_for_next_attempt"])
        if generation is None:
            raise ProxyExperimentError("challenger lineage requires one coverage-eligible attempt")
        if (
            generation["attempt_digest"] != provenance["generation_attempt_digest"]
            or generation["call_id"] != provenance["generation_call_id"]
            or generation["code"] != evidence["code"]
        ):
            raise ProxyExperimentError("challenger generation reference differs")
        preview = generation["scientific_preview"]
        if (
            not isinstance(preview, Mapping)
            or preview.get("environment_digest") != evidence["environment_digest"]
            or preview.get("qualification_digest")
            != evidence["qualification"]["qualification_digest"]
            or preview.get("probes_digest") != _digest(evidence["probes"])
            or preview.get("viability_digest") != evidence["one_turn_viability"]["viability_digest"]
            or preview.get("cwa_evidence_digest") != evidence["cwa"]["evidence_digest"]
            or preview.get("descriptor") != evidence["cwa"]["descriptor"]
            or preview.get("quality_score") != evidence["cwa"]["quality_score"]
        ):
            raise ProxyExperimentError("challenger scientific preview differs from evidence")
        for seed in SEEDS:
            hint = evidence["hints"][str(seed)]
            hint_root = self.run_dir / "challenger-hints" / candidate_id / str(seed)
            locked: Mapping[str, Any] | None = None
            hint_feedback = ""
            for attempt in range(1, HINT_ATTEMPTS + 1):
                hint_path = hint_root / f"attempt-{attempt:02d}.json"
                if not hint_path.is_file():
                    if any(
                        (hint_root / f"attempt-{later:02d}.json").exists()
                        for later in range(attempt + 1, HINT_ATTEMPTS + 1)
                    ):
                        raise ProxyExperimentError("challenger hint attempts are not contiguous")
                    break
                leaf = self._validate_hint_attempt(
                    base._read_json(hint_path),
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=str(evidence["probes"][str(seed)]["observation"]),
                    solution=evidence["probes"][str(seed)]["solution"],
                    attempt=attempt,
                    feedback=hint_feedback,
                )
                if leaf["status"] == "accepted":
                    if any(
                        (hint_root / f"attempt-{later:02d}.json").exists()
                        for later in range(attempt + 1, HINT_ATTEMPTS + 1)
                    ):
                        raise ProxyExperimentError("hint attempt exists after terminal acceptance")
                    locked = leaf
                    break
                hint_feedback = str(leaf["feedback_for_next_attempt"])
            if locked != hint:
                raise ProxyExperimentError("challenger hint differs from canonical attempt chain")

    def _reaudit_candidate(self, evidence: Mapping[str, Any]) -> None:
        """Recompute all scientific pre-actor evidence from the sealed source."""
        code = str(evidence["code"])
        candidate_id = str(evidence["candidate_id"])
        qualification = self._qualification(code, candidate_id)
        probes = self._probe_candidate(code, candidate_id)
        viability = self._one_turn_viability(code, candidate_id, probes)
        candidate_for_cwa = {
            "candidate_id": candidate_id,
            "stratum_id": evidence["stratum_id"],
            "source_arm": evidence["source_arm"],
            "code": code,
            "environment_digest": qualification["environment_digest"],
            "qualification": qualification,
            "probes": probes,
        }
        cwa_evidence = self._cwa_evidence(candidate_for_cwa)
        if (
            qualification != evidence["qualification"]
            or probes != evidence["probes"]
            or viability != evidence["one_turn_viability"]
            or cwa_evidence != evidence["cwa"]
        ):
            raise ProxyExperimentError(
                f"candidate scientific evidence changed on sealed re-audit: {candidate_id}"
            )

    def _proxy_candidate(self, evidence: Mapping[str, Any]) -> ProxyCandidate:
        cwa_value = evidence["cwa"]
        return ProxyCandidate.create(
            candidate_id=str(evidence["candidate_id"]),
            stratum_id=str(evidence["stratum_id"]),
            source_arm=str(evidence["source_arm"]),
            quality_score=float(cwa_value["quality_score"]),
            descriptor=cwa_value["descriptor"],
            environment_digest=str(evidence["environment_digest"]),
            # Portfolio locking is scientific and must not depend on subsequently
            # generated hint bytes. The CWA digest is the stable matching evidence.
            evidence_digest=str(cwa_value["evidence_digest"]),
        )

    def _actor_requests_exist(self) -> bool:
        calls = self.run_dir / "calls"
        for path in calls.glob("*/request.json") if calls.is_dir() else ():
            if base._read_json(path).get("purpose", {}).get("phase") == "actor":
                return True
        return False

    def _validate_complete_pre_hint_quality_gate(self) -> tuple[LockedPortfolios, ...]:
        candidates: list[ProxyCandidate] = []
        for stratum in self.intent["strata"]:
            stratum_id = str(stratum["stratum_id"])
            candidates.extend(self._historical_proxy_candidates(stratum_id))
            candidate_id = str(stratum["challenger_candidate_id"])
            feedback = ""
            terminal: Mapping[str, Any] | None = None
            for attempt in range(1, DESIGN_ATTEMPTS + 1):
                path = (
                    self.run_dir
                    / "challenger-generation"
                    / candidate_id
                    / f"attempt-{attempt:02d}.json"
                )
                if not path.is_file():
                    break
                leaf = self._validate_design_attempt(
                    base._read_json(path),
                    stratum=stratum,
                    attempt=attempt,
                    feedback=feedback,
                )
                if leaf["status"] == "coverage_eligible":
                    terminal = leaf
                    break
                feedback = str(leaf["feedback_for_next_attempt"])
            if terminal is None:
                raise ProxyExperimentError(
                    "challenger hint requires six terminal scientific design records"
                )
            preview = terminal["scientific_preview"]
            candidates.append(
                ProxyCandidate.create(
                    candidate_id=candidate_id,
                    stratum_id=stratum_id,
                    source_arm="challenger",
                    quality_score=float(preview["quality_score"]),
                    descriptor=preview["descriptor"],
                    environment_digest=str(preview["environment_digest"]),
                    evidence_digest=str(preview["cwa_evidence_digest"]),
                )
            )
        return lock_all_portfolios(
            candidates,
            maximum_stratum_absolute_quality_gap=float(
                self.config["maximum_stratum_absolute_quality_gap"]
            ),
            maximum_mean_absolute_quality_gap=float(
                self.config["maximum_mean_absolute_quality_gap"]
            ),
        )

    def _validate_pre_actor_call_inventory(self, *, require_closed: bool) -> None:
        allowed: dict[str, Mapping[str, Any]] = {}
        for stratum in self.intent["strata"]:
            candidate_id = str(stratum["challenger_candidate_id"])
            for attempt in range(1, DESIGN_ATTEMPTS + 1):
                purpose = {
                    "phase": "challenger-design",
                    "stratum_id": stratum["stratum_id"],
                    "candidate_id": candidate_id,
                    "attempt": attempt,
                }
                allowed[base._call_id(purpose)] = purpose
            for seed in SEEDS:
                for attempt in range(1, HINT_ATTEMPTS + 1):
                    purpose = {
                        "phase": "challenger-hint",
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "attempt": attempt,
                    }
                    allowed[base._call_id(purpose)] = purpose
        observed: dict[str, Mapping[str, Any]] = {}
        calls_root = self.run_dir / "calls"
        for path in calls_root.glob("*/request.json") if calls_root.is_dir() else ():
            request = self._validate_call_request(base._read_json(path))
            if request["purpose"]["phase"] == "actor":
                continue
            call_id = str(request["call_id"])
            if call_id not in allowed or request["purpose"] != allowed[call_id]:
                raise ProxyExperimentError("pre-actor call is outside the sealed 36-slot schedule")
            result_path = path.parent / "result.json"
            if not result_path.is_file():
                raise AmbiguousProviderCall(f"pre-actor call {call_id} has no durable result")
            self._validate_call_result(base._read_json(result_path), request)
            observed[call_id] = request
        if any(request["purpose"]["phase"] == "challenger-hint" for request in observed.values()):
            self._validate_complete_pre_hint_quality_gate()
        stratum_ordinals = {
            str(item["stratum_id"]): index for index, item in enumerate(self.intent["strata"])
        }
        ordered_requests = sorted(observed.values(), key=lambda item: int(item["local_ordinal"]))
        generation_root = self.run_dir / "challenger-generation"

        def terminal_design_exists(candidate: str) -> bool:
            directory = generation_root / candidate
            return any(
                path.is_file() and base._read_json(path).get("status") == "coverage_eligible"
                for path in (
                    directory / f"attempt-{attempt:02d}.json"
                    for attempt in range(1, DESIGN_ATTEMPTS + 1)
                )
            )

        chronology_keys: list[tuple[int, int, int, int]] = []
        for request in ordered_requests:
            purpose = request["purpose"]
            candidate_id = str(purpose["candidate_id"])
            stratum_id = candidate_id.split("--", 1)[0]
            ordinal = stratum_ordinals.get(stratum_id)
            if ordinal is None:
                raise ProxyExperimentError("pre-actor call candidate stratum is unknown")
            if any(
                not self._candidate_path(_candidate_id(stratum_id, source_arm)).is_file()
                for source_arm in ("v3", "v4")
            ):
                raise ProxyExperimentError("challenger call precedes historical requalification")
            phase = str(purpose["phase"])
            if phase == "challenger-design" and any(
                not terminal_design_exists(str(prior["challenger_candidate_id"]))
                for prior in self.intent["strata"][:ordinal]
            ):
                raise ProxyExperimentError("challenger design skipped a prior sealed stratum")
            if phase == "challenger-hint" and any(
                not terminal_design_exists(str(item["challenger_candidate_id"]))
                for item in self.intent["strata"]
            ):
                raise ProxyExperimentError("challenger hint precedes the complete scientific panel")
            if phase == "challenger-hint" and any(
                not self._candidate_path(
                    _candidate_id(str(prior["stratum_id"]), "challenger")
                ).is_file()
                for prior in self.intent["strata"][:ordinal]
            ):
                raise ProxyExperimentError("challenger hints skipped an unfinished prior stratum")
            key = (
                0 if phase == "challenger-design" else 1,
                ordinal,
                0 if phase == "challenger-design" else list(SEEDS).index(int(purpose["seed"])),
                int(purpose["attempt"]),
            )
            chronology_keys.append(key)
        if chronology_keys != sorted(set(chronology_keys)):
            raise ProxyExperimentError("pre-actor calls violate sealed stratum/attempt chronology")
        referenced: set[str] = set()
        legal_frontier_ids: set[str] = set()
        hints_root = self.run_dir / "challenger-hints"
        expected_candidates = {
            str(item["challenger_candidate_id"]) for item in self.intent["strata"]
        }
        for root, label in ((generation_root, "generation"), (hints_root, "hint")):
            if root.exists():
                if (
                    root.is_symlink()
                    or not root.is_dir()
                    or any(
                        path.name not in expected_candidates
                        or path.is_symlink()
                        or not path.is_dir()
                        for path in root.iterdir()
                    )
                ):
                    raise ProxyExperimentError(
                        f"challenger {label} root contains a noncanonical candidate"
                    )
        for stratum in self.intent["strata"]:
            candidate_id = str(stratum["challenger_candidate_id"])
            design_dir = generation_root / candidate_id
            qualified: Mapping[str, Any] | None = None
            feedback = ""
            if design_dir.is_dir():
                allowed_names = {
                    f"attempt-{attempt:02d}.json" for attempt in range(1, DESIGN_ATTEMPTS + 1)
                }
                if any(
                    path.name not in allowed_names or not path.is_file()
                    for path in design_dir.iterdir()
                ):
                    raise ProxyExperimentError("challenger generation inventory is noncanonical")
            for attempt in range(1, DESIGN_ATTEMPTS + 1):
                path = design_dir / f"attempt-{attempt:02d}.json"
                if not path.is_file():
                    if any(
                        (design_dir / f"attempt-{later:02d}.json").exists()
                        for later in range(attempt + 1, DESIGN_ATTEMPTS + 1)
                    ):
                        raise ProxyExperimentError(
                            "challenger generation attempts are not contiguous"
                        )
                    purpose = {
                        "phase": "challenger-design",
                        "stratum_id": stratum["stratum_id"],
                        "candidate_id": candidate_id,
                        "attempt": attempt,
                    }
                    legal_frontier_ids.add(base._call_id(purpose))
                    break
                leaf = self._validate_design_attempt(
                    base._read_json(path), stratum=stratum, attempt=attempt, feedback=feedback
                )
                call_id = str(leaf["call_id"])
                if call_id in referenced:
                    raise ProxyExperimentError(
                        "pre-actor provider call is referenced more than once"
                    )
                referenced.add(call_id)
                if leaf["status"] == "coverage_eligible":
                    if any(
                        (design_dir / f"attempt-{later:02d}.json").exists()
                        for later in range(attempt + 1, DESIGN_ATTEMPTS + 1)
                    ):
                        raise ProxyExperimentError(
                            "design attempt exists after terminal eligibility"
                        )
                    qualified = leaf
                    break
                feedback = str(leaf["feedback_for_next_attempt"])
            hint_candidate_root = hints_root / candidate_id
            if hint_candidate_root.exists():
                if qualified is None:
                    raise ProxyExperimentError(
                        "challenger hint exists before terminal design qualification"
                    )
                if (
                    hint_candidate_root.is_symlink()
                    or not hint_candidate_root.is_dir()
                    or any(
                        path.name not in {"0", "42"} or path.is_symlink() or not path.is_dir()
                        for path in hint_candidate_root.iterdir()
                    )
                ):
                    raise ProxyExperimentError("challenger hint seed inventory is noncanonical")
                candidate_path = self._candidate_path(candidate_id)
                if candidate_path.is_file():
                    evidence = self._validate_candidate_evidence(base._read_json(candidate_path))
                    probes = evidence["probes"]
                else:
                    probes = self._probe_candidate(str(qualified["code"]), candidate_id)
                for seed in SEEDS:
                    seed_root = hint_candidate_root / str(seed)
                    hint_feedback = ""
                    if seed_root.is_dir():
                        allowed_names = {
                            f"attempt-{attempt:02d}.json" for attempt in range(1, HINT_ATTEMPTS + 1)
                        }
                        if any(
                            path.name not in allowed_names or not path.is_file()
                            for path in seed_root.iterdir()
                        ):
                            raise ProxyExperimentError(
                                "challenger hint attempt inventory is noncanonical"
                            )
                    for attempt in range(1, HINT_ATTEMPTS + 1):
                        path = seed_root / f"attempt-{attempt:02d}.json"
                        if not path.is_file():
                            if any(
                                (seed_root / f"attempt-{later:02d}.json").exists()
                                for later in range(attempt + 1, HINT_ATTEMPTS + 1)
                            ):
                                raise ProxyExperimentError(
                                    "challenger hint attempts are not contiguous"
                                )
                            if seed == 0 or any(
                                item.get("status") == "accepted"
                                for item in (
                                    base._read_json(candidate_hint)
                                    for candidate_hint in sorted(
                                        (hint_candidate_root / "0").glob("attempt-*.json")
                                    )
                                )
                            ):
                                purpose = {
                                    "phase": "challenger-hint",
                                    "candidate_id": candidate_id,
                                    "seed": seed,
                                    "attempt": attempt,
                                }
                                legal_frontier_ids.add(base._call_id(purpose))
                            break
                        leaf = self._validate_hint_attempt(
                            base._read_json(path),
                            candidate_id=candidate_id,
                            seed=seed,
                            observation=str(probes[str(seed)]["observation"]),
                            solution=probes[str(seed)]["solution"],
                            attempt=attempt,
                            feedback=hint_feedback,
                        )
                        call_id = str(leaf["call_id"])
                        if call_id in referenced:
                            raise ProxyExperimentError(
                                "pre-actor provider call is referenced more than once"
                            )
                        referenced.add(call_id)
                        if leaf["status"] == "accepted":
                            if any(
                                (seed_root / f"attempt-{later:02d}.json").exists()
                                for later in range(attempt + 1, HINT_ATTEMPTS + 1)
                            ):
                                raise ProxyExperimentError(
                                    "hint attempt exists after terminal acceptance"
                                )
                            break
                        hint_feedback = str(leaf["feedback_for_next_attempt"])
            elif qualified is not None:
                legal_frontier_ids.add(
                    base._call_id(
                        {
                            "phase": "challenger-hint",
                            "candidate_id": candidate_id,
                            "seed": 0,
                            "attempt": 1,
                        }
                    )
                )
        if not referenced.issubset(observed):
            raise ProxyExperimentError("pre-actor artifact references a missing provider call")
        unreferenced = set(observed) - referenced
        if require_closed and unreferenced:
            raise ProxyExperimentError("actor seal would leave orphaned pre-actor calls")
        if not require_closed and len(unreferenced) > 1:
            raise ProxyExperimentError(
                "multiple orphaned pre-actor calls are not a resume frontier"
            )
        if unreferenced:
            frontier = observed[next(iter(unreferenced))]
            if int(frontier["local_ordinal"]) != max(
                int(request["local_ordinal"]) for request in observed.values()
            ):
                raise ProxyExperimentError(
                    "unreferenced pre-actor call is not the latest resume frontier"
                )
            if frontier["call_id"] not in legal_frontier_ids:
                raise ProxyExperimentError(
                    "unreferenced pre-actor call is not the next legal attempt"
                )

    async def prepare_actor_plan(self) -> Path:
        existing = self.run_dir / "actor-plan.json"
        self._validate_pre_actor_call_inventory(require_closed=existing.is_file())
        if existing.is_file():
            validate_actor_plan(base._read_json(existing), self.intent, run_dir=self.run_dir)
            return existing
        if self._actor_requests_exist():
            raise ProxyExperimentError("actor requests exist before the concrete actor seal")
        by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
        # Establish every historical CWA anchor before any challenger design call.
        for stratum in self.intent["strata"]:
            for source_arm in ("v3", "v4"):
                item = await self._prepare_candidate(stratum, source_arm)
                by_identity[(str(stratum["stratum_id"]), source_arm)] = item
        # Establish all six challenger scientific records, including exact target
        # cell/CWA/per-stratum matching, without spending any hint calls.
        for stratum in self.intent["strata"]:
            item = await self._prepare_candidate(stratum, "challenger", defer_challenger_hints=True)
            by_identity[(str(stratum["stratum_id"]), "challenger")] = item

        provisional = [
            by_identity[(str(stratum["stratum_id"]), source_arm)]
            for stratum in self.intent["strata"]
            for source_arm in ("v3", "v4", "challenger")
        ]
        provisional_locks = lock_all_portfolios(
            [self._proxy_candidate(item) for item in provisional],
            maximum_stratum_absolute_quality_gap=float(
                self.config["maximum_stratum_absolute_quality_gap"]
            ),
            maximum_mean_absolute_quality_gap=float(
                self.config["maximum_mean_absolute_quality_gap"]
            ),
        )
        # Only after the complete panel passes both prospective quality-match
        # gates may challenger hint calls begin.
        for stratum in self.intent["strata"]:
            key = (str(stratum["stratum_id"]), "challenger")
            item = by_identity[key]
            if item["hints"] == {}:
                by_identity[key] = await self._finalize_deferred_challenger(item)
        evidence = [
            by_identity[(str(stratum["stratum_id"]), source_arm)]
            for stratum in self.intent["strata"]
            for source_arm in ("v3", "v4", "challenger")
        ]
        if len(evidence) != SHARED_STRATA * CANDIDATES_PER_STRATUM:
            raise ProxyExperimentError("candidate panel is incomplete")
        candidates = [self._proxy_candidate(item) for item in evidence]
        locks = lock_all_portfolios(
            candidates,
            maximum_stratum_absolute_quality_gap=float(
                self.config["maximum_stratum_absolute_quality_gap"]
            ),
            maximum_mean_absolute_quality_gap=float(
                self.config["maximum_mean_absolute_quality_gap"]
            ),
        )
        # Hint bytes do not enter quality or descriptors; nevertheless assert the
        # post-hint locks are identical to the provisional scientific lock.
        if [item.to_dict() for item in locks] != [item.to_dict() for item in provisional_locks]:
            raise ProxyExperimentError("portfolio lock changed after nonmatching hint generation")
        grouped = {
            stratum["stratum_id"]: tuple(
                _candidate_id(str(stratum["stratum_id"]), source_arm)
                for source_arm in ("v3", "v4", "challenger")
            )
            for stratum in self.intent["strata"]
        }
        schedule = counterbalanced_pairs(grouped, SEEDS)
        body = {
            "schema_version": ACTOR_PLAN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "chronology": "sealed-after-candidates-cwa-hints-before-actor",
            "provider": "agy",
            "model": ACTOR_MODEL,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "candidate_evidence": [
                {
                    "stratum_id": item["stratum_id"],
                    "candidate_id": item["candidate_id"],
                    "source_arm": item["source_arm"],
                    "path": str(
                        self._candidate_path(str(item["candidate_id"])).relative_to(self.run_dir)
                    ),
                    "evidence_digest": item["evidence_digest"],
                    "qualification_digest": item["qualification"]["qualification_digest"],
                    "cwa_evidence_digest": item["cwa"]["evidence_digest"],
                    "viability_digest": item["one_turn_viability"]["viability_digest"],
                    "hint_digests": {seed: _digest(item["hints"][seed]) for seed in ("0", "42")},
                }
                for item in evidence
            ],
            "portfolios": [item.to_dict() for item in locks],
            "portfolio_quality_diagnostics": portfolio_quality_diagnostics(locks),
            "pair_schedule": list(schedule),
            "pair_schedule_digest": _digest(schedule),
            "attempt_policy": {
                "whole_pair_retries": True,
                "waves_1_and_2": "all-unresolved",
                "wave_3": "only-if-unresolved-after-wave-2-at-most-14",
                "retryable_failure_categories": ["empty_response", "provider_timeout"],
                "nonempty_parser_misses_are_zero_reward_and_not_retried": True,
                "environment_runtime_integrity_failures_are_fatal": True,
            },
            "actor_call_ceiling": ACTOR_CALL_CEILING,
            "success_rule": "reward-positive-even-if-simultaneously-truncated",
            "analysis_gates": self.intent["analysis_gates"],
            "analysis_interpretation": (
                "quality-matched coverage-forced portfolio-swap association; exact 3!^6 "
                "label-permutation and 2^6 sign-flip sensitivity analyses under strong "
                "exchangeability/symmetry assumptions; neither is design-based"
            ),
        }
        actor_plan = {**body, "actor_plan_digest": _digest(body)}
        self._validate_pre_actor_call_inventory(require_closed=True)
        validate_actor_plan(actor_plan, self.intent, run_dir=self.run_dir)
        if self._actor_requests_exist():
            raise ProxyExperimentError("actor request raced actor-plan sealing")
        base._write_json(existing, actor_plan)
        return existing


def validate_actor_plan(
    value: object,
    intent: Mapping[str, Any],
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProxyExperimentError("actor plan must be an object")
    required = {
        "schema_version",
        "protocol_id",
        "intent_digest",
        "chronology",
        "provider",
        "model",
        "backend_identity_attested",
        "route_authority",
        "candidate_evidence",
        "portfolios",
        "portfolio_quality_diagnostics",
        "pair_schedule",
        "pair_schedule_digest",
        "attempt_policy",
        "actor_call_ceiling",
        "success_rule",
        "analysis_gates",
        "analysis_interpretation",
        "actor_plan_digest",
    }
    if set(value) != required:
        raise ProxyExperimentError("actor plan fields differ from schema")
    if (
        value["schema_version"] != ACTOR_PLAN_SCHEMA
        or value["protocol_id"] != PROTOCOL_ID
        or value["intent_digest"] != intent["intent_digest"]
        or value["chronology"] != "sealed-after-candidates-cwa-hints-before-actor"
        or value["provider"] != "agy"
        or value["model"] != ACTOR_MODEL
        or value["backend_identity_attested"] is not False
        or value["route_authority"] != "requested-route-only"
        or value["actor_call_ceiling"] != ACTOR_CALL_CEILING
        or value["success_rule"] != "reward-positive-even-if-simultaneously-truncated"
    ):
        raise ProxyExperimentError("actor plan identity differs")
    references = value["candidate_evidence"]
    if not isinstance(references, list) or len(references) != SHARED_STRATA * 3:
        raise ProxyExperimentError("actor plan must bind all 18 candidates")
    expected_identities = [
        (
            str(stratum["stratum_id"]),
            source_arm,
            _candidate_id(str(stratum["stratum_id"]), source_arm),
        )
        for stratum in intent["strata"]
        for source_arm in ("v3", "v4", "challenger")
    ]
    observed_identities = [
        (str(item.get("stratum_id")), str(item.get("source_arm")), str(item.get("candidate_id")))
        for item in references
    ]
    if observed_identities != expected_identities:
        raise ProxyExperimentError("actor plan candidate topology/order differs from 6 x 3 seal")
    grouped: dict[str, list[str]] = {}
    proxy_candidates: list[ProxyCandidate] = []
    for item in references:
        if set(item) != {
            "stratum_id",
            "candidate_id",
            "source_arm",
            "path",
            "evidence_digest",
            "qualification_digest",
            "cwa_evidence_digest",
            "viability_digest",
            "hint_digests",
        }:
            raise ProxyExperimentError("actor candidate reference fields differ")
        canonical_path = f"candidate-evidence/{item['candidate_id']}.json"
        if item["path"] != canonical_path:
            raise ProxyExperimentError("actor candidate reference path is noncanonical")
        grouped.setdefault(str(item["stratum_id"]), []).append(str(item["candidate_id"]))
        if run_dir is not None:
            relative = Path(str(item["path"]))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != item["path"]
            ):
                raise ProxyExperimentError("candidate evidence reference path is unsafe")
            evidence = _Engine._validate_candidate_evidence(base._read_json(run_dir / relative))
            if (
                evidence["intent_digest"] != intent["intent_digest"]
                or evidence["stratum_id"] != item["stratum_id"]
                or evidence["candidate_id"] != item["candidate_id"]
                or evidence["source_arm"] != item["source_arm"]
            ):
                raise ProxyExperimentError(
                    "actor candidate reference identity differs from evidence"
                )
            expected = {
                "evidence_digest": evidence["evidence_digest"],
                "qualification_digest": evidence["qualification"]["qualification_digest"],
                "cwa_evidence_digest": evidence["cwa"]["evidence_digest"],
                "viability_digest": evidence["one_turn_viability"]["viability_digest"],
                "hint_digests": {seed: _digest(evidence["hints"][seed]) for seed in ("0", "42")},
            }
            if any(item.get(key) != expected_value for key, expected_value in expected.items()):
                raise ProxyExperimentError("actor plan candidate reference drifted")
            proxy_candidates.append(
                ProxyCandidate.create(
                    candidate_id=str(evidence["candidate_id"]),
                    stratum_id=str(evidence["stratum_id"]),
                    source_arm=str(evidence["source_arm"]),
                    quality_score=float(evidence["cwa"]["quality_score"]),
                    descriptor=evidence["cwa"]["descriptor"],
                    environment_digest=str(evidence["environment_digest"]),
                    evidence_digest=str(evidence["cwa"]["evidence_digest"]),
                )
            )
    expected_schedule = list(
        counterbalanced_pairs(
            {key: tuple(items) for key, items in grouped.items()},
            SEEDS,
        )
    )
    if value["pair_schedule"] != expected_schedule or value["pair_schedule_digest"] != _digest(
        expected_schedule
    ):
        raise ProxyExperimentError("actor plan schedule differs from the candidate panel")
    portfolios = value["portfolios"]
    if not isinstance(portfolios, list) or len(portfolios) != SHARED_STRATA:
        raise ProxyExperimentError("actor plan portfolios are incomplete")
    if run_dir is not None:
        locked = lock_all_portfolios(
            proxy_candidates,
            maximum_stratum_absolute_quality_gap=float(
                intent["configuration"]["maximum_stratum_absolute_quality_gap"]
            ),
            maximum_mean_absolute_quality_gap=float(
                intent["configuration"]["maximum_mean_absolute_quality_gap"]
            ),
        )
        recomputed = [item.to_dict() for item in locked]
        if portfolios != recomputed:
            raise ProxyExperimentError("actor portfolios differ from sealed candidate evidence")
        if value["portfolio_quality_diagnostics"] != portfolio_quality_diagnostics(locked):
            raise ProxyExperimentError("actor portfolio quality diagnostics differ")
    elif not isinstance(value["portfolio_quality_diagnostics"], dict):
        raise ProxyExperimentError("actor portfolio quality diagnostics are absent")
    if not all(item.get("differs") is True for item in portfolios):
        raise ProxyExperimentError("coverage/redundant portfolios must differ in all six strata")
    if value["analysis_gates"] != intent["analysis_gates"]:
        raise ProxyExperimentError("actor plan analysis gates differ from intent")
    if value["attempt_policy"] != {
        "whole_pair_retries": True,
        "waves_1_and_2": "all-unresolved",
        "wave_3": "only-if-unresolved-after-wave-2-at-most-14",
        "retryable_failure_categories": ["empty_response", "provider_timeout"],
        "nonempty_parser_misses_are_zero_reward_and_not_retried": True,
        "environment_runtime_integrity_failures_are_fatal": True,
    }:
        raise ProxyExperimentError("actor attempt policy differs")
    if value["analysis_interpretation"] != (
        "quality-matched coverage-forced portfolio-swap association; exact 3!^6 "
        "label-permutation and 2^6 sign-flip sensitivity analyses under strong "
        "exchangeability/symmetry assumptions; neither is design-based"
    ):
        raise ProxyExperimentError("actor analysis interpretation differs")
    body = {key: item for key, item in value.items() if key != "actor_plan_digest"}
    if value["actor_plan_digest"] != _digest(body):
        raise ProxyExperimentError("actor plan digest mismatch")
    return value


def _sealed_artifact(value: Mapping[str, Any], digest_field: str) -> None:
    body = {key: item for key, item in value.items() if key != digest_field}
    if value.get(digest_field) != _digest(body):
        raise ProxyExperimentError(f"{digest_field} does not bind its artifact")


def _pair_by_id(actor_plan: Mapping[str, Any], pair_id: str) -> Mapping[str, Any]:
    matches = [item for item in actor_plan["pair_schedule"] if item["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ProxyExperimentError(f"unknown or duplicate pair id: {pair_id}")
    return matches[0]


class _ActorEngine(_Engine):
    def __init__(
        self,
        intent: dict[str, Any],
        intent_bytes: bytes,
        run_dir: Path,
        dependencies: RunnerDependencies,
        actor_plan: dict[str, Any],
    ) -> None:
        super().__init__(intent, intent_bytes, run_dir, dependencies)
        self.actor_plan = actor_plan
        self._candidate_by_id = {
            str(item["candidate_id"]): self._validate_candidate_evidence(
                base._read_json(run_dir / str(item["path"]))
            )
            for item in actor_plan["candidate_evidence"]
        }
        self._scientific_reaudit_complete = False

    def _arm_path(self, pair_id: str, attempt: int, arm: str) -> Path:
        return self.run_dir / "pair-attempts" / pair_id / f"attempt-{attempt:02d}" / f"{arm}.json"

    def _attempt_path(self, pair_id: str, attempt: int) -> Path:
        return self.run_dir / "pair-attempts" / pair_id / f"attempt-{attempt:02d}" / "attempt.json"

    def _resolution_path(self, pair_id: str) -> Path:
        return self.run_dir / "pair-resolutions" / f"{pair_id}.json"

    def _validate_resolution(
        self, value: Mapping[str, Any], pair: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if (
            set(value)
            != {
                "schema_version",
                "intent_digest",
                "actor_plan_digest",
                "pair",
                "selected_attempt",
                "selected_attempt_digest",
                "attempts",
                "first_attempt_exogenous_failure",
                "arms",
                "resolution_digest",
            }
            or value.get("schema_version") != PAIR_RESOLUTION_SCHEMA
        ):
            raise ProxyExperimentError("pair resolution fields differ from schema")
        _sealed_artifact(value, "resolution_digest")
        if (
            value["intent_digest"] != self.intent["intent_digest"]
            or value["actor_plan_digest"] != self.actor_plan["actor_plan_digest"]
            or value["pair"] != dict(pair)
        ):
            raise ProxyExperimentError("pair resolution identity differs")
        selected = value["selected_attempt"]
        if type(selected) is not int or not 1 <= selected <= 3:
            raise ProxyExperimentError("pair resolution selected attempt is invalid")
        attempt_refs = value["attempts"]
        if not isinstance(attempt_refs, list) or [
            item.get("pair_attempt") for item in attempt_refs
        ] != list(range(1, selected + 1)):
            raise ProxyExperimentError("pair resolution attempt closure is not contiguous")
        attempts: list[Mapping[str, Any]] = []
        for index, reference in enumerate(attempt_refs, start=1):
            if set(reference) != {"pair_attempt", "status", "attempt_digest"}:
                raise ProxyExperimentError("pair resolution attempt reference fields differ")
            attempt_path = self._attempt_path(str(pair["pair_id"]), index)
            attempt = self._validate_attempt(base._read_json(attempt_path), pair, index)
            if (
                reference["status"] != attempt["status"]
                or reference["attempt_digest"] != attempt["attempt_digest"]
            ):
                raise ProxyExperimentError("pair resolution attempt reference drifted")
            expected_status = "completed" if index == selected else "retryable_failure"
            if attempt["status"] != expected_status:
                raise ProxyExperimentError("pair resolution selected a noncanonical attempt chain")
            attempts.append(attempt)
        selected_attempt = attempts[-1]
        if (
            value["selected_attempt_digest"] != selected_attempt["attempt_digest"]
            or value["arms"] != selected_attempt["arms"]
            or value["first_attempt_exogenous_failure"]
            is not (attempts[0]["status"] == "retryable_failure")
        ):
            raise ProxyExperimentError("pair resolution selected evidence differs")
        pair_root = self.run_dir / "pair-attempts" / str(pair["pair_id"])
        if pair_root.is_dir():
            expected_dirs = {f"attempt-{index:02d}" for index in range(1, selected + 1)}
            if {path.name for path in pair_root.iterdir()} != expected_dirs:
                raise ProxyExperimentError("attempt artifacts exist after pair resolution")
        return value

    def _preflight_actor_state(self) -> None:
        """Validate the complete actor frontier before any new paid request."""
        self._validate_pre_actor_call_inventory(require_closed=True)
        if not self._scientific_reaudit_complete:
            for evidence in self._candidate_by_id.values():
                self._validate_candidate_lineage(evidence)
                self._reaudit_candidate(evidence)
            self._scientific_reaudit_complete = True
        schedule = self.actor_plan["pair_schedule"]
        pair_ids = {str(item["pair_id"]) for item in schedule}
        artifact_actor_calls: set[str] = set()
        attempts_root = self.run_dir / "pair-attempts"
        if attempts_root.exists():
            if attempts_root.is_symlink() or not attempts_root.is_dir():
                raise ProxyExperimentError("pair-attempts root is unsafe")
            if any(
                path.name not in pair_ids or not path.is_dir() for path in attempts_root.iterdir()
            ):
                raise ProxyExperimentError("unknown pair-attempt artifact exists")
        resolutions_root = self.run_dir / "pair-resolutions"
        if resolutions_root.exists():
            if resolutions_root.is_symlink() or not resolutions_root.is_dir():
                raise ProxyExperimentError("pair-resolutions root is unsafe")
            expected_names = {f"{pair_id}.json" for pair_id in pair_ids}
            if any(
                path.name not in expected_names or not path.is_file()
                for path in resolutions_root.iterdir()
            ):
                raise ProxyExperimentError("unknown pair-resolution artifact exists")
        for pair in schedule:
            pair_id = str(pair["pair_id"])
            pair_root = attempts_root / pair_id
            attempt_numbers: list[int] = []
            if pair_root.is_dir():
                for path in pair_root.iterdir():
                    match = re.fullmatch(r"attempt-([0-9]{2})", path.name)
                    if path.is_symlink() or not path.is_dir() or match is None:
                        raise ProxyExperimentError("pair attempt directory is noncanonical")
                    attempt_numbers.append(int(match.group(1)))
                attempt_numbers.sort()
                if attempt_numbers != list(range(1, len(attempt_numbers) + 1)) or any(
                    item > 3 for item in attempt_numbers
                ):
                    raise ProxyExperimentError("pair attempt directories are not a sealed prefix")
                for attempt in attempt_numbers:
                    attempt_root = pair_root / f"attempt-{attempt:02d}"
                    allowed = {"attempt.json", "hinted.json", "unhinted.json"}
                    if any(
                        path.name not in allowed or not path.is_file()
                        for path in attempt_root.iterdir()
                    ):
                        raise ProxyExperimentError("pair attempt contains an unknown leaf")
                    arm_names = [
                        f"{arm}.json"
                        for arm in pair["arm_order"]
                        if (attempt_root / f"{arm}.json").is_file()
                    ]
                    present_names = [f"{arm}.json" for arm in pair["arm_order"][: len(arm_names)]]
                    if arm_names != present_names:
                        raise ProxyExperimentError("partial pair arms are not an order prefix")
                    for arm in pair["arm_order"][: len(arm_names)]:
                        arm_leaf = self._validate_arm(
                            base._read_json(attempt_root / f"{arm}.json"),
                            pair=pair,
                            attempt=attempt,
                            arm=str(arm),
                        )
                        artifact_actor_calls.add(str(arm_leaf["call_id"]))
                    summary_path = attempt_root / "attempt.json"
                    if summary_path.is_file():
                        summary = self._validate_attempt(
                            base._read_json(summary_path), pair, attempt
                        )
                        if summary["failed_call_id"] is not None:
                            artifact_actor_calls.add(str(summary["failed_call_id"]))
                    elif attempt != attempt_numbers[-1]:
                        raise ProxyExperimentError("unfinished attempt is not the actor frontier")
            resolution_path = self._resolution_path(pair_id)
            if resolution_path.is_file():
                self._validate_resolution(base._read_json(resolution_path), pair)
            elif attempt_numbers:
                for prior in attempt_numbers[:-1]:
                    summary = self._validate_attempt(
                        base._read_json(self._attempt_path(pair_id, prior)), pair, prior
                    )
                    if summary["status"] != "retryable_failure":
                        raise ProxyExperimentError(
                            "later attempt follows a completed unresolved pair"
                        )

        calls_root = self.run_dir / "calls"
        requests = (
            [
                self._validate_call_request(base._read_json(path))
                for path in calls_root.glob("*/request.json")
            ]
            if calls_root.is_dir()
            else []
        )
        requests.sort(key=lambda item: int(item["local_ordinal"]))
        actor_seen = False
        actor_requests: list[Mapping[str, Any]] = []
        actor_results: dict[str, Mapping[str, Any]] = {}
        prior_key: tuple[int, int, int] | None = None
        schedule_by_id = {str(item["pair_id"]): item for item in schedule}
        for request in requests:
            purpose = request["purpose"]
            if purpose["phase"] != "actor":
                if actor_seen:
                    raise ProxyExperimentError(
                        "pre-actor call appears after actor chronology began"
                    )
                continue
            actor_seen = True
            result_path = calls_root / str(request["call_id"]) / "result.json"
            if not result_path.is_file():
                raise AmbiguousProviderCall(
                    f"actor call {request['call_id']} has unknown provider disposition"
                )
            actor_results[str(request["call_id"])] = self._validate_call_result(
                base._read_json(result_path), request
            )
            actor_requests.append(request)
            pair = schedule_by_id.get(str(purpose.get("pair_id")))
            if pair is None:
                raise ProxyExperimentError("actor call references an unknown pair")
            attempt = purpose.get("pair_attempt")
            arm = purpose.get("arm")
            if type(attempt) is not int or attempt not in {1, 2, 3} or arm not in pair["arm_order"]:
                raise ProxyExperimentError("actor call attempt/arm is outside its pair")
            expected_purpose = {
                "phase": "actor",
                "actor_plan_digest": self.actor_plan["actor_plan_digest"],
                "pair_id": pair["pair_id"],
                "pair_ordinal": pair["pair_ordinal"],
                "pair_attempt": attempt,
                "stratum_id": pair["stratum_id"],
                "candidate_id": pair["candidate_id"],
                "seed": pair["seed"],
                "arm": arm,
                "turn": 1,
                "horizon": 1,
            }
            candidate = self._candidate_by_id[str(pair["candidate_id"])]
            probe = candidate["probes"][str(pair["seed"])]
            hint = str(candidate["hints"][str(pair["seed"])]["hint"]) if arm == "hinted" else ""
            prompt = self._prompt(str(probe["observation"]), hint)
            self._validate_call_request(
                request,
                {
                    "purpose": expected_purpose,
                    "prompt": prompt,
                    "prompt_digest": _digest(prompt),
                    "system": "",
                    "system_digest": _digest(""),
                },
            )
            if int(attempt) > 1:
                previous_path = self._attempt_path(str(pair["pair_id"]), int(attempt) - 1)
                if not previous_path.is_file():
                    raise ProxyExperimentError("later-wave actor call lacks prior attempt summary")
                previous = self._validate_attempt(
                    base._read_json(previous_path), pair, int(attempt) - 1
                )
                if previous["status"] != "retryable_failure":
                    raise ProxyExperimentError("later-wave actor call follows nonretryable attempt")
                if int(attempt) == 2 and any(
                    not self._attempt_path(str(other["pair_id"]), 1).is_file() for other in schedule
                ):
                    raise ProxyExperimentError("wave 2 actor call began before wave 1 closed")
                if int(attempt) == 3:
                    unresolved = 0
                    for other in schedule:
                        if self._resolution_path(str(other["pair_id"])).is_file():
                            continue
                        second_path = self._attempt_path(str(other["pair_id"]), 2)
                        if not second_path.is_file():
                            raise ProxyExperimentError(
                                "wave 3 actor call began before wave 2 closed"
                            )
                        second = self._validate_attempt(base._read_json(second_path), other, 2)
                        if second["status"] == "retryable_failure":
                            unresolved += 1
                    if unresolved > THIRD_WAVE_LIMIT:
                        raise ProxyExperimentError("wave 3 actor call exceeds unresolved gate")
            key = (int(attempt), int(pair["pair_ordinal"]), list(pair["arm_order"]).index(str(arm)))
            if prior_key is not None and key <= prior_key:
                raise ProxyExperimentError("actor calls violate sealed wave/pair/arm order")
            prior_key = key
        unreferenced = {
            str(request["call_id"]) for request in actor_requests
        } - artifact_actor_calls
        if len(unreferenced) > 1:
            raise ProxyExperimentError("multiple actor calls are outside the legal resume frontier")
        if unreferenced:
            call_id = next(iter(unreferenced))
            request = next(item for item in actor_requests if item["call_id"] == call_id)
            if request is not actor_requests[-1]:
                raise ProxyExperimentError("unmaterialized actor call is not the latest request")
            purpose = request["purpose"]
            pair = schedule_by_id[str(purpose["pair_id"])]
            attempt = int(purpose["pair_attempt"])
            arm = str(purpose["arm"])
            arm_index = list(pair["arm_order"]).index(arm)
            if (
                self._attempt_path(str(pair["pair_id"]), attempt).exists()
                or self._arm_path(str(pair["pair_id"]), attempt, arm).exists()
                or any(
                    not self._arm_path(str(pair["pair_id"]), attempt, str(prior_arm)).is_file()
                    for prior_arm in pair["arm_order"][:arm_index]
                )
                or any(
                    self._arm_path(str(pair["pair_id"]), attempt, str(later_arm)).exists()
                    for later_arm in pair["arm_order"][arm_index + 1 :]
                )
            ):
                raise ProxyExperimentError(
                    "actor call is outside the exact artifact resume frontier"
                )
            result = actor_results[call_id]
            if result["status"] == "error" and result["failure_category"] not in {
                "empty_response",
                "provider_timeout",
            }:
                raise ProxyExperimentIncomplete("fatal actor transport cannot be resumed")
        if any(request["purpose"]["phase"] == "actor" for request in requests):
            self._validate_actor_wave_eligibility(schedule)

    def _validate_actor_wave_eligibility(self, schedule: Sequence[Mapping[str, Any]]) -> None:
        """Prove later waves follow retryable whole-pair dispositions."""
        attempt2_seen = False
        attempt3_seen = False
        unresolved_after_two = 0
        for pair in schedule:
            first_path = self._attempt_path(str(pair["pair_id"]), 1)
            second_path = self._attempt_path(str(pair["pair_id"]), 2)
            third_path = self._attempt_path(str(pair["pair_id"]), 3)
            first = (
                self._validate_attempt(base._read_json(first_path), pair, 1)
                if first_path.is_file()
                else None
            )
            second = (
                self._validate_attempt(base._read_json(second_path), pair, 2)
                if second_path.is_file()
                else None
            )
            if second is not None:
                attempt2_seen = True
                if first is None or first["status"] != "retryable_failure":
                    raise ProxyExperimentError("attempt 2 lacks a retryable attempt 1")
                if second["status"] == "retryable_failure":
                    unresolved_after_two += 1
            if third_path.is_file() or (
                third_path.parent.is_dir() and any(third_path.parent.iterdir())
            ):
                attempt3_seen = True
                if second is None or second["status"] != "retryable_failure":
                    raise ProxyExperimentError("attempt 3 lacks a retryable attempt 2")
        if attempt2_seen and any(
            not self._attempt_path(str(pair["pair_id"]), 1).is_file() for pair in schedule
        ):
            raise ProxyExperimentError("attempt wave 2 began before wave 1 completed")
        if attempt3_seen:
            if any(
                self._resolution_path(str(pair["pair_id"])).is_file() is False
                and not self._attempt_path(str(pair["pair_id"]), 2).is_file()
                for pair in schedule
            ):
                raise ProxyExperimentError("attempt wave 3 began before wave 2 completed")
            if unresolved_after_two > THIRD_WAVE_LIMIT:
                raise ProxyExperimentError("attempt wave 3 exceeds unresolved-pair gate")

    @staticmethod
    def _prompt(observation: str, hint: str) -> str:
        history = f"Initial observation: {observation}"
        if hint:
            history += f"\n\nPrivileged strategy hint:\n{hint}"
        return (
            "You are playing an interactive reasoning environment.\n"
            f"{history}\n\nTurn 1/1: reason about the state, then provide exactly one next "
            "action with the required answer format."
        )

    def _validate_arm(
        self,
        value: Mapping[str, Any],
        *,
        pair: Mapping[str, Any],
        attempt: int,
        arm: str,
    ) -> Mapping[str, Any]:
        required = {
            "schema_version",
            "intent_digest",
            "actor_plan_digest",
            "pair_id",
            "pair_ordinal",
            "pair_attempt",
            "stratum_id",
            "candidate_id",
            "seed",
            "arm",
            "call_id",
            "raw_response",
            "raw_response_digest",
            "parser_miss",
            "clean_action",
            "pre_observation",
            "pre_observation_digest",
            "post_observation",
            "post_observation_digest",
            "raw_reward",
            "binary_return",
            "terminated",
            "truncated",
            "success_rule",
            "arm_digest",
        }
        if set(value) != required or value.get("schema_version") != "spade-coverage-forced-arm/v1":
            raise ProxyExperimentError("actor arm fields differ from schema")
        _sealed_artifact(value, "arm_digest")
        expected = {
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "pair_attempt": attempt,
            "stratum_id": pair["stratum_id"],
            "candidate_id": pair["candidate_id"],
            "seed": pair["seed"],
            "arm": arm,
            "success_rule": "reward-positive-even-if-simultaneously-truncated",
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ProxyExperimentError("actor arm identity differs")
        raw = value["raw_response"]
        if not isinstance(raw, str) or not raw or value["raw_response_digest"] != _digest(raw):
            raise ProxyExperimentError("actor response digest differs")
        parser_miss = live.extract_boxed_answer(raw) is None
        if value["parser_miss"] is not parser_miss or value[
            "clean_action"
        ] != live.extract_clean_action(raw, "boxed"):
            raise ProxyExperimentError("actor parse evidence differs from response")
        if value["pre_observation_digest"] != _digest(value["pre_observation"]) or value[
            "post_observation_digest"
        ] != _digest(value["post_observation"]):
            raise ProxyExperimentError("actor observation digest differs")
        reward = value["raw_reward"]
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise ProxyExperimentError("actor reward is invalid")
        if value["binary_return"] != (1.0 if float(reward) > 0 else 0.0):
            raise ProxyExperimentError("actor binary return violates reward-positive rule")
        if type(value["terminated"]) is not bool or type(value["truncated"]) is not bool:
            raise ProxyExperimentError("actor termination flags are invalid")
        candidate = self._candidate_by_id[str(pair["candidate_id"])]
        probe = candidate["probes"][str(pair["seed"])]
        hint = str(candidate["hints"][str(pair["seed"])]["hint"]) if arm == "hinted" else ""
        prompt = self._prompt(str(probe["observation"]), hint)
        purpose = {
            "phase": "actor",
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "pair_attempt": attempt,
            "stratum_id": pair["stratum_id"],
            "candidate_id": pair["candidate_id"],
            "seed": pair["seed"],
            "arm": arm,
            "turn": 1,
            "horizon": 1,
        }
        if value["call_id"] != base._call_id(purpose):
            raise ProxyExperimentError("actor arm call id differs")
        request_path = self.run_dir / "calls" / str(value["call_id"]) / "request.json"
        request = self._validate_call_request(
            base._read_json(request_path),
            {
                "purpose": purpose,
                "prompt": prompt,
                "prompt_digest": _digest(prompt),
                "system": "",
                "system_digest": _digest(""),
            },
        )
        result = self._validate_call_result(
            base._read_json(request_path.parent / "result.json"), request
        )
        if result["status"] != "success" or result["response"] != raw:
            raise ProxyExperimentError("actor arm does not link a successful provider result")
        try:
            target = self._proofpack_call(
                self.dependencies.target_factory,
                candidate["code"],
                action_format="boxed",
                max_turns=1,
                operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
            )
            env = self._proofpack_call(target.instantiate)
            observation, _ = self._proofpack_call(env.reset, seed=int(pair["seed"]))
            if str(observation) != probe["observation"]:
                raise ProxyExperimentError("persisted actor reset differs from locked probe")
            step = self._proofpack_call(env.step, str(value["clean_action"]))
            if not isinstance(step, tuple) or len(step) != 5:
                raise ProxyExperimentError("persisted actor replay returned invalid step shape")
            post, reward, terminated, truncated, _info = step
        except Exception as exc:
            if isinstance(exc, ProxyExperimentError):
                raise
            raise ProxyExperimentError(f"persisted actor replay failed: {exc}") from exc
        replay_fields = {
            "pre_observation": str(observation),
            "post_observation": str(post),
            "raw_reward": float(reward),
            "binary_return": 1.0 if float(reward) > 0 else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if any(value.get(key) != item for key, item in replay_fields.items()):
            raise ProxyExperimentError("persisted actor arm diverges under deterministic replay")
        return value

    def _validate_attempt(
        self, value: Mapping[str, Any], pair: Mapping[str, Any], attempt: int
    ) -> Mapping[str, Any]:
        if (
            set(value)
            != {
                "schema_version",
                "intent_digest",
                "actor_plan_digest",
                "pair_id",
                "pair_ordinal",
                "pair_attempt",
                "status",
                "failure_category",
                "failed_call_id",
                "arms",
                "attempt_digest",
            }
            or value.get("schema_version") != PAIR_ATTEMPT_SCHEMA
        ):
            raise ProxyExperimentError("pair attempt schema differs")
        _sealed_artifact(value, "attempt_digest")
        fixed = {
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "pair_attempt": attempt,
        }
        if any(value.get(key) != item for key, item in fixed.items()):
            raise ProxyExperimentError("pair attempt identity differs")
        status = value.get("status")
        arms = value.get("arms")
        if not isinstance(arms, list):
            raise ProxyExperimentError("pair attempt arms are invalid")
        if status == "completed":
            if value.get("failure_category") is not None or value.get("failed_call_id") is not None:
                raise ProxyExperimentError("completed pair attempt has failure metadata")
            if [item.get("arm") for item in arms] != pair["arm_order"]:
                raise ProxyExperimentError("completed pair attempt arm order differs")
            for item in arms:
                arm = str(item["arm"])
                expected_path = str(
                    self._arm_path(str(pair["pair_id"]), attempt, arm).relative_to(self.run_dir)
                )
                if set(item) != {"arm", "path", "arm_digest"} or item.get("path") != expected_path:
                    raise ProxyExperimentError("pair attempt arm path is noncanonical")
                leaf = self._validate_arm(
                    base._read_json(self.run_dir / str(item["path"])),
                    pair=pair,
                    attempt=attempt,
                    arm=arm,
                )
                if item.get("arm_digest") != leaf["arm_digest"]:
                    raise ProxyExperimentError("pair attempt arm reference differs")
        elif status == "retryable_failure":
            if value.get("failure_category") not in {"empty_response", "provider_timeout"}:
                raise ProxyExperimentError("pair retry uses a nonretryable failure")
            if not isinstance(value.get("failed_call_id"), str):
                raise ProxyExperimentError("pair retry lacks a failed call")
            if [item.get("arm") for item in arms] != pair["arm_order"][: len(arms)]:
                raise ProxyExperimentError("partial pair attempt arm order differs")
            for item in arms:
                arm = str(item["arm"])
                expected_path = str(
                    self._arm_path(str(pair["pair_id"]), attempt, arm).relative_to(self.run_dir)
                )
                if set(item) != {"arm", "path", "arm_digest"} or item.get("path") != expected_path:
                    raise ProxyExperimentError("partial pair arm path is noncanonical")
                leaf = self._validate_arm(
                    base._read_json(self.run_dir / str(item["path"])),
                    pair=pair,
                    attempt=attempt,
                    arm=arm,
                )
                if item.get("arm_digest") != leaf["arm_digest"]:
                    raise ProxyExperimentError("partial pair arm reference differs")
            failed_request = self.run_dir / "calls" / str(value["failed_call_id"]) / "request.json"
            if len(arms) >= len(pair["arm_order"]):
                raise ProxyExperimentError("retryable pair has no next failing arm")
            failed_arm = str(pair["arm_order"][len(arms)])
            candidate = self._candidate_by_id[str(pair["candidate_id"])]
            probe = candidate["probes"][str(pair["seed"])]
            hint = (
                str(candidate["hints"][str(pair["seed"])]["hint"]) if failed_arm == "hinted" else ""
            )
            prompt = self._prompt(str(probe["observation"]), hint)
            purpose = {
                "phase": "actor",
                "actor_plan_digest": self.actor_plan["actor_plan_digest"],
                "pair_id": pair["pair_id"],
                "pair_ordinal": pair["pair_ordinal"],
                "pair_attempt": attempt,
                "stratum_id": pair["stratum_id"],
                "candidate_id": pair["candidate_id"],
                "seed": pair["seed"],
                "arm": failed_arm,
                "turn": 1,
                "horizon": 1,
            }
            if value["failed_call_id"] != base._call_id(purpose):
                raise ProxyExperimentError("pair retry failed call id differs from next arm")
            request = self._validate_call_request(
                base._read_json(failed_request),
                {
                    "purpose": purpose,
                    "prompt": prompt,
                    "prompt_digest": _digest(prompt),
                    "system": "",
                    "system_digest": _digest(""),
                },
            )
            result = self._validate_call_result(
                base._read_json(failed_request.parent / "result.json"), request
            )
            if (
                result["status"] != "error"
                or result["failure_category"] != value["failure_category"]
            ):
                raise ProxyExperimentError("pair retry failure does not match its call")
        else:
            raise ProxyExperimentError("pair attempt status differs")
        return value

    async def _run_pair_attempt(self, pair: Mapping[str, Any], attempt: int) -> Mapping[str, Any]:
        attempt_path = self._attempt_path(str(pair["pair_id"]), attempt)
        if attempt_path.is_file():
            return self._validate_attempt(base._read_json(attempt_path), pair, attempt)
        candidate = self._candidate_by_id[str(pair["candidate_id"])]
        probe = candidate["probes"][str(pair["seed"])]
        arm_references: list[dict[str, Any]] = []
        for arm in pair["arm_order"]:
            arm_path = self._arm_path(str(pair["pair_id"]), attempt, str(arm))
            hint = str(candidate["hints"][str(pair["seed"])]["hint"]) if arm == "hinted" else ""
            prompt = self._prompt(str(probe["observation"]), hint)
            purpose = {
                "phase": "actor",
                "actor_plan_digest": self.actor_plan["actor_plan_digest"],
                "pair_id": pair["pair_id"],
                "pair_ordinal": pair["pair_ordinal"],
                "pair_attempt": attempt,
                "stratum_id": pair["stratum_id"],
                "candidate_id": pair["candidate_id"],
                "seed": pair["seed"],
                "arm": arm,
                "turn": 1,
                "horizon": 1,
            }
            try:
                raw, call_id = await self.call(purpose=purpose, prompt=prompt)
            except _RecordedCallFailure as exc:
                if not exc.retryable:
                    raise ProxyExperimentIncomplete(f"fatal actor transport: {exc}") from exc
                body = {
                    "schema_version": PAIR_ATTEMPT_SCHEMA,
                    "intent_digest": self.intent["intent_digest"],
                    "actor_plan_digest": self.actor_plan["actor_plan_digest"],
                    "pair_id": pair["pair_id"],
                    "pair_ordinal": pair["pair_ordinal"],
                    "pair_attempt": attempt,
                    "status": "retryable_failure",
                    "failure_category": exc.category,
                    "failed_call_id": exc.call_id,
                    "arms": arm_references,
                }
                value = {**body, "attempt_digest": _digest(body)}
                base._write_json(attempt_path, value)
                return self._validate_attempt(value, pair, attempt)
            if arm_path.is_file():
                arm_leaf = self._validate_arm(
                    base._read_json(arm_path), pair=pair, attempt=attempt, arm=str(arm)
                )
            else:
                try:
                    target = self._proofpack_call(
                        self.dependencies.target_factory,
                        candidate["code"],
                        action_format="boxed",
                        max_turns=1,
                        operation_timeout_seconds=float(
                            self.config["qualification_timeout_seconds"]
                        ),
                    )
                    env = self._proofpack_call(target.instantiate)
                    observation, _ = self._proofpack_call(env.reset, seed=int(pair["seed"]))
                    if str(observation) != probe["observation"]:
                        raise ProxyExperimentError("actor reset differs from locked probe")
                    clean_action = live.extract_clean_action(raw, "boxed")
                    step = self._proofpack_call(env.step, clean_action)
                    if not isinstance(step, tuple) or len(step) != 5:
                        raise ProxyExperimentError("actor environment step must return five values")
                    post, reward, terminated, truncated, _info = step
                    if (
                        isinstance(reward, bool)
                        or not isinstance(reward, (int, float))
                        or not math.isfinite(reward)
                    ):
                        raise ProxyExperimentError("actor environment reward is invalid")
                except Exception as exc:
                    raise ProxyExperimentIncomplete(
                        f"fatal actor environment/integrity failure for {pair['pair_id']}/{arm}: {exc}"
                    ) from exc
                arm_body = {
                    "schema_version": "spade-coverage-forced-arm/v1",
                    "intent_digest": self.intent["intent_digest"],
                    "actor_plan_digest": self.actor_plan["actor_plan_digest"],
                    "pair_id": pair["pair_id"],
                    "pair_ordinal": pair["pair_ordinal"],
                    "pair_attempt": attempt,
                    "stratum_id": pair["stratum_id"],
                    "candidate_id": pair["candidate_id"],
                    "seed": pair["seed"],
                    "arm": arm,
                    "call_id": call_id,
                    "raw_response": raw,
                    "raw_response_digest": _digest(raw),
                    "parser_miss": live.extract_boxed_answer(raw) is None,
                    "clean_action": clean_action,
                    "pre_observation": str(observation),
                    "pre_observation_digest": _digest(str(observation)),
                    "post_observation": str(post),
                    "post_observation_digest": _digest(str(post)),
                    "raw_reward": float(reward),
                    "binary_return": 1.0 if float(reward) > 0 else 0.0,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success_rule": "reward-positive-even-if-simultaneously-truncated",
                }
                arm_leaf = {**arm_body, "arm_digest": _digest(arm_body)}
                self._validate_arm(arm_leaf, pair=pair, attempt=attempt, arm=str(arm))
                base._write_json(arm_path, arm_leaf)
            arm_references.append(
                {
                    "arm": arm,
                    "path": str(arm_path.relative_to(self.run_dir)),
                    "arm_digest": arm_leaf["arm_digest"],
                }
            )
        body = {
            "schema_version": PAIR_ATTEMPT_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "pair_attempt": attempt,
            "status": "completed",
            "failure_category": None,
            "failed_call_id": None,
            "arms": arm_references,
        }
        value = {**body, "attempt_digest": _digest(body)}
        base._write_json(attempt_path, value)
        return self._validate_attempt(value, pair, attempt)

    def _resolution(
        self,
        pair: Mapping[str, Any],
        selected_attempt: Mapping[str, Any],
        attempts: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        path = self._resolution_path(str(pair["pair_id"]))
        body = {
            "schema_version": PAIR_RESOLUTION_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "pair": dict(pair),
            "selected_attempt": selected_attempt["pair_attempt"],
            "selected_attempt_digest": selected_attempt["attempt_digest"],
            "attempts": [
                {
                    "pair_attempt": item["pair_attempt"],
                    "status": item["status"],
                    "attempt_digest": item["attempt_digest"],
                }
                for item in attempts
            ],
            "first_attempt_exogenous_failure": attempts[0]["status"] == "retryable_failure",
            "arms": selected_attempt["arms"],
        }
        value = {**body, "resolution_digest": _digest(body)}
        if path.is_file():
            existing = base._read_json(path)
            if existing != value:
                raise ProxyExperimentError("pair resolution differs on resume")
            return self._validate_resolution(existing, pair)
        base._write_json(path, value)
        return self._validate_resolution(value, pair)

    async def run_actor_waves(self) -> list[Mapping[str, Any]]:
        self._preflight_actor_state()
        schedule = self.actor_plan["pair_schedule"]
        unresolved = [dict(item) for item in schedule]
        attempts_by_pair: dict[str, list[Mapping[str, Any]]] = {
            item["pair_id"]: [] for item in schedule
        }
        resolutions: dict[str, Mapping[str, Any]] = {}
        for pair in schedule:
            path = self._resolution_path(str(pair["pair_id"]))
            if path.is_file():
                resolution = self._validate_resolution(base._read_json(path), pair)
                resolutions[str(pair["pair_id"])] = resolution
                unresolved = [item for item in unresolved if item["pair_id"] != pair["pair_id"]]
        for wave in (1, 2, 3):
            if not unresolved:
                break
            if wave == 3 and len(unresolved) > THIRD_WAVE_LIMIT:
                raise ProxyExperimentIncomplete(
                    f"third actor wave forbidden with {len(unresolved)} unresolved pairs"
                )
            next_unresolved: list[dict[str, Any]] = []
            for pair in unresolved:
                pair_id = str(pair["pair_id"])
                attempt = await self._run_pair_attempt(pair, wave)
                attempts_by_pair[pair_id].append(attempt)
                if attempt["status"] == "completed":
                    resolutions[pair_id] = self._resolution(
                        pair, attempt, attempts_by_pair[pair_id]
                    )
                else:
                    next_unresolved.append(pair)
            unresolved = next_unresolved
        if unresolved:
            raise ProxyExperimentIncomplete(
                f"{len(unresolved)} pairs remain unresolved after all permitted waves"
            )
        ordered = [resolutions[str(pair["pair_id"])] for pair in schedule]
        if len(ordered) != PAIR_COUNT:
            raise ProxyExperimentError("actor resolution panel is incomplete")
        self._preflight_actor_state()
        actor_requests = [
            base._read_json(path)
            for path in (self.run_dir / "calls").glob("*/request.json")
            if base._read_json(path).get("purpose", {}).get("phase") == "actor"
        ]
        if len(actor_requests) > ACTOR_CALL_CEILING:
            raise CallCapExceeded("actor calls exceed the sealed 172-call ceiling")
        return ordered

    def _paired_outcomes(self, resolutions: Sequence[Mapping[str, Any]]) -> list[PairedOutcome]:
        outcomes: list[PairedOutcome] = []
        for resolution in resolutions:
            pair = resolution["pair"]
            arm_values: dict[str, Mapping[str, Any]] = {}
            for reference in resolution["arms"]:
                arm_values[str(reference["arm"])] = self._validate_arm(
                    base._read_json(self.run_dir / str(reference["path"])),
                    pair=pair,
                    attempt=int(resolution["selected_attempt"]),
                    arm=str(reference["arm"]),
                )
            outcomes.append(
                PairedOutcome(
                    candidate_id=str(pair["candidate_id"]),
                    stratum_id=str(pair["stratum_id"]),
                    seed=int(pair["seed"]),
                    unhinted=float(arm_values["unhinted"]["binary_return"]),
                    hinted=float(arm_values["hinted"]["binary_return"]),
                    first_attempt_exogenous_failure=bool(
                        resolution["first_attempt_exogenous_failure"]
                    ),
                    parser_failure=bool(
                        arm_values["unhinted"]["parser_miss"] or arm_values["hinted"]["parser_miss"]
                    ),
                    task_failure=False,
                )
            )
        return outcomes

    def aggregate(self, resolutions: Sequence[Mapping[str, Any]]) -> Path:
        _validate_runtime_identity(self.intent["runtime_identity"])
        self._validate_complete_actor_closure(resolutions)
        outcomes = self._paired_outcomes(resolutions)
        locks = tuple(
            LockedPortfolios(
                stratum_id=str(item["stratum_id"]),
                candidate_ids=tuple(item["candidate_ids"]),
                coverage_forced=tuple(item["coverage_forced"]),
                redundant_historical=tuple(item["redundant_historical"]),
                challenger_id=str(item["challenger_id"]),
                retained_historical_id=str(item["retained_historical_id"]),
                displaced_historical_id=str(item["displaced_historical_id"]),
                coverage_forced_quality=float(item["coverage_forced_quality"]),
                redundant_historical_quality=float(item["redundant_historical_quality"]),
                signed_quality_gap=float(item["signed_quality_gap"]),
                absolute_quality_gap=float(item["absolute_quality_gap"]),
            )
            for item in self.actor_plan["portfolios"]
        )
        gates = self.intent["analysis_gates"]
        analysis = analyze_proxy_pilot(
            locks,
            outcomes,
            baseline_min=float(gates["pooled_unhinted_min"]),
            baseline_max=float(gates["pooled_unhinted_max"]),
            minimum_discordant=int(gates["minimum_discordant_pairs"]),
            minimum_delta=float(gates["minimum_coverage_forced_delta"]),
            alpha=float(gates["one_sided_label_permutation_alpha"]),
            maximum_first_attempt_failure_rate=float(
                gates["maximum_first_attempt_exogenous_failure_rate"]
            ),
        )
        evidence_inventory = self._complete_evidence_inventory()
        body = {
            "schema_version": AGGREGATE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "analysis_role": self.intent["analysis_role"],
            "claim_exclusions": self.intent["claim_exclusions"],
            "provider": "agy",
            "model": ACTOR_MODEL,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "estimand": (
                "association over locked realized environment plus source-specific hint packages; "
                "environment effects are not isolated from hint quality, epoch, or source arm"
            ),
            "resolution_digests": [item["resolution_digest"] for item in resolutions],
            "portfolios": self.actor_plan["portfolios"],
            "portfolio_quality_diagnostics": self.actor_plan["portfolio_quality_diagnostics"],
            "outcomes": [_plain(item) for item in outcomes],
            "analysis": analysis.to_dict(),
            "analysis_interpretation": (
                "exploratory quality-matched coverage-forced portfolio-swap association; exact "
                "label-permutation and sign-flip sensitivity analyses under strong assumptions; "
                "neither is design-based and there is no causal or learner-improvement claim"
            ),
            "new_charged_calls": self.call_count,
            "global_charged_calls": PRIOR_CHARGED_CALLS + self.call_count,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "global_budget_governance_scope": self.intent["budget"]["governance_scope"],
            "evidence_inventory": evidence_inventory,
            "evidence_root_digest": _digest(evidence_inventory),
            "assay_decision": "not-run-not-applicable",
            "release_authorized": False,
            "model_lock_status": "absent",
        }
        value = {**body, "aggregate_digest": _digest(body)}
        path = self.run_dir / "aggregate.json"
        base._write_json(path, value)
        self._validate_aggregate(base._read_json(path), resolutions)
        return path

    def _validate_complete_actor_closure(self, resolutions: Sequence[Mapping[str, Any]]) -> None:
        if len(resolutions) != PAIR_COUNT:
            raise ProxyExperimentError("complete actor closure requires 36 resolutions")
        referenced_calls: set[str] = set()
        expected_resolution_digests: list[str] = []
        for pair, resolution in zip(self.actor_plan["pair_schedule"], resolutions):
            checked = self._validate_resolution(resolution, pair)
            expected_resolution_digests.append(str(checked["resolution_digest"]))
            for attempt_ref in checked["attempts"]:
                attempt = self._validate_attempt(
                    base._read_json(
                        self._attempt_path(str(pair["pair_id"]), int(attempt_ref["pair_attempt"]))
                    ),
                    pair,
                    int(attempt_ref["pair_attempt"]),
                )
                for arm_ref in attempt["arms"]:
                    arm_leaf = base._read_json(self.run_dir / str(arm_ref["path"]))
                    referenced_calls.add(str(arm_leaf["call_id"]))
                if attempt["failed_call_id"] is not None:
                    referenced_calls.add(str(attempt["failed_call_id"]))
        actor_requests: set[str] = set()
        calls_root = self.run_dir / "calls"
        for path in calls_root.glob("*/request.json") if calls_root.is_dir() else ():
            request = self._validate_call_request(base._read_json(path))
            if request["purpose"]["phase"] == "actor":
                actor_requests.add(str(request["call_id"]))
        if actor_requests != referenced_calls:
            raise ProxyExperimentError("complete actor evidence has orphaned or missing calls")
        if len(actor_requests) > ACTOR_CALL_CEILING:
            raise CallCapExceeded("complete actor closure exceeds 172 calls")

    def _complete_evidence_inventory(self) -> list[dict[str, Any]]:
        if any(
            path.name == "model.lock"
            for root in (self.run_dir, self.ledger_root)
            for path in root.rglob("*")
        ):
            raise ProxyExperimentError("physical model.lock is forbidden for the proxy pilot")
        records: list[dict[str, Any]] = []
        roots = (("run", self.run_dir), ("shared-ledger", self.ledger_root))
        for namespace, root in roots:
            for current, directories, files in os.walk(root, topdown=True, followlinks=False):
                current_path = Path(current)
                for name in (*directories, *files):
                    if (current_path / name).is_symlink():
                        raise ProxyExperimentError("symlink found while rooting final evidence")
                for name in files:
                    path = current_path / name
                    if path == root / ".writer.lock" or (
                        namespace == "run" and path == self.run_dir / "aggregate.json"
                    ):
                        continue
                    if not path.is_file():
                        raise ProxyExperimentError("non-regular final evidence leaf")
                    content = path.read_bytes()
                    records.append(
                        {
                            "namespace": namespace,
                            "path": path.relative_to(root).as_posix(),
                            "digest": _bytes_digest(content),
                            "size_bytes": len(content),
                        }
                    )
        return sorted(records, key=lambda item: (item["namespace"], item["path"]))

    def _validate_aggregate(
        self, value: Mapping[str, Any], resolutions: Sequence[Mapping[str, Any]]
    ) -> None:
        required = {
            "schema_version",
            "protocol_id",
            "intent_digest",
            "actor_plan_digest",
            "analysis_role",
            "claim_exclusions",
            "provider",
            "model",
            "backend_identity_attested",
            "route_authority",
            "estimand",
            "resolution_digests",
            "portfolios",
            "portfolio_quality_diagnostics",
            "outcomes",
            "analysis",
            "analysis_interpretation",
            "new_charged_calls",
            "global_charged_calls",
            "authorized_global_call_cap",
            "global_budget_governance_scope",
            "evidence_inventory",
            "evidence_root_digest",
            "assay_decision",
            "release_authorized",
            "model_lock_status",
            "aggregate_digest",
        }
        if set(value) != required or value.get("schema_version") != AGGREGATE_SCHEMA:
            raise ProxyExperimentError("aggregate fields/schema differ")
        _sealed_artifact(value, "aggregate_digest")
        fixed = {
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "actor_plan_digest": self.actor_plan["actor_plan_digest"],
            "analysis_role": self.intent["analysis_role"],
            "claim_exclusions": self.intent["claim_exclusions"],
            "provider": "agy",
            "model": ACTOR_MODEL,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "estimand": (
                "association over locked realized environment plus source-specific hint packages; "
                "environment effects are not isolated from hint quality, epoch, or source arm"
            ),
            "resolution_digests": [item["resolution_digest"] for item in resolutions],
            "portfolios": self.actor_plan["portfolios"],
            "portfolio_quality_diagnostics": self.actor_plan["portfolio_quality_diagnostics"],
            "analysis_interpretation": (
                "exploratory quality-matched coverage-forced portfolio-swap association; exact "
                "label-permutation and sign-flip sensitivity analyses under strong assumptions; "
                "neither is design-based and there is no causal or learner-improvement claim"
            ),
            "new_charged_calls": self.call_count,
            "global_charged_calls": PRIOR_CHARGED_CALLS + self.call_count,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "global_budget_governance_scope": self.intent["budget"]["governance_scope"],
            "assay_decision": "not-run-not-applicable",
            "release_authorized": False,
            "model_lock_status": "absent",
        }
        if any(value.get(key) != item for key, item in fixed.items()):
            raise ProxyExperimentError("aggregate identity/release boundary differs")
        inventory = self._complete_evidence_inventory()
        if value.get("evidence_inventory") != inventory or value.get(
            "evidence_root_digest"
        ) != _digest(inventory):
            raise ProxyExperimentError("aggregate evidence root differs from disk")
        if not isinstance(value.get("outcomes"), list) or len(value["outcomes"]) != PAIR_COUNT:
            raise ProxyExperimentError("aggregate paired outcome panel is incomplete")
        expected_outcomes = [_plain(item) for item in self._paired_outcomes(resolutions)]
        if value["outcomes"] != expected_outcomes:
            raise ProxyExperimentError("aggregate outcomes differ from resolved actor evidence")
        try:
            outcomes = [PairedOutcome(**item) for item in value["outcomes"]]
            locks = tuple(
                LockedPortfolios(
                    stratum_id=str(item["stratum_id"]),
                    candidate_ids=tuple(item["candidate_ids"]),
                    coverage_forced=tuple(item["coverage_forced"]),
                    redundant_historical=tuple(item["redundant_historical"]),
                    challenger_id=str(item["challenger_id"]),
                    retained_historical_id=str(item["retained_historical_id"]),
                    displaced_historical_id=str(item["displaced_historical_id"]),
                    coverage_forced_quality=float(item["coverage_forced_quality"]),
                    redundant_historical_quality=float(item["redundant_historical_quality"]),
                    signed_quality_gap=float(item["signed_quality_gap"]),
                    absolute_quality_gap=float(item["absolute_quality_gap"]),
                )
                for item in self.actor_plan["portfolios"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProxyExperimentError("aggregate outcomes/portfolios are invalid") from exc
        gates = self.intent["analysis_gates"]
        recomputed = analyze_proxy_pilot(
            locks,
            outcomes,
            baseline_min=float(gates["pooled_unhinted_min"]),
            baseline_max=float(gates["pooled_unhinted_max"]),
            minimum_discordant=int(gates["minimum_discordant_pairs"]),
            minimum_delta=float(gates["minimum_coverage_forced_delta"]),
            alpha=float(gates["one_sided_label_permutation_alpha"]),
            maximum_first_attempt_failure_rate=float(
                gates["maximum_first_attempt_exogenous_failure_rate"]
            ),
        ).to_dict()
        if value.get("analysis") != recomputed:
            raise ProxyExperimentError(
                "aggregate analysis differs from deterministic recomputation"
            )


async def prepare_experiment(
    intent_path: Path | str,
    *,
    execute: bool = False,
    acknowledged_new_call_cap: int | None = None,
    dependencies: RunnerDependencies | None = None,
) -> RunResult:
    """Validate the generation intent or create the concrete pre-actor seal."""
    path = Path(intent_path)
    intent = load_generation_intent(path)
    run_dir = derive_run_dir(intent)
    actor_plan_path = run_dir / "actor-plan.json"
    if not execute:
        if actor_plan_path.is_file():
            validate_actor_plan(base._read_json(actor_plan_path), intent, run_dir=run_dir)
        return RunResult(
            status="validated",
            intent_digest=str(intent["intent_digest"]),
            run_dir=run_dir,
            call_count=0,
            actor_plan_path=actor_plan_path if actor_plan_path.is_file() else None,
        )
    if acknowledged_new_call_cap != NEW_CALL_CAP:
        raise ProxyExperimentError(f"--acknowledge-new-call-cap must exactly equal {NEW_CALL_CAP}")
    resolved = dependencies or _default_dependencies(intent)
    engine = _Engine(intent, base._pretty_json(intent), run_dir, resolved)
    try:
        with base._single_writer(run_dir):
            engine.initialize()
            actor_plan_path = await engine.prepare_actor_plan()
            engine._validate_existing_calls()
    except base.ExperimentError as exc:
        raise ProxyExperimentError(str(exc)) from exc
    return RunResult(
        status="actor-plan-sealed",
        intent_digest=str(intent["intent_digest"]),
        run_dir=run_dir,
        call_count=engine.call_count,
        actor_plan_path=actor_plan_path,
    )


async def run_experiment(
    intent_path: Path | str,
    *,
    execute: bool = False,
    acknowledged_new_call_cap: int | None = None,
    dependencies: RunnerDependencies | None = None,
) -> RunResult:
    """Validate or execute the actor stage from an already sealed actor plan."""
    path = Path(intent_path)
    intent = load_generation_intent(path)
    run_dir = derive_run_dir(intent)
    actor_plan_path = run_dir / "actor-plan.json"
    if not actor_plan_path.is_file():
        raise ProxyExperimentError("prepare must seal actor-plan.json before actor execution")
    actor_plan = validate_actor_plan(base._read_json(actor_plan_path), intent, run_dir=run_dir)
    if not execute:
        return RunResult(
            status="validated",
            intent_digest=str(intent["intent_digest"]),
            run_dir=run_dir,
            call_count=0,
            actor_plan_path=actor_plan_path,
        )
    if acknowledged_new_call_cap != NEW_CALL_CAP:
        raise ProxyExperimentError(f"--acknowledge-new-call-cap must exactly equal {NEW_CALL_CAP}")
    resolved = dependencies or _default_dependencies(intent)
    engine = _ActorEngine(
        intent,
        base._pretty_json(intent),
        run_dir,
        resolved,
        actor_plan,
    )
    try:
        with base._single_writer(run_dir):
            engine.initialize()
            resolutions = await engine.run_actor_waves()
            aggregate_path = engine.aggregate(resolutions)
            engine._validate_existing_calls()
    except base.ExperimentError as exc:
        raise ProxyExperimentError(str(exc)) from exc
    return RunResult(
        status="complete",
        intent_digest=str(intent["intent_digest"]),
        run_dir=run_dir,
        call_count=engine.call_count,
        actor_plan_path=actor_plan_path,
        aggregate_path=aggregate_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    intent = commands.add_parser("intent-plan", help="seal response-independent generation intent")
    intent.add_argument("--experiment-id", required=True)
    intent.add_argument("--v3-run", required=True)
    intent.add_argument("--v4-run", required=True)
    intent.add_argument("--output-root", required=True)
    intent.add_argument("--shared-ledger-root", required=True)
    intent.add_argument("--output", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="generate exact-cell challengers, quality-match portfolios, and seal actor plan",
    )
    prepare.add_argument("--intent", required=True)
    prepare.add_argument("--execute", action="store_true")
    prepare.add_argument("--acknowledge-new-call-cap", type=int)
    run = commands.add_parser(
        "run", help="execute the sealed noncausal coverage-forced matched-swap panel"
    )
    run.add_argument("--intent", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--acknowledge-new-call-cap", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "intent-plan":
            value = build_generation_intent(
                experiment_id=args.experiment_id,
                v3_run=Path(args.v3_run),
                v4_run=Path(args.v4_run),
                output_root=Path(args.output_root),
                shared_ledger_root=Path(args.shared_ledger_root),
            )
            base._write_json(Path(args.output), value)
            print(value["intent_digest"])
            return 0
        if args.command == "prepare":
            result = asyncio.run(
                prepare_experiment(
                    args.intent,
                    execute=args.execute,
                    acknowledged_new_call_cap=args.acknowledge_new_call_cap,
                )
            )
        else:
            result = asyncio.run(
                run_experiment(
                    args.intent,
                    execute=args.execute,
                    acknowledged_new_call_cap=args.acknowledge_new_call_cap,
                )
            )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "intent_digest": result.intent_digest,
                    "run_dir": str(result.run_dir),
                    "call_count": result.call_count,
                    "actor_plan_path": (
                        str(result.actor_plan_path) if result.actor_plan_path else None
                    ),
                    "aggregate_path": str(result.aggregate_path) if result.aggregate_path else None,
                },
                sort_keys=True,
            )
        )
        return 0
    except (ProxyExperimentError, base.ExperimentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
