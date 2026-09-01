#!/usr/bin/env python3
"""Run the sealed, offline counterfactual-witness falsification experiment.

The experiment consumes only the allowlisted pre-outcome evidence from the
authorized Google-v4 SPADE cohort.  Candidate source is executed exclusively by
ProofPack's macOS sandbox worker.  No AGY, provider, actor, or Assay call is
available from this runner.

The protocol is intentionally two-phase:

* ``plan`` validates and seals the source cohort, generated variants, probes,
  runtime, and exact sandbox-operation ceiling before any response is observed;
* ``run`` revalidates the seal, executes every source/probe pair twice, and
  writes immutable per-probe leaves followed by certificates and one aggregate
  decision.

Passing this exploratory benchmark establishes only that the witness
representation detects operator-held-out mutations efficiently without
rejecting the admitted controls.  The held-out operators share observable
channels with the training catalog, so this is not an independent semantic
generalization test and does not establish downstream learner improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spade.core.counterfactual_witness import (  # noqa: E402
    SourceVariant,
    TraceSignature,
    WitnessMatrix,
    WitnessProbe,
    build_candidate_probes,
    build_certificate,
    generate_source_variants,
    score_probe_ids,
    select_witnesses,
    trace_signature,
)
from spade.core.witness_archive import behavior_descriptor  # noqa: E402
from tools import run_spade_agy_experiment as base  # noqa: E402
from tools import run_spade_agy_outcome_replay as source_import  # noqa: E402


PLAN_SCHEMA = "spade-counterfactual-witness-plan/v1"
RUN_SCHEMA = "spade-counterfactual-witness-run/v1"
TRACE_LEAF_SCHEMA = "spade-counterfactual-witness-trace/v1"
OPERATION_REQUEST_SCHEMA = "spade-counterfactual-witness-operation-request/v1"
OPERATION_RESULT_SCHEMA = "spade-counterfactual-witness-operation-result/v1"
CLUSTER_RESULT_SCHEMA = "spade-counterfactual-witness-cluster-result/v1"
AGGREGATE_SCHEMA = "spade-counterfactual-witness-aggregate/v1"
PROTOCOL_ID = "spade-counterfactual-witness-falsification/v1"

AUTHORIZED_SOURCE_PLAN_DIGEST = (
    "sha256:8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc"
)
AUTHORIZED_SOURCE_COHORT_DIGEST = (
    "sha256:161353ebd4454516e3379414444323dd13aeab95640eb130ec7414f23876b84b"
)
AUTHORIZED_CLUSTERS = 18
AUTHORIZED_SEEDS = (0, 1, 42)
ACTION_FORMAT = "boxed"
MAX_TURNS = 5
DEFAULT_OPERATION_TIMEOUT_SECONDS = 5.0
DEFAULT_WITNESS_BUDGET = 16
DEFAULT_RANDOM_BASELINE_DRAWS = 512
DEFAULT_REPETITIONS = 2
PRIMARY_RECALL_MARGIN = 0.15
MAX_EQUIVALENT_FALSE_REJECTION_RATE = 0.05
MIN_TRAIN_RECALL = 0.90
MIN_SAFE_BANK_HELDOUT_RECALL = 0.90
EARLY_TERMINATION_ERROR = "WorkerError: action sequence continues after the episode ended"


class WitnessExperimentError(RuntimeError):
    """Fail-closed protocol or evidence error."""


class _ExecutionTarget(Protocol):
    def inspect(self, seed: int, *, timeout_seconds: float | None = None) -> Any: ...

    def run_actions(
        self,
        seed: int | None,
        actions: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> Any: ...

    def run_oracle(self, seed: int = 0, timeout_seconds: float | None = None) -> Any: ...


@dataclass(frozen=True)
class RunnerDependencies:
    """Injected offline boundaries used by tests and the pinned production run."""

    load_source_snapshot: Callable[[Path], Any]
    target_factory: Callable[..., _ExecutionTarget]
    runtime_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    """Result of plan validation or execution."""

    status: str
    plan_digest: str
    run_dir: Path
    aggregate_path: Path | None = None


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    return _bytes_digest(path.read_bytes())


def _plain(value: Any) -> Any:
    """Convert supported dataclass/protocol values into canonical JSON data."""
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
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WitnessExperimentError("non-finite number cannot enter witness evidence")
        return value
    raise WitnessExperimentError(f"non-JSON witness value: {type(value).__name__}")


def _required_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise WitnessExperimentError(f"{where} must be non-empty text")
    return value


def _sha256_text(value: object, where: str) -> str:
    text = _required_text(value, where)
    if not text.startswith("sha256:") or len(text) != 71:
        raise WitnessExperimentError(f"{where} must be a sha256 digest")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise WitnessExperimentError(f"{where} must be a sha256 digest") from exc
    return text


def _canonical_absolute_dir(value: Path | str, where: str, *, must_exist: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise WitnessExperimentError(f"{where} must be absolute")
    base._reject_symlink_ancestors(path)
    resolved = path.resolve(strict=False)
    if path != resolved or path.is_symlink():
        raise WitnessExperimentError(f"{where} must be a canonical non-symlink path")
    if must_exist and not path.is_dir():
        raise WitnessExperimentError(f"{where} does not exist")
    if not must_exist and path.exists() and not path.is_dir():
        raise WitnessExperimentError(f"{where} exists but is not a directory")
    return path


def _git_root(path: Path) -> Path:
    start = path.parent if path.is_file() else path
    try:
        text = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise WitnessExperimentError(f"cannot resolve git root for {path}") from exc
    return Path(text).resolve()


def _git_head(root: Path) -> str:
    try:
        text = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise WitnessExperimentError(f"cannot resolve git revision for {root}") from exc
    if len(text) != 40:
        raise WitnessExperimentError(f"invalid git revision for {root}")
    return text


def _require_tracked_clean(root: Path) -> None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise WitnessExperimentError(f"cannot inspect worktree state for {root}") from exc
    if output:
        raise WitnessExperimentError(f"tracked source worktree is dirty: {root}")


def _module_file(module: Any, where: str) -> Path:
    source = inspect.getsourcefile(module)
    if source is None:
        raise WitnessExperimentError(f"cannot locate {where} source")
    path = Path(source).resolve()
    base._reject_symlink_ancestors(path)
    if path.is_symlink() or not path.is_file():
        raise WitnessExperimentError(f"unsafe {where} source path")
    return path


def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise WitnessExperimentError("provider and Assay boundaries are forbidden in witness mode")


async def _forbidden_async(*_args: Any, **_kwargs: Any) -> str:
    raise WitnessExperimentError("provider calls are forbidden in witness mode")


def _default_dependencies(source_run_dir: Path) -> RunnerDependencies:
    """Resolve only the pinned ProofPack boundary; never resolve AGY or Assay."""
    source_plan = base.load_plan(source_run_dir / "plan.json")
    expected_qualifier = Path(
        source_plan["runtime_identity"]["imported_sources"]["proofpack_qualifier"]["path"]
    ).resolve()

    from proofpack_env import spade_qualification, spade_target, spade_worker
    from proofpack_env.spade_qualification import qualify_spade_environment
    from proofpack_env.spade_target import SpadeEnvironmentTarget

    qualifier_path = _module_file(spade_qualification, "ProofPack qualifier")
    target_path = _module_file(spade_target, "ProofPack target")
    worker_path = _module_file(spade_worker, "ProofPack worker")
    launcher_path = worker_path.with_name("spade_launcher.py")
    if not launcher_path.is_file() or launcher_path.is_symlink():
        raise WitnessExperimentError("ProofPack sandbox launcher is unavailable or unsafe")
    if qualifier_path != expected_qualifier:
        raise WitnessExperimentError(
            "imported ProofPack qualifier is not the source cohort's pinned checkout"
        )
    proofpack_root = _git_root(qualifier_path)
    if _git_head(proofpack_root) != source_plan["source_revisions"]["proofpack"]:
        raise WitnessExperimentError("imported ProofPack revision differs from source cohort")
    _require_tracked_clean(proofpack_root)

    source_dependencies = base.RunnerDependencies(
        llm_call=_forbidden_async,
        qualify=qualify_spade_environment,
        target_factory=SpadeEnvironmentTarget,
        assay_writer=_forbidden,
        task_factory=_forbidden,
        cluster_factory=_forbidden,
        run_metadata_factory=_forbidden,
        client_or_bin=None,
        source_revisions=source_plan["source_revisions"],
        runtime_identity=source_plan["runtime_identity"],
    )

    def load_snapshot(path: Path) -> Any:
        return source_import._validate_source_snapshot(path, dependencies=source_dependencies)

    spade_root = _git_root(ROOT_DIR)
    _require_tracked_clean(spade_root)
    python_path = Path(sys.executable).resolve()
    runtime_files = {
        "runner": Path(__file__).resolve(),
        "counterfactual_witness": ROOT_DIR / "spade" / "core" / "counterfactual_witness.py",
        "witness_archive": ROOT_DIR / "spade" / "core" / "witness_archive.py",
        "source_importer": Path(source_import.__file__).resolve(),
        "proofpack_qualifier": qualifier_path,
        "proofpack_target": target_path,
        "proofpack_worker": worker_path,
        "proofpack_launcher": launcher_path,
    }
    runtime_identity = {
        "python_executable": str(python_path),
        "python_executable_digest": _file_digest(python_path),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "spade_revision": _git_head(spade_root),
        "proofpack_revision": _git_head(proofpack_root),
        "files": {
            name: {"path": str(path), "digest": _file_digest(path)}
            for name, path in sorted(runtime_files.items())
        },
        "operation_runtime_files": ["proofpack_launcher", "proofpack_worker"],
        "execution_boundary": "macos-sandbox-exec-worker/v1",
        "network_or_provider_calls": False,
    }
    return RunnerDependencies(
        load_source_snapshot=load_snapshot,
        target_factory=SpadeEnvironmentTarget,
        runtime_identity=runtime_identity,
    )


def _validate_runtime_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WitnessExperimentError("runtime_identity must be an object")
    expected = {
        "python_executable",
        "python_executable_digest",
        "python_version",
        "spade_revision",
        "proofpack_revision",
        "files",
        "operation_runtime_files",
        "execution_boundary",
        "network_or_provider_calls",
    }
    if set(value) != expected:
        raise WitnessExperimentError("runtime_identity fields differ from schema")
    _sha256_text(value["python_executable_digest"], "python executable digest")
    if value["execution_boundary"] != "macos-sandbox-exec-worker/v1":
        raise WitnessExperimentError("unsupported witness execution boundary")
    if value["network_or_provider_calls"] is not False:
        raise WitnessExperimentError("witness runtime must forbid network/provider calls")
    files = value["files"]
    if not isinstance(files, dict) or not files:
        raise WitnessExperimentError("runtime files must be a non-empty object")
    for name, record in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, dict)
            or set(record)
            != {
                "path",
                "digest",
            }
        ):
            raise WitnessExperimentError("runtime file record is malformed")
        _required_text(record["path"], f"runtime file {name} path")
        _sha256_text(record["digest"], f"runtime file {name} digest")
    operation_files = value["operation_runtime_files"]
    if (
        not isinstance(operation_files, list)
        or not operation_files
        or len(operation_files) != len(set(operation_files))
        or any(not isinstance(name, str) or name not in files for name in operation_files)
    ):
        raise WitnessExperimentError("operation runtime files are empty, duplicated, or unknown")
    return value


def _revalidate_runtime_identity(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if _validate_runtime_identity(dict(expected)) != _validate_runtime_identity(dict(observed)):
        raise WitnessExperimentError("witness runtime identity drifted")
    python_path = Path(str(expected["python_executable"]))
    if python_path.resolve() != python_path or not python_path.is_file():
        raise WitnessExperimentError("sealed Python executable is unavailable")
    if _file_digest(python_path) != expected["python_executable_digest"]:
        raise WitnessExperimentError("sealed Python executable bytes drifted")
    for record in expected["files"].values():
        path = Path(str(record["path"]))
        base._reject_symlink_ancestors(path)
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise WitnessExperimentError(f"sealed runtime file is unavailable: {path}")
        if _file_digest(path) != record["digest"]:
            raise WitnessExperimentError(f"sealed runtime file bytes drifted: {path}")


def _revalidate_operation_runtime(expected: Mapping[str, Any]) -> None:
    """Rehash disk-loaded sandbox components immediately before every replay."""
    validated = _validate_runtime_identity(dict(expected))
    files = validated["files"]
    for name in validated["operation_runtime_files"]:
        record = files[name]
        path = Path(str(record["path"]))
        base._reject_symlink_ancestors(path)
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise WitnessExperimentError(f"operation runtime file is unavailable: {path}")
        if _file_digest(path) != record["digest"]:
            raise WitnessExperimentError(f"operation runtime file drifted: {path}")


def _variant_id(variant: SourceVariant) -> str:
    value = getattr(variant, "variant_id", None)
    if value is None:
        value = getattr(variant, "source_digest", None)
    return _required_text(value, "variant id")


def _variant_source(variant: SourceVariant) -> str:
    return _required_text(getattr(variant, "source", None), "variant source")


def _variant_catalog_record(variant: SourceVariant) -> dict[str, Any]:
    record = _plain(variant)
    if not isinstance(record, dict):
        raise WitnessExperimentError("variant must serialize as an object")
    source = record.pop("source", None)
    if source is None:
        source = _variant_source(variant)
    record["variant_id"] = _variant_id(variant)
    record["source_digest"] = _bytes_digest(str(source).encode("utf-8"))
    return record


def _probe_id(probe: WitnessProbe) -> str:
    value = getattr(probe, "probe_id", None)
    if value is None:
        value = getattr(probe, "trace_id", None)
    return _required_text(value, "probe id")


def _probe_record(probe: WitnessProbe) -> dict[str, Any]:
    record = _plain(probe)
    if not isinstance(record, dict):
        raise WitnessExperimentError("probe must serialize as an object")
    record["probe_id"] = _probe_id(probe)
    return record


def _semantic_families(variants: Sequence[SourceVariant], *, partition: str) -> list[str]:
    values = {
        _required_text(getattr(variant, "family", None), "semantic mutant family")
        for variant in variants
        if getattr(variant, "kind", None) == "semantic_mutant"
        and getattr(variant, "partition", None) == partition
    }
    if not values:
        raise WitnessExperimentError(f"semantic {partition} mutation families are required")
    return sorted(values)


def _control_operators(variants: Sequence[SourceVariant], *, partition: str) -> list[str]:
    values = {
        _required_text(getattr(variant, "operator", None), "equivalent control operator")
        for variant in variants
        if getattr(variant, "kind", None) == "equivalent_control"
        and getattr(variant, "partition", None) == partition
    }
    if not values:
        raise WitnessExperimentError(f"equivalent {partition} controls are required")
    return sorted(values)


def _oracle_actions(selection: Mapping[str, Any], seeds: Sequence[int]) -> dict[int, list[str]]:
    probes = selection.get("probes")
    if not isinstance(probes, Mapping):
        raise WitnessExperimentError("selection lacks locked probes")
    values: dict[int, list[str]] = {}
    for seed in seeds:
        probe = probes.get(str(seed))
        if not isinstance(probe, Mapping) or "solution" not in probe:
            raise WitnessExperimentError(f"selection lacks locked solution for seed {seed}")
        values[seed] = source_import._boxed_oracle_actions(probe["solution"])
    return values


def _build_cluster_catalog(
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], list[SourceVariant]]:
    code = _required_text(selection.get("code"), "selected environment source")
    code_digest = _digest(code)
    if code_digest != selection.get("code_digest"):
        raise WitnessExperimentError("selected environment source digest mismatch")
    environment_digest = _bytes_digest(code.encode("utf-8"))
    if environment_digest != selection.get("environment_digest"):
        raise WitnessExperimentError("selected environment raw source digest mismatch")
    variants = list(generate_source_variants(code))
    variant_ids = [_variant_id(variant) for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise WitnessExperimentError("variant generator produced duplicate identifiers")
    probes = list(
        build_candidate_probes(
            action_format=ACTION_FORMAT,
            seeds=list(AUTHORIZED_SEEDS),
            base_oracle_actions=_oracle_actions(selection, AUTHORIZED_SEEDS),
            max_turns=MAX_TURNS,
        )
    )
    probe_ids = [_probe_id(probe) for probe in probes]
    if len(probe_ids) != len(set(probe_ids)):
        raise WitnessExperimentError("probe generator produced duplicate identifiers")
    if not probes:
        raise WitnessExperimentError("probe generator produced an empty panel")
    record = {
        "cluster_id": _required_text(selection.get("cluster_id"), "cluster id"),
        "candidate_id": _required_text(selection.get("candidate_id"), "candidate id"),
        "skill": _required_text(selection.get("skill"), "skill"),
        "difficulty": _required_text(selection.get("difficulty"), "difficulty"),
        "environment_name": _required_text(selection.get("environment_name"), "environment name"),
        "code_digest": code_digest,
        "environment_digest": environment_digest,
        "qualification_digest": _sha256_text(
            selection.get("qualification_digest"), "qualification digest"
        ),
        "variants": [_variant_catalog_record(variant) for variant in variants],
        "probes": [_probe_record(probe) for probe in probes],
    }
    return record, variants


def build_plan(
    *,
    source_run_dir: Path | str,
    output_root: Path | str,
    witness_budget: int = DEFAULT_WITNESS_BUDGET,
    repetitions: int = DEFAULT_REPETITIONS,
    random_baseline_draws: int = DEFAULT_RANDOM_BASELINE_DRAWS,
    operation_timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS,
    dependencies: RunnerDependencies | None = None,
) -> dict[str, Any]:
    """Validate the source cohort and seal a response-free witness plan."""
    source_run = _canonical_absolute_dir(source_run_dir, "source_run_dir", must_exist=True)
    output = _canonical_absolute_dir(output_root, "output_root", must_exist=False)
    if type(witness_budget) is not int or not 1 <= witness_budget <= 64:
        raise WitnessExperimentError("witness_budget must be within 1..64")
    if type(repetitions) is not int or repetitions != DEFAULT_REPETITIONS:
        raise WitnessExperimentError("the v1 protocol requires exactly two repetitions")
    if type(random_baseline_draws) is not int or not 1 <= random_baseline_draws <= 10_000:
        raise WitnessExperimentError("random_baseline_draws must be within 1..10000")
    if (
        isinstance(operation_timeout_seconds, bool)
        or not isinstance(operation_timeout_seconds, (int, float))
        or not math.isfinite(float(operation_timeout_seconds))
        or not 0 < float(operation_timeout_seconds) <= 3_600
    ):
        raise WitnessExperimentError("operation_timeout_seconds must be finite in (0,3600]")

    resolved = dependencies or _default_dependencies(source_run)
    snapshot = resolved.load_source_snapshot(source_run)
    if snapshot.plan["plan_digest"] != AUTHORIZED_SOURCE_PLAN_DIGEST:
        raise WitnessExperimentError("source plan is not the authorized Google-v4 plan")
    if snapshot.cohort["cohort_digest"] != AUTHORIZED_SOURCE_COHORT_DIGEST:
        raise WitnessExperimentError("source cohort is not the authorized Google-v4 cohort")
    if len(snapshot.selections) != AUTHORIZED_CLUSTERS:
        raise WitnessExperimentError("authorized source cohort must contain exactly 18 selections")

    clusters: list[dict[str, Any]] = []
    family_catalog: dict[str, list[str]] | None = None
    operation_ceiling = 0
    for schedule_ordinal, scheduled in enumerate(snapshot.plan["cluster_schedule"], start=1):
        cluster_id = str(scheduled["cluster_id"])
        selection = snapshot.selections[cluster_id]
        cluster, variants = _build_cluster_catalog(selection)
        observed_catalog = {
            "training_semantic": _semantic_families(variants, partition="train"),
            "heldout_semantic": _semantic_families(variants, partition="heldout"),
            "training_controls": _control_operators(variants, partition="train"),
            "heldout_controls": _control_operators(variants, partition="heldout"),
        }
        if family_catalog is None:
            family_catalog = observed_catalog
        elif observed_catalog != family_catalog:
            raise WitnessExperimentError("variant family catalog differs across environments")
        cluster["schedule_ordinal"] = schedule_ordinal
        clusters.append(cluster)
        operation_ceiling += repetitions * (1 + len(variants)) * len(cluster["probes"])

    assert family_catalog is not None
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "spade-counterfactual-witness-v1",
        "source": {
            "run_dir": str(source_run),
            "plan_digest": AUTHORIZED_SOURCE_PLAN_DIGEST,
            "cohort_digest": AUTHORIZED_SOURCE_COHORT_DIGEST,
            "import_manifest_digest": _digest(snapshot.manifest),
            "import_leaf_count": len(snapshot.manifest),
        },
        "output_root": str(output),
        "runtime_identity": _plain(resolved.runtime_identity),
        "configuration": {
            "action_format": ACTION_FORMAT,
            "seeds": list(AUTHORIZED_SEEDS),
            "max_turns": MAX_TURNS,
            "operation_timeout_seconds": float(operation_timeout_seconds),
            "repetitions": repetitions,
            "witness_budget": witness_budget,
            "random_baseline_draws": random_baseline_draws,
            "sandbox_operation_ceiling": operation_ceiling,
            "primary_recall_margin": PRIMARY_RECALL_MARGIN,
            "max_equivalent_false_rejection_rate": MAX_EQUIVALENT_FALSE_REJECTION_RATE,
            "minimum_training_recall": MIN_TRAIN_RECALL,
            "minimum_safe_bank_heldout_recall": MIN_SAFE_BANK_HELDOUT_RECALL,
        },
        "family_split": family_catalog,
        "clusters": clusters,
        "analysis_role": "offline-representation-falsification-only",
        "provider_calls": 0,
        "learner_updates": 0,
    }
    return {**body, "plan_digest": _digest(body)}


def validate_plan(value: object) -> dict[str, Any]:
    """Validate the sealed plan without touching the source tree or runtime."""
    if not isinstance(value, dict):
        raise WitnessExperimentError("plan must be an object")
    expected_top = {
        "schema_version",
        "protocol_id",
        "experiment_id",
        "source",
        "output_root",
        "runtime_identity",
        "configuration",
        "family_split",
        "clusters",
        "analysis_role",
        "provider_calls",
        "learner_updates",
        "plan_digest",
    }
    if set(value) != expected_top:
        raise WitnessExperimentError("plan fields differ from schema")
    if (
        value["schema_version"] != PLAN_SCHEMA
        or value["protocol_id"] != PROTOCOL_ID
        or value["experiment_id"] != "spade-counterfactual-witness-v1"
        or value["analysis_role"] != "offline-representation-falsification-only"
        or value["provider_calls"] != 0
        or value["learner_updates"] != 0
    ):
        raise WitnessExperimentError("plan protocol identity is invalid")
    body = {key: item for key, item in value.items() if key != "plan_digest"}
    if value["plan_digest"] != _digest(body):
        raise WitnessExperimentError("plan self-digest mismatch")
    source = value["source"]
    if not isinstance(source, dict) or source.get("plan_digest") != AUTHORIZED_SOURCE_PLAN_DIGEST:
        raise WitnessExperimentError("plan source binding is invalid")
    if source.get("cohort_digest") != AUTHORIZED_SOURCE_COHORT_DIGEST:
        raise WitnessExperimentError("plan cohort binding is invalid")
    _sha256_text(source.get("import_manifest_digest"), "source import manifest digest")
    if source.get("import_leaf_count") != 321:
        raise WitnessExperimentError("source import leaf count is not the authorized closure")
    _canonical_absolute_dir(source.get("run_dir", ""), "source run", must_exist=True)
    _canonical_absolute_dir(value["output_root"], "output root", must_exist=False)
    _validate_runtime_identity(value["runtime_identity"])
    config = value["configuration"]
    if not isinstance(config, dict):
        raise WitnessExperimentError("configuration must be an object")
    if (
        config.get("action_format") != ACTION_FORMAT
        or config.get("seeds") != list(AUTHORIZED_SEEDS)
        or config.get("max_turns") != MAX_TURNS
        or config.get("repetitions") != DEFAULT_REPETITIONS
        or config.get("primary_recall_margin") != PRIMARY_RECALL_MARGIN
        or config.get("max_equivalent_false_rejection_rate") != MAX_EQUIVALENT_FALSE_REJECTION_RATE
        or config.get("minimum_training_recall") != MIN_TRAIN_RECALL
        or config.get("minimum_safe_bank_heldout_recall") != MIN_SAFE_BANK_HELDOUT_RECALL
    ):
        raise WitnessExperimentError("configuration differs from the v1 protocol")
    if type(config.get("witness_budget")) is not int or not 1 <= config["witness_budget"] <= 64:
        raise WitnessExperimentError("invalid witness budget")
    if (
        type(config.get("random_baseline_draws")) is not int
        or not 1 <= config["random_baseline_draws"] <= 10_000
    ):
        raise WitnessExperimentError("invalid random baseline draw count")
    timeout = config.get("operation_timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 3_600
    ):
        raise WitnessExperimentError("invalid operation timeout")
    clusters = value["clusters"]
    if not isinstance(clusters, list) or len(clusters) != AUTHORIZED_CLUSTERS:
        raise WitnessExperimentError("plan must contain exactly 18 clusters")
    operation_ceiling = 0
    cluster_ids: set[str] = set()
    for ordinal, cluster in enumerate(clusters, start=1):
        if not isinstance(cluster, dict) or cluster.get("schedule_ordinal") != ordinal:
            raise WitnessExperimentError("cluster schedule is malformed")
        cluster_id = _required_text(cluster.get("cluster_id"), "cluster id")
        if cluster_id in cluster_ids:
            raise WitnessExperimentError("cluster ids must be unique")
        cluster_ids.add(cluster_id)
        _sha256_text(cluster.get("code_digest"), "cluster code digest")
        _sha256_text(cluster.get("environment_digest"), "cluster environment digest")
        _sha256_text(cluster.get("qualification_digest"), "qualification digest")
        variants = cluster.get("variants")
        probes = cluster.get("probes")
        if not isinstance(variants, list) or not variants:
            raise WitnessExperimentError("cluster variant catalog is empty")
        if not isinstance(probes, list) or not probes:
            raise WitnessExperimentError("cluster probe catalog is empty")
        variant_ids = [_required_text(item.get("variant_id"), "variant id") for item in variants]
        probe_ids = [_required_text(item.get("probe_id"), "probe id") for item in probes]
        if len(variant_ids) != len(set(variant_ids)) or len(probe_ids) != len(set(probe_ids)):
            raise WitnessExperimentError("cluster catalog contains duplicate identifiers")
        for item in variants:
            _sha256_text(item.get("source_digest"), "variant source digest")
        operation_ceiling += config["repetitions"] * (1 + len(variants)) * len(probes)
    if config.get("sandbox_operation_ceiling") != operation_ceiling:
        raise WitnessExperimentError("sandbox operation ceiling is inconsistent")
    split = value["family_split"]
    if not isinstance(split, dict) or set(split) != {
        "training_semantic",
        "heldout_semantic",
        "training_controls",
        "heldout_controls",
    }:
        raise WitnessExperimentError("family split is malformed")
    for key, items in split.items():
        if not isinstance(items, list) or not items or len(items) != len(set(items)):
            raise WitnessExperimentError(f"family split {key} is empty or duplicated")
    if set(split["training_semantic"]) & set(split["heldout_semantic"]):
        raise WitnessExperimentError("semantic train/holdout families overlap")
    if set(split["training_controls"]) & set(split["heldout_controls"]):
        raise WitnessExperimentError("control train/holdout families overlap")
    return value


def write_plan(path: Path | str, plan: Mapping[str, Any]) -> Path:
    """Publish one immutable plan file."""
    validated = validate_plan(dict(plan))
    target = Path(path)
    if not target.is_absolute():
        raise WitnessExperimentError("plan path must be absolute")
    base._write_json(target, validated)
    return target


def load_plan(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        raise WitnessExperimentError("plan path must be absolute")
    return validate_plan(base._read_json(target))


def derive_run_dir(plan: Mapping[str, Any]) -> Path:
    validated = validate_plan(dict(plan))
    suffix = str(validated["plan_digest"]).removeprefix("sha256:")
    return Path(str(validated["output_root"])) / f"{validated['experiment_id']}-{suffix}"


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


def _safe_component(value: str, where: str) -> str:
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise WitnessExperimentError(f"{where} is not a safe artifact identifier")
    return value


def _probe_mode(probe: WitnessProbe) -> str:
    value = getattr(probe, "mode", None)
    if value is None:
        value = getattr(probe, "operation", None)
    if value is None:
        return "replay"
    if value not in {"inspect", "oracle", "replay"}:
        raise WitnessExperimentError(f"unsupported witness probe mode: {value!r}")
    return str(value)


def _probe_seed(probe: WitnessProbe) -> int:
    value = getattr(probe, "seed", None)
    if type(value) is not int:
        raise WitnessExperimentError("witness probe seed must be an integer")
    return value


def _probe_actions(probe: WitnessProbe) -> tuple[str, ...]:
    value = getattr(probe, "actions", ())
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise WitnessExperimentError("witness probe actions must be text")
    return tuple(value)


def _execute_probe(
    target: _ExecutionTarget,
    probe: WitnessProbe,
    *,
    timeout_seconds: float,
) -> Any:
    mode = _probe_mode(probe)
    seed = _probe_seed(probe)
    if mode == "inspect":
        return target.inspect(seed=seed, timeout_seconds=timeout_seconds)
    if mode == "oracle":
        return target.run_oracle(seed=seed, timeout_seconds=timeout_seconds)
    return target.run_actions(
        seed=seed,
        actions=_probe_actions(probe),
        timeout_seconds=timeout_seconds,
    )


def _execution_error(result: Any) -> str | None:
    if isinstance(result, Mapping):
        value = result.get("error")
    else:
        value = getattr(result, "error", None)
    if value is None:
        return None
    return str(value)


def _validate_execution_outcome(result: Any, *, variant_id: str, probe_id: str) -> None:
    """Reject infrastructure/interface failures but retain early-stop behavior.

    ProofPack's bounded ``run_actions`` returns a structured failed result when
    an action suffix follows a terminal transition.  Early termination is
    precisely one of the semantic relations this experiment must observe, so
    that one exact worker error is a trace outcome.  All other errors remain a
    fail-closed execution failure.
    """
    error = _execution_error(result)
    if error is None:
        return
    projected = _plain(result)
    allowed_early_stop = isinstance(projected, dict) and projected == {
        "success": False,
        "reward": 0.0,
        "turn_count": 0,
        "terminated": False,
        "truncated": False,
        "trajectory": [],
        "error": EARLY_TERMINATION_ERROR,
    }
    if not allowed_early_stop:
        raise WitnessExperimentError(
            f"sandbox execution failed for {variant_id}/{probe_id}: {error}"
        )


def _trace_leaf_path(run_dir: Path, cluster_id: str, variant_id: str, probe_id: str) -> Path:
    def digest_component(value: str, where: str) -> str:
        return _sha256_text(value, where).removeprefix("sha256:")

    return (
        run_dir
        / "traces"
        / _safe_component(cluster_id, "cluster id")
        / digest_component(variant_id, "variant id")
        / f"{digest_component(probe_id, 'probe id')}.json"
    )


def _operation_paths(
    run_dir: Path,
    cluster_id: str,
    variant_id: str,
    probe_id: str,
    repetition: int,
) -> tuple[Path, Path]:
    trace_path = _trace_leaf_path(run_dir, cluster_id, variant_id, probe_id)
    stem = trace_path.with_suffix("")
    return (
        stem / f"repetition-{repetition:02d}-request.json",
        stem / f"repetition-{repetition:02d}-result.json",
    )


def _sealed_body(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != digest_field}
    if value.get(digest_field) != _digest(body):
        raise WitnessExperimentError(f"{digest_field} does not bind its artifact")
    return body


def _operation_request(
    *,
    plan: Mapping[str, Any],
    cluster: Mapping[str, Any],
    variant: Mapping[str, Any],
    probe: WitnessProbe,
    repetition: int,
) -> dict[str, Any]:
    body = {
        "schema_version": OPERATION_REQUEST_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "cluster_id": cluster["cluster_id"],
        "candidate_id": cluster["candidate_id"],
        "variant_id": variant["variant_id"],
        "variant_source_digest": variant["source_digest"],
        "probe": _probe_record(probe),
        "repetition": repetition,
        "timeout_seconds": plan["configuration"]["operation_timeout_seconds"],
    }
    return {**body, "request_digest": _digest(body)}


def _validate_operation_request(value: object, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value != dict(expected):
        raise WitnessExperimentError("operation request differs from sealed schedule")
    _sealed_body(value, "request_digest")
    return value


def _operation_result(request: Mapping[str, Any], result: Any) -> dict[str, Any]:
    body = {
        "schema_version": OPERATION_RESULT_SCHEMA,
        "request_digest": request["request_digest"],
        "result": _plain(result),
    }
    return {**body, "result_digest": _digest(body)}


def _validate_operation_result(value: object, request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "request_digest",
        "result",
        "result_digest",
    }:
        raise WitnessExperimentError("operation result fields differ from schema")
    if (
        value["schema_version"] != OPERATION_RESULT_SCHEMA
        or value["request_digest"] != request["request_digest"]
    ):
        raise WitnessExperimentError("operation result is not bound to its request")
    _sealed_body(value, "result_digest")
    # This validates the complete externally visible result shape now rather
    # than waiting until aggregation.
    try:
        trace_signature(value["result"])
    except (TypeError, ValueError, KeyError) as exc:
        raise WitnessExperimentError("operation result has an invalid trace shape") from exc
    return value


def _trace_leaf_body(
    *,
    plan: Mapping[str, Any],
    cluster: Mapping[str, Any],
    variant: Mapping[str, Any],
    probe: WitnessProbe,
    result: Any,
    signature: TraceSignature,
    operation_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_LEAF_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "source_plan_digest": plan["source"]["plan_digest"],
        "source_cohort_digest": plan["source"]["cohort_digest"],
        "cluster_id": cluster["cluster_id"],
        "candidate_id": cluster["candidate_id"],
        "base_code_digest": cluster["code_digest"],
        "variant": dict(variant),
        "probe": _probe_record(probe),
        "repetitions": plan["configuration"]["repetitions"],
        "operation_evidence": [dict(item) for item in operation_evidence],
        "result": _plain(result),
        "signature": _plain(signature),
    }


def _validate_trace_leaf(
    value: object,
    *,
    expected_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "plan_digest",
        "source_plan_digest",
        "source_cohort_digest",
        "cluster_id",
        "candidate_id",
        "base_code_digest",
        "variant",
        "probe",
        "repetitions",
        "operation_evidence",
        "result",
        "signature",
        "leaf_digest",
    }:
        raise WitnessExperimentError("trace leaf fields differ from schema")
    body = {key: item for key, item in value.items() if key != "leaf_digest"}
    if value["schema_version"] != TRACE_LEAF_SCHEMA or value["leaf_digest"] != _digest(body):
        raise WitnessExperimentError("trace leaf self-digest mismatch")
    if expected_body is not None and body != dict(expected_body):
        raise WitnessExperimentError("persisted trace leaf differs from deterministic replay")
    return value


def _execute_trace_leaf(
    *,
    plan: Mapping[str, Any],
    run_dir: Path,
    cluster: Mapping[str, Any],
    source: str,
    variant: Mapping[str, Any],
    probe: WitnessProbe,
    dependencies: RunnerDependencies,
) -> dict[str, Any]:
    variant_id = _required_text(variant.get("variant_id"), "variant id")
    if _bytes_digest(source.encode("utf-8")) != variant.get("source_digest"):
        raise WitnessExperimentError("runtime variant source differs from sealed digest")
    cluster_id = str(cluster["cluster_id"])
    probe_id = _probe_id(probe)
    path = _trace_leaf_path(run_dir, cluster_id, variant_id, probe_id)
    config = plan["configuration"]
    target: _ExecutionTarget | None = None
    observations: list[tuple[Any, TraceSignature]] = []
    operation_evidence: list[dict[str, Any]] = []
    for repetition in range(1, int(config["repetitions"]) + 1):
        request_path, result_path = _operation_paths(
            run_dir, cluster_id, variant_id, probe_id, repetition
        )
        request = _operation_request(
            plan=plan,
            cluster=cluster,
            variant=variant,
            probe=probe,
            repetition=repetition,
        )
        if request_path.is_file():
            _validate_operation_request(base._read_json(request_path), request)
            if not result_path.is_file():
                raise WitnessExperimentError(
                    f"ambiguous sandbox operation cannot be replayed: {request_path}"
                )
            result_leaf = _validate_operation_result(base._read_json(result_path), request)
            result = result_leaf["result"]
        else:
            if result_path.exists() or result_path.is_symlink():
                raise WitnessExperimentError("operation result exists without its request")
            if target is None:
                try:
                    target = dependencies.target_factory(
                        source,
                        action_format=ACTION_FORMAT,
                        max_turns=MAX_TURNS,
                        operation_timeout_seconds=float(config["operation_timeout_seconds"]),
                    )
                except Exception as exc:
                    raise WitnessExperimentError(
                        f"variant {variant_id} failed the static/sandbox target gate: {exc}"
                    ) from exc
            _revalidate_operation_runtime(plan["runtime_identity"])
            # Publish the reservation only after every local preflight passes,
            # immediately before the external sandbox boundary. A surviving
            # request without a result therefore means the operation may have
            # started and is genuinely ambiguous.
            base._write_json(request_path, request)
            try:
                result = _execute_probe(
                    target,
                    probe,
                    timeout_seconds=float(config["operation_timeout_seconds"]),
                )
            except Exception as exc:
                raise WitnessExperimentError(
                    f"sandbox operation raised for {variant_id}/{probe_id}: {exc}"
                ) from exc
            result_leaf = _operation_result(request, result)
            base._write_json(result_path, result_leaf)
            _validate_operation_result(result_leaf, request)
        operation_evidence.append(
            {
                "repetition": repetition,
                "request_path": str(request_path.relative_to(run_dir)),
                "request_digest": request["request_digest"],
                "result_path": str(result_path.relative_to(run_dir)),
                "result_digest": result_leaf["result_digest"],
            }
        )
        _validate_execution_outcome(result, variant_id=variant_id, probe_id=probe_id)
        observations.append((result, trace_signature(result)))
    first_result, first_signature = observations[0]
    if any(_plain(signature) != _plain(first_signature) for _, signature in observations[1:]):
        raise WitnessExperimentError(
            f"nondeterministic witness response for {variant_id}/{_probe_id(probe)}"
        )
    body = _trace_leaf_body(
        plan=plan,
        cluster=cluster,
        variant=variant,
        probe=probe,
        result=first_result,
        signature=first_signature,
        operation_evidence=operation_evidence,
    )
    if path.is_file():
        return _validate_trace_leaf(base._read_json(path), expected_body=body)
    leaf = {**body, "leaf_digest": _digest(body)}
    base._write_json(path, leaf)
    return _validate_trace_leaf(leaf, expected_body=body)


def _regenerate_cluster(
    snapshot: Any,
    sealed_cluster: Mapping[str, Any],
) -> tuple[str, list[SourceVariant], list[WitnessProbe]]:
    cluster_id = str(sealed_cluster["cluster_id"])
    selection = snapshot.selections.get(cluster_id)
    if not isinstance(selection, Mapping):
        raise WitnessExperimentError(f"source snapshot lacks cluster {cluster_id}")
    regenerated, variants = _build_cluster_catalog(selection)
    regenerated["schedule_ordinal"] = sealed_cluster["schedule_ordinal"]
    if regenerated != dict(sealed_cluster):
        raise WitnessExperimentError(f"regenerated catalog drifted for {cluster_id}")
    probes = list(
        build_candidate_probes(
            action_format=ACTION_FORMAT,
            seeds=list(AUTHORIZED_SEEDS),
            base_oracle_actions=_oracle_actions(selection, AUTHORIZED_SEEDS),
            max_turns=MAX_TURNS,
        )
    )
    return _required_text(selection.get("code"), "selected source"), variants, probes


def _run_manifest(plan: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan["plan_digest"],
        "source": plan["source"],
        "runtime_identity": plan["runtime_identity"],
        "sandbox_operation_ceiling": plan["configuration"]["sandbox_operation_ceiling"],
        "provider_calls": 0,
        "learner_updates": 0,
    }
    return {**body, "manifest_digest": _digest(body)}


def _initialize_run(
    plan: Mapping[str, Any],
    plan_path: Path,
    dependencies: RunnerDependencies,
) -> tuple[Path, Any]:
    run_dir = derive_run_dir(plan)
    base._reject_symlink_ancestors(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    base._reject_symlink_ancestors(run_dir)
    if run_dir.is_symlink() or run_dir.resolve() != run_dir:
        raise WitnessExperimentError("run directory is unsafe")
    _revalidate_runtime_identity(plan["runtime_identity"], dependencies.runtime_identity)
    snapshot = dependencies.load_source_snapshot(Path(str(plan["source"]["run_dir"])))
    if (
        snapshot.plan["plan_digest"] != plan["source"]["plan_digest"]
        or snapshot.cohort["cohort_digest"] != plan["source"]["cohort_digest"]
        or _digest(snapshot.manifest) != plan["source"]["import_manifest_digest"]
        or len(snapshot.manifest) != plan["source"]["import_leaf_count"]
    ):
        raise WitnessExperimentError("source snapshot drifted after plan sealing")
    plan_copy = run_dir / "plan.json"
    plan_bytes = plan_path.read_bytes()
    if plan_bytes != base._pretty_json(plan):
        # The canonical plan reader already verified semantic content.  This
        # additional check keeps the copied bytes deterministic.
        raise WitnessExperimentError("plan file bytes are not canonical pretty JSON")
    base._write_immutable_bytes(plan_copy, plan_bytes)
    base._write_json(run_dir / "run-manifest.json", _run_manifest(plan))
    return run_dir, snapshot


def _base_variant_record(cluster: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": cluster["environment_digest"],
        "kind": "base",
        "partition": "base",
        "family": "base",
        "operator": "base",
        "source_digest": cluster["environment_digest"],
        "entrypoint": cluster["environment_name"],
        "expected_effect": "reference",
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise WitnessExperimentError("cannot average an empty metric sequence")
    return math.fsum(values) / len(values)


def _score_dict(score: Any) -> dict[str, Any]:
    value = _plain(score)
    if not isinstance(value, dict):
        raise WitnessExperimentError("selection score did not serialize as an object")
    return value


def _detection_profile(
    matrix: WitnessMatrix,
    *,
    selected_probe_ids: Sequence[str],
    safe_probe_ids: Sequence[str],
    partition: str,
) -> dict[str, Any]:
    mutants = [
        variant
        for variant in matrix.variants
        if variant.kind == "semantic_mutant" and variant.partition == partition
    ]

    def detected(variant: SourceVariant, probe_ids: Sequence[str]) -> bool:
        return any(
            matrix.signature(probe_id, variant.variant_id).digest
            != matrix.signature(probe_id, matrix.base_key).digest
            for probe_id in probe_ids
        )

    observable = [variant for variant in mutants if detected(variant, safe_probe_ids)]
    selected = [variant for variant in observable if detected(variant, selected_probe_ids)]
    body = {
        "partition": partition,
        "raw_mutant_count": len(mutants),
        "safe_bank_observable_count": len(observable),
        "selected_observable_kill_count": len(selected),
        "applicable_recall": len(selected) / len(observable) if observable else 0.0,
        "unobservable_variant_ids": sorted(
            variant.variant_id for variant in mutants if variant not in observable
        ),
    }
    return {**body, "profile_digest": _digest(body)}


def _fixed_baseline_ids(
    probes: Sequence[WitnessProbe], safe_probe_ids: Sequence[str], count: int
) -> tuple[str, ...]:
    safe = set(safe_probe_ids)
    by_id = {_probe_id(probe): probe for probe in probes}
    role_order = {
        "base_oracle": 0,
        "well_formed_wrong": 1,
        "reset": 2,
        "malformed": 3,
        "blank_noop": 4,
    }
    seed_order = {seed: index for index, seed in enumerate(AUTHORIZED_SEEDS)}
    ordered = sorted(
        safe,
        key=lambda probe_id: (
            seed_order.get(by_id[probe_id].seed, len(seed_order)),
            role_order.get(by_id[probe_id].role, 5),
            len(by_id[probe_id].actions),
            probe_id,
        ),
    )
    return tuple(ordered[:count])


def _random_baseline(
    *,
    matrix: WitnessMatrix,
    safe_probe_ids: Sequence[str],
    count: int,
    draws: int,
    seed_text: str,
) -> dict[str, Any]:
    safe = tuple(sorted(safe_probe_ids))
    if count > len(safe):
        raise WitnessExperimentError("random baseline sample exceeds its safe probe bank")
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest(), "big")
    generator = random.Random(seed)
    recall: list[float] = []
    macro_recall: list[float] = []
    false_rejection: list[float] = []
    applicable_recall: list[float] = []
    selection_digests: list[str] = []
    for _ in range(draws):
        chosen = tuple(sorted(generator.sample(safe, count))) if count else ()
        score = score_probe_ids(matrix, chosen, partition="heldout")
        recall.append(score.mutant_recall)
        macro_recall.append(score.family_macro_recall)
        false_rejection.append(score.equivalent_false_rejection_rate)
        applicable_recall.append(
            _detection_profile(
                matrix,
                selected_probe_ids=chosen,
                safe_probe_ids=safe,
                partition="heldout",
            )["applicable_recall"]
        )
        selection_digests.append(_digest(list(chosen)))
    body = {
        "draws": draws,
        "sample_size": count,
        "seed_digest": _bytes_digest(seed_text.encode("utf-8")),
        "selection_sequence_digest": _digest(selection_digests),
        "heldout_mutant_recall_mean": _mean(recall),
        "heldout_family_macro_recall_mean": _mean(macro_recall),
        "heldout_equivalent_false_rejection_rate_mean": _mean(false_rejection),
        "heldout_applicable_recall_mean": _mean(applicable_recall),
    }
    return {**body, "baseline_digest": _digest(body)}


def _trace_inventory_record(path: Path, run_dir: Path, leaf: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "leaf_digest": leaf["leaf_digest"],
    }


def _run_cluster(
    *,
    plan: Mapping[str, Any],
    run_dir: Path,
    snapshot: Any,
    cluster: Mapping[str, Any],
    dependencies: RunnerDependencies,
) -> dict[str, Any]:
    code, variants, probes = _regenerate_cluster(snapshot, cluster)
    base_variant = _base_variant_record(cluster)
    sources: list[tuple[str, dict[str, Any], str]] = [("base", base_variant, code)] + [
        (_variant_id(variant), _variant_catalog_record(variant), _variant_source(variant))
        for variant in variants
    ]
    signatures: dict[str, dict[str, TraceSignature]] = {_probe_id(probe): {} for probe in probes}
    trace_inventory: list[dict[str, Any]] = []
    for matrix_key, variant, source in sources:
        for probe in probes:
            leaf = _execute_trace_leaf(
                plan=plan,
                run_dir=run_dir,
                cluster=cluster,
                source=source,
                variant=variant,
                probe=probe,
                dependencies=dependencies,
            )
            signature = trace_signature(leaf["result"])
            if _plain(signature) != leaf["signature"]:
                raise WitnessExperimentError("trace signature differs from persisted result")
            signatures[_probe_id(probe)][matrix_key] = signature
            trace_path = _trace_leaf_path(
                run_dir,
                str(cluster["cluster_id"]),
                str(variant["variant_id"]),
                _probe_id(probe),
            )
            trace_inventory.append(_trace_inventory_record(trace_path, run_dir, leaf))

    matrix = WitnessMatrix(
        base_environment_digest=str(cluster["environment_digest"]),
        probes=tuple(probes),
        variants=tuple(variants),
        signatures=signatures,
    )
    matrix.validate()
    selection = select_witnesses(matrix, budget=int(plan["configuration"]["witness_budget"]))
    selected_ids = selection.selected_probe_ids
    train_score = score_probe_ids(matrix, selected_ids, partition="train")
    heldout_score = score_probe_ids(matrix, selected_ids, partition="heldout")
    safe_bank_score = score_probe_ids(matrix, selection.safe_probe_ids, partition="heldout")
    fixed_ids = _fixed_baseline_ids(probes, selection.safe_probe_ids, len(selected_ids))
    fixed_score = score_probe_ids(matrix, fixed_ids, partition="heldout")
    random_baseline = _random_baseline(
        matrix=matrix,
        safe_probe_ids=selection.safe_probe_ids,
        count=len(selected_ids),
        draws=int(plan["configuration"]["random_baseline_draws"]),
        seed_text=f"{plan['plan_digest']}:{cluster['cluster_id']}:random-baseline/v1",
    )
    base_signatures = {probe.probe_id: signatures[probe.probe_id]["base"] for probe in probes}
    descriptor = behavior_descriptor(
        action_format=ACTION_FORMAT,
        probes=probes,
        signatures=base_signatures,
    )
    certificate = build_certificate(
        matrix,
        selection,
        metadata={
            "plan_digest": plan["plan_digest"],
            "source_plan_digest": plan["source"]["plan_digest"],
            "source_cohort_digest": plan["source"]["cohort_digest"],
            "cluster_id": cluster["cluster_id"],
            "candidate_id": cluster["candidate_id"],
            "qualification_digest": cluster["qualification_digest"],
            "analysis_role": plan["analysis_role"],
        },
    )
    certificate_path = run_dir / "certificates" / f"{cluster['cluster_id']}.json"
    base._write_json(certificate_path, _plain(certificate))
    body = {
        "schema_version": CLUSTER_RESULT_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "cluster": dict(cluster),
        "selection": _plain(selection),
        "scores": {
            "training": _score_dict(train_score),
            "heldout": _score_dict(heldout_score),
            "safe_bank_heldout": _score_dict(safe_bank_score),
            "proofpack_fixed_heldout": _score_dict(fixed_score),
        },
        "applicability": {
            "training": _detection_profile(
                matrix,
                selected_probe_ids=selected_ids,
                safe_probe_ids=selection.safe_probe_ids,
                partition="train",
            ),
            "heldout": _detection_profile(
                matrix,
                selected_probe_ids=selected_ids,
                safe_probe_ids=selection.safe_probe_ids,
                partition="heldout",
            ),
        },
        "proofpack_fixed_probe_ids": list(fixed_ids),
        "random_baseline": random_baseline,
        "behavior_descriptor": _plain(descriptor),
        "certificate": {
            "path": str(certificate_path.relative_to(run_dir)),
            "digest": certificate.certificate_digest,
        },
        "trace_inventory": sorted(trace_inventory, key=lambda item: item["path"]),
        "provider_calls": 0,
        "learner_updates": 0,
    }
    result = {**body, "cluster_result_digest": _digest(body)}
    result_path = run_dir / "clusters" / f"{cluster['cluster_id']}.json"
    base._write_json(result_path, result)
    return _validate_cluster_result(result, plan=plan, expected_cluster=cluster)


def _validate_cluster_result(
    value: object,
    *,
    plan: Mapping[str, Any],
    expected_cluster: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "plan_digest",
        "cluster",
        "selection",
        "scores",
        "applicability",
        "proofpack_fixed_probe_ids",
        "random_baseline",
        "behavior_descriptor",
        "certificate",
        "trace_inventory",
        "provider_calls",
        "learner_updates",
        "cluster_result_digest",
    }:
        raise WitnessExperimentError("cluster result fields differ from schema")
    if (
        value["schema_version"] != CLUSTER_RESULT_SCHEMA
        or value["plan_digest"] != plan["plan_digest"]
        or value["cluster"] != dict(expected_cluster)
        or value["provider_calls"] != 0
        or value["learner_updates"] != 0
    ):
        raise WitnessExperimentError("cluster result binding is invalid")
    _sealed_body(value, "cluster_result_digest")
    scores = value["scores"]
    if not isinstance(scores, dict) or set(scores) != {
        "training",
        "heldout",
        "safe_bank_heldout",
        "proofpack_fixed_heldout",
    }:
        raise WitnessExperimentError("cluster score fields differ from schema")
    applicability = value["applicability"]
    if not isinstance(applicability, dict) or set(applicability) != {"training", "heldout"}:
        raise WitnessExperimentError("cluster applicability fields differ from schema")
    return value


def _aggregate_results(
    plan: Mapping[str, Any], cluster_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(cluster_results) != AUTHORIZED_CLUSTERS:
        raise WitnessExperimentError("aggregate requires all 18 cluster results")
    train_recall = [float(item["scores"]["training"]["mutant_recall"]) for item in cluster_results]
    heldout_recall = [float(item["scores"]["heldout"]["mutant_recall"]) for item in cluster_results]
    heldout_macro = [
        float(item["scores"]["heldout"]["family_macro_recall"]) for item in cluster_results
    ]
    heldout_fpr = [
        float(item["scores"]["heldout"]["equivalent_false_rejection_rate"])
        for item in cluster_results
    ]
    applicable_recall = [
        float(item["applicability"]["heldout"]["applicable_recall"]) for item in cluster_results
    ]
    safe_bank_recall = [
        float(item["scores"]["safe_bank_heldout"]["mutant_recall"]) for item in cluster_results
    ]
    fixed_recall = [
        float(item["scores"]["proofpack_fixed_heldout"]["mutant_recall"])
        for item in cluster_results
    ]
    random_recall = [
        float(item["random_baseline"]["heldout_mutant_recall_mean"]) for item in cluster_results
    ]
    random_applicable_recall = [
        float(item["random_baseline"]["heldout_applicable_recall_mean"]) for item in cluster_results
    ]
    selected_counts = [
        float(len(item["selection"]["selected_probe_ids"])) for item in cluster_results
    ]
    metrics = {
        "training_mutant_recall_macro": _mean(train_recall),
        "heldout_mutant_recall_macro": _mean(heldout_recall),
        "heldout_family_macro_recall": _mean(heldout_macro),
        "heldout_equivalent_false_rejection_rate": _mean(heldout_fpr),
        "heldout_applicable_mutant_recall_macro": _mean(applicable_recall),
        "heldout_applicable_mutant_count": sum(
            int(item["applicability"]["heldout"]["safe_bank_observable_count"])
            for item in cluster_results
        ),
        "heldout_raw_mutant_count": sum(
            int(item["applicability"]["heldout"]["raw_mutant_count"]) for item in cluster_results
        ),
        "safe_bank_heldout_mutant_recall_macro": _mean(safe_bank_recall),
        "proofpack_fixed_heldout_mutant_recall_macro": _mean(fixed_recall),
        "random_safe_probe_heldout_mutant_recall_macro": _mean(random_recall),
        "random_safe_probe_heldout_applicable_recall_macro": _mean(random_applicable_recall),
        "witness_minus_random_recall": _mean(
            [left - right for left, right in zip(heldout_recall, random_recall)]
        ),
        "witness_minus_proofpack_fixed_recall": _mean(
            [left - right for left, right in zip(heldout_recall, fixed_recall)]
        ),
        "average_selected_probes": _mean(selected_counts),
    }
    gates = {
        "training_recall": metrics["training_mutant_recall_macro"] >= MIN_TRAIN_RECALL,
        "heldout_advantage_over_random": metrics["witness_minus_random_recall"]
        >= PRIMARY_RECALL_MARGIN,
        "heldout_equivalent_control_safety": metrics["heldout_equivalent_false_rejection_rate"]
        <= MAX_EQUIVALENT_FALSE_REJECTION_RATE,
        "safe_bank_heldout_killability": metrics["safe_bank_heldout_mutant_recall_macro"]
        >= float(plan["configuration"]["minimum_safe_bank_heldout_recall"]),
    }
    status = "pass" if all(gates.values()) else "fail"
    body = {
        "schema_version": AGGREGATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan["plan_digest"],
        "source_plan_digest": plan["source"]["plan_digest"],
        "source_cohort_digest": plan["source"]["cohort_digest"],
        "status": status,
        "metrics": metrics,
        "gates": gates,
        "thresholds": {
            "primary_recall_margin": plan["configuration"]["primary_recall_margin"],
            "max_equivalent_false_rejection_rate": plan["configuration"][
                "max_equivalent_false_rejection_rate"
            ],
            "minimum_training_recall": plan["configuration"]["minimum_training_recall"],
            "minimum_safe_bank_heldout_recall": plan["configuration"][
                "minimum_safe_bank_heldout_recall"
            ],
        },
        "cluster_result_digests": [item["cluster_result_digest"] for item in cluster_results],
        "sandbox_operations_completed": plan["configuration"]["sandbox_operation_ceiling"],
        "provider_calls": 0,
        "learner_updates": 0,
        "claim_boundary": (
            "exploratory operator-heldout offline representation falsification only; "
            "observable channels and admitted controls overlap the training catalog, "
            "and this result does not establish downstream learner improvement"
        ),
    }
    return {**body, "aggregate_digest": _digest(body)}


def _validate_aggregate(value: object, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "protocol_id",
        "plan_digest",
        "source_plan_digest",
        "source_cohort_digest",
        "status",
        "metrics",
        "gates",
        "thresholds",
        "cluster_result_digests",
        "sandbox_operations_completed",
        "provider_calls",
        "learner_updates",
        "claim_boundary",
        "aggregate_digest",
    }:
        raise WitnessExperimentError("aggregate fields differ from schema")
    if (
        value["schema_version"] != AGGREGATE_SCHEMA
        or value["protocol_id"] != PROTOCOL_ID
        or value["plan_digest"] != plan["plan_digest"]
        or value["source_plan_digest"] != plan["source"]["plan_digest"]
        or value["source_cohort_digest"] != plan["source"]["cohort_digest"]
        or value["status"] not in {"pass", "fail"}
        or value["provider_calls"] != 0
        or value["learner_updates"] != 0
        or value["thresholds"]
        != {
            "primary_recall_margin": plan["configuration"]["primary_recall_margin"],
            "max_equivalent_false_rejection_rate": plan["configuration"][
                "max_equivalent_false_rejection_rate"
            ],
            "minimum_training_recall": plan["configuration"]["minimum_training_recall"],
            "minimum_safe_bank_heldout_recall": plan["configuration"][
                "minimum_safe_bank_heldout_recall"
            ],
        }
    ):
        raise WitnessExperimentError("aggregate binding is invalid")
    _sealed_body(value, "aggregate_digest")
    return value


def _summary_markdown(aggregate: Mapping[str, Any]) -> bytes:
    metrics = aggregate["metrics"]
    gates = aggregate["gates"]
    lines = [
        "# Counterfactual witness falsification result",
        "",
        f"Status: **{str(aggregate['status']).upper()}**",
        "",
        f"- Training mutant recall: {metrics['training_mutant_recall_macro']:.3f}",
        f"- Held-out mutant recall: {metrics['heldout_mutant_recall_macro']:.3f}",
        f"- Held-out applicable-mutant recall: {metrics['heldout_applicable_mutant_recall_macro']:.3f}",
        f"- Held-out equivalent-control false rejection: {metrics['heldout_equivalent_false_rejection_rate']:.3f}",
        f"- Recall advantage over random safe probes: {metrics['witness_minus_random_recall']:.3f}",
        f"- Recall advantage over fixed ProofPack-like probes: {metrics['witness_minus_proofpack_fixed_recall']:.3f}",
        f"- Mean selected probes: {metrics['average_selected_probes']:.2f}",
        "",
        "Gates:",
        *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items()],
        "",
        "This is an exploratory operator-heldout falsification result. Its observable channels and admitted controls overlap the training catalog; it is not evidence of independent semantic generalization or downstream learner improvement.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _expected_inventory(plan: Mapping[str, Any]) -> set[Path]:
    run_dir = derive_run_dir(plan)
    expected = {
        Path(".writer.lock"),
        Path("plan.json"),
        Path("run-manifest.json"),
        Path("aggregate.json"),
        Path("summary.md"),
    }
    repetitions = int(plan["configuration"]["repetitions"])
    for cluster in plan["clusters"]:
        cluster_id = str(cluster["cluster_id"])
        expected.add(Path("clusters") / f"{cluster_id}.json")
        expected.add(Path("certificates") / f"{cluster_id}.json")
        variants = [
            cluster["environment_digest"],
            *[item["variant_id"] for item in cluster["variants"]],
        ]
        for variant_id in variants:
            for probe in cluster["probes"]:
                probe_id = str(probe["probe_id"])
                trace_path = _trace_leaf_path(run_dir, cluster_id, str(variant_id), probe_id)
                expected.add(trace_path.relative_to(run_dir))
                for repetition in range(1, repetitions + 1):
                    request, result = _operation_paths(
                        run_dir, cluster_id, str(variant_id), probe_id, repetition
                    )
                    expected.add(request.relative_to(run_dir))
                    expected.add(result.relative_to(run_dir))
    return expected


def _validate_run_inventory(run_dir: Path, plan: Mapping[str, Any], *, complete: bool) -> None:
    expected = _expected_inventory(plan)
    observed: set[Path] = set()
    for path in run_dir.rglob("*"):
        base._reject_symlink_ancestors(path)
        if path.is_symlink():
            raise WitnessExperimentError(f"symlinked run artifact is forbidden: {path}")
        if path.is_file():
            observed.add(path.relative_to(run_dir))
    unknown = observed - expected
    if unknown:
        raise WitnessExperimentError(
            f"run tree contains unrecognized artifacts: {sorted(map(str, unknown))[:3]}"
        )
    if complete and observed != expected:
        missing = expected - observed
        raise WitnessExperimentError(
            f"completed run inventory is missing artifacts: {sorted(map(str, missing))[:3]}"
        )


def _revalidate_source(plan: Mapping[str, Any], dependencies: RunnerDependencies) -> Any:
    snapshot = dependencies.load_source_snapshot(Path(str(plan["source"]["run_dir"])))
    if (
        snapshot.plan["plan_digest"] != plan["source"]["plan_digest"]
        or snapshot.cohort["cohort_digest"] != plan["source"]["cohort_digest"]
        or _digest(snapshot.manifest) != plan["source"]["import_manifest_digest"]
        or len(snapshot.manifest) != plan["source"]["import_leaf_count"]
    ):
        raise WitnessExperimentError("source evidence drifted during the witness run")
    return snapshot


def run_experiment(
    plan_path: Path | str,
    *,
    execute: bool = False,
    dependencies: RunnerDependencies | None = None,
) -> RunResult:
    """Validate a plan, or execute/resume its provider-free sandbox matrix."""
    path = Path(plan_path)
    if not path.is_absolute():
        raise WitnessExperimentError("plan path must be absolute")
    plan = load_plan(path)
    run_dir = derive_run_dir(plan)
    if not execute:
        return RunResult("validated", str(plan["plan_digest"]), run_dir)
    resolved = dependencies or _default_dependencies(Path(str(plan["source"]["run_dir"])))
    with base._single_writer(run_dir):
        run_dir, snapshot = _initialize_run(plan, path, resolved)
        _validate_run_inventory(run_dir, plan, complete=False)
        cluster_results: list[dict[str, Any]] = []
        for cluster in plan["clusters"]:
            _revalidate_runtime_identity(plan["runtime_identity"], resolved.runtime_identity)
            try:
                cluster_result = _run_cluster(
                    plan=plan,
                    run_dir=run_dir,
                    snapshot=snapshot,
                    cluster=cluster,
                    dependencies=resolved,
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise WitnessExperimentError(
                    f"invalid witness evidence for cluster {cluster['cluster_id']}"
                ) from exc
            cluster_results.append(cluster_result)
        _revalidate_runtime_identity(plan["runtime_identity"], resolved.runtime_identity)
        _revalidate_source(plan, resolved)
        aggregate = _aggregate_results(plan, cluster_results)
        aggregate_path = run_dir / "aggregate.json"
        base._write_json(aggregate_path, aggregate)
        base._write_immutable_bytes(run_dir / "summary.md", _summary_markdown(aggregate))
        _validate_aggregate(base._read_json(aggregate_path), plan)
        _validate_run_inventory(run_dir, plan, complete=True)
        return RunResult(
            str(aggregate["status"]),
            str(plan["plan_digest"]),
            run_dir,
            aggregate_path,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="seal an offline witness plan")
    plan_parser.add_argument("--source-run", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--witness-budget", type=int, default=DEFAULT_WITNESS_BUDGET)
    plan_parser.add_argument(
        "--random-baseline-draws", type=int, default=DEFAULT_RANDOM_BASELINE_DRAWS
    )
    plan_parser.add_argument(
        "--operation-timeout-seconds",
        type=float,
        default=DEFAULT_OPERATION_TIMEOUT_SECONDS,
    )
    run_parser = subparsers.add_parser("run", help="validate or execute a sealed plan")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_plan(
                source_run_dir=args.source_run,
                output_root=args.output_root,
                witness_budget=args.witness_budget,
                random_baseline_draws=args.random_baseline_draws,
                operation_timeout_seconds=args.operation_timeout_seconds,
            )
            output = write_plan(args.output.resolve(), plan)
            print(
                json.dumps(
                    {
                        "status": "sealed",
                        "plan": str(output),
                        "plan_digest": plan["plan_digest"],
                        "sandbox_operation_ceiling": plan["configuration"][
                            "sandbox_operation_ceiling"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = run_experiment(args.plan.resolve(), execute=args.execute)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "plan_digest": result.plan_digest,
                    "run_dir": str(result.run_dir),
                    "aggregate": str(result.aggregate_path) if result.aggregate_path else None,
                },
                sort_keys=True,
            )
        )
        return 0
    except (WitnessExperimentError, base.ExperimentError) as exc:
        print(f"counterfactual witness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
