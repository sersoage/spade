#!/usr/bin/env python3
"""Run a sealed, resumable multi-environment SPADE ``agy`` experiment.

The local plan is an integrity seal, not witnessed preregistration.  Execution is
opt-in twice: ``--execute`` must be accompanied by an acknowledgement equal to
the sealed total-call cap.  Every provider request is reserved durably before it
is sent, and an ambiguous in-flight request is never replayed.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import run_live_spade_eval as live  # noqa: E402
from spade.core.proofpack_bridge import validate_positive_proofpack_receipt  # noqa: E402


PLAN_SCHEMA = "spade-agy-experiment-plan/v1"
RUN_SCHEMA = "spade-agy-experiment-run/v1"
CALL_REQUEST_SCHEMA = "spade-agy-call-request/v1"
CALL_RESULT_SCHEMA = "spade-agy-call-result/v1"
COHORT_SCHEMA = "spade-agy-cohort-lock/v1"
OUTCOME_SCHEMA = "spade-agy-outcome/v1"
OUTCOME_TURN_SCHEMA = "spade-agy-outcome-turn/v1"
SELECTION_SCHEMA = "spade-agy-selected-candidate/v1"
LEDGER_SCHEMA = "spade-agy-leaf-ledger/v1"
ASSAY_REQUEST_SCHEMA = "spade-agy-assay-request/v1"
ASSAY_RESULT_SCHEMA = "spade-agy-assay-result/v1"
_ZERO_DIGEST = "sha256:" + "0" * 64

PILOT_SKILLS = (
    "Pattern Recognition",
    "Mathematical Reasoning",
    "Logical Deduction",
    "Strategic Planning",
    "Spatial Reasoning",
    "Causal Inference",
    "Memory Recall",
    "Optimization",
    "Language Understanding",
)
PILOT_DIFFICULTIES = ("medium", "hard")
ARMS = ("unhinted", "hinted")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ExperimentError(RuntimeError):
    """Fail-closed experiment error with a stable command exit code."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ExperimentIncomplete(ExperimentError):
    """The sealed schedule did not complete and Assay must not be called."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 9)


class CallFailed(ExperimentIncomplete):
    """One recorded provider attempt completed with an error."""


class AmbiguousCall(ExperimentIncomplete):
    """A request reservation exists without a durable result."""


class CallCapExceeded(ExperimentIncomplete):
    """The sealed hard provider-call cap has been reached."""


LlmCall = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class RunnerDependencies:
    """Injectable live boundaries; tests provide deterministic offline fakes."""

    llm_call: LlmCall
    qualify: Callable[..., Any]
    target_factory: Callable[..., Any]
    assay_writer: Callable[..., Any]
    task_factory: Callable[..., Any]
    cluster_factory: Callable[..., Any]
    run_metadata_factory: Callable[..., Any]
    client_or_bin: Any
    source_revisions: Mapping[str, str]
    runtime_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    """Validation or execution result."""

    status: str
    plan_digest: str
    run_dir: Path
    call_count: int
    cohort_lock_path: Path | None = None
    assay_result_path: Path | None = None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_text(value: object, where: str) -> str:
    text = _required_text(value, where)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ExperimentError(f"{where} must be a lowercase SHA-256 digest")
    return text


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ExperimentError(f"non-finite JSON number: {value}")


def _decode_json(data: bytes, where: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExperimentError(f"invalid JSON in {where}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"{where} must contain one JSON object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ExperimentError(f"{where} keys differ; missing={missing}, extra={extra}")


def _required_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{where} must be a non-empty string")
    return value


def _safe_id(value: object, where: str) -> str:
    text = _required_text(value, where)
    if _SAFE_ID.fullmatch(text) is None:
        raise ExperimentError(f"{where} must match {_SAFE_ID.pattern}")
    return text


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentError(f"{where} must be a positive integer")
    return value


def _finite_number(value: object, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise ExperimentError(f"{where} must be {qualifier}finite")
    return result


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ExperimentError(f"cannot derive an identifier from {value!r}")
    return slug[:48]


def _outcome_schedule(
    clusters: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    ordinal = 0
    for cluster_index, cluster in enumerate(clusters):
        cluster_id = str(cluster["cluster_id"])
        for seed_index, seed in enumerate(seeds):
            order = ARMS if (cluster_index + seed_index) % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                schedule.append(
                    {
                        "ordinal": ordinal,
                        "outcome_id": f"{cluster_id}-seed-{seed}-{arm}",
                        "cluster_id": cluster_id,
                        "seed": seed,
                        "arm": arm,
                    }
                )
                ordinal += 1
    return schedule


def _source_revisions() -> dict[str, str]:
    roots = {
        "spade": ROOT_DIR,
        "proofpack": ROOT_DIR.parent / "proofpack",
        "assay": ROOT_DIR.parent / "assay",
    }
    revisions: dict[str, str] = {}
    for name, root in roots.items():
        try:
            revisions[name] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
            clean = subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
                check=False,
                timeout=10,
            )
            if clean.returncode == 1:
                raise ExperimentError(
                    f"{name} has tracked or staged changes; seal only a clean source revision"
                )
            if clean.returncode != 0:
                raise ExperimentError(f"cannot inspect {name} worktree state at {root}")
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ExperimentError(f"cannot resolve {name} source revision at {root}") from exc
    return revisions


def _runtime_identity() -> dict[str, Any]:
    """Resolve the local execution identity without making a model call."""
    live._require_integrations()
    executable = shutil.which("agy")
    if executable is None:
        raise ExperimentError("cannot resolve the agy executable for the sealed runtime identity")
    executable_path = Path(executable).resolve()
    if not executable_path.is_file():
        raise ExperimentError(f"agy executable is not a regular file: {executable_path}")
    try:
        version = subprocess.check_output(
            [str(executable_path), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ExperimentError("cannot resolve the agy executable version") from exc
    python_executable = Path(sys.executable).resolve()
    imported_objects = {
        "spade_live_runner": live.call_llm,
        "proofpack_qualifier": live.qualify_spade_environment,
        "proofpack_receipt_validator": validate_positive_proofpack_receipt,
        "assay_writer": live.write_spade_evaluation,
    }
    imported_sources: dict[str, dict[str, str]] = {}
    expected_roots = {
        "spade_live_runner": ROOT_DIR,
        "proofpack_qualifier": ROOT_DIR.parent / "proofpack",
        "proofpack_receipt_validator": ROOT_DIR,
        "assay_writer": ROOT_DIR.parent / "assay",
    }
    for name, imported in imported_objects.items():
        source_name = inspect.getsourcefile(imported)
        if source_name is None:
            raise ExperimentError(f"cannot locate the imported source for {name}")
        source = Path(source_name).resolve()
        if not source.is_file() or not source.is_relative_to(expected_roots[name].resolve()):
            raise ExperimentError(f"{name} was imported outside its expected source repository")
        imported_sources[name] = {
            "path": str(source),
            "digest": _bytes_digest(source.read_bytes()),
        }
    return {
        "runner_digest": _bytes_digest(Path(__file__).read_bytes()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(python_executable),
        "python_executable_digest": _bytes_digest(python_executable.read_bytes()),
        "platform": platform.platform(),
        "agy_executable": str(executable_path),
        "agy_executable_digest": _bytes_digest(executable_path.read_bytes()),
        "agy_version": _required_text(version, "agy version"),
        "imported_sources": imported_sources,
    }


def _validate_runtime_identity(value: object, where: str = "runtime_identity") -> None:
    if not isinstance(value, dict):
        raise ExperimentError(f"{where} must be an object")
    _require_keys(
        value,
        {
            "runner_digest",
            "python_implementation",
            "python_version",
            "python_executable",
            "python_executable_digest",
            "platform",
            "agy_executable",
            "agy_executable_digest",
            "agy_version",
            "imported_sources",
        },
        where,
    )
    _sha256_text(value["runner_digest"], f"{where}.runner_digest")
    _sha256_text(value["python_executable_digest"], f"{where}.python_executable_digest")
    _sha256_text(value["agy_executable_digest"], f"{where}.agy_executable_digest")
    for key in (
        "python_implementation",
        "python_version",
        "python_executable",
        "platform",
        "agy_executable",
        "agy_version",
    ):
        _required_text(value[key], f"{where}.{key}")
    sources = value["imported_sources"]
    if not isinstance(sources, dict):
        raise ExperimentError(f"{where}.imported_sources must be an object")
    _require_keys(
        sources,
        {
            "spade_live_runner",
            "proofpack_qualifier",
            "proofpack_receipt_validator",
            "assay_writer",
        },
        f"{where}.imported_sources",
    )
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise ExperimentError(f"{where}.imported_sources.{name} must be an object")
        _require_keys(source, {"path", "digest"}, f"{where}.imported_sources.{name}")
        _required_text(source["path"], f"{where}.imported_sources.{name}.path")
        _sha256_text(source["digest"], f"{where}.imported_sources.{name}.digest")


def _canonical_output_root(value: Path | str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ExperimentError("run_output_root must be an absolute path")
    return str(path.resolve())


def derive_run_dir(output_root: Path | str, plan: Mapping[str, Any]) -> Path:
    """Return the deterministic run location for a sealed plan under a root."""
    validate_plan(plan)
    canonical_root = _canonical_output_root(output_root)
    if canonical_root != plan["run_output_root"]:
        raise ExperimentError("requested output root differs from the sealed run_output_root")
    digest_hex = str(plan["plan_digest"]).removeprefix("sha256:")
    return Path(canonical_root) / f"{plan['experiment_id']}-{digest_hex}"


@contextmanager
def _single_writer(run_dir: Path):
    """Hold a non-blocking exclusive lock for the entire execution/resume."""
    _mkdir_durable(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ExperimentError(f"run directory is not a safe directory: {run_dir}")
    lock_path = run_dir / ".writer.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExperimentIncomplete(
                f"another writer already owns the experiment run: {run_dir}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def build_plan(
    *,
    experiment_id: str,
    model: str,
    skills: Sequence[str],
    difficulties: Sequence[str],
    qualification_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    total_call_cap: int,
    reserve_count: int = 0,
    max_turns: int = 5,
    design_attempts_per_slot: int = 3,
    hint_attempts: int = 2,
    llm_timeout_seconds: float = 180.0,
    qualification_timeout_seconds: float = 5.0,
    minimum_certification_clusters: int = 4,
    alpha: float = 0.05,
    non_inferiority_margin: float = 0.0,
    source_revisions: Mapping[str, str] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    stage: str = "pilot",
    analysis_role: str = "calibration-only",
    protocol_id: str = "spade-agy-generic-grid/v1",
    run_output_root: Path | str,
) -> dict[str, Any]:
    """Construct a deterministic sealed plan over a skill × difficulty grid."""
    _safe_id(experiment_id, "experiment_id")
    _safe_id(stage, "stage")
    _safe_id(analysis_role, "analysis_role")
    _required_text(protocol_id, "protocol_id")
    if protocol_id != protocol_id.strip():
        raise ExperimentError("protocol_id cannot contain leading/trailing whitespace")
    _required_text(model, "model")
    if model != model.strip():
        raise ExperimentError("model cannot contain leading/trailing whitespace")
    if model == "agy-subscription":
        raise ExperimentError("model must name an explicit agy route")
    if isinstance(reserve_count, bool) or not isinstance(reserve_count, int) or reserve_count < 0:
        raise ExperimentError("reserve_count must be a non-negative integer")
    checked_skills = [_required_text(item, "skills[]") for item in skills]
    checked_difficulties = [_required_text(item, "difficulties[]") for item in difficulties]
    if any(item != item.strip() for item in checked_skills + checked_difficulties):
        raise ExperimentError("skills/difficulties cannot contain leading/trailing whitespace")
    if not checked_skills or len(set(checked_skills)) != len(checked_skills):
        raise ExperimentError("skills must be non-empty and unique")
    if not checked_difficulties or len(set(checked_difficulties)) != len(checked_difficulties):
        raise ExperimentError("difficulties must be non-empty and unique")
    qualification = sorted(set(qualification_seeds))
    evaluation = sorted(set(evaluation_seeds))
    if (
        not qualification
        or not evaluation
        or any(isinstance(item, bool) or not isinstance(item, int) for item in qualification)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in evaluation)
    ):
        raise ExperimentError("qualification/evaluation seeds must be non-empty integer sets")
    if not set(evaluation).issubset(qualification):
        raise ExperimentError("evaluation seeds must be a subset of qualification seeds")

    clusters: list[dict[str, Any]] = []
    for skill_index, skill in enumerate(checked_skills, start=1):
        for difficulty_index, difficulty in enumerate(checked_difficulties, start=1):
            cluster_id = (
                f"c{len(clusters) + 1:03d}-{_slug(skill)}-{_slug(difficulty)}"
            )
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "skill": skill,
                    "difficulty": difficulty,
                    "candidate_slots": [
                        {
                            "candidate_id": f"{cluster_id}-primary",
                            "role": "primary",
                        }
                    ],
                    "skill_ordinal": skill_index,
                    "difficulty_ordinal": difficulty_index,
                }
            )

    # Pilot defaults allocate one same-skill/difficulty reserve to each hard cluster
    # before cycling over the remaining fixed schedule.
    reserve_order = [
        index
        for index, cluster in enumerate(clusters)
        if cluster["difficulty"] == checked_difficulties[-1]
    ] + [
        index
        for index, cluster in enumerate(clusters)
        if cluster["difficulty"] != checked_difficulties[-1]
    ]
    for reserve_index in range(reserve_count):
        cluster = clusters[reserve_order[reserve_index % len(reserve_order)]]
        slot_number = len(cluster["candidate_slots"])
        cluster["candidate_slots"].append(
            {
                "candidate_id": f"{cluster['cluster_id']}-reserve-{slot_number:02d}",
                "role": "reserve",
            }
        )

    revisions = dict(source_revisions or _source_revisions())
    runtime = dict(runtime_identity or _runtime_identity())
    sealed_output_root = _canonical_output_root(run_output_root)
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": experiment_id,
        "stage": stage,
        "analysis_role": analysis_role,
        "protocol_id": protocol_id,
        "run_output_root": sealed_output_root,
        "provider": "agy",
        "model": model,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "source_revisions": revisions,
        "runtime_identity": runtime,
        "skills": checked_skills,
        "difficulties": checked_difficulties,
        "qualification_seeds": qualification,
        "evaluation_seeds": evaluation,
        "configuration": {
            "action_format": "boxed",
            "max_turns": max_turns,
            "design_attempts_per_slot": design_attempts_per_slot,
            "hint_attempts": hint_attempts,
            "llm_timeout_seconds": llm_timeout_seconds,
            "qualification_timeout_seconds": qualification_timeout_seconds,
            "total_call_cap": total_call_cap,
            "minimum_certification_clusters": minimum_certification_clusters,
            "alpha": alpha,
            "non_inferiority_margin": non_inferiority_margin,
        },
        "cluster_schedule": clusters,
        "outcome_schedule": _outcome_schedule(clusters, evaluation),
    }
    plan = {**body, "plan_digest": _digest(body)}
    validate_plan(plan)
    return plan


def build_pilot_plan(
    *,
    experiment_id: str,
    model: str,
    run_output_root: Path | str,
    total_call_cap: int = 450,
) -> dict[str, Any]:
    """Build the documented 18-cluster/27-candidate-slot calibration pilot."""
    return build_plan(
        experiment_id=experiment_id,
        model=model,
        skills=PILOT_SKILLS,
        difficulties=PILOT_DIFFICULTIES,
        qualification_seeds=live.QUALIFIED_SEEDS,
        evaluation_seeds=live.QUALIFIED_SEEDS,
        total_call_cap=total_call_cap,
        reserve_count=9,
        minimum_certification_clusters=18,
        non_inferiority_margin=0.10,
        protocol_id="spade-agy-18-cluster-pilot/v1",
        run_output_root=run_output_root,
    )


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Validate every plan field and its deterministic schedules."""
    _require_keys(
        plan,
        {
            "schema_version",
            "experiment_id",
            "stage",
            "analysis_role",
            "protocol_id",
            "run_output_root",
            "provider",
            "model",
            "backend_identity_attested",
            "route_authority",
            "source_revisions",
            "runtime_identity",
            "skills",
            "difficulties",
            "qualification_seeds",
            "evaluation_seeds",
            "configuration",
            "cluster_schedule",
            "outcome_schedule",
            "plan_digest",
        },
        "plan",
    )
    if plan["schema_version"] != PLAN_SCHEMA or plan["provider"] != "agy":
        raise ExperimentError("unsupported plan schema/provider")
    _safe_id(plan["experiment_id"], "experiment_id")
    _safe_id(plan["stage"], "stage")
    _safe_id(plan["analysis_role"], "analysis_role")
    protocol_id = _required_text(plan["protocol_id"], "protocol_id")
    if protocol_id not in {
        "spade-agy-generic-grid/v1",
        "spade-agy-18-cluster-pilot/v1",
    }:
        raise ExperimentError("unsupported experiment protocol_id")
    if plan["stage"] != "pilot" or plan["analysis_role"] != "calibration-only":
        raise ExperimentError("this runner supports only pilot/calibration-only plans")
    if plan["run_output_root"] != _canonical_output_root(plan["run_output_root"]):
        raise ExperimentError("run_output_root is not canonical")
    model = _required_text(plan["model"], "model")
    if model != model.strip():
        raise ExperimentError("model cannot contain leading/trailing whitespace")
    if model == "agy-subscription":
        raise ExperimentError("plan must bind an explicit agy model")
    if plan["backend_identity_attested"] is not False:
        raise ExperimentError("agy backend identity must remain unattested")
    if plan["route_authority"] != "requested-route-only":
        raise ExperimentError("agy route authority must be requested-route-only")

    revisions = plan["source_revisions"]
    if not isinstance(revisions, dict):
        raise ExperimentError("source_revisions must be an object")
    _require_keys(revisions, {"spade", "proofpack", "assay"}, "source_revisions")
    for name, revision in revisions.items():
        text = _required_text(revision, f"source_revisions.{name}")
        if re.fullmatch(r"[0-9a-f]{40}", text) is None:
            raise ExperimentError(f"source_revisions.{name} must be a 40-character commit")

    _validate_runtime_identity(plan["runtime_identity"])

    for field in ("skills", "difficulties", "qualification_seeds", "evaluation_seeds"):
        if not isinstance(plan[field], list) or not plan[field]:
            raise ExperimentError(f"{field} must be a non-empty list")
    skills = plan["skills"]
    difficulties = plan["difficulties"]
    if any(not isinstance(item, str) or not item.strip() for item in skills) or len(
        set(skills)
    ) != len(skills) or any(item != item.strip() for item in skills):
        raise ExperimentError("skills must contain unique non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in difficulties) or len(
        set(difficulties)
    ) != len(difficulties) or any(item != item.strip() for item in difficulties):
        raise ExperimentError("difficulties must contain unique non-empty strings")
    qualification = plan["qualification_seeds"]
    evaluation = plan["evaluation_seeds"]
    if (
        any(isinstance(item, bool) or not isinstance(item, int) for item in qualification)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in evaluation)
        or qualification != sorted(set(qualification))
        or evaluation != sorted(set(evaluation))
        or not set(evaluation).issubset(qualification)
    ):
        raise ExperimentError("seeds must be sorted unique integers; evaluation ⊆ qualification")

    config = plan["configuration"]
    if not isinstance(config, dict):
        raise ExperimentError("configuration must be an object")
    _require_keys(
        config,
        {
            "action_format",
            "max_turns",
            "design_attempts_per_slot",
            "hint_attempts",
            "llm_timeout_seconds",
            "qualification_timeout_seconds",
            "total_call_cap",
            "minimum_certification_clusters",
            "alpha",
            "non_inferiority_margin",
        },
        "configuration",
    )
    if config["action_format"] != "boxed":
        raise ExperimentError("only boxed action format is supported")
    for field in (
        "max_turns",
        "design_attempts_per_slot",
        "hint_attempts",
        "total_call_cap",
        "minimum_certification_clusters",
    ):
        _positive_int(config[field], f"configuration.{field}")
    if config["minimum_certification_clusters"] < 4:
        raise ExperimentError("minimum_certification_clusters cannot be below 4")
    _finite_number(config["llm_timeout_seconds"], "llm_timeout_seconds", positive=True)
    _finite_number(
        config["qualification_timeout_seconds"],
        "qualification_timeout_seconds",
        positive=True,
    )
    alpha = _finite_number(config["alpha"], "alpha", positive=True)
    if alpha >= 1.0:
        raise ExperimentError("alpha must be below 1")
    _finite_number(config["non_inferiority_margin"], "non_inferiority_margin")

    clusters = plan["cluster_schedule"]
    if not isinstance(clusters, list) or not clusters:
        raise ExperimentError("cluster_schedule must be non-empty")
    if config["minimum_certification_clusters"] > len(clusters):
        raise ExperimentError(
            "minimum_certification_clusters exceeds the sealed cluster schedule"
        )
    seen_clusters: set[str] = set()
    seen_candidates: set[str] = set()
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            raise ExperimentError(f"cluster_schedule[{index}] must be an object")
        _require_keys(
            cluster,
            {
                "cluster_id",
                "skill",
                "difficulty",
                "candidate_slots",
                "skill_ordinal",
                "difficulty_ordinal",
            },
            f"cluster_schedule[{index}]",
        )
        cluster_id = _safe_id(cluster["cluster_id"], f"cluster_schedule[{index}].cluster_id")
        if cluster_id in seen_clusters:
            raise ExperimentError(f"duplicate cluster_id: {cluster_id}")
        seen_clusters.add(cluster_id)
        if cluster["skill"] not in skills or cluster["difficulty"] not in difficulties:
            raise ExperimentError(f"cluster {cluster_id} uses undeclared skill/difficulty")
        _positive_int(cluster["skill_ordinal"], "skill_ordinal")
        _positive_int(cluster["difficulty_ordinal"], "difficulty_ordinal")
        slots = cluster["candidate_slots"]
        if not isinstance(slots, list) or not slots:
            raise ExperimentError(f"cluster {cluster_id} must have candidate slots")
        for slot_index, slot in enumerate(slots):
            if not isinstance(slot, dict):
                raise ExperimentError("candidate slot must be an object")
            _require_keys(slot, {"candidate_id", "role"}, "candidate slot")
            candidate_id = _safe_id(slot["candidate_id"], "candidate_id")
            expected_role = "primary" if slot_index == 0 else "reserve"
            if slot["role"] != expected_role:
                raise ExperimentError(f"candidate {candidate_id} must have role {expected_role}")
            if candidate_id in seen_candidates:
                raise ExperimentError(f"duplicate candidate_id: {candidate_id}")
            seen_candidates.add(candidate_id)

    expected_outcomes = _outcome_schedule(clusters, evaluation)
    if plan["outcome_schedule"] != expected_outcomes:
        raise ExperimentError("outcome_schedule is not the deterministic counterbalanced schedule")
    if protocol_id == "spade-agy-18-cluster-pilot/v1":
        expected_clusters: list[dict[str, Any]] = []
        for skill_index, skill in enumerate(PILOT_SKILLS, start=1):
            for difficulty_index, difficulty in enumerate(PILOT_DIFFICULTIES, start=1):
                cluster_id = f"c{len(expected_clusters) + 1:03d}-{_slug(skill)}-{_slug(difficulty)}"
                slots = [{"candidate_id": f"{cluster_id}-primary", "role": "primary"}]
                if difficulty == "hard":
                    slots.append(
                        {
                            "candidate_id": f"{cluster_id}-reserve-01",
                            "role": "reserve",
                        }
                    )
                expected_clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "skill": skill,
                        "difficulty": difficulty,
                        "candidate_slots": slots,
                        "skill_ordinal": skill_index,
                        "difficulty_ordinal": difficulty_index,
                    }
                )
        if (
            tuple(skills) != PILOT_SKILLS
            or tuple(difficulties) != PILOT_DIFFICULTIES
            or qualification != sorted(live.QUALIFIED_SEEDS)
            or evaluation != sorted(live.QUALIFIED_SEEDS)
            or clusters != expected_clusters
            or config["max_turns"] != 5
            or config["design_attempts_per_slot"] != 3
            or config["hint_attempts"] != 2
            or config["llm_timeout_seconds"] != 180.0
            or config["qualification_timeout_seconds"] != 5.0
            or config["minimum_certification_clusters"] != 18
            or config["alpha"] != 0.05
            or config["non_inferiority_margin"] != 0.10
        ):
            raise ExperimentError("18-cluster pilot protocol fields differ from its template")
    body = {key: value for key, value in plan.items() if key != "plan_digest"}
    if plan["plan_digest"] != _digest(body):
        raise ExperimentError("plan_digest does not match the canonical plan body")


def _write_immutable_bytes(target: Path, content: bytes) -> None:
    _reject_symlink_ancestors(target)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise ExperimentError(f"immutable path has conflicting bytes: {target}")
        return
    _mkdir_durable(target.parent)
    _reject_symlink_ancestors(target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o644)
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError(f"zero-byte write staging {target}")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary_name, target, follow_symlinks=False)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                raise ExperimentError(f"immutable path raced with different bytes: {target}")
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _write_json(target: Path, value: object) -> None:
    _write_immutable_bytes(target, _pretty_json(value))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    """Create a directory chain and durably publish every newly created entry."""
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.is_symlink():
        raise ExperimentError(f"symlinked artifact paths are forbidden: {cursor}")
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


def _reject_symlink_ancestors(target: Path) -> None:
    """Reject a target whose path itself or any existing ancestor is a symlink."""
    candidate = target
    while True:
        if candidate.is_symlink():
            raise ExperimentError(f"symlinked artifact paths are forbidden: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _read_json(target: Path) -> dict[str, Any]:
    _reject_symlink_ancestors(target)
    if target.is_symlink() or not target.is_file():
        raise ExperimentError(f"required immutable JSON leaf is missing or unsafe: {target}")
    content = target.read_bytes()
    value = _decode_json(content, str(target))
    if content != _pretty_json(value):
        raise ExperimentError(f"JSON leaf is not in canonical pretty form: {target}")
    return value


def _validate_leaf(
    value: Mapping[str, Any],
    *,
    schema: str,
    keys: set[str],
    where: str,
    expected: Mapping[str, Any] | None = None,
) -> None:
    _require_keys(value, keys, where)
    if value["schema_version"] != schema:
        raise ExperimentError(f"{where} has unsupported schema_version")
    for key, expected_value in (expected or {}).items():
        if value[key] != expected_value:
            raise ExperimentError(f"{where} has conflicting {key}")


def _validate_timestamp(value: object, where: str) -> datetime:
    text = _required_text(value, where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExperimentError(f"{where} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ExperimentError(f"{where} must include a timezone")
    return parsed


def write_plan(path: Path | str, plan: Mapping[str, Any]) -> Path:
    validate_plan(plan)
    target = Path(path)
    _write_json(target, plan)
    return target


def load_plan(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    plan = _read_json(target)
    validate_plan(plan)
    return plan


def _normalize_solution(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentError("oracle solution contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_solution(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _normalize_solution(item) for key, item in sorted(value.items())}
    return str(value)


def _call_id(purpose: Mapping[str, Any]) -> str:
    phase = _slug(str(purpose.get("phase", "call")))[:24]
    return f"{phase}-{_digest(purpose).removeprefix('sha256:')[:24]}"


def _default_dependencies(
    plan: Mapping[str, Any],
    *,
    source_revisions: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
) -> RunnerDependencies:
    live._require_integrations()
    client_or_bin, resolved_model = live.get_llm_client(
        "agy",
        str(plan["model"]),
        timeout_seconds=float(plan["configuration"]["llm_timeout_seconds"]),
    )
    if resolved_model != plan["model"]:
        raise ExperimentError("agy resolved a different model than the sealed plan")
    resolved_executable = Path(str(client_or_bin)).resolve()
    if str(resolved_executable) != runtime_identity["agy_executable"]:
        raise ExperimentError("agy resolved an executable path different from the sealed runtime")
    if _bytes_digest(resolved_executable.read_bytes()) != runtime_identity[
        "agy_executable_digest"
    ]:
        raise ExperimentError("agy executable bytes differ from the sealed runtime")
    return RunnerDependencies(
        llm_call=live.call_llm,
        qualify=live.qualify_spade_environment,
        target_factory=live.SpadeEnvironmentTarget,
        assay_writer=live.write_spade_evaluation,
        task_factory=live.SpadeTaskPayload,
        cluster_factory=live.SpadeClusterData,
        run_metadata_factory=live.SpadeRunMetadata,
        client_or_bin=client_or_bin,
        source_revisions=dict(source_revisions),
        runtime_identity=dict(runtime_identity),
    )


class _Engine:
    def __init__(
        self,
        plan: dict[str, Any],
        plan_bytes: bytes,
        run_dir: Path,
        dependencies: RunnerDependencies,
    ) -> None:
        self.plan = plan
        self.plan_bytes = plan_bytes
        self.run_dir = run_dir
        self.dependencies = dependencies
        self.config = plan["configuration"]

    @property
    def call_count(self) -> int:
        calls = self.run_dir / "calls"
        return len(list(calls.glob("*/request.json"))) if calls.is_dir() else 0

    def _verify_provider_executable(self) -> None:
        client = self.dependencies.client_or_bin
        if not isinstance(client, (str, os.PathLike)):
            return
        executable = Path(client).resolve()
        if str(executable) != self.plan["runtime_identity"]["agy_executable"]:
            raise ExperimentIncomplete("agy executable path drifted after plan sealing")
        if not executable.is_file() or _bytes_digest(executable.read_bytes()) != self.plan[
            "runtime_identity"
        ]["agy_executable_digest"]:
            raise ExperimentIncomplete("agy executable bytes drifted after plan sealing")

    def _validate_run_tree(self) -> None:
        """Reject every symlink below the run root before reading or writing leaves."""
        if self.run_dir.is_symlink() or not self.run_dir.is_dir():
            raise ExperimentError(f"run directory is missing or unsafe: {self.run_dir}")
        for current, directory_names, file_names in os.walk(
            self.run_dir, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            for name in (*directory_names, *file_names):
                path = current_path / name
                if path.is_symlink():
                    raise ExperimentError(f"symlinks are forbidden in experiment runs: {path}")
        lock_path = self.run_dir / ".writer.lock"
        if lock_path.exists() and not lock_path.is_file():
            raise ExperimentError("writer lock must be a regular file")

    def initialize(self) -> None:
        if dict(self.dependencies.source_revisions) != self.plan["source_revisions"]:
            raise ExperimentIncomplete("current source revisions differ from the sealed plan")
        if dict(self.dependencies.runtime_identity) != self.plan["runtime_identity"]:
            raise ExperimentIncomplete("current runtime identity differs from the sealed plan")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._validate_run_tree()
        _write_immutable_bytes(self.run_dir / "plan.json", self.plan_bytes)
        _write_json(
            self.run_dir / "run-manifest.json",
            {
                "schema_version": RUN_SCHEMA,
                "experiment_id": self.plan["experiment_id"],
                "plan_digest": self.plan["plan_digest"],
                "provider": "agy",
                "model": self.plan["model"],
                "backend_identity_attested": False,
                "route_authority": "requested-route-only",
                "total_call_cap": self.config["total_call_cap"],
                "stage": self.plan["stage"],
                "analysis_role": self.plan["analysis_role"],
                "protocol_id": self.plan["protocol_id"],
                "source_revisions": self.plan["source_revisions"],
                "runtime_identity": self.plan["runtime_identity"],
            },
        )
        calls_root = self.run_dir / "calls"
        for artifact_root_name in ("calls", "candidates", "selections", "outcomes"):
            artifact_root = self.run_dir / artifact_root_name
            if artifact_root.is_symlink():
                raise ExperimentError(f"unsafe symlinked artifact root: {artifact_root}")
        if calls_root.is_dir():
            for directory in calls_root.iterdir():
                if directory.is_symlink() or not directory.is_dir():
                    raise ExperimentError(f"unsafe call directory entry: {directory}")
                inventory = {item.name for item in directory.iterdir()}
                if not inventory.issubset({"request.json", "result.json"}):
                    raise ExperimentError(f"unexpected call artifacts in {directory}")
        request_paths = sorted(calls_root.glob("*/request.json")) if calls_root.is_dir() else []
        result_paths = sorted(calls_root.glob("*/result.json")) if calls_root.is_dir() else []
        if {path.parent for path in result_paths} - {path.parent for path in request_paths}:
            raise ExperimentError("a call result exists without its request reservation")
        ordinals: list[int] = []
        for request_path in request_paths:
            request = self._validate_call_request(_read_json(request_path))
            if request_path.parent.name != request["call_id"]:
                raise ExperimentError("call request is stored under a non-canonical directory")
            ordinals.append(int(request["call_ordinal"]))
            result_path = request_path.parent / "result.json"
            if not result_path.is_file():
                raise AmbiguousCall(
                    f"call {request_path.parent.name} has a request reservation but no result; "
                    "refusing to replay an attempt of unknown provider disposition"
                )
            self._validate_call_result(_read_json(result_path), request)
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            raise ExperimentError("call reservations do not have one contiguous ordinal sequence")

    def _validate_call_request(
        self,
        request: Mapping[str, Any],
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        where = f"call request {request.get('call_id', '<unknown>')}"
        _validate_leaf(
            request,
            schema=CALL_REQUEST_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "call_id",
                "call_ordinal",
                "reservation_status",
                "reserved_at_utc",
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
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "provider": "agy",
                "model": self.plan["model"],
                "backend_identity_attested": False,
                "route_authority": "requested-route-only",
                "runtime_identity_digest": _digest(self.plan["runtime_identity"]),
                "agy_executable_digest": self.plan["runtime_identity"][
                    "agy_executable_digest"
                ],
                "reservation_status": "reserved-before-spawn",
                "timeout_seconds": self.config["llm_timeout_seconds"],
                **dict(expected or {}),
            },
        )
        _safe_id(request["call_id"], f"{where}.call_id")
        _positive_int(request["call_ordinal"], f"{where}.call_ordinal")
        if not isinstance(request["purpose"], dict):
            raise ExperimentError(f"{where}.purpose must be an object")
        prompt = _required_text(request["prompt"], f"{where}.prompt")
        if request["prompt_digest"] != _digest(prompt):
            raise ExperimentError(f"{where} prompt digest mismatch")
        if not isinstance(request["system"], str):
            raise ExperimentError(f"{where}.system must be a string")
        if request["system_digest"] != _digest(request["system"]):
            raise ExperimentError(f"{where} system digest mismatch")
        if request["call_id"] != _call_id(request["purpose"]):
            raise ExperimentError(f"{where} call_id does not bind purpose")
        _validate_timestamp(request["reserved_at_utc"], f"{where}.reserved_at_utc")
        return request

    def _validate_call_result(
        self, result: Mapping[str, Any], request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        call_id = str(request["call_id"])
        where = f"call result {call_id}"
        _validate_leaf(
            result,
            schema=CALL_RESULT_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "call_id",
                "call_ordinal",
                "status",
                "started_at_utc",
                "finished_at_utc",
                "duration_seconds",
                "exit_status",
                "input_tokens",
                "output_tokens",
                "cost_usd",
                "error",
                "response",
                "response_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "call_id": call_id,
                "call_ordinal": request["call_ordinal"],
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
            },
        )
        reserved = _validate_timestamp(
            request["reserved_at_utc"], f"call request {call_id}.reserved_at_utc"
        )
        started = _validate_timestamp(result["started_at_utc"], f"{where}.started_at_utc")
        finished = _validate_timestamp(result["finished_at_utc"], f"{where}.finished_at_utc")
        if not reserved <= started <= finished:
            raise ExperimentError(f"{where} timestamps are out of order")
        duration = _finite_number(result["duration_seconds"], f"{where}.duration_seconds")
        if duration < 0:
            raise ExperimentError(f"{where}.duration_seconds cannot be negative")
        if result["status"] == "success":
            if result["exit_status"] != 0:
                raise ExperimentError(f"{where} success must record exit_status=0")
            if result["error"] is not None:
                raise ExperimentError(f"{where} success cannot contain an error")
            response = _required_text(result["response"], f"{where}.response")
            if result["response_digest"] != _digest(response):
                raise ExperimentError(f"{where} response digest mismatch")
        elif result["status"] == "error":
            if result["exit_status"] is not None:
                raise ExperimentError(f"{where} unknown error exit status must be null")
            _required_text(result["error"], f"{where}.error")
            if result["response"] is not None or result["response_digest"] is not None:
                raise ExperimentError(f"{where} error cannot contain a response")
        else:
            raise ExperimentError(f"{where} has invalid status")
        return result

    async def call(
        self,
        *,
        purpose: Mapping[str, Any],
        prompt: str,
        system: str = "",
    ) -> tuple[str, str]:
        call_id = _call_id(purpose)
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
        }
        if request_path.exists():
            request = self._validate_call_request(_read_json(request_path), expected=expected)
            if not result_path.is_file():
                raise AmbiguousCall(f"call {call_id} is ambiguous and will not be replayed")
            result = self._validate_call_result(_read_json(result_path), request)
            if result["status"] != "success":
                raise CallFailed(f"recorded call {call_id} failed: {result['error']}")
            response = str(result["response"])
            return response, call_id

        if self.call_count >= self.config["total_call_cap"]:
            raise CallCapExceeded(
                f"sealed total LLM call cap {self.config['total_call_cap']} reached"
            )
        self._verify_provider_executable()
        self._validate_run_tree()
        request = {
            "schema_version": CALL_REQUEST_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "call_id": call_id,
            "call_ordinal": self.call_count + 1,
            "reservation_status": "reserved-before-spawn",
            "reserved_at_utc": _utc_now(),
            "purpose": dict(purpose),
            "provider": "agy",
            "model": self.plan["model"],
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "runtime_identity_digest": _digest(self.plan["runtime_identity"]),
            "agy_executable_digest": self.plan["runtime_identity"][
                "agy_executable_digest"
            ],
            "prompt": prompt,
            "prompt_digest": _digest(prompt),
            "system": system,
            "system_digest": _digest(system),
            "timeout_seconds": self.config["llm_timeout_seconds"],
        }
        self._validate_call_request(request, expected=expected)
        _write_json(request_path, request)
        started_at = _utc_now()
        started_clock = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="spade-agy-call-") as temporary_workdir:
                response = await self.dependencies.llm_call(
                    self.dependencies.client_or_bin,
                    str(self.plan["model"]),
                    prompt,
                    system=system,
                    provider="agy",
                    workdir=Path(temporary_workdir),
                    timeout_seconds=float(self.config["llm_timeout_seconds"]),
                )
            response = _required_text(response, f"call {call_id} response")
        except Exception as exc:
            result = {
                "schema_version": CALL_RESULT_SCHEMA,
                "plan_digest": self.plan["plan_digest"],
                "call_id": call_id,
                "call_ordinal": request["call_ordinal"],
                "status": "error",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "duration_seconds": time.monotonic() - started_clock,
                "exit_status": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "error": f"{type(exc).__name__}: {exc}",
                "response": None,
                "response_digest": None,
            }
            self._validate_call_result(result, request)
            _write_json(result_path, result)
            raise CallFailed(f"call {call_id} failed: {type(exc).__name__}: {exc}") from exc
        result = {
            "schema_version": CALL_RESULT_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "call_id": call_id,
            "call_ordinal": request["call_ordinal"],
            "status": "success",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "duration_seconds": time.monotonic() - started_clock,
            "exit_status": 0,
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": None,
            "error": None,
            "response": response,
            "response_digest": _digest(response),
        }
        self._validate_call_result(result, request)
        _write_json(result_path, result)
        return response, call_id

    def _candidate_directory(self, cluster_id: str, candidate_id: str) -> Path:
        return self.run_dir / "candidates" / cluster_id / candidate_id

    def _cluster(self, cluster_id: str) -> Mapping[str, Any]:
        for cluster in self.plan["cluster_schedule"]:
            if cluster["cluster_id"] == cluster_id:
                return cluster
        raise ExperimentError(f"unknown cluster_id in persisted leaf: {cluster_id}")

    def _validate_design_attempt(
        self,
        leaf: Mapping[str, Any],
        *,
        cluster_id: str,
        candidate_id: str,
        attempt: int,
    ) -> Mapping[str, Any]:
        where = f"design attempt {cluster_id}/{candidate_id}/{attempt}"
        common = {
            "schema_version",
            "plan_digest",
            "cluster_id",
            "candidate_id",
            "attempt",
            "status",
            "failure_category",
            "reason",
            "feedback_for_next_attempt",
            "attempt_digest",
        }
        status = leaf.get("status")
        keys = common if status == "call_error" else common | {
            "call_id",
            "code",
            "code_digest",
            "qualification",
            "qualification_digest",
            "environment_name",
            "environment_digest",
        }
        _validate_leaf(
            leaf,
            schema="spade-agy-design-attempt/v1",
            keys=keys,
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "attempt": attempt,
            },
        )
        body = {key: value for key, value in leaf.items() if key != "attempt_digest"}
        if leaf["attempt_digest"] != _digest(body):
            raise ExperimentError(f"{where} digest mismatch")
        _required_text(leaf["reason"], f"{where}.reason")
        if not isinstance(leaf["feedback_for_next_attempt"], str):
            raise ExperimentError(f"{where}.feedback_for_next_attempt must be text")
        purpose = {
            "phase": "designer",
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "attempt": attempt,
        }
        call_id = _call_id(purpose)
        request_path = self.run_dir / "calls" / call_id / "request.json"
        request = self._validate_call_request(
            _read_json(request_path), expected={"call_id": call_id, "purpose": purpose}
        )
        result = self._validate_call_result(_read_json(request_path.parent / "result.json"), request)
        if status == "call_error":
            if leaf["failure_category"] != "designer_exhaustion" or result["status"] != "error":
                raise ExperimentError(f"{where} call-error semantics are contradictory")
            if leaf["feedback_for_next_attempt"] != (
                "\n\nThe prior provider attempt failed; return a complete environment."
            ) or leaf["reason"] != f"call {call_id} failed: {result['error']}":
                raise ExperimentError(f"{where} call-error feedback/reason is contradictory")
        elif status in {"qualified", "rejected"}:
            if leaf["call_id"] != call_id or result["status"] != "success":
                raise ExperimentError(f"{where} does not link a successful designer call")
            code = leaf["code"]
            if not isinstance(code, str) or leaf["code_digest"] != (
                _digest(code) if code else None
            ):
                raise ExperimentError(f"{where} code digest mismatch")
            if code and live.extract_python_code(str(result["response"])) != code:
                raise ExperimentError(f"{where} code differs from designer response")
            qualification = leaf["qualification"]
            if leaf["qualification_digest"] != (
                _digest(qualification) if qualification is not None else None
            ):
                raise ExperimentError(f"{where} qualification digest mismatch")
            if status == "qualified" and leaf["failure_category"] is not None:
                raise ExperimentError(f"{where} qualified status has a failure category")
            if status == "rejected" and leaf["failure_category"] != "proofpack_rejection":
                raise ExperimentError(f"{where} rejected status has the wrong category")
        else:
            raise ExperimentError(f"{where} has invalid status")
        return leaf

    def _validate_probe(
        self,
        probe: Mapping[str, Any],
        *,
        cluster_id: str,
        candidate_id: str,
        seed: int,
    ) -> None:
        where = f"probe {cluster_id}/{candidate_id}/{seed}"
        _validate_leaf(
            probe,
            schema="spade-agy-probe/v1",
            keys={
                "schema_version",
                "plan_digest",
                "cluster_id",
                "candidate_id",
                "seed",
                "observation",
                "observation_digest",
                "solution",
                "solution_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "seed": seed,
            },
        )
        if not isinstance(probe["observation"], str):
            raise ExperimentError(f"{where}.observation must be a string")
        if probe["observation_digest"] != _digest(probe["observation"]):
            raise ExperimentError(f"{where} observation digest mismatch")
        if probe["solution_digest"] != _digest(probe["solution"]):
            raise ExperimentError(f"{where} solution digest mismatch")

    def _validate_hint(
        self,
        hint: Mapping[str, Any],
        *,
        cluster_id: str,
        candidate_id: str,
        seed: int,
        observation: str,
        solution: Any,
    ) -> None:
        where = f"accepted hint {cluster_id}/{candidate_id}/{seed}"
        _validate_leaf(
            hint,
            schema="spade-agy-hint-attempt/v1",
            keys={
                "schema_version",
                "plan_digest",
                "cluster_id",
                "candidate_id",
                "seed",
                "attempt",
                "call_id",
                "status",
                "reason",
                "hint",
                "hint_digest",
                "feedback_for_next_attempt",
                "attempt_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "seed": seed,
                "status": "accepted",
            },
        )
        _positive_int(hint["attempt"], f"{where}.attempt")
        if hint["attempt"] > self.config["hint_attempts"]:
            raise ExperimentError(f"{where}.attempt exceeds the sealed hint limit")
        value = _required_text(hint["hint"], f"{where}.hint")
        if hint["hint_digest"] != _digest(value):
            raise ExperimentError(f"{where} digest mismatch")
        purpose = {
            "phase": "hint",
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "attempt": hint["attempt"],
        }
        if hint["call_id"] != _call_id(purpose):
            raise ExperimentError(f"{where} call_id mismatch")
        request_path = self.run_dir / "calls" / str(hint["call_id"]) / "request.json"
        result_path = request_path.parent / "result.json"
        request = self._validate_call_request(
            _read_json(request_path),
            expected={
                "call_id": hint["call_id"],
                "purpose": purpose,
                "system": "Provide strategy only; never solve the puzzle for the player.",
            },
        )
        result = self._validate_call_result(_read_json(result_path), request)
        if result["status"] != "success" or result["response"] != value:
            raise ExperimentError(f"{where} does not match its successful call result")
        if live.hint_reveals_solution(value, solution, observation):
            raise ExperimentError(f"{where} reveals the locked oracle solution")
        body = {key: value for key, value in hint.items() if key != "attempt_digest"}
        if hint["attempt_digest"] != _digest(body):
            raise ExperimentError(f"{where} attempt digest mismatch")
        if hint["feedback_for_next_attempt"] != (
            "\n\nThe previous hint was unusable. Return only general strategy and no exact "
            "answer values."
        ):
            raise ExperimentError(f"{where} feedback differs from the fixed retry policy")

    def _validate_failed_hint(
        self,
        hint: Mapping[str, Any],
        *,
        cluster_id: str,
        candidate_id: str,
        seed: int,
        observation: str,
        solution: Any,
    ) -> None:
        where = f"failed hint {cluster_id}/{candidate_id}/{seed}"
        _validate_leaf(
            hint,
            schema="spade-agy-hint-attempt/v1",
            keys={
                "schema_version",
                "plan_digest",
                "cluster_id",
                "candidate_id",
                "seed",
                "attempt",
                "call_id",
                "status",
                "reason",
                "hint",
                "hint_digest",
                "feedback_for_next_attempt",
                "attempt_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "seed": seed,
            },
        )
        attempt = _positive_int(hint["attempt"], f"{where}.attempt")
        if attempt > self.config["hint_attempts"]:
            raise ExperimentError(f"{where}.attempt exceeds the sealed hint limit")
        purpose = {
            "phase": "hint",
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "attempt": attempt,
        }
        if hint["call_id"] != _call_id(purpose):
            raise ExperimentError(f"{where} call_id mismatch")
        request_path = self.run_dir / "calls" / str(hint["call_id"]) / "request.json"
        request = self._validate_call_request(
            _read_json(request_path), expected={"call_id": hint["call_id"], "purpose": purpose}
        )
        result = self._validate_call_result(_read_json(request_path.parent / "result.json"), request)
        if hint["status"] == "leaked":
            value = _required_text(hint["hint"], f"{where}.hint")
            if (
                hint["hint_digest"] != _digest(value)
                or result["status"] != "success"
                or result["response"] != value
                or not live.hint_reveals_solution(value, solution, observation)
            ):
                raise ExperimentError(f"{where} leakage semantics are contradictory")
        elif hint["status"] == "call_error":
            if (
                hint["hint"] is not None
                or hint["hint_digest"] is not None
                or result["status"] != "error"
            ):
                raise ExperimentError(f"{where} call-error semantics are contradictory")
        else:
            raise ExperimentError(f"{where} has invalid status")
        body = {key: value for key, value in hint.items() if key != "attempt_digest"}
        if hint["attempt_digest"] != _digest(body):
            raise ExperimentError(f"{where} attempt digest mismatch")
        if hint["feedback_for_next_attempt"] != (
            "\n\nThe previous hint was unusable. Return only general strategy and no exact "
            "answer values."
        ):
            raise ExperimentError(f"{where} feedback differs from the fixed retry policy")

    def _validate_selection(
        self, selection: Mapping[str, Any], cluster: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        cluster_id = str(cluster["cluster_id"])
        where = f"selection {cluster_id}"
        _validate_leaf(
            selection,
            schema=SELECTION_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "cluster_id",
                "candidate_id",
                "candidate_role",
                "skill",
                "difficulty",
                "code",
                "code_digest",
                "environment_name",
                "environment_digest",
                "qualification_digest",
                "probes",
                "hints",
                "selection_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "skill": cluster["skill"],
                "difficulty": cluster["difficulty"],
            },
        )
        slots = {slot["candidate_id"]: slot for slot in cluster["candidate_slots"]}
        candidate_id = str(selection["candidate_id"])
        if candidate_id not in slots:
            raise ExperimentError(f"{where} candidate is outside the sealed reserve schedule")
        if selection["candidate_role"] != slots[candidate_id]["role"]:
            raise ExperimentError(f"{where} candidate role mismatch")
        slot_ids = [slot["candidate_id"] for slot in cluster["candidate_slots"]]
        for earlier_id in slot_ids[: slot_ids.index(candidate_id)]:
            disposition = self._validate_disposition(
                _read_json(
                    self._candidate_directory(cluster_id, str(earlier_id))
                    / "disposition.json"
                ),
                cluster_id=cluster_id,
                candidate_id=str(earlier_id),
            )
            if disposition["status"] != "rejected":
                raise ExperimentError(f"{where} bypasses an earlier eligible candidate slot")
        code = _required_text(selection["code"], f"{where}.code")
        if selection["code_digest"] != _digest(code):
            raise ExperimentError(f"{where} code digest mismatch")
        _required_text(selection["environment_name"], f"{where}.environment_name")
        _sha256_text(selection["environment_digest"], f"{where}.environment_digest")
        _sha256_text(selection["qualification_digest"], f"{where}.qualification_digest")
        raw_code_digest = "sha256:" + hashlib.sha256(
            code.encode("utf-8", errors="replace")
        ).hexdigest()
        if selection["environment_digest"] != raw_code_digest:
            raise ExperimentError(f"{where} ProofPack environment digest does not match code")
        qualified_attempts: list[tuple[int, Mapping[str, Any]]] = []
        for attempt in range(1, self.config["design_attempts_per_slot"] + 1):
            path = (
                self._candidate_directory(cluster_id, candidate_id)
                / f"design-attempt-{attempt:02d}.json"
            )
            if not path.is_file():
                continue
            leaf = _read_json(path)
            if leaf.get("status") == "qualified":
                qualified_attempts.append((attempt, leaf))
        if len(qualified_attempts) != 1:
            raise ExperimentError(f"{where} must link exactly one qualified design attempt")
        design_attempt, design = qualified_attempts[0]
        self._validate_design_attempt(
            design,
            cluster_id=cluster_id,
            candidate_id=candidate_id,
            attempt=design_attempt,
        )
        _validate_leaf(
            design,
            schema="spade-agy-design-attempt/v1",
            keys={
                "schema_version",
                "plan_digest",
                "cluster_id",
                "candidate_id",
                "attempt",
                "call_id",
                "status",
                "failure_category",
                "reason",
                "code",
                "code_digest",
                "qualification",
                "qualification_digest",
                "environment_name",
                "environment_digest",
                "feedback_for_next_attempt",
                "attempt_digest",
            },
            where=f"{where} qualified design attempt",
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "status": "qualified",
                "failure_category": None,
                "code": code,
                "code_digest": selection["code_digest"],
                "qualification_digest": selection["qualification_digest"],
                "environment_name": selection["environment_name"],
                "environment_digest": selection["environment_digest"],
            },
        )
        attempt = _positive_int(design["attempt"], f"{where} design attempt")
        purpose = {
            "phase": "designer",
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "attempt": attempt,
        }
        if design["call_id"] != _call_id(purpose):
            raise ExperimentError(f"{where} designer call_id mismatch")
        request_path = self.run_dir / "calls" / str(design["call_id"]) / "request.json"
        request = self._validate_call_request(
            _read_json(request_path),
            expected={"call_id": design["call_id"], "purpose": purpose},
        )
        call_result = self._validate_call_result(
            _read_json(request_path.parent / "result.json"), request
        )
        if call_result["status"] != "success" or live.extract_python_code(
            str(call_result["response"])
        ) != code:
            raise ExperimentError(f"{where} does not match its successful designer call")
        qualification = design["qualification"]
        if not isinstance(qualification, dict) or _digest(qualification) != design[
            "qualification_digest"
        ]:
            raise ExperimentError(f"{where} qualification receipt digest mismatch")
        _require_keys(
            qualification,
            {
                "schema_version",
                "passed",
                "environment_name",
                "environment_digest",
                "clauses",
                "metadata",
            },
            f"{where} qualification receipt",
        )
        if (
            qualification["schema_version"] != "proofpack-spade-qualification/v2"
            or qualification["passed"] is not True
            or qualification["environment_name"] != selection["environment_name"]
            or qualification["environment_digest"] != selection["environment_digest"]
        ):
            raise ExperimentError(f"{where} persisted qualification receipt is contradictory")
        expected_metadata = {
            "action_format": "boxed",
            "seeds": self.plan["qualification_seeds"],
            "max_turns": self.config["max_turns"],
            "timeout_seconds": self.config["qualification_timeout_seconds"],
            "execution_boundary": "macos-sandbox-exec-worker/v1",
        }
        if qualification["metadata"] != expected_metadata:
            raise ExperimentError(f"{where} qualification metadata differs from the plan")
        expected_clauses = {
            "v0_syntax",
            "v1_sandbox_smoke",
            "v2_oracle_solvable",
            "v3_no_agent_unwinnable",
            "v4_mutation_robustness",
        }
        clauses = qualification["clauses"]
        if not isinstance(clauses, dict) or set(clauses) != expected_clauses:
            raise ExperimentError(f"{where} qualification receipt lacks exact V0-V4 clauses")
        for clause_id, clause in clauses.items():
            if not isinstance(clause, dict) or clause.get("clause_id") != clause_id or clause.get(
                "status"
            ) != "pass":
                raise ExperimentError(f"{where} qualification clause {clause_id} is invalid")
        seeds = {str(seed) for seed in self.plan["evaluation_seeds"]}
        probes = selection["probes"]
        hints = selection["hints"]
        if not isinstance(probes, dict) or set(probes) != seeds:
            raise ExperimentError(f"{where} probe seeds differ from the sealed schedule")
        if not isinstance(hints, dict) or set(hints) != seeds:
            raise ExperimentError(f"{where} hint seeds differ from the sealed schedule")
        for seed in self.plan["evaluation_seeds"]:
            probe = probes[str(seed)]
            hint = hints[str(seed)]
            if not isinstance(probe, dict) or not isinstance(hint, dict):
                raise ExperimentError(f"{where} probe/hint entries must be objects")
            self._validate_probe(
                probe, cluster_id=cluster_id, candidate_id=candidate_id, seed=seed
            )
            self._validate_hint(
                hint,
                cluster_id=cluster_id,
                candidate_id=candidate_id,
                seed=seed,
                observation=str(probe["observation"]),
                solution=probe["solution"],
            )
        body = {key: value for key, value in selection.items() if key != "selection_digest"}
        if selection["selection_digest"] != _digest(body):
            raise ExperimentError(f"{where} digest mismatch")
        return selection

    def _load_selection(self, cluster_id: str) -> Mapping[str, Any]:
        cluster = self._cluster(cluster_id)
        selection = _read_json(self.run_dir / "selections" / f"{cluster_id}.json")
        return self._validate_selection(selection, cluster)

    async def _generate_candidate(
        self, cluster: Mapping[str, Any], slot: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str, str]:
        cluster_id = str(cluster["cluster_id"])
        candidate_id = str(slot["candidate_id"])
        directory = self._candidate_directory(cluster_id, candidate_id)
        feedback = ""
        last_category = "designer_exhaustion"
        last_reason = "all design attempts exhausted"
        for attempt in range(1, self.config["design_attempts_per_slot"] + 1):
            attempt_path = directory / f"design-attempt-{attempt:02d}.json"
            if attempt_path.is_file():
                leaf = self._validate_design_attempt(
                    _read_json(attempt_path),
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                )
                if leaf.get("status") == "qualified":
                    return leaf, "", ""
                last_category = str(leaf.get("failure_category") or last_category)
                last_reason = str(leaf.get("reason") or last_reason)
                feedback = str(leaf.get("feedback_for_next_attempt") or "")
                continue
            prompt = (
                live.DESIGNER_PROMPT.format(skill=cluster["skill"])
                + f"\nSealed candidate slot: {candidate_id} ({slot['role']})."
                + (
                    " Generate a substantively distinct puzzle from the primary slot."
                    if slot["role"] == "reserve"
                    else ""
                )
                + (
                    "\nMedium difficulty: require 2-3 dependent reasoning steps and one "
                    "unique boxed answer."
                    if cluster["difficulty"] == "medium"
                    else "\nHard difficulty: require 4-6 dependent reasoning steps, plausible "
                    "distractors, and one unique boxed answer."
                    if cluster["difficulty"] == "hard"
                    else f"\nRequested difficulty: {cluster['difficulty']}."
                )
                + feedback
            )
            purpose = {
                "phase": "designer",
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "attempt": attempt,
            }
            try:
                raw, call_id = await self.call(
                    purpose=purpose,
                    prompt=prompt,
                    system="Return secure, deterministic environment source code only.",
                )
            except CallFailed as exc:
                feedback = "\n\nThe prior provider attempt failed; return a complete environment."
                attempt_body = {
                        "schema_version": "spade-agy-design-attempt/v1",
                        "plan_digest": self.plan["plan_digest"],
                        "cluster_id": cluster_id,
                        "candidate_id": candidate_id,
                        "attempt": attempt,
                        "status": "call_error",
                        "failure_category": "designer_exhaustion",
                        "reason": str(exc),
                        "feedback_for_next_attempt": feedback,
                }
                leaf = {**attempt_body, "attempt_digest": _digest(attempt_body)}
                self._validate_design_attempt(
                    leaf,
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                )
                _write_json(attempt_path, leaf)
                last_category = "designer_exhaustion"
                last_reason = str(exc)
                continue
            code = ""
            try:
                code = live.extract_python_code(raw)
                report = self.dependencies.qualify(
                    code,
                    seeds=list(self.plan["qualification_seeds"]),
                    timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                    max_turns=int(self.config["max_turns"]),
                )
                report_value = _decode_json(
                    (report.to_json() + "\n").encode("utf-8"),
                    f"qualification report for {candidate_id}",
                )
                passed, reason = validate_positive_proofpack_receipt(
                    report,
                    game_code=code,
                    action_format="boxed",
                    seeds=self.plan["qualification_seeds"],
                    timeout_seconds=float(self.config["qualification_timeout_seconds"]),
                    max_turns=int(self.config["max_turns"]),
                )
            except Exception as exc:
                report_value = None
                passed = False
                reason = f"{type(exc).__name__}: {exc}"
            failure_category = None if passed else "proofpack_rejection"
            feedback = (
                "\n\nThe previous candidate failed pre-outcome qualification:\n"
                f"{reason}\nReturn a complete corrected environment, not a patch."
            )
            attempt_body = {
                "schema_version": "spade-agy-design-attempt/v1",
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "attempt": attempt,
                "call_id": call_id,
                "status": "qualified" if passed else "rejected",
                "failure_category": failure_category,
                "reason": reason,
                "code": code,
                "code_digest": _digest(code) if code else None,
                "qualification": report_value,
                "qualification_digest": _digest(report_value) if report_value is not None else None,
                "environment_name": getattr(report, "environment_name", None)
                if report_value is not None
                else None,
                "environment_digest": getattr(report, "environment_digest", None)
                if report_value is not None
                else None,
                "feedback_for_next_attempt": feedback,
            }
            leaf = {**attempt_body, "attempt_digest": _digest(attempt_body)}
            self._validate_design_attempt(
                leaf,
                cluster_id=cluster_id,
                candidate_id=candidate_id,
                attempt=attempt,
            )
            _write_json(attempt_path, leaf)
            if passed:
                return leaf, "", ""
            last_category = "proofpack_rejection"
            last_reason = reason
        return None, last_category, last_reason

    def _reject_candidate(
        self,
        cluster_id: str,
        candidate_id: str,
        category: str,
        reason: str,
    ) -> None:
        body = {
            "schema_version": "spade-agy-candidate-disposition/v1",
            "plan_digest": self.plan["plan_digest"],
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "status": "rejected",
            "category": category,
            "reason": reason,
        }
        _write_json(
            self._candidate_directory(cluster_id, candidate_id) / "disposition.json",
            {**body, "disposition_digest": _digest(body)},
        )

    def _validate_disposition(
        self,
        disposition: Mapping[str, Any],
        *,
        cluster_id: str,
        candidate_id: str,
    ) -> Mapping[str, Any]:
        where = f"candidate disposition {cluster_id}/{candidate_id}"
        _validate_leaf(
            disposition,
            schema="spade-agy-candidate-disposition/v1",
            keys={
                "schema_version",
                "plan_digest",
                "cluster_id",
                "candidate_id",
                "status",
                "category",
                "reason",
                "disposition_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
            },
        )
        body = {key: value for key, value in disposition.items() if key != "disposition_digest"}
        if disposition["disposition_digest"] != _digest(body):
            raise ExperimentError(f"{where} digest mismatch")
        _required_text(disposition["reason"], f"{where}.reason")
        if disposition["status"] == "selected":
            if disposition["category"] is not None:
                raise ExperimentError(f"{where} selected status has a failure category")
        elif disposition["status"] == "rejected":
            if disposition["category"] not in {
                "designer_exhaustion",
                "proofpack_rejection",
                "probe_failure",
                "hint_lock_failure",
                "duplicate_environment",
            }:
                raise ExperimentError(f"{where} has an invalid rejection category")
        else:
            raise ExperimentError(f"{where} has an invalid status")
        return disposition

    async def _hint(
        self,
        *,
        cluster_id: str,
        candidate_id: str,
        seed: int,
        observation: str,
        solution: Any,
    ) -> dict[str, Any] | None:
        directory = self._candidate_directory(cluster_id, candidate_id) / "hints" / str(seed)
        feedback = ""
        for attempt in range(1, self.config["hint_attempts"] + 1):
            path = directory / f"attempt-{attempt:02d}.json"
            if path.is_file():
                leaf = _read_json(path)
                if leaf.get("status") == "accepted":
                    self._validate_hint(
                        leaf,
                        cluster_id=cluster_id,
                        candidate_id=candidate_id,
                        seed=seed,
                        observation=observation,
                        solution=solution,
                    )
                    return leaf
                self._validate_failed_hint(
                    leaf,
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=observation,
                    solution=solution,
                )
                feedback = str(leaf.get("feedback_for_next_attempt") or "")
                continue
            prompt = live.HINT_PROMPT.format(observation=observation) + feedback
            purpose = {
                "phase": "hint",
                "cluster_id": cluster_id,
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
                reason = "explicit-answer leakage" if leaked else "accepted"
            except CallFailed as exc:
                hint = None
                call_id = _call_id(purpose)
                status = "call_error"
                reason = str(exc)
            feedback = (
                "\n\nThe previous hint was unusable. Return only general strategy and "
                "no exact answer values."
            )
            attempt_body = {
                "schema_version": "spade-agy-hint-attempt/v1",
                "plan_digest": self.plan["plan_digest"],
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "seed": seed,
                "attempt": attempt,
                "call_id": call_id,
                "status": status,
                "reason": reason,
                "hint": hint,
                "hint_digest": _digest(hint) if hint is not None else None,
                "feedback_for_next_attempt": feedback,
            }
            leaf = {**attempt_body, "attempt_digest": _digest(attempt_body)}
            if status == "accepted":
                self._validate_hint(
                    leaf,
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=observation,
                    solution=solution,
                )
            else:
                self._validate_failed_hint(
                    leaf,
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=observation,
                    solution=solution,
                )
            _write_json(path, leaf)
            if status == "accepted":
                return leaf
        return None

    async def _prepare_candidate(
        self,
        cluster: Mapping[str, Any],
        slot: Mapping[str, Any],
        selected_code_digests: set[str],
    ) -> dict[str, Any] | None:
        cluster_id = str(cluster["cluster_id"])
        candidate_id = str(slot["candidate_id"])
        selection_path = self.run_dir / "selections" / f"{cluster_id}.json"
        if selection_path.is_file():
            return dict(self._load_selection(cluster_id))
        disposition_path = self._candidate_directory(cluster_id, candidate_id) / "disposition.json"
        if disposition_path.is_file():
            disposition = self._validate_disposition(
                _read_json(disposition_path),
                cluster_id=cluster_id,
                candidate_id=candidate_id,
            )
            if disposition.get("status") == "rejected":
                return None

        generated, failure_category, failure_reason = await self._generate_candidate(cluster, slot)
        if generated is None:
            self._reject_candidate(
                cluster_id, candidate_id, failure_category, failure_reason
            )
            return None
        code = str(generated["code"])
        code_digest = str(generated["code_digest"])
        if code_digest in selected_code_digests:
            self._reject_candidate(
                cluster_id,
                candidate_id,
                "duplicate_environment",
                "environment code duplicates an earlier selected cluster",
            )
            return None

        probes: dict[str, Any] = {}
        hints: dict[str, Any] = {}
        try:
            for seed in self.plan["evaluation_seeds"]:
                probe_path = (
                    self._candidate_directory(cluster_id, candidate_id)
                    / "probes"
                    / f"seed-{seed}.json"
                )
                if probe_path.is_file():
                    probe_leaf = _read_json(probe_path)
                    self._validate_probe(
                        probe_leaf,
                        cluster_id=cluster_id,
                        candidate_id=candidate_id,
                        seed=seed,
                    )
                else:
                    target = self.dependencies.target_factory(
                        code,
                        action_format="boxed",
                        max_turns=int(self.config["max_turns"]),
                        operation_timeout_seconds=float(
                            self.config["qualification_timeout_seconds"]
                        ),
                    )
                    env = target.instantiate()
                    try:
                        observation, _info = env.reset(seed=seed)
                        solution = _normalize_solution(env.solution())
                    finally:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
                    probe_leaf = {
                        "schema_version": "spade-agy-probe/v1",
                        "plan_digest": self.plan["plan_digest"],
                        "cluster_id": cluster_id,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "observation": str(observation),
                        "observation_digest": _digest(str(observation)),
                        "solution": solution,
                        "solution_digest": _digest(solution),
                    }
                    _write_json(probe_path, probe_leaf)
                probes[str(seed)] = probe_leaf
                hint_leaf = await self._hint(
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    observation=str(probe_leaf["observation"]),
                    solution=probe_leaf["solution"],
                )
                if hint_leaf is None:
                    self._reject_candidate(
                        cluster_id,
                        candidate_id,
                        "hint_lock_failure",
                        f"no acceptable hint for seed {seed}",
                    )
                    return None
                hints[str(seed)] = hint_leaf
        except (AmbiguousCall, CallCapExceeded, ExperimentError):
            raise
        except Exception as exc:
            self._reject_candidate(
                cluster_id,
                candidate_id,
                "probe_failure",
                f"{type(exc).__name__}: {exc}",
            )
            return None

        selection_body = {
            "schema_version": "spade-agy-selected-candidate/v1",
            "plan_digest": self.plan["plan_digest"],
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "candidate_role": slot["role"],
            "skill": cluster["skill"],
            "difficulty": cluster["difficulty"],
            "code": code,
            "code_digest": code_digest,
            "environment_name": generated["environment_name"],
            "environment_digest": generated["environment_digest"],
            "qualification_digest": generated["qualification_digest"],
            "probes": probes,
            "hints": hints,
        }
        selection = {
            **selection_body,
            "selection_digest": _digest(selection_body),
        }
        self._validate_selection(selection, cluster)
        _write_json(selection_path, selection)
        self._ensure_selected_disposition(selection)
        return selection

    def _actor_evidence_exists(self) -> bool:
        outcomes = self.run_dir / "outcomes"
        if outcomes.exists():
            return True
        calls = self.run_dir / "calls"
        for request_path in calls.glob("*/request.json") if calls.is_dir() else ():
            request = self._validate_call_request(_read_json(request_path))
            if request["purpose"].get("phase") == "actor":
                return True
        return False

    def _ensure_selected_disposition(self, selection: Mapping[str, Any]) -> None:
        cluster_id = str(selection["cluster_id"])
        candidate_id = str(selection["candidate_id"])
        body = {
            "schema_version": "spade-agy-candidate-disposition/v1",
            "plan_digest": self.plan["plan_digest"],
            "cluster_id": cluster_id,
            "candidate_id": candidate_id,
            "status": "selected",
            "category": None,
            "reason": "qualified, probed, and hint-locked before outcomes",
        }
        value = {**body, "disposition_digest": _digest(body)}
        self._validate_disposition(
            value, cluster_id=cluster_id, candidate_id=candidate_id
        )
        _write_json(
            self._candidate_directory(cluster_id, candidate_id) / "disposition.json",
            value,
        )

    def _validate_cohort(self, cohort: Mapping[str, Any]) -> Mapping[str, Any]:
        _validate_leaf(
            cohort,
            schema=COHORT_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "experiment_id",
                "selections",
                "outcome_schedule_digest",
                "cohort_digest",
            },
            where="cohort lock",
            expected={
                "plan_digest": self.plan["plan_digest"],
                "experiment_id": self.plan["experiment_id"],
                "outcome_schedule_digest": _digest(self.plan["outcome_schedule"]),
            },
        )
        references = cohort["selections"]
        if not isinstance(references, list) or len(references) != len(
            self.plan["cluster_schedule"]
        ):
            raise ExperimentError("cohort lock does not contain the complete sealed cohort")
        seen_code_digests: set[str] = set()
        for cluster, reference in zip(self.plan["cluster_schedule"], references):
            if not isinstance(reference, dict):
                raise ExperimentError("cohort selection reference must be an object")
            _require_keys(
                reference,
                {
                    "cluster_id",
                    "candidate_id",
                    "selection_digest",
                    "code_digest",
                    "qualification_digest",
                    "probe_digests",
                    "hint_digests",
                },
                "cohort selection reference",
            )
            cluster_id = str(cluster["cluster_id"])
            selection = self._load_selection(cluster_id)
            expected = {
                "cluster_id": cluster_id,
                "candidate_id": selection["candidate_id"],
                "selection_digest": selection["selection_digest"],
                "code_digest": selection["code_digest"],
                "qualification_digest": selection["qualification_digest"],
                "probe_digests": {
                    seed: _digest(probe) for seed, probe in selection["probes"].items()
                },
                "hint_digests": {
                    seed: _digest(hint) for seed, hint in selection["hints"].items()
                },
            }
            if reference != expected:
                raise ExperimentError(f"cohort lock differs from selection {cluster_id}")
            code_digest = str(reference["code_digest"])
            if code_digest in seen_code_digests:
                raise ExperimentError("cohort lock contains duplicate environment bytes")
            seen_code_digests.add(code_digest)
        body = {key: value for key, value in cohort.items() if key != "cohort_digest"}
        if cohort["cohort_digest"] != _digest(body):
            raise ExperimentError("cohort lock digest mismatch")
        return cohort

    def _load_cohort(self) -> Mapping[str, Any]:
        return self._validate_cohort(_read_json(self.run_dir / "cohort-lock.json"))

    async def prepare_cohort(self) -> dict[str, Any]:
        lock_path = self.run_dir / "cohort-lock.json"
        if lock_path.is_file():
            return dict(self._load_cohort())
        if self._actor_evidence_exists():
            raise ExperimentIncomplete("actor evidence exists without a cohort lock")
        selections: list[dict[str, Any]] = []
        selected_code_digests: set[str] = set()
        for cluster in self.plan["cluster_schedule"]:
            cluster_id = str(cluster["cluster_id"])
            selection_path = self.run_dir / "selections" / f"{cluster_id}.json"
            selection: dict[str, Any] | None = None
            if selection_path.is_file():
                selection = dict(self._load_selection(cluster_id))
            else:
                for slot in cluster["candidate_slots"]:
                    selection = await self._prepare_candidate(
                        cluster, slot, selected_code_digests
                    )
                    if selection is not None:
                        break
            if selection is None:
                raise ExperimentIncomplete(
                    f"cluster {cluster_id} exhausted its sealed primary/reserve schedule"
                )
            code_digest = str(selection["code_digest"])
            if code_digest in selected_code_digests:
                raise ExperimentIncomplete("selected cohort contains duplicate environment bytes")
            selected_code_digests.add(code_digest)
            self._ensure_selected_disposition(selection)
            selections.append(selection)

        body = {
            "schema_version": COHORT_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "experiment_id": self.plan["experiment_id"],
            "selections": [
                {
                    "cluster_id": item["cluster_id"],
                    "candidate_id": item["candidate_id"],
                    "selection_digest": item["selection_digest"],
                    "code_digest": item["code_digest"],
                    "qualification_digest": item["qualification_digest"],
                    "probe_digests": {
                        seed: _digest(probe) for seed, probe in item["probes"].items()
                    },
                    "hint_digests": {
                        seed: _digest(hint) for seed, hint in item["hints"].items()
                    },
                }
                for item in selections
            ],
            "outcome_schedule_digest": _digest(self.plan["outcome_schedule"]),
        }
        cohort = {**body, "cohort_digest": _digest(body)}
        self._validate_cohort(cohort)
        _write_json(lock_path, cohort)
        return dict(self._load_cohort())

    def _selection(
        self, cluster_id: str, cohort: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        locked = self._load_cohort() if cohort is None else self._validate_cohort(cohort)
        selection = self._load_selection(cluster_id)
        reference = next(
            (
                item
                for item in locked["selections"]
                if item["cluster_id"] == cluster_id
            ),
            None,
        )
        if reference is None or reference["selection_digest"] != selection["selection_digest"]:
            raise ExperimentError(f"selection {cluster_id} is not bound by the cohort lock")
        return selection

    def _validate_turn_leaf(
        self,
        leaf: Mapping[str, Any],
        *,
        item: Mapping[str, Any],
        cohort: Mapping[str, Any],
        turn: int,
        expected_prompt: str | None = None,
    ) -> Mapping[str, Any]:
        outcome_id = str(item["outcome_id"])
        where = f"outcome {outcome_id} turn {turn}"
        _validate_leaf(
            leaf,
            schema=OUTCOME_TURN_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "cohort_digest",
                "outcome_id",
                "cluster_id",
                "seed",
                "arm",
                "turn",
                "call_id",
                "raw_response",
                "raw_response_digest",
                "clean_action",
                "pre_observation_digest",
                "post_observation",
                "post_observation_digest",
                "reward",
                "terminated",
                "truncated",
                "turn_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cohort_digest": cohort["cohort_digest"],
                "outcome_id": outcome_id,
                "cluster_id": item["cluster_id"],
                "seed": item["seed"],
                "arm": item["arm"],
                "turn": turn,
            },
        )
        raw = _required_text(leaf["raw_response"], f"{where}.raw_response")
        if leaf["raw_response_digest"] != _digest(raw):
            raise ExperimentError(f"{where} raw response digest mismatch")
        if leaf["clean_action"] != live.extract_clean_action(raw, "boxed"):
            raise ExperimentError(f"{where} clean action does not match raw response")
        _sha256_text(leaf["pre_observation_digest"], f"{where}.pre_observation_digest")
        if not isinstance(leaf["post_observation"], str):
            raise ExperimentError(f"{where}.post_observation must be a string")
        if leaf["post_observation_digest"] != _digest(leaf["post_observation"]):
            raise ExperimentError(f"{where} post observation digest mismatch")
        _finite_number(leaf["reward"], f"{where}.reward")
        if not isinstance(leaf["terminated"], bool) or not isinstance(
            leaf["truncated"], bool
        ):
            raise ExperimentError(f"{where} termination flags must be booleans")
        purpose = {
            "phase": "actor",
            "outcome_id": outcome_id,
            "cluster_id": item["cluster_id"],
            "seed": item["seed"],
            "arm": item["arm"],
            "turn": turn,
        }
        if leaf["call_id"] != _call_id(purpose):
            raise ExperimentError(f"{where} call_id mismatch")
        request_path = self.run_dir / "calls" / str(leaf["call_id"]) / "request.json"
        result_path = request_path.parent / "result.json"
        request = self._validate_call_request(
            _read_json(request_path),
            expected={
                "call_id": leaf["call_id"],
                "purpose": purpose,
                **(
                    {
                        "prompt": expected_prompt,
                        "prompt_digest": _digest(expected_prompt),
                        "system": "",
                        "system_digest": _digest(""),
                    }
                    if expected_prompt is not None
                    else {}
                ),
            },
        )
        result = self._validate_call_result(_read_json(result_path), request)
        if result["status"] != "success" or result["response"] != raw:
            raise ExperimentError(f"{where} does not match its successful call result")
        body = {key: value for key, value in leaf.items() if key != "turn_digest"}
        if leaf["turn_digest"] != _digest(body):
            raise ExperimentError(f"{where} digest mismatch")
        return leaf

    def _validate_completed_outcome(
        self,
        outcome: Mapping[str, Any],
        *,
        item: Mapping[str, Any],
        cohort: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        outcome_id = str(item["outcome_id"])
        where = f"outcome {outcome_id}"
        _validate_leaf(
            outcome,
            schema=OUTCOME_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "cohort_digest",
                "outcome_id",
                "cluster_id",
                "seed",
                "arm",
                "ordinal",
                "status",
                "return",
                "terminated",
                "truncated",
                "trajectory",
                "outcome_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cohort_digest": cohort["cohort_digest"],
                "outcome_id": outcome_id,
                "cluster_id": item["cluster_id"],
                "seed": item["seed"],
                "arm": item["arm"],
                "ordinal": item["ordinal"],
                "status": "completed",
            },
        )
        _finite_number(outcome["return"], f"{where}.return")
        if not isinstance(outcome["terminated"], bool) or not isinstance(
            outcome["truncated"], bool
        ):
            raise ExperimentError(f"{where} termination flags must be booleans")
        trajectory = outcome["trajectory"]
        if (
            not isinstance(trajectory, list)
            or not 1 <= len(trajectory) <= self.config["max_turns"]
        ):
            raise ExperimentError(f"{where} trajectory length is invalid")
        turn_paths = sorted(
            (self.run_dir / "outcomes" / outcome_id).glob("turn-*.json"),
            key=lambda path: int(path.stem.removeprefix("turn-")),
        )
        expected_paths = [
            self.run_dir / "outcomes" / outcome_id / f"turn-{index:02d}.json"
            for index in range(1, len(trajectory) + 1)
        ]
        if turn_paths != expected_paths:
            raise ExperimentError(f"{where} has missing or extra turn leaves")
        selection = self._selection(str(item["cluster_id"]), cohort)
        probe = selection["probes"][str(item["seed"])]
        observation = str(probe["observation"])
        hint = (
            str(selection["hints"][str(item["seed"])]["hint"])
            if item["arm"] == "hinted"
            else ""
        )
        history = f"Initial observation: {observation}"
        if hint:
            history += f"\n\nPrivileged strategy hint:\n{hint}"
        prior_done = False
        for index, embedded in enumerate(trajectory, start=1):
            if prior_done:
                raise ExperimentError(f"{where} contains a turn after episode completion")
            if not isinstance(embedded, dict):
                raise ExperimentError(f"{where} trajectory entry must be an object")
            disk = _read_json(
                self.run_dir / "outcomes" / outcome_id / f"turn-{index:02d}.json"
            )
            if embedded != disk:
                raise ExperimentError(f"{where} embedded trajectory differs from turn {index}")
            prompt = (
                "You are playing an interactive reasoning environment.\n"
                f"{history}\n\nTurn {index}/{self.config['max_turns']}: reason about the "
                "state, then provide exactly one next action with the required answer format."
            )
            self._validate_turn_leaf(
                disk,
                item=item,
                cohort=cohort,
                turn=index,
                expected_prompt=prompt,
            )
            if disk["pre_observation_digest"] != _digest(observation):
                raise ExperimentError(f"{where} turn {index} breaks the observation chain")
            observation = str(disk["post_observation"])
            history += (
                f"\n\nAction at turn {index}: {disk['clean_action']}\n"
                f"Environment response: {observation}"
            )
            prior_done = bool(disk["terminated"] or disk["truncated"])
        if not prior_done and len(trajectory) != self.config["max_turns"]:
            raise ExperimentError(f"{where} ended before termination or the sealed turn limit")
        self._replay_completed_trajectory(
            selection=selection,
            item=item,
            trajectory=trajectory,
        )
        if trajectory:
            final = trajectory[-1]
            expected_return = float(final["reward"]) if final["terminated"] else 0.0
            if outcome["return"] != expected_return:
                raise ExperimentError(f"{where} return does not match its final turn")
            if outcome["terminated"] != final["terminated"] or outcome["truncated"] != final[
                "truncated"
            ]:
                raise ExperimentError(f"{where} final status differs from its trajectory")
        body = {key: value for key, value in outcome.items() if key != "outcome_digest"}
        if outcome["outcome_digest"] != _digest(body):
            raise ExperimentError(f"{where} digest mismatch")
        return outcome

    def _replay_completed_trajectory(
        self,
        *,
        selection: Mapping[str, Any],
        item: Mapping[str, Any],
        trajectory: Sequence[Mapping[str, Any]],
    ) -> None:
        """Deterministically replay persisted actions before accepting outcome returns."""
        outcome_id = str(item["outcome_id"])
        seed = int(item["seed"])
        probe = selection["probes"][str(seed)]
        target = self.dependencies.target_factory(
            selection["code"],
            action_format="boxed",
            max_turns=int(self.config["max_turns"]),
            operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
        )
        env = target.instantiate()
        try:
            observation, _info = env.reset(seed=seed)
            if str(observation) != probe["observation"]:
                raise ExperimentError(
                    f"outcome {outcome_id} replay reset differs from the locked probe"
                )
            for turn, leaf in enumerate(trajectory, start=1):
                step = env.step(str(leaf["clean_action"]))
                if len(step) == 5:
                    post, reward, terminated, truncated, _step_info = step
                elif len(step) == 4:
                    post, reward, terminated, _step_info = step
                    truncated = False
                else:
                    raise ExperimentError(
                        f"outcome {outcome_id} replay turn {turn} returned {len(step)} values"
                    )
                expected = {
                    "post_observation": str(post),
                    "post_observation_digest": _digest(str(post)),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                for key, value in expected.items():
                    if leaf[key] != value:
                        raise ExperimentError(
                            f"outcome {outcome_id} deterministic replay diverged at "
                            f"turn {turn} {key}"
                        )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _preflight_partial_outcome_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        if directory.is_symlink() or not directory.is_dir():
            raise ExperimentError(f"partial outcome path is unsafe: {directory}")
        entries = list(directory.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ExperimentError(f"partial outcome contains an unsafe entry: {directory}")
        turn_paths = [entry for entry in entries if entry.name != "outcome.json"]
        if any(re.fullmatch(r"turn-[0-9]+\.json", entry.name) is None for entry in turn_paths):
            raise ExperimentError(f"partial outcome contains an unexpected artifact: {directory}")
        indices = sorted(int(entry.stem.removeprefix("turn-")) for entry in turn_paths)
        if indices != list(range(1, len(indices) + 1)):
            raise ExperimentError("partial outcome turn leaves are not one contiguous prefix")
        if indices and indices[-1] > int(self.config["max_turns"]):
            raise ExperimentError("partial outcome exceeds the sealed turn limit")
        expected_names = {f"turn-{index:02d}.json" for index in indices}
        if {entry.name for entry in turn_paths} != expected_names:
            raise ExperimentError("partial outcome turn filenames are not canonical")

    async def _run_outcome(self, item: Mapping[str, Any], cohort: Mapping[str, Any]) -> dict[str, Any]:
        cohort = self._load_cohort()
        outcome_id = str(item["outcome_id"])
        directory = self.run_dir / "outcomes" / outcome_id
        outcome_path = directory / "outcome.json"
        if outcome_path.is_file():
            outcome = _read_json(outcome_path)
            if outcome.get("status") != "completed":
                raise ExperimentIncomplete(f"outcome {outcome_id} previously failed")
            return dict(self._validate_completed_outcome(outcome, item=item, cohort=cohort))
        self._preflight_partial_outcome_directory(directory)

        cluster_id = str(item["cluster_id"])
        seed = int(item["seed"])
        arm = str(item["arm"])
        selection = self._selection(cluster_id, cohort)
        probe = selection["probes"][str(seed)]
        hint = selection["hints"][str(seed)]["hint"] if arm == "hinted" else ""
        target = self.dependencies.target_factory(
            selection["code"],
            action_format="boxed",
            max_turns=int(self.config["max_turns"]),
            operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
        )
        env = target.instantiate()
        trajectory: list[dict[str, Any]] = []
        try:
            observation, _info = env.reset(seed=seed)
            if str(observation) != probe["observation"]:
                raise ExperimentIncomplete(
                    f"outcome {outcome_id} reset diverged from the locked probe"
                )
            history = f"Initial observation: {observation}"
            if hint:
                history += f"\n\nPrivileged strategy hint:\n{hint}"
            terminated = False
            truncated = False
            last_reward = 0.0
            for turn in range(1, int(self.config["max_turns"]) + 1):
                if terminated or truncated:
                    break
                prompt = (
                    "You are playing an interactive reasoning environment.\n"
                    f"{history}\n\nTurn {turn}/{self.config['max_turns']}: reason about the "
                    "state, then provide exactly one next action with the required answer format."
                )
                turn_path = directory / f"turn-{turn:02d}.json"
                purpose = {
                    "phase": "actor",
                    "outcome_id": outcome_id,
                    "cluster_id": cluster_id,
                    "seed": seed,
                    "arm": arm,
                    "turn": turn,
                }
                if turn_path.is_file():
                    turn_leaf = _read_json(turn_path)
                    self._validate_turn_leaf(
                        turn_leaf,
                        item=item,
                        cohort=cohort,
                        turn=turn,
                        expected_prompt=prompt,
                    )
                    if turn_leaf["pre_observation_digest"] != _digest(str(observation)):
                        raise ExperimentIncomplete(
                            f"outcome {outcome_id} turn {turn} pre-state diverged"
                        )
                    action = str(turn_leaf["clean_action"])
                else:
                    try:
                        raw, call_id = await self.call(purpose=purpose, prompt=prompt)
                    except CallFailed as exc:
                        _write_json(
                            outcome_path,
                            {
                                "schema_version": OUTCOME_SCHEMA,
                                "plan_digest": self.plan["plan_digest"],
                                "cohort_digest": cohort["cohort_digest"],
                                "outcome_id": outcome_id,
                                "status": "failed",
                                "error": str(exc),
                            },
                        )
                        raise
                    action = live.extract_clean_action(raw, "boxed")
                    turn_leaf = {
                        "schema_version": OUTCOME_TURN_SCHEMA,
                        "plan_digest": self.plan["plan_digest"],
                        "cohort_digest": cohort["cohort_digest"],
                        "outcome_id": outcome_id,
                        "cluster_id": cluster_id,
                        "seed": seed,
                        "arm": arm,
                        "turn": turn,
                        "call_id": call_id,
                        "raw_response": raw,
                        "raw_response_digest": _digest(raw),
                        "clean_action": action,
                        "pre_observation_digest": _digest(str(observation)),
                    }
                step = env.step(action)
                if len(step) == 5:
                    next_observation, reward, terminated, truncated, _step_info = step
                elif len(step) == 4:
                    next_observation, reward, terminated, _step_info = step
                    truncated = False
                else:
                    raise ExperimentIncomplete(
                        f"outcome {outcome_id} environment returned {len(step)} step values"
                    )
                last_reward = float(reward)
                replay_fields = {
                    "post_observation": str(next_observation),
                    "post_observation_digest": _digest(str(next_observation)),
                    "reward": last_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                if turn_path.is_file():
                    for key, value in replay_fields.items():
                        if turn_leaf.get(key) != value:
                            raise ExperimentIncomplete(
                                f"outcome {outcome_id} turn {turn} replay diverged at {key}"
                            )
                else:
                    turn_leaf.update(replay_fields)
                    turn_leaf["turn_digest"] = _digest(turn_leaf)
                    self._validate_turn_leaf(
                        turn_leaf,
                        item=item,
                        cohort=cohort,
                        turn=turn,
                        expected_prompt=prompt,
                    )
                    _write_json(turn_path, turn_leaf)
                trajectory.append(turn_leaf)
                observation = next_observation
                history += (
                    f"\n\nAction at turn {turn}: {action}\n"
                    f"Environment response: {next_observation}"
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        outcome_body = {
            "schema_version": OUTCOME_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "outcome_id": outcome_id,
            "cluster_id": cluster_id,
            "seed": seed,
            "arm": arm,
            "ordinal": item["ordinal"],
            "status": "completed",
            "return": last_reward if terminated else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "trajectory": trajectory,
        }
        outcome = {**outcome_body, "outcome_digest": _digest(outcome_body)}
        self._validate_completed_outcome(outcome, item=item, cohort=cohort)
        _write_json(outcome_path, outcome)
        return outcome

    async def run_outcomes(self, cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for item in self.plan["outcome_schedule"]:
            outcomes.append(await self._run_outcome(item, cohort))
        expected = len(self.plan["cluster_schedule"]) * len(
            self.plan["evaluation_seeds"]
        ) * 2
        if len(outcomes) != expected or any(item.get("status") != "completed" for item in outcomes):
            raise ExperimentIncomplete("the sealed outcome schedule is incomplete")
        return outcomes

    def _ledger_root(self) -> dict[str, Any]:
        roots = (
            "plan.json",
            "run-manifest.json",
            "calls",
            "candidates",
            "selections",
            "cohort-lock.json",
            "outcomes",
        )
        files: list[dict[str, Any]] = []
        for root_name in roots:
            root = self.run_dir / root_name
            paths = [root] if root.is_file() else sorted(root.rglob("*.json")) if root.is_dir() else []
            for path in paths:
                if path.is_symlink() or not path.is_file():
                    raise ExperimentError(f"unsafe ledger leaf: {path}")
                content = path.read_bytes()
                files.append(
                    {
                        "path": path.relative_to(self.run_dir).as_posix(),
                        "digest": _bytes_digest(content),
                        "size_bytes": len(content),
                    }
                )
        files.sort(key=lambda item: item["path"])
        body = {
            "schema_version": LEDGER_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "leaf_count": len(files),
            "leaves": files,
        }
        result = {**body, "ledger_root_digest": _digest(body)}
        _write_json(self.run_dir / "ledger-root.json", result)
        return result

    def _plain_assay_inputs(
        self,
        cohort: Mapping[str, Any],
        outcomes: Sequence[Mapping[str, Any]],
        ledger: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        by_key = {
            (str(item["cluster_id"]), int(item["seed"]), str(item["arm"])): item
            for item in outcomes
        }
        tasks: list[dict[str, Any]] = []
        clusters: list[dict[str, Any]] = []
        for cluster in self.plan["cluster_schedule"]:
            cluster_id = str(cluster["cluster_id"])
            selection = self._selection(cluster_id, cohort)
            solutions = {
                str(seed): selection["probes"][str(seed)]["solution"]
                for seed in self.plan["evaluation_seeds"]
            }
            solution_json = _canonical_json(solutions)
            solution_digests = {
                str(seed): selection["probes"][str(seed)]["solution_digest"]
                for seed in self.plan["evaluation_seeds"]
            }
            hinted = [
                float(by_key[(cluster_id, seed, "hinted")]["return"])
                for seed in self.plan["evaluation_seeds"]
            ]
            unhinted = [
                float(by_key[(cluster_id, seed, "unhinted")]["return"])
                for seed in self.plan["evaluation_seeds"]
            ]
            tasks.append(
                {
                    "task_id": cluster_id,
                    "environment_name": selection["environment_name"],
                    "skill": cluster["skill"],
                    "code": selection["code"],
                    "solution": solution_json,
                    "max_turns": self.config["max_turns"],
                    "seed": min(self.plan["evaluation_seeds"]),
                    "metadata": {
                        "plan_digest": self.plan["plan_digest"],
                        "cohort_digest": cohort["cohort_digest"],
                        "experiment_ledger_root_digest": ledger["ledger_root_digest"],
                        "solution_digests_by_seed": solution_digests,
                        "selected_candidate_id": selection["candidate_id"],
                        "proofpack_environment_digest": selection["environment_digest"],
                        "proofpack_qualification_digest": selection["qualification_digest"],
                        "requested_provider": "agy",
                        "requested_model": self.plan["model"],
                        "backend_identity_attested": False,
                        "route_authority": "requested-route-only",
                        "stage": self.plan["stage"],
                        "analysis_role": self.plan["analysis_role"],
                        "experiment_ledger_leaf_count": ledger["leaf_count"],
                    },
                }
            )
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "candidate_returns": hinted,
                    "base_returns": unhinted,
                    "hinted_returns": hinted,
                    "regret": max(
                        0.0,
                        (sum(hinted) / len(hinted)) - (sum(unhinted) / len(unhinted)),
                    ),
                }
            )
        return tasks, clusters

    def _validate_assay_result(
        self,
        result: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if result.get("status") != "complete":
            raise ExperimentIncomplete("the prior aggregate Assay write failed")
        _validate_leaf(
            result,
            schema=ASSAY_RESULT_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "status",
                "assay_request_digest",
                "cohort_digest",
                "ledger_root_digest",
                "bundle_digest",
                "report",
                "release_authorized",
                "model_lock_path",
                "assay_file_inventory",
                "evaluation_relative_path",
                "certification_relative_path",
            },
            where="Assay result",
            expected={
                "plan_digest": self.plan["plan_digest"],
                "status": "complete",
                "assay_request_digest": _digest(request),
                "cohort_digest": request["cohort_digest"],
                "ledger_root_digest": request["ledger_root_digest"],
                "release_authorized": False,
                "model_lock_path": None,
            },
        )
        _sha256_text(result["bundle_digest"], "Assay result bundle_digest")
        report = result["report"]
        if not isinstance(report, dict) or report.get("release_authorized") is not False:
            raise ExperimentError("Assay report must explicitly refuse release authorization")
        self._validate_assay_bundle(result)
        return result

    def _assay_file_inventory(self) -> list[dict[str, Any]]:
        root = self.run_dir / "assay"
        if root.is_symlink() or not root.is_dir():
            raise ExperimentError("Assay output root is missing or unsafe")
        inventory: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ExperimentError(f"unsafe symlink in Assay output: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ExperimentError(f"unsafe non-file in Assay output: {path}")
            content = path.read_bytes()
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "digest": _bytes_digest(content),
                    "size_bytes": len(content),
                }
            )
        if not inventory:
            raise ExperimentError("Assay output inventory cannot be empty")
        return inventory

    def _assay_relative_path(self, value: object, where: str) -> str:
        path = Path(value) if isinstance(value, (str, os.PathLike)) else None
        if path is None:
            raise ExperimentError(f"{where} must be a path")
        root = (self.run_dir / "assay").resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ExperimentError(f"{where} escapes the Assay output root")
        return resolved.relative_to(root).as_posix()

    def _validate_assay_bundle(self, result: Mapping[str, Any]) -> None:
        inventory = result["assay_file_inventory"]
        if not isinstance(inventory, list):
            raise ExperimentError("Assay file inventory must be a list")
        seen: set[str] = set()
        for index, entry in enumerate(inventory):
            if not isinstance(entry, dict):
                raise ExperimentError(f"Assay inventory entry {index} must be an object")
            _require_keys(entry, {"path", "digest", "size_bytes"}, "Assay inventory entry")
            relative = _required_text(entry["path"], "Assay inventory path")
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
                raise ExperimentError(f"unsafe Assay inventory path: {relative!r}")
            if relative in seen:
                raise ExperimentError(f"duplicate Assay inventory path: {relative}")
            seen.add(relative)
            _sha256_text(entry["digest"], f"Assay inventory {relative} digest")
            size = entry["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ExperimentError(
                    f"Assay inventory {relative} size_bytes must be a non-negative integer"
                )
        actual = self._assay_file_inventory()
        if inventory != actual:
            raise ExperimentError("Assay output file inventory or bytes changed across resume")
        evaluation_relative = _required_text(
            result["evaluation_relative_path"], "evaluation_relative_path"
        )
        certification_relative = _required_text(
            result["certification_relative_path"], "certification_relative_path"
        )
        if evaluation_relative not in seen or certification_relative not in seen:
            raise ExperimentError("Assay evaluation/certification is absent from the inventory")
        root = self.run_dir / "assay"
        evaluation = _read_json(root / evaluation_relative)
        certification = _read_json(root / certification_relative)
        if evaluation.get("report") != result["report"]:
            raise ExperimentError("Assay result report differs from evaluation.json")
        if certification.get("evaluation_digest") != _digest(evaluation):
            raise ExperimentError("Assay certification does not bind evaluation.json")
        if certification.get("artifact_digest") != result["bundle_digest"]:
            raise ExperimentError("Assay certification artifact_digest differs from bundle_digest")
        if certification.get("release_authorized") is not False:
            raise ExperimentError("Assay certification must explicitly refuse release authorization")
        sealed = {**certification, "artifact_digest": _ZERO_DIGEST}
        if _digest(sealed) != result["bundle_digest"]:
            raise ExperimentError("Assay certification artifact digest is invalid")

    def write_assay(
        self,
        cohort: Mapping[str, Any],
        outcomes: Sequence[Mapping[str, Any]],
    ) -> Path:
        ledger = self._ledger_root()
        tasks_plain, clusters_plain = self._plain_assay_inputs(cohort, outcomes, ledger)
        request = {
            "schema_version": ASSAY_REQUEST_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "ledger_root_digest": ledger["ledger_root_digest"],
            "tasks_digest": _digest(tasks_plain),
            "clusters_digest": _digest(clusters_plain),
            "task_count": len(tasks_plain),
            "cluster_count": len(clusters_plain),
        }
        request_path = self.run_dir / "assay-request.json"
        result_path = self.run_dir / "assay-result.json"
        existed = request_path.is_file()
        _write_json(request_path, request)
        if result_path.is_file():
            result = _read_json(result_path)
            self._validate_assay_result(result, request)
            return result_path
        if existed:
            raise ExperimentIncomplete(
                "aggregate Assay request exists without a result; refusing a second write"
            )

        tasks = [self.dependencies.task_factory(**item) for item in tasks_plain]
        clusters = [self.dependencies.cluster_factory(**item) for item in clusters_plain]
        try:
            artifact = self.dependencies.assay_writer(
                output_dir=self.run_dir / "assay",
                curriculum_id=self.plan["experiment_id"],
                tasks=tuple(tasks),
                clusters=tuple(clusters),
                candidate_arm=f"{self.plan['model']}:hinted",
                base_arm=f"{self.plan['model']}:unhinted",
                run_metadata=self.dependencies.run_metadata_factory(
                    run_id=self.plan["experiment_id"]
                ),
                non_inferiority_margin=self.config["non_inferiority_margin"],
                alpha=self.config["alpha"],
                minimum_clusters=self.config["minimum_certification_clusters"],
            )
            report = artifact.report.to_dict()
            if (
                getattr(artifact.report, "release_authorized", None) is not False
                or not isinstance(report, dict)
                or report.get("release_authorized") is not False
            ):
                raise ExperimentIncomplete(
                    "Assay must explicitly return release_authorized=false for a SPADE signal"
                )
            if not hasattr(artifact, "model_lock_path") or artifact.model_lock_path is not None:
                raise ExperimentIncomplete("Assay unexpectedly emitted a model.lock")
            evaluation_relative = self._assay_relative_path(
                getattr(artifact, "evaluation_path", None), "Assay evaluation_path"
            )
            certification_relative = self._assay_relative_path(
                getattr(artifact, "certification_path", None), "Assay certification_path"
            )
            inventory = self._assay_file_inventory()
            value = {
                "schema_version": ASSAY_RESULT_SCHEMA,
                "plan_digest": self.plan["plan_digest"],
                "status": "complete",
                "assay_request_digest": _digest(request),
                "cohort_digest": request["cohort_digest"],
                "ledger_root_digest": request["ledger_root_digest"],
                "bundle_digest": artifact.bundle_digest,
                "report": report,
                "release_authorized": False,
                "model_lock_path": None,
                "assay_file_inventory": inventory,
                "evaluation_relative_path": evaluation_relative,
                "certification_relative_path": certification_relative,
            }
        except Exception as exc:
            value = {
                "schema_version": ASSAY_RESULT_SCHEMA,
                "plan_digest": self.plan["plan_digest"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(result_path, value)
            raise ExperimentIncomplete(f"aggregate Assay write failed: {exc}") from exc
        self._validate_assay_result(value, request)
        _write_json(result_path, value)
        return result_path

    async def execute(self) -> RunResult:
        self.initialize()
        cohort = await self.prepare_cohort()
        outcomes = await self.run_outcomes(cohort)
        assay_result_path = self.write_assay(cohort, outcomes)
        self._validate_run_tree()
        return RunResult(
            status="complete",
            plan_digest=self.plan["plan_digest"],
            run_dir=self.run_dir,
            call_count=self.call_count,
            cohort_lock_path=self.run_dir / "cohort-lock.json",
            assay_result_path=assay_result_path,
        )


async def run_experiment(
    plan_path: Path | str,
    output_root: Path | str,
    *,
    execute: bool = False,
    acknowledged_call_cap: int | None = None,
    dependencies: RunnerDependencies | None = None,
) -> RunResult:
    """Validate by default; execute only after an exact sealed-cap acknowledgement."""
    plan_file = Path(plan_path)
    plan = load_plan(plan_file)
    destination = derive_run_dir(output_root, plan)
    if not execute:
        return RunResult(
            status="validated",
            plan_digest=plan["plan_digest"],
            run_dir=destination,
            call_count=0,
        )
    expected_cap = int(plan["configuration"]["total_call_cap"])
    if acknowledged_call_cap != expected_cap:
        raise ExperimentError(
            "--acknowledge-call-cap must exactly equal the sealed total_call_cap "
            f"({expected_cap}) before --execute can spend calls"
        )
    if dependencies is None:
        current_sources = _source_revisions()
        current_runtime = _runtime_identity()
        resolved_dependencies = _default_dependencies(
            plan,
            source_revisions=current_sources,
            runtime_identity=current_runtime,
        )
    else:
        resolved_dependencies = dependencies
    engine = _Engine(plan, _pretty_json(plan), destination, resolved_dependencies)
    with _single_writer(destination):
        return await engine.execute()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sealed, resumable multi-environment SPADE agy experiment runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot-plan", help="write the deterministic 18-cluster pilot")
    pilot.add_argument("--output", required=True)
    pilot.add_argument(
        "--output-root",
        required=True,
        help="canonical absolute root sealed into the plan for all run artifacts",
    )
    pilot.add_argument("--experiment-id", required=True)
    pilot.add_argument("--model", required=True, help="explicit agy model route")
    pilot.add_argument("--total-call-cap", type=int, default=450)

    run = subparsers.add_parser("run", help="validate by default or explicitly execute a plan")
    run.add_argument("--plan", required=True)
    run.add_argument(
        "--output-root",
        required=True,
        help="root under which experiment_id + plan digest determines the run directory",
    )
    run.add_argument("--execute", action="store_true")
    run.add_argument("--acknowledge-call-cap", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pilot-plan":
            plan = build_pilot_plan(
                experiment_id=args.experiment_id,
                model=args.model,
                run_output_root=args.output_root,
                total_call_cap=args.total_call_cap,
            )
            target = write_plan(args.output, plan)
            print(
                f"sealed pilot plan: {target} ({len(plan['cluster_schedule'])} clusters, "
                f"{sum(len(item['candidate_slots']) for item in plan['cluster_schedule'])} "
                f"candidate slots, digest {plan['plan_digest']})"
            )
            return 0
        result = asyncio.run(
            run_experiment(
                args.plan,
                args.output_root,
                execute=args.execute,
                acknowledged_call_cap=args.acknowledge_call_cap,
            )
        )
        print(
            f"{result.status}: plan={result.plan_digest}, calls={result.call_count}, "
            f"run_dir={result.run_dir}"
        )
        return 0
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
