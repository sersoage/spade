#!/usr/bin/env python3
"""Run the prospective, outcome-only SPADE AGY replay protocol.

This runner imports a completed *pre-outcome* cohort from the v4 experiment and
never imports actor outcomes.  Its experimental unit is a sealed
``(cluster, seed)`` pair.  A retryable transport failure discards the whole pair
attempt; hinted and unhinted arms are never combined across attempts.

The legacy experiment runner is intentionally left untouched.  This module
reuses its immutable JSON, ProofPack, deterministic replay, and Assay helpers,
but owns a separate plan and artifact schema.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import run_spade_agy_experiment as base  # noqa: E402


PLAN_SCHEMA = "spade-agy-outcome-replay-plan/v1"
RUN_SCHEMA = "spade-agy-outcome-replay-run/v1"
COHORT_SCHEMA = "spade-agy-imported-cohort-lock/v1"
CALL_REQUEST_SCHEMA = "spade-agy-outcome-replay-call-request/v1"
CALL_RESULT_SCHEMA = "spade-agy-outcome-replay-call-result/v1"
TURN_SCHEMA = "spade-agy-outcome-replay-turn/v1"
ATTEMPT_OUTCOME_SCHEMA = "spade-agy-outcome-replay-attempt-outcome/v1"
PAIR_ATTEMPT_SCHEMA = "spade-agy-outcome-replay-pair-attempt/v1"
PAIR_RESOLUTION_SCHEMA = "spade-agy-outcome-replay-pair-resolution/v1"
RESOLUTION_MANIFEST_SCHEMA = "spade-agy-outcome-replay-resolution-manifest/v1"
OUTCOME_REFERENCE_SCHEMA = "spade-agy-outcome-replay-selected-outcome/v1"
QUALIFICATION_REVALIDATION_SCHEMA = "spade-agy-qualification-revalidation/v1"
HORIZON_VIABILITY_SCHEMA = "spade-agy-actor-horizon-viability/v1"
ASSAY_REQUEST_SCHEMA = "spade-agy-outcome-replay-assay-request/v1"

PROTOCOL_ID = "spade-agy-v4-cohort-outcome-replay/v1"
CANONICAL_MODEL = "gemini-3.1-pro-high"
CANONICAL_TIMEOUT_SECONDS = 180.0
SOURCE_MAX_TURNS = 5
PAIR_ATTEMPTS = 2
ARMS = ("unhinted", "hinted")
HORIZON_POLICY_ID = "locked-probe-boxed-oracle-horizon/v1"
IMPORT_ROOT = "imported-v4"
PRIOR_CHARGED_CALLS = 178
AUTHORIZED_GLOBAL_CALL_CAP = 450


@dataclass(frozen=True)
class SourceSnapshot:
    """Validated, allowlisted v4 evidence selected for import."""

    run_dir: Path
    plan: dict[str, Any]
    cohort: dict[str, Any]
    selections: dict[str, dict[str, Any]]
    manifest: list[dict[str, Any]]


def _file_digest(path: Path) -> str:
    return base._bytes_digest(path.read_bytes())


def _safe_relative(value: object, where: str) -> str:
    text = base._required_text(value, where)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise base.ExperimentError(f"{where} is not a canonical safe relative path")
    return text


def _canonical_source_run(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise base.ExperimentError("source_run_dir must be absolute")
    base._reject_symlink_ancestors(path)
    resolved = path.resolve()
    if path != resolved or path.is_symlink() or not path.is_dir():
        raise base.ExperimentError("source_run_dir must be a canonical regular directory")
    return path


def _safe_tree(root: Path, *, json_only: bool = True) -> list[Path]:
    """Inventory a tree without following or accepting any symlink."""
    base._reject_symlink_ancestors(root)
    if root.is_symlink() or not root.is_dir():
        raise base.ExperimentError(f"required source evidence directory is unsafe: {root}")
    leaves: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            if path.is_symlink():
                raise base.ExperimentError(f"symlink in source evidence tree: {path}")
        for name in files:
            path = current_path / name
            if not path.is_file() or (json_only and path.suffix != ".json"):
                raise base.ExperimentError(f"unexpected source evidence leaf: {path}")
            leaves.append(path)
    return sorted(leaves)


def _manifest_entry(path: Path, *, relative_to: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "digest": base._bytes_digest(content),
        "size_bytes": len(content),
    }


def _validate_runtime_files(runtime: Mapping[str, Any]) -> None:
    """Verify every file byte identity sealed by a source runtime."""
    base._validate_runtime_identity(runtime)
    checks = [
        (Path(str(runtime["python_executable"])), runtime["python_executable_digest"]),
        (Path(str(runtime["agy_executable"])), runtime["agy_executable_digest"]),
    ]
    for source in runtime["imported_sources"].values():
        checks.append((Path(str(source["path"])), source["digest"]))
    for path, expected_digest in checks:
        base._reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file() or path != path.resolve():
            raise base.ExperimentError(f"sealed runtime file is missing or unsafe: {path}")
        if _file_digest(path) != expected_digest:
            raise base.ExperimentError(f"sealed runtime file bytes drifted: {path}")
    source_runner = Path(base.__file__).resolve()
    if _file_digest(source_runner) != runtime["runner_digest"]:
        raise base.ExperimentError("source v4 runner bytes differ from its sealed runtime")


def _git_head_for_source(path: Path) -> str:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise base.ExperimentError(f"cannot verify source revision for {path}") from exc


def _validate_source_revisions(plan: Mapping[str, Any]) -> None:
    sources = plan["runtime_identity"]["imported_sources"]
    paths = {
        "spade": Path(sources["spade_live_runner"]["path"]),
        "proofpack": Path(sources["proofpack_qualifier"]["path"]),
        "assay": Path(sources["assay_writer"]["path"]),
    }
    for name, path in paths.items():
        try:
            root = subprocess.check_output(
                ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
            check = subprocess.run(
                [
                    "git",
                    "-C",
                    root,
                    "cat-file",
                    "-e",
                    f"{plan['source_revisions'][name]}^{{commit}}",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise base.ExperimentError(f"cannot verify sealed source revision for {path}") from exc
        if check.returncode != 0:
            raise base.ExperimentError(f"sealed source v4 {name} commit is unavailable")


def _source_dependencies(
    plan: Mapping[str, Any], dependencies: base.RunnerDependencies | None
) -> base.RunnerDependencies:
    if dependencies is not None:
        return dependencies
    return base._default_dependencies(
        plan,
        source_revisions=plan["source_revisions"],
        runtime_identity=plan["runtime_identity"],
    )


def _validate_source_protocol(plan: Mapping[str, Any]) -> None:
    base.validate_plan(plan)
    config = plan["configuration"]
    if (
        plan["protocol_id"] != "spade-agy-18-cluster-pilot/v1"
        or plan["provider"] != "agy"
        or plan["model"] != CANONICAL_MODEL
        or plan["route_authority"] != "requested-route-only"
        or plan["backend_identity_attested"] is not False
        or config["max_turns"] != SOURCE_MAX_TURNS
        or config["llm_timeout_seconds"] != CANONICAL_TIMEOUT_SECONDS
        or len(plan["cluster_schedule"]) != 18
        or len(plan["evaluation_seeds"]) != 3
        or len(plan["outcome_schedule"]) != 108
    ):
        raise base.ExperimentError("source run is not the sealed Google v4 pilot cohort")


def _find_qualified_design(
    source_run: Path,
    source_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = source_run / "candidates" / str(selection["cluster_id"]) / str(selection["candidate_id"])
    matches: list[tuple[Path, dict[str, Any]]] = []
    for attempt in range(1, int(source_plan["configuration"]["design_attempts_per_slot"]) + 1):
        path = root / f"design-attempt-{attempt:02d}.json"
        if path.is_file():
            leaf = base._read_json(path)
            if leaf.get("status") == "qualified":
                matches.append((path, leaf))
    if len(matches) != 1:
        raise base.ExperimentError(
            f"selected candidate {selection['candidate_id']} lacks one qualified design"
        )
    return matches[0]


def _validate_candidate_history(
    source_engine: base._Engine,
    run_dir: Path,
    plan: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
) -> tuple[set[Path], set[str]]:
    """Validate the exact candidate artifact topology and every linked call leaf."""
    root = run_dir / "candidates"
    base._reject_symlink_ancestors(root)
    if root.is_symlink() or not root.is_dir():
        raise base.ExperimentError("source candidate history root is missing or unsafe")
    cluster_map = {str(item["cluster_id"]): item for item in plan["cluster_schedule"]}
    if {path.name for path in root.iterdir()} != set(cluster_map):
        raise base.ExperimentError("source candidate cluster inventory differs from schedule")
    leaves: set[Path] = set()
    call_ids: set[str] = set()
    evaluation_seeds = {str(seed) for seed in plan["evaluation_seeds"]}
    for cluster_id, cluster in cluster_map.items():
        cluster_root = root / cluster_id
        if cluster_root.is_symlink() or not cluster_root.is_dir():
            raise base.ExperimentError(f"unsafe source candidate cluster: {cluster_root}")
        slot_ids = {str(slot["candidate_id"]) for slot in cluster["candidate_slots"]}
        candidate_entries = list(cluster_root.iterdir())
        if any(
            entry.is_symlink() or not entry.is_dir() or entry.name not in slot_ids
            for entry in candidate_entries
        ):
            raise base.ExperimentError(f"source candidate slot inventory is invalid: {cluster_id}")
        ordered_slots = [str(slot["candidate_id"]) for slot in cluster["candidate_slots"]]
        selected_id = str(selections[cluster_id]["candidate_id"])
        expected_present = set(ordered_slots[: ordered_slots.index(selected_id) + 1])
        if {entry.name for entry in candidate_entries} != expected_present:
            raise base.ExperimentError("source candidate history continues after selection")
        for candidate_root in candidate_entries:
            candidate_id = candidate_root.name
            allowed_names = {"disposition.json", "probes", "hints"} | {
                f"design-attempt-{attempt:02d}.json"
                for attempt in range(1, int(plan["configuration"]["design_attempts_per_slot"]) + 1)
            }
            entries = list(candidate_root.iterdir())
            if any(entry.name not in allowed_names or entry.is_symlink() for entry in entries):
                raise base.ExperimentError(
                    f"unexpected source candidate artifact: {candidate_root}"
                )
            design_paths = sorted(candidate_root.glob("design-attempt-*.json"))
            expected_design_names = [
                f"design-attempt-{attempt:02d}.json" for attempt in range(1, len(design_paths) + 1)
            ]
            if [path.name for path in design_paths] != expected_design_names:
                raise base.ExperimentError("source design attempts are not a contiguous prefix")
            qualified_seen = False
            for attempt, path in enumerate(design_paths, start=1):
                design = source_engine._validate_design_attempt(
                    base._read_json(path),
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                )
                if qualified_seen:
                    raise base.ExperimentError("source design attempt follows qualification")
                qualified_seen = design["status"] == "qualified"
                call_ids.add(
                    base._call_id(
                        {
                            "phase": "designer",
                            "cluster_id": cluster_id,
                            "candidate_id": candidate_id,
                            "attempt": attempt,
                        }
                    )
                )
                leaves.add(path)
            disposition_path = candidate_root / "disposition.json"
            if disposition_path.is_file():
                disposition = source_engine._validate_disposition(
                    base._read_json(disposition_path),
                    cluster_id=cluster_id,
                    candidate_id=candidate_id,
                )
                selected_candidate = str(selections[cluster_id]["candidate_id"])
                expected_status = "selected" if candidate_id == selected_candidate else "rejected"
                if disposition["status"] != expected_status:
                    raise base.ExperimentError(
                        "source candidate disposition contradicts the cohort selection"
                    )
                leaves.add(disposition_path)
            probes: dict[str, Mapping[str, Any]] = {}
            probes_root = candidate_root / "probes"
            if probes_root.exists():
                if probes_root.is_symlink() or not probes_root.is_dir():
                    raise base.ExperimentError("source probes root is unsafe")
                for path in probes_root.iterdir():
                    match = re.fullmatch(r"seed-(-?[0-9]+)\.json", path.name)
                    if path.is_symlink() or not path.is_file() or match is None:
                        raise base.ExperimentError(f"unexpected source probe artifact: {path}")
                    seed_text = match.group(1)
                    if seed_text not in evaluation_seeds:
                        raise base.ExperimentError("source probe seed is outside the schedule")
                    probe = base._read_json(path)
                    source_engine._validate_probe(
                        probe,
                        cluster_id=cluster_id,
                        candidate_id=candidate_id,
                        seed=int(seed_text),
                    )
                    probes[seed_text] = probe
                    leaves.add(path)
            hints_root = candidate_root / "hints"
            if hints_root.exists():
                if hints_root.is_symlink() or not hints_root.is_dir():
                    raise base.ExperimentError("source hints root is unsafe")
                for seed_root in hints_root.iterdir():
                    if (
                        seed_root.is_symlink()
                        or not seed_root.is_dir()
                        or seed_root.name not in evaluation_seeds
                        or seed_root.name not in probes
                    ):
                        raise base.ExperimentError(f"unexpected source hint seed path: {seed_root}")
                    paths = sorted(seed_root.glob("attempt-*.json"))
                    if {path.name for path in seed_root.iterdir()} != {
                        path.name for path in paths
                    } or any(path.is_symlink() or not path.is_file() for path in paths):
                        raise base.ExperimentError("source hint directory contains an extra leaf")
                    expected_names = [
                        f"attempt-{attempt:02d}.json" for attempt in range(1, len(paths) + 1)
                    ]
                    if [path.name for path in paths] != expected_names or len(paths) > int(
                        plan["configuration"]["hint_attempts"]
                    ):
                        raise base.ExperimentError("source hint attempts are not a sealed prefix")
                    probe = probes[seed_root.name]
                    accepted_seen = False
                    for attempt, path in enumerate(paths, start=1):
                        hint = base._read_json(path)
                        if accepted_seen:
                            raise base.ExperimentError("source hint attempt follows acceptance")
                        if hint.get("status") == "accepted":
                            source_engine._validate_hint(
                                hint,
                                cluster_id=cluster_id,
                                candidate_id=candidate_id,
                                seed=int(seed_root.name),
                                observation=str(probe["observation"]),
                                solution=probe["solution"],
                            )
                        else:
                            source_engine._validate_failed_hint(
                                hint,
                                cluster_id=cluster_id,
                                candidate_id=candidate_id,
                                seed=int(seed_root.name),
                                observation=str(probe["observation"]),
                                solution=probe["solution"],
                            )
                        accepted_seen = hint.get("status") == "accepted"
                        call_ids.add(str(hint["call_id"]))
                        leaves.add(path)
    return leaves, call_ids


def _validate_source_snapshot(
    source_run: Path | str,
    *,
    dependencies: base.RunnerDependencies | None = None,
) -> SourceSnapshot:
    """Validate v4 and select only its immutable pre-outcome evidence."""
    run_dir = _canonical_source_run(source_run)
    plan_path = run_dir / "plan.json"
    plan = base.load_plan(plan_path)
    _validate_source_protocol(plan)
    if base.derive_run_dir(plan["run_output_root"], plan) != run_dir:
        raise base.ExperimentError("source run path is not canonical for its sealed plan")
    _validate_runtime_files(plan["runtime_identity"])
    _validate_source_revisions(plan)

    resolved_dependencies = _source_dependencies(plan, dependencies)
    source_engine = base._Engine(plan, base._pretty_json(plan), run_dir, resolved_dependencies)
    cohort = dict(source_engine._load_cohort())

    manifest_paths: set[Path] = {
        plan_path,
        run_dir / "run-manifest.json",
        run_dir / "cohort-lock.json",
    }
    run_manifest = base._read_json(run_dir / "run-manifest.json")
    expected_run_manifest = {
        "schema_version": base.RUN_SCHEMA,
        "experiment_id": plan["experiment_id"],
        "plan_digest": plan["plan_digest"],
        "provider": "agy",
        "model": plan["model"],
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "total_call_cap": plan["configuration"]["total_call_cap"],
        "stage": plan["stage"],
        "analysis_role": plan["analysis_role"],
        "protocol_id": plan["protocol_id"],
        "source_revisions": plan["source_revisions"],
        "runtime_identity": plan["runtime_identity"],
    }
    if run_manifest != expected_run_manifest:
        raise base.ExperimentError("source v4 run manifest differs from its sealed plan")
    selections: dict[str, dict[str, Any]] = {}
    for cluster in plan["cluster_schedule"]:
        cluster_id = str(cluster["cluster_id"])
        selection_path = run_dir / "selections" / f"{cluster_id}.json"
        selection = dict(source_engine._load_selection(cluster_id))
        selections[cluster_id] = selection
        manifest_paths.add(selection_path)

        candidate_root = run_dir / "candidates" / cluster_id / str(selection["candidate_id"])
        _safe_tree(candidate_root)
        design_path, design = _find_qualified_design(run_dir, plan, selection)
        manifest_paths.add(design_path)
        selected_disposition = candidate_root / "disposition.json"
        source_engine._validate_disposition(
            base._read_json(selected_disposition),
            cluster_id=cluster_id,
            candidate_id=str(selection["candidate_id"]),
        )
        manifest_paths.add(selected_disposition)

        slot_ids = [str(slot["candidate_id"]) for slot in cluster["candidate_slots"]]
        for earlier_id in slot_ids[: slot_ids.index(str(selection["candidate_id"]))]:
            disposition_path = run_dir / "candidates" / cluster_id / earlier_id / "disposition.json"
            disposition = source_engine._validate_disposition(
                base._read_json(disposition_path),
                cluster_id=cluster_id,
                candidate_id=earlier_id,
            )
            if disposition["status"] != "rejected":
                raise base.ExperimentError("source selection bypasses a non-rejected slot")
            manifest_paths.add(disposition_path)

        linked_call_ids = {str(design["call_id"])}
        for seed in plan["evaluation_seeds"]:
            seed_text = str(seed)
            probe_path = candidate_root / "probes" / f"seed-{seed}.json"
            hint = selection["hints"][seed_text]
            hint_path = (
                candidate_root / "hints" / seed_text / f"attempt-{int(hint['attempt']):02d}.json"
            )
            if base._read_json(probe_path) != selection["probes"][seed_text]:
                raise base.ExperimentError(
                    f"source probe disk/selection mismatch: {cluster_id}/{seed}"
                )
            if base._read_json(hint_path) != hint:
                raise base.ExperimentError(
                    f"source hint disk/selection mismatch: {cluster_id}/{seed}"
                )
            manifest_paths.update({probe_path, hint_path})
            linked_call_ids.add(str(hint["call_id"]))
        for call_id in linked_call_ids:
            call_root = run_dir / "calls" / call_id
            request_path = call_root / "request.json"
            result_path = call_root / "result.json"
            request = source_engine._validate_call_request(base._read_json(request_path))
            source_engine._validate_call_result(base._read_json(result_path), request)
            if request["purpose"].get("phase") not in {"designer", "hint"}:
                raise base.ExperimentError("source import attempted to include an actor call")
            manifest_paths.update({request_path, result_path})

    # Preserve and semantically validate the full pre-outcome history, including
    # failed designer/hint attempts, rather than presenting only selected calls.
    candidate_paths, referenced_nonactor_calls = _validate_candidate_history(
        source_engine, run_dir, plan, selections
    )
    manifest_paths.update(candidate_paths)
    nonactor_ordinals: list[int] = []
    observed_nonactor_calls: set[str] = set()
    calls_root = run_dir / "calls"
    base._reject_symlink_ancestors(calls_root)
    if calls_root.is_symlink() or not calls_root.is_dir():
        raise base.ExperimentError("source v4 calls root is missing or unsafe")
    for directory in sorted(calls_root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise base.ExperimentError(f"unsafe source v4 call directory: {directory}")
        for leaf in directory.iterdir():
            if (
                leaf.is_symlink()
                or not leaf.is_file()
                or leaf.name
                not in {
                    "request.json",
                    "result.json",
                }
            ):
                raise base.ExperimentError(f"unexpected source v4 call leaf: {leaf}")
        request_path = directory / "request.json"
        if not request_path.is_file():
            raise base.ExperimentError("source v4 call directory lacks a request reservation")
        request = source_engine._validate_call_request(base._read_json(request_path))
        phase = request["purpose"].get("phase")
        if phase in {"designer", "hint"}:
            result_path = directory / "result.json"
            if not result_path.is_file():
                raise base.ExperimentError("source pre-outcome call lacks a durable result")
            source_engine._validate_call_result(base._read_json(result_path), request)
            manifest_paths.update({request_path, result_path})
            nonactor_ordinals.append(int(request["call_ordinal"]))
            observed_nonactor_calls.add(str(request["call_id"]))
        elif phase != "actor":
            raise base.ExperimentError(f"unknown source call phase: {phase!r}")
    if sorted(nonactor_ordinals) != list(range(1, len(nonactor_ordinals) + 1)):
        raise base.ExperimentError("source pre-outcome call history is not a contiguous prefix")
    if observed_nonactor_calls != referenced_nonactor_calls:
        raise base.ExperimentError("source pre-outcome calls are orphaned or incompletely linked")

    for path in manifest_paths:
        base._reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file():
            raise base.ExperimentError(f"source import leaf is missing or unsafe: {path}")
        relative = path.relative_to(run_dir).as_posix()
        if (
            relative in {"outcomes", "assay"}
            or relative.startswith(("outcomes/", "assay/"))
            or relative
            in {
                "assay-request.json",
                "assay-result.json",
                "ledger-root.json",
            }
        ):
            raise base.ExperimentError("source actor/outcome/Assay evidence is forbidden")
        if relative.startswith("calls/"):
            request_path = path.parent / "request.json"
            request = base._read_json(request_path)
            if request.get("purpose", {}).get("phase") == "actor":
                raise base.ExperimentError("source actor calls are forbidden from import")

    manifest = sorted(
        (_manifest_entry(path, relative_to=run_dir) for path in manifest_paths),
        key=lambda item: item["path"],
    )
    return SourceSnapshot(
        run_dir=run_dir,
        plan=plan,
        cohort=cohort,
        selections=selections,
        manifest=manifest,
    )


def _boxed_oracle_actions(solution: Any) -> list[str]:
    """Match ProofPack's boxed oracle action interpretation exactly."""
    if isinstance(solution, list):
        values = solution
    elif isinstance(solution, str) and "\n" in solution.strip():
        values = [line.strip() for line in solution.splitlines() if line.strip()]
    else:
        values = [solution]
    if not values:
        raise base.ExperimentError("locked solution contains no oracle action")
    actions: list[str] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise base.ExperimentError("locked boxed solution has an unsupported action type")
        action = str(value).strip()
        if not action:
            raise base.ExperimentError("locked boxed solution contains an empty action")
        actions.append(action if re.search(r"\\boxed\{", action) else rf"\boxed{{{action}}}")
    return actions


def _derive_cluster_horizons(
    source_plan: Mapping[str, Any], selections: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for cluster in source_plan["cluster_schedule"]:
        cluster_id = str(cluster["cluster_id"])
        selection = selections[cluster_id]
        seed_horizons: dict[str, int] = {}
        solution_digests: dict[str, str] = {}
        for seed in source_plan["evaluation_seeds"]:
            probe = selection["probes"][str(seed)]
            if probe["solution_digest"] != base._digest(probe["solution"]):
                raise base.ExperimentError(f"locked solution digest mismatch: {cluster_id}/{seed}")
            actions = _boxed_oracle_actions(probe["solution"])
            seed_horizons[str(seed)] = max(1, len(actions))
            solution_digests[str(seed)] = str(probe["solution_digest"])
        horizon = max(seed_horizons.values())
        if len(set(seed_horizons.values())) != 1:
            raise base.ExperimentError(
                f"locked oracle horizons disagree across seeds for {cluster_id}"
            )
        if horizon > int(source_plan["configuration"]["max_turns"]):
            raise base.ExperimentError(f"locked oracle horizon exceeds source limit: {cluster_id}")
        values.append(
            {
                "cluster_id": cluster_id,
                "horizon": horizon,
                "seed_horizons": seed_horizons,
                "solution_digests": solution_digests,
            }
        )
    return values


def _report_value(report: Any, where: str) -> dict[str, Any]:
    return base._decode_json((report.to_json() + "\n").encode("utf-8"), where)


def _qualification_revalidations(
    snapshot: SourceSnapshot,
    dependencies: base.RunnerDependencies,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_config = snapshot.plan["configuration"]
    for cluster in snapshot.plan["cluster_schedule"]:
        cluster_id = str(cluster["cluster_id"])
        selection = snapshot.selections[cluster_id]
        report = dependencies.qualify(
            selection["code"],
            seeds=list(snapshot.plan["qualification_seeds"]),
            timeout_seconds=float(source_config["qualification_timeout_seconds"]),
            max_turns=SOURCE_MAX_TURNS,
        )
        receipt = _report_value(report, f"qualification revalidation {cluster_id}")
        passed, reason = base.validate_positive_proofpack_receipt(
            report,
            game_code=str(selection["code"]),
            action_format="boxed",
            seeds=snapshot.plan["qualification_seeds"],
            timeout_seconds=float(source_config["qualification_timeout_seconds"]),
            max_turns=SOURCE_MAX_TURNS,
        )
        if not passed:
            raise base.ExperimentError(
                f"source environment no longer passes V0-V4 at five turns: {cluster_id}: {reason}"
            )
        body = {
            "schema_version": QUALIFICATION_REVALIDATION_SCHEMA,
            "cluster_id": cluster_id,
            "code_digest": selection["code_digest"],
            "source_qualification_digest": selection["qualification_digest"],
            "max_turns": SOURCE_MAX_TURNS,
            "seeds": list(snapshot.plan["qualification_seeds"]),
            "timeout_seconds": source_config["qualification_timeout_seconds"],
            "receipt": receipt,
            "receipt_digest": base._digest(receipt),
        }
        records.append({**body, "record_digest": base._digest(body)})
    return records


def _normalize_trace_value(value: Any) -> Any:
    return base._normalize_solution(value)


def _viability_trace(
    *,
    selection: Mapping[str, Any],
    seed: int,
    horizon: int,
    timeout_seconds: float,
    target_factory: Any,
) -> dict[str, Any]:
    probe = selection["probes"][str(seed)]
    actions = _boxed_oracle_actions(probe["solution"])
    if len(actions) != horizon:
        raise base.ExperimentError("locked oracle action count differs from cluster horizon")
    target = target_factory(
        selection["code"],
        action_format="boxed",
        max_turns=horizon,
        operation_timeout_seconds=timeout_seconds,
    )
    env = target.instantiate()
    steps: list[dict[str, Any]] = []
    try:
        observation, reset_info = env.reset(seed=seed)
        solution = _normalize_trace_value(env.solution())
        if str(observation) != probe["observation"] or solution != probe["solution"]:
            raise base.ExperimentError("actor-horizon replay differs from locked probe")
        terminated = False
        truncated = False
        for turn, action in enumerate(actions, start=1):
            if terminated or truncated:
                raise base.ExperimentError("oracle action sequence continues after episode end")
            result = env.step(action)
            if not isinstance(result, tuple) or len(result) != 5:
                raise base.ExperimentError("actor-horizon replay requires a five-item Gym step")
            post, reward, terminated, truncated, info = result
            numeric_reward = base._finite_number(reward, "actor-horizon reward")
            steps.append(
                {
                    "turn": turn,
                    "action": action,
                    "pre_observation": str(observation),
                    "post_observation": str(post),
                    "reward": numeric_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _normalize_trace_value(info),
                }
            )
            observation = post
        if not terminated or truncated or steps[-1]["reward"] < 1.0:
            raise base.ExperimentError(
                "actor-horizon oracle did not terminate successfully within its bound"
            )
        return {
            "initial_observation": str(probe["observation"]),
            "reset_info": _normalize_trace_value(reset_info),
            "solution": probe["solution"],
            "oracle_actions": actions,
            "steps": steps,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _horizon_viability_records(
    snapshot: SourceSnapshot,
    cluster_horizons: Sequence[Mapping[str, Any]],
    dependencies: base.RunnerDependencies,
) -> list[dict[str, Any]]:
    horizon_by_cluster = {
        str(item["cluster_id"]): int(item["horizon"]) for item in cluster_horizons
    }
    timeout_seconds = float(snapshot.plan["configuration"]["qualification_timeout_seconds"])
    records: list[dict[str, Any]] = []
    for cluster in snapshot.plan["cluster_schedule"]:
        cluster_id = str(cluster["cluster_id"])
        selection = snapshot.selections[cluster_id]
        horizon = horizon_by_cluster[cluster_id]
        seeds: dict[str, Any] = {}
        for seed in snapshot.plan["evaluation_seeds"]:
            traces = [
                _viability_trace(
                    selection=selection,
                    seed=seed,
                    horizon=horizon,
                    timeout_seconds=timeout_seconds,
                    target_factory=dependencies.target_factory,
                )
                for _replay in range(2)
            ]
            if traces[0] != traces[1]:
                raise base.ExperimentError(
                    f"actor-horizon replay is not deterministic: {cluster_id}/{seed}"
                )
            seeds[str(seed)] = {
                "probe_digest": base._digest(selection["probes"][str(seed)]),
                "trace": traces[0],
                "trace_digest": base._digest(traces[0]),
                "replay_count": 2,
            }
        body = {
            "schema_version": HORIZON_VIABILITY_SCHEMA,
            "cluster_id": cluster_id,
            "code_digest": selection["code_digest"],
            "horizon": horizon,
            "seeds": seeds,
        }
        records.append({**body, "record_digest": base._digest(body)})
    return records


def _build_schedules(
    source_plan: Mapping[str, Any], cluster_horizons: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    horizons = {str(item["cluster_id"]): int(item["horizon"]) for item in cluster_horizons}
    source_outcomes = source_plan["outcome_schedule"]
    outcomes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    if len(source_outcomes) % 2:
        raise base.ExperimentError("source outcome schedule cannot form complete pairs")
    for offset in range(0, len(source_outcomes), 2):
        source_pair = source_outcomes[offset : offset + 2]
        cluster_id = str(source_pair[0]["cluster_id"])
        seed = int(source_pair[0]["seed"])
        if any(
            str(item["cluster_id"]) != cluster_id or int(item["seed"]) != seed
            for item in source_pair
        ) or {str(item["arm"]) for item in source_pair} != set(ARMS):
            raise base.ExperimentError("source outcome schedule breaks paired arm adjacency")
        pair_ordinal = len(pairs) + 1
        pair_id = f"{cluster_id}-seed-{seed}"
        logical: list[dict[str, Any]] = []
        for source_item in source_pair:
            item = {
                "ordinal": len(outcomes) + 1,
                "outcome_id": str(source_item["outcome_id"]),
                "pair_id": pair_id,
                "pair_ordinal": pair_ordinal,
                "cluster_id": cluster_id,
                "seed": seed,
                "arm": str(source_item["arm"]),
                "horizon": horizons[cluster_id],
            }
            outcomes.append(item)
            logical.append(item)
        attempt_slots = []
        for attempt in range(1, PAIR_ATTEMPTS + 1):
            call_schedule = []
            for logical_item in logical:
                for turn in range(1, horizons[cluster_id] + 1):
                    call_schedule.append(
                        {
                            "pair_attempt": attempt,
                            "outcome_id": logical_item["outcome_id"],
                            "outcome_ordinal": logical_item["ordinal"],
                            "arm": logical_item["arm"],
                            "turn": turn,
                        }
                    )
            attempt_slots.append({"pair_attempt": attempt, "call_schedule": call_schedule})
        pairs.append(
            {
                "pair_ordinal": pair_ordinal,
                "pair_id": pair_id,
                "cluster_id": cluster_id,
                "seed": seed,
                "horizon": horizons[cluster_id],
                "arm_order": [item["arm"] for item in logical],
                "logical_outcomes": logical,
                "attempt_slots": attempt_slots,
            }
        )
    return outcomes, pairs


def build_outcome_replay_plan(
    *,
    experiment_id: str,
    source_run_dir: Path | str,
    run_output_root: Path | str,
    expected_source_plan_digest: str,
    expected_source_cohort_digest: str,
    total_call_cap: int | None = None,
    dependencies: base.RunnerDependencies | None = None,
    source_revisions: Mapping[str, str] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and seal an outcome-only plan after offline source validation."""
    base._safe_id(experiment_id, "experiment_id")
    base._sha256_text(expected_source_plan_digest, "expected_source_plan_digest")
    base._sha256_text(expected_source_cohort_digest, "expected_source_cohort_digest")
    snapshot = _validate_source_snapshot(source_run_dir, dependencies=dependencies)
    if snapshot.plan["plan_digest"] != expected_source_plan_digest:
        raise base.ExperimentError("source plan digest differs from explicit authorization")
    if snapshot.cohort["cohort_digest"] != expected_source_cohort_digest:
        raise base.ExperimentError("source cohort digest differs from explicit authorization")
    resolved_dependencies = _source_dependencies(snapshot.plan, dependencies)
    horizons = _derive_cluster_horizons(snapshot.plan, snapshot.selections)
    outcomes, pairs = _build_schedules(snapshot.plan, horizons)
    ceiling = sum(len(slot["call_schedule"]) for pair in pairs for slot in pair["attempt_slots"])
    cap = ceiling if total_call_cap is None else total_call_cap
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= ceiling:
        raise base.ExperimentError(f"total_call_cap must be within 1..{ceiling}")
    qualification = _qualification_revalidations(snapshot, resolved_dependencies)
    viability = _horizon_viability_records(snapshot, horizons, resolved_dependencies)
    revisions = dict(source_revisions or base._source_revisions())
    runtime = dict(runtime_identity or base._runtime_identity())
    manifest_digest = base._digest(snapshot.manifest)
    source_evidence = {
        "source_run_dir": str(snapshot.run_dir),
        "source_plan_digest": snapshot.plan["plan_digest"],
        "source_cohort_digest": snapshot.cohort["cohort_digest"],
        "source_runtime_identity_digest": base._digest(snapshot.plan["runtime_identity"]),
        "source_revisions_digest": base._digest(snapshot.plan["source_revisions"]),
        "import_manifest": snapshot.manifest,
        "import_manifest_digest": manifest_digest,
        "imported_actor_call_count": 0,
        "imported_outcome_count": 0,
        "excluded_source_roots": [
            "outcomes",
            "assay",
            "assay-request.json",
            "assay-result.json",
            "ledger-root.json",
        ],
    }
    horizon_policy = {
        "policy_id": HORIZON_POLICY_ID,
        "basis": "locked pre-outcome probe solution shape only",
        "boxed_list_rule": "one turn per list item",
        "boxed_multiline_rule": "one turn per nonblank line",
        "boxed_scalar_rule": "one turn",
        "seed_consistency_required": True,
        "source_actor_outcomes_consulted": False,
    }
    config = snapshot.plan["configuration"]
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": experiment_id,
        "stage": "post-v4-exploratory",
        "analysis_role": "outcome-replay-calibration-only",
        "protocol_id": PROTOCOL_ID,
        "run_output_root": base._canonical_output_root(run_output_root),
        "provider": "agy",
        "model": snapshot.plan["model"],
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "source_revisions": revisions,
        "runtime_identity": runtime,
        "replay_runner_digest": _file_digest(Path(__file__).resolve()),
        "source_evidence": source_evidence,
        "source_authorization": {
            "expected_source_plan_digest": expected_source_plan_digest,
            "expected_source_cohort_digest": expected_source_cohort_digest,
        },
        "skills": snapshot.plan["skills"],
        "difficulties": snapshot.plan["difficulties"],
        "evaluation_seeds": snapshot.plan["evaluation_seeds"],
        "configuration": {
            "action_format": "boxed",
            "pair_attempts": PAIR_ATTEMPTS,
            "llm_timeout_seconds": config["llm_timeout_seconds"],
            "qualification_timeout_seconds": config["qualification_timeout_seconds"],
            "source_qualification_max_turns": SOURCE_MAX_TURNS,
            "total_call_cap": cap,
            "computed_call_ceiling": ceiling,
            "minimum_certification_clusters": config["minimum_certification_clusters"],
            "alpha": config["alpha"],
            "non_inferiority_margin": config["non_inferiority_margin"],
        },
        "budget_context": {
            "prior_charged_calls": PRIOR_CHARGED_CALLS,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "planned_max_global_calls": PRIOR_CHARGED_CALLS + cap,
            "headroom_calls": AUTHORIZED_GLOBAL_CALL_CAP - PRIOR_CHARGED_CALLS - cap,
            "canonical_full_run": cap == ceiling,
        },
        "horizon_policy": horizon_policy,
        "cluster_horizons": horizons,
        "cluster_horizons_digest": base._digest(horizons),
        "cluster_schedule": snapshot.plan["cluster_schedule"],
        "outcome_schedule": outcomes,
        "pair_schedule": pairs,
        "qualification_revalidations": qualification,
        "qualification_revalidations_digest": base._digest(qualification),
        "horizon_viability": viability,
        "horizon_viability_digest": base._digest(viability),
    }
    plan = {**body, "plan_digest": base._digest(body)}
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Validate the complete prospective replay contract without live calls."""
    base._require_keys(
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
            "replay_runner_digest",
            "source_evidence",
            "source_authorization",
            "skills",
            "difficulties",
            "evaluation_seeds",
            "configuration",
            "budget_context",
            "horizon_policy",
            "cluster_horizons",
            "cluster_horizons_digest",
            "cluster_schedule",
            "outcome_schedule",
            "pair_schedule",
            "qualification_revalidations",
            "qualification_revalidations_digest",
            "horizon_viability",
            "horizon_viability_digest",
            "plan_digest",
        },
        "outcome replay plan",
    )
    if (
        plan["schema_version"] != PLAN_SCHEMA
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["stage"] != "post-v4-exploratory"
        or plan["analysis_role"] != "outcome-replay-calibration-only"
        or plan["provider"] != "agy"
        or plan["model"] != CANONICAL_MODEL
        or plan["backend_identity_attested"] is not False
        or plan["route_authority"] != "requested-route-only"
    ):
        raise base.ExperimentError("unsupported outcome replay protocol identity")
    base._safe_id(plan["experiment_id"], "experiment_id")
    if plan["run_output_root"] != base._canonical_output_root(plan["run_output_root"]):
        raise base.ExperimentError("run_output_root is not canonical")
    base._sha256_text(plan["replay_runner_digest"], "replay_runner_digest")
    revisions = plan["source_revisions"]
    if not isinstance(revisions, dict):
        raise base.ExperimentError("source_revisions must be an object")
    base._require_keys(revisions, {"spade", "proofpack", "assay"}, "source_revisions")
    for name, revision in revisions.items():
        if re.fullmatch(r"[0-9a-f]{40}", base._required_text(revision, name)) is None:
            raise base.ExperimentError(f"source_revisions.{name} is not a commit")
    base._validate_runtime_identity(plan["runtime_identity"])

    source = plan["source_evidence"]
    if not isinstance(source, dict):
        raise base.ExperimentError("source_evidence must be an object")
    base._require_keys(
        source,
        {
            "source_run_dir",
            "source_plan_digest",
            "source_cohort_digest",
            "source_runtime_identity_digest",
            "source_revisions_digest",
            "import_manifest",
            "import_manifest_digest",
            "imported_actor_call_count",
            "imported_outcome_count",
            "excluded_source_roots",
        },
        "source_evidence",
    )
    if source["source_run_dir"] != str(Path(str(source["source_run_dir"])).resolve()):
        raise base.ExperimentError("source_evidence.source_run_dir is not canonical")
    for field in (
        "source_plan_digest",
        "source_cohort_digest",
        "source_runtime_identity_digest",
        "source_revisions_digest",
        "import_manifest_digest",
    ):
        base._sha256_text(source[field], f"source_evidence.{field}")
    if source["imported_actor_call_count"] != 0 or source["imported_outcome_count"] != 0:
        raise base.ExperimentError("source import must attest zero actor/outcome artifacts")
    expected_exclusions = [
        "outcomes",
        "assay",
        "assay-request.json",
        "assay-result.json",
        "ledger-root.json",
    ]
    if source["excluded_source_roots"] != expected_exclusions:
        raise base.ExperimentError("source import exclusions differ from protocol")
    manifest = source["import_manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise base.ExperimentError("source import manifest must be non-empty")
    prior = ""
    for index, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            raise base.ExperimentError("source import manifest entry must be an object")
        base._require_keys(entry, {"path", "digest", "size_bytes"}, "import manifest entry")
        relative = _safe_relative(entry["path"], f"import_manifest[{index}].path")
        if relative <= prior:
            raise base.ExperimentError("source import manifest paths must be sorted and unique")
        prior = relative
        if (
            relative in {"outcomes", "assay"}
            or relative.startswith(("outcomes/", "assay/"))
            or relative in expected_exclusions[2:]
        ):
            raise base.ExperimentError(
                "source import manifest includes actor/outcome/Assay evidence"
            )
        base._sha256_text(entry["digest"], f"import manifest {relative} digest")
        if (
            isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] < 0
        ):
            raise base.ExperimentError("import manifest size must be non-negative")
    if source["import_manifest_digest"] != base._digest(manifest):
        raise base.ExperimentError("source import manifest digest mismatch")
    authorization = plan["source_authorization"]
    expected_authorization = {
        "expected_source_plan_digest": source["source_plan_digest"],
        "expected_source_cohort_digest": source["source_cohort_digest"],
    }
    if authorization != expected_authorization:
        raise base.ExperimentError("source evidence differs from explicit source authorization")

    if plan["skills"] != list(base.PILOT_SKILLS):
        raise base.ExperimentError("outcome replay skill schedule differs from v4")
    if plan["difficulties"] != list(base.PILOT_DIFFICULTIES):
        raise base.ExperimentError("outcome replay difficulty schedule differs from v4")
    if plan["evaluation_seeds"] != sorted(base.live.QUALIFIED_SEEDS):
        raise base.ExperimentError("outcome replay seed schedule differs from v4")
    clusters = plan["cluster_schedule"]
    if not isinstance(clusters, list) or len(clusters) != 18:
        raise base.ExperimentError("outcome replay requires exactly 18 locked clusters")
    expected_clusters: list[dict[str, Any]] = []
    for skill_index, skill in enumerate(base.PILOT_SKILLS, start=1):
        for difficulty_index, difficulty in enumerate(base.PILOT_DIFFICULTIES, start=1):
            cluster_id = (
                f"c{len(expected_clusters) + 1:03d}-{base._slug(skill)}-{base._slug(difficulty)}"
            )
            slots = [{"candidate_id": f"{cluster_id}-primary", "role": "primary"}]
            if difficulty == "hard":
                slots.append({"candidate_id": f"{cluster_id}-reserve-01", "role": "reserve"})
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
    if clusters != expected_clusters:
        raise base.ExperimentError("outcome replay cluster schedule differs from v4 template")

    config = plan["configuration"]
    if not isinstance(config, dict):
        raise base.ExperimentError("configuration must be an object")
    base._require_keys(
        config,
        {
            "action_format",
            "pair_attempts",
            "llm_timeout_seconds",
            "qualification_timeout_seconds",
            "source_qualification_max_turns",
            "total_call_cap",
            "computed_call_ceiling",
            "minimum_certification_clusters",
            "alpha",
            "non_inferiority_margin",
        },
        "configuration",
    )
    if (
        config["action_format"] != "boxed"
        or config["pair_attempts"] != PAIR_ATTEMPTS
        or config["llm_timeout_seconds"] != CANONICAL_TIMEOUT_SECONDS
        or config["source_qualification_max_turns"] != SOURCE_MAX_TURNS
        or config["minimum_certification_clusters"] != 18
        or config["qualification_timeout_seconds"] != 5.0
        or config["alpha"] != 0.05
        or config["non_inferiority_margin"] != 0.10
    ):
        raise base.ExperimentError("outcome replay configuration differs from protocol")
    base._finite_number(
        config["qualification_timeout_seconds"], "qualification timeout", positive=True
    )
    base._finite_number(config["alpha"], "alpha", positive=True)
    base._finite_number(config["non_inferiority_margin"], "non-inferiority margin")

    policy = plan["horizon_policy"]
    expected_policy = {
        "policy_id": HORIZON_POLICY_ID,
        "basis": "locked pre-outcome probe solution shape only",
        "boxed_list_rule": "one turn per list item",
        "boxed_multiline_rule": "one turn per nonblank line",
        "boxed_scalar_rule": "one turn",
        "seed_consistency_required": True,
        "source_actor_outcomes_consulted": False,
    }
    if policy != expected_policy:
        raise base.ExperimentError("horizon policy differs from the sealed exploratory policy")
    horizons = plan["cluster_horizons"]
    if not isinstance(horizons, list) or len(horizons) != 18:
        raise base.ExperimentError("cluster_horizons must cover all 18 clusters")
    if plan["cluster_horizons_digest"] != base._digest(horizons):
        raise base.ExperimentError("cluster horizon digest mismatch")
    horizon_ids: list[str] = []
    for cluster, item in zip(clusters, horizons):
        if not isinstance(item, dict):
            raise base.ExperimentError("cluster horizon entry must be an object")
        base._require_keys(
            item,
            {"cluster_id", "horizon", "seed_horizons", "solution_digests"},
            "cluster horizon entry",
        )
        cluster_id = str(cluster["cluster_id"])
        if item["cluster_id"] != cluster_id:
            raise base.ExperimentError("cluster horizon order differs from cluster schedule")
        horizon = base._positive_int(item["horizon"], "cluster horizon")
        expected_seed_keys = {str(seed) for seed in plan["evaluation_seeds"]}
        if (
            set(item["seed_horizons"]) != expected_seed_keys
            or set(item["solution_digests"]) != expected_seed_keys
        ):
            raise base.ExperimentError("cluster horizon seed evidence is incomplete")
        if set(item["seed_horizons"].values()) != {horizon}:
            raise base.ExperimentError("cluster horizon seed values disagree")
        for digest in item["solution_digests"].values():
            base._sha256_text(digest, "locked solution digest")
        horizon_ids.append(cluster_id)

    source_schedule = base._outcome_schedule(clusters, plan["evaluation_seeds"])
    outcomes, pairs = _build_schedules(
        {
            "cluster_schedule": clusters,
            "outcome_schedule": source_schedule,
        },
        horizons,
    )
    if plan["outcome_schedule"] != outcomes or plan["pair_schedule"] != pairs:
        raise base.ExperimentError("outcome/pair schedules are not deterministic")
    if len(outcomes) != 108 or len(pairs) != 54:
        raise base.ExperimentError("outcome replay requires 54 pairs and 108 outcomes")
    ceiling = sum(len(slot["call_schedule"]) for pair in pairs for slot in pair["attempt_slots"])
    if config["computed_call_ceiling"] != ceiling:
        raise base.ExperimentError("computed call ceiling differs from pair horizons")
    cap = config["total_call_cap"]
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= ceiling:
        raise base.ExperimentError("total_call_cap exceeds the computed ceiling")
    budget = plan["budget_context"]
    expected_budget = {
        "prior_charged_calls": PRIOR_CHARGED_CALLS,
        "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
        "planned_max_global_calls": PRIOR_CHARGED_CALLS + cap,
        "headroom_calls": AUTHORIZED_GLOBAL_CALL_CAP - PRIOR_CHARGED_CALLS - cap,
        "canonical_full_run": cap == ceiling,
    }
    if budget != expected_budget or budget["planned_max_global_calls"] > AUTHORIZED_GLOBAL_CALL_CAP:
        raise base.ExperimentError("global call budget context is invalid")

    for field, schema in (
        ("qualification_revalidations", QUALIFICATION_REVALIDATION_SCHEMA),
        ("horizon_viability", HORIZON_VIABILITY_SCHEMA),
    ):
        records = plan[field]
        if not isinstance(records, list) or len(records) != 18:
            raise base.ExperimentError(f"{field} must cover every cluster")
        for cluster, horizon_item, record in zip(clusters, horizons, records):
            if not isinstance(record, dict) or record.get("schema_version") != schema:
                raise base.ExperimentError(f"{field} contains an invalid record")
            if record.get("cluster_id") != cluster["cluster_id"]:
                raise base.ExperimentError(f"{field} order differs from cluster schedule")
            body = {key: value for key, value in record.items() if key != "record_digest"}
            if record.get("record_digest") != base._digest(body):
                raise base.ExperimentError(f"{field} record digest mismatch")
            if field == "qualification_revalidations":
                if record.get("max_turns") != SOURCE_MAX_TURNS:
                    raise base.ExperimentError("qualification revalidation changed max_turns")
                if record.get("receipt_digest") != base._digest(record.get("receipt")):
                    raise base.ExperimentError("qualification receipt digest mismatch")
            else:
                if record.get("horizon") != horizon_item["horizon"]:
                    raise base.ExperimentError("viability horizon differs from sealed horizon")
                if set(record.get("seeds", {})) != {str(seed) for seed in plan["evaluation_seeds"]}:
                    raise base.ExperimentError("viability receipt seed evidence is incomplete")
                for seed_record in record["seeds"].values():
                    if seed_record.get("replay_count") != 2 or seed_record.get(
                        "trace_digest"
                    ) != base._digest(seed_record.get("trace")):
                        raise base.ExperimentError("viability trace binding is invalid")
        digest_field = f"{field}_digest"
        if plan[digest_field] != base._digest(records):
            raise base.ExperimentError(f"{field} aggregate digest mismatch")

    body = {key: value for key, value in plan.items() if key != "plan_digest"}
    if plan["plan_digest"] != base._digest(body):
        raise base.ExperimentError("outcome replay plan digest mismatch")


def write_plan(path: Path | str, plan: Mapping[str, Any]) -> Path:
    validate_plan(plan)
    target = Path(path)
    base._write_json(target, plan)
    return target


def load_plan(path: Path | str) -> dict[str, Any]:
    plan = base._read_json(Path(path))
    validate_plan(plan)
    return plan


def derive_run_dir(output_root: Path | str, plan: Mapping[str, Any]) -> Path:
    validate_plan(plan)
    canonical = base._canonical_output_root(output_root)
    if canonical != plan["run_output_root"]:
        raise base.ExperimentError("requested output root differs from sealed run_output_root")
    return Path(canonical) / (
        f"{plan['experiment_id']}-{str(plan['plan_digest']).removeprefix('sha256:')}"
    )


class _RecordedProviderFailure(base.ExperimentIncomplete):
    """A durable provider result that did not contain a response."""

    def __init__(self, call_id: str, category: str, error: str) -> None:
        super().__init__(f"call {call_id} failed [{category}]: {error}")
        self.call_id = call_id
        self.category = category
        self.error = error

    @property
    def retryable(self) -> bool:
        return self.category in {"empty_response", "provider_timeout"}


class _ArmExecutionFailure(base.ExperimentIncomplete):
    """A non-provider episode failure that cannot authorize replacement."""

    def __init__(self, message: str, *, call_id: str | None = None) -> None:
        super().__init__(message)
        self.call_id = call_id


class _ReplayEngine(base._Engine):
    """Outcome-only engine with whole-pair replacement semantics."""

    def __init__(
        self,
        plan: dict[str, Any],
        plan_bytes: bytes,
        run_dir: Path,
        dependencies: base.RunnerDependencies,
    ) -> None:
        super().__init__(plan, plan_bytes, run_dir, dependencies)
        self._source_snapshot: SourceSnapshot | None = None

    def _pair(self, pair_id: str) -> Mapping[str, Any]:
        pair = next(
            (item for item in self.plan["pair_schedule"] if item["pair_id"] == pair_id),
            None,
        )
        if pair is None:
            raise base.ExperimentError(f"unknown pair_id: {pair_id}")
        return pair

    def _logical_outcome(self, outcome_id: str) -> Mapping[str, Any]:
        item = next(
            (item for item in self.plan["outcome_schedule"] if item["outcome_id"] == outcome_id),
            None,
        )
        if item is None:
            raise base.ExperimentError(f"unknown outcome_id: {outcome_id}")
        return item

    def _validate_import_manifest_bytes(self, root: Path) -> None:
        expected = self.plan["source_evidence"]["import_manifest"]
        actual_paths = _safe_tree(root)
        actual = sorted(
            (_manifest_entry(path, relative_to=root) for path in actual_paths),
            key=lambda item: item["path"],
        )
        if actual != expected:
            raise base.ExperimentError("imported v4 evidence inventory or bytes changed")
        for entry in actual:
            relative = str(entry["path"])
            if relative.startswith(("outcomes/", "assay/")):
                raise base.ExperimentError("imported v4 tree contains actor outcomes or Assay")
            if relative.startswith("calls/") and relative.endswith("request.json"):
                request = base._read_json(root / relative)
                if request.get("purpose", {}).get("phase") == "actor":
                    raise base.ExperimentError("imported v4 tree contains an actor call")

    def _validate_import_manifest_subset(self, root: Path) -> bool:
        """Validate a crash-interrupted import without trusting any partial byte."""
        expected = {
            str(entry["path"]): entry for entry in self.plan["source_evidence"]["import_manifest"]
        }
        allowed_directories = {"."}
        for relative in expected:
            parent = Path(relative).parent
            while parent != Path("."):
                allowed_directories.add(parent.as_posix())
                parent = parent.parent
        actual_paths = _safe_tree(root)
        for current, directories, _files in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in directories:
                relative = (current_path / name).relative_to(root).as_posix()
                if relative not in allowed_directories:
                    raise base.ExperimentError(
                        f"partial imported v4 tree contains an extra directory: {relative}"
                    )
        for path in actual_paths:
            entry = _manifest_entry(path, relative_to=root)
            relative = str(entry["path"])
            if relative not in expected:
                raise base.ExperimentError(
                    f"partial imported v4 tree contains an extra leaf: {relative}"
                )
            if entry != expected[relative]:
                raise base.ExperimentError(
                    f"partial imported v4 leaf conflicts with the seal: {relative}"
                )
        return len(actual_paths) == len(expected)

    def _copy_source_evidence(self, snapshot: SourceSnapshot) -> Path:
        root = self.run_dir / IMPORT_ROOT
        if root.is_symlink():
            raise base.ExperimentError("imported v4 root cannot be a symlink")
        for entry in snapshot.manifest:
            relative = _safe_relative(entry["path"], "source import path")
            source = snapshot.run_dir / relative
            content = source.read_bytes()
            if (
                len(content) != entry["size_bytes"]
                or base._bytes_digest(content) != entry["digest"]
            ):
                raise base.ExperimentError("source evidence changed after validation")
            base._write_immutable_bytes(root / relative, content)
        self._validate_import_manifest_bytes(root)
        return root

    def _validate_imported_cohort(self, root: Path) -> dict[str, dict[str, Any]]:
        source_plan = base.load_plan(root / "plan.json")
        if source_plan["plan_digest"] != self.plan["source_evidence"]["source_plan_digest"]:
            raise base.ExperimentError("imported source plan digest differs from v5 binding")
        source_engine = base._Engine(
            source_plan,
            base._pretty_json(source_plan),
            root,
            self.dependencies,
        )
        cohort = source_engine._load_cohort()
        if cohort["cohort_digest"] != self.plan["source_evidence"]["source_cohort_digest"]:
            raise base.ExperimentError("imported source cohort digest differs from v5 binding")
        selections = {
            str(cluster["cluster_id"]): dict(
                source_engine._load_selection(str(cluster["cluster_id"]))
            )
            for cluster in source_plan["cluster_schedule"]
        }
        derived = _derive_cluster_horizons(source_plan, selections)
        if derived != self.plan["cluster_horizons"]:
            raise base.ExperimentError("imported locked probes derive different actor horizons")
        return selections

    def _rerun_offline_assurance(self, snapshot: SourceSnapshot) -> None:
        qualification = _qualification_revalidations(snapshot, self.dependencies)
        viability = _horizon_viability_records(
            snapshot, self.plan["cluster_horizons"], self.dependencies
        )
        if qualification != self.plan["qualification_revalidations"]:
            raise base.ExperimentIncomplete("five-turn ProofPack revalidation differs from seal")
        if viability != self.plan["horizon_viability"]:
            raise base.ExperimentIncomplete("actor-horizon viability differs from seal")
        for record in qualification:
            base._write_json(
                self.run_dir / "qualification-revalidation" / f"{record['cluster_id']}.json",
                record,
            )
        for record in viability:
            base._write_json(
                self.run_dir / "horizon-viability" / f"{record['cluster_id']}.json",
                record,
            )

    def _cohort_value(self, selections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        qualification = {
            str(item["cluster_id"]): str(item["record_digest"])
            for item in self.plan["qualification_revalidations"]
        }
        viability = {
            str(item["cluster_id"]): str(item["record_digest"])
            for item in self.plan["horizon_viability"]
        }
        horizons = {
            str(item["cluster_id"]): int(item["horizon"]) for item in self.plan["cluster_horizons"]
        }
        body = {
            "schema_version": COHORT_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "experiment_id": self.plan["experiment_id"],
            "source_plan_digest": self.plan["source_evidence"]["source_plan_digest"],
            "source_cohort_digest": self.plan["source_evidence"]["source_cohort_digest"],
            "import_manifest_digest": self.plan["source_evidence"]["import_manifest_digest"],
            "cluster_horizons_digest": self.plan["cluster_horizons_digest"],
            "qualification_revalidations_digest": self.plan["qualification_revalidations_digest"],
            "horizon_viability_digest": self.plan["horizon_viability_digest"],
            "outcome_schedule_digest": base._digest(self.plan["outcome_schedule"]),
            "pair_schedule_digest": base._digest(self.plan["pair_schedule"]),
            "selections": [
                {
                    "cluster_id": str(cluster["cluster_id"]),
                    "candidate_id": selections[str(cluster["cluster_id"])]["candidate_id"],
                    "selection_digest": selections[str(cluster["cluster_id"])]["selection_digest"],
                    "code_digest": selections[str(cluster["cluster_id"])]["code_digest"],
                    "source_qualification_digest": selections[str(cluster["cluster_id"])][
                        "qualification_digest"
                    ],
                    "qualification_revalidation_digest": qualification[str(cluster["cluster_id"])],
                    "horizon_viability_digest": viability[str(cluster["cluster_id"])],
                    "horizon": horizons[str(cluster["cluster_id"])],
                }
                for cluster in self.plan["cluster_schedule"]
            ],
        }
        return {**body, "cohort_digest": base._digest(body)}

    def _validate_cohort(self, cohort: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._source_snapshot is None:
            raise base.ExperimentError("source snapshot is not initialized")
        expected = self._cohort_value(self._source_snapshot.selections)
        if cohort != expected:
            raise base.ExperimentError("v5 imported cohort lock differs from sealed evidence")
        return cohort

    def _load_cohort(self) -> Mapping[str, Any]:
        return self._validate_cohort(base._read_json(self.run_dir / "cohort-lock.json"))

    def _selection(
        self, cluster_id: str, cohort: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        if self._source_snapshot is None or cluster_id not in self._source_snapshot.selections:
            raise base.ExperimentError(f"unknown imported selection: {cluster_id}")
        locked = self._load_cohort() if cohort is None else self._validate_cohort(cohort)
        selection = self._source_snapshot.selections[cluster_id]
        reference = next(
            (item for item in locked["selections"] if item["cluster_id"] == cluster_id),
            None,
        )
        if reference is None or reference["selection_digest"] != selection["selection_digest"]:
            raise base.ExperimentError("imported selection is not bound by the v5 cohort")
        return selection

    def _validate_call_request(
        self,
        request: Mapping[str, Any],
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        where = f"replay call request {request.get('call_id', '<unknown>')}"
        base._validate_leaf(
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
                "workdir_policy",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "provider": "agy",
                "model": self.plan["model"],
                "backend_identity_attested": False,
                "route_authority": "requested-route-only",
                "runtime_identity_digest": base._digest(self.plan["runtime_identity"]),
                "agy_executable_digest": self.plan["runtime_identity"]["agy_executable_digest"],
                "reservation_status": "reserved-before-spawn",
                "timeout_seconds": self.config["llm_timeout_seconds"],
                "workdir_policy": "fresh-temporary-directory-per-call",
                **dict(expected or {}),
            },
        )
        base._safe_id(request["call_id"], f"{where}.call_id")
        base._positive_int(request["call_ordinal"], f"{where}.call_ordinal")
        purpose = request["purpose"]
        if not isinstance(purpose, dict):
            raise base.ExperimentError(f"{where}.purpose must be an object")
        self._validate_call_purpose(purpose)
        if request["call_id"] != base._call_id(purpose):
            raise base.ExperimentError(f"{where} call_id does not bind purpose")
        prompt = base._required_text(request["prompt"], f"{where}.prompt")
        if request["prompt_digest"] != base._digest(prompt):
            raise base.ExperimentError(f"{where} prompt digest mismatch")
        if not isinstance(request["system"], str) or request["system_digest"] != base._digest(
            request["system"]
        ):
            raise base.ExperimentError(f"{where} system digest mismatch")
        base._validate_timestamp(request["reserved_at_utc"], f"{where}.reserved_at_utc")
        return request

    def _validate_call_purpose(self, purpose: Mapping[str, Any]) -> None:
        base._require_keys(
            purpose,
            {
                "phase",
                "pair_id",
                "pair_ordinal",
                "pair_attempt",
                "outcome_id",
                "outcome_ordinal",
                "cluster_id",
                "seed",
                "arm",
                "turn",
                "horizon",
            },
            "replay call purpose",
        )
        if purpose["phase"] != "actor-replay":
            raise base.ExperimentError("replay call purpose phase is invalid")
        pair = self._pair(str(purpose["pair_id"]))
        if purpose["pair_ordinal"] != pair["pair_ordinal"]:
            raise base.ExperimentError("replay call pair ordinal mismatch")
        attempt = base._positive_int(purpose["pair_attempt"], "pair attempt")
        if attempt > PAIR_ATTEMPTS:
            raise base.ExperimentError("replay call exceeds pair attempt limit")
        logical = next(
            (
                item
                for item in pair["logical_outcomes"]
                if item["outcome_id"] == purpose["outcome_id"]
            ),
            None,
        )
        if logical is None:
            raise base.ExperimentError("replay call outcome is outside its pair")
        expected = {
            "outcome_ordinal": logical["ordinal"],
            "cluster_id": logical["cluster_id"],
            "seed": logical["seed"],
            "arm": logical["arm"],
            "horizon": logical["horizon"],
        }
        for key, value in expected.items():
            if purpose[key] != value:
                raise base.ExperimentError(f"replay call purpose {key} mismatch")
        turn = base._positive_int(purpose["turn"], "replay call turn")
        if turn > logical["horizon"]:
            raise base.ExperimentError("replay call turn exceeds sealed horizon")

    @staticmethod
    def _failure_category(exc: Exception) -> str:
        message = str(exc)
        if isinstance(exc, base.live.LiveEvalError):
            if message == "agy returned an empty response":
                return "empty_response"
            if message == f"agy call timed out after {CANONICAL_TIMEOUT_SECONDS:.0f}s":
                return "provider_timeout"
            if message == "agy failed with exit 1: Error: timeout waiting for response":
                return "provider_timeout"
        return "fatal_transport"

    def _validate_call_result(
        self, result: Mapping[str, Any], request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        call_id = str(request["call_id"])
        where = f"replay call result {call_id}"
        base._validate_leaf(
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
                "failure_category",
                "exception_type",
                "error",
                "response",
                "response_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "call_id": call_id,
                "call_ordinal": request["call_ordinal"],
            },
        )
        reserved = base._validate_timestamp(request["reserved_at_utc"], "reserved_at_utc")
        started = base._validate_timestamp(result["started_at_utc"], "started_at_utc")
        finished = base._validate_timestamp(result["finished_at_utc"], "finished_at_utc")
        if not reserved <= started <= finished:
            raise base.ExperimentError(f"{where} timestamps are out of order")
        duration = base._finite_number(result["duration_seconds"], "duration_seconds")
        if duration < 0:
            raise base.ExperimentError(f"{where} duration cannot be negative")
        if result["status"] == "success":
            if (
                result["exit_status"] != 0
                or result["failure_category"] is not None
                or result["exception_type"] is not None
                or result["error"] is not None
            ):
                raise base.ExperimentError(f"{where} success metadata is contradictory")
            response = base._required_text(result["response"], f"{where}.response")
            if result["response_digest"] != base._digest(response):
                raise base.ExperimentError(f"{where} response digest mismatch")
        elif result["status"] == "error":
            error = base._required_text(result["error"], f"{where}.error")
            exception_type = base._required_text(
                result["exception_type"], f"{where}.exception_type"
            )
            if result["response"] is not None or result["response_digest"] is not None:
                raise base.ExperimentError(f"{where} error cannot contain a response")
            synthetic = (
                base.live.LiveEvalError(error, 4)
                if exception_type == "LiveEvalError"
                else RuntimeError(error)
            )
            expected_category = self._failure_category(synthetic)
            if result["failure_category"] != expected_category:
                raise base.ExperimentError(f"{where} failure category is not derivable")
            match = re.fullmatch(r"agy failed with exit (-?[0-9]+):.*", error, re.DOTALL)
            expected_exit = int(match.group(1)) if match else None
            if result["exit_status"] != expected_exit:
                raise base.ExperimentError(f"{where} transport exit status mismatch")
        else:
            raise base.ExperimentError(f"{where} status is invalid")
        return result

    async def _call_once(
        self,
        *,
        purpose: Mapping[str, Any],
        prompt: str,
        system: str = "",
    ) -> tuple[str, str]:
        call_id = base._call_id(purpose)
        directory = self.run_dir / "calls" / call_id
        request_path = directory / "request.json"
        result_path = directory / "result.json"
        expected = {
            "call_id": call_id,
            "purpose": dict(purpose),
            "prompt": prompt,
            "prompt_digest": base._digest(prompt),
            "system": system,
            "system_digest": base._digest(system),
        }
        if request_path.exists():
            request = self._validate_call_request(base._read_json(request_path), expected=expected)
            if not result_path.is_file():
                raise base.AmbiguousCall(f"call {call_id} is ambiguous and will not be replayed")
            result = self._validate_call_result(base._read_json(result_path), request)
            if result["status"] == "error":
                raise _RecordedProviderFailure(
                    call_id, str(result["failure_category"]), str(result["error"])
                )
            return str(result["response"]), call_id

        if self.call_count >= self.config["total_call_cap"]:
            raise base.CallCapExceeded(
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
            "reserved_at_utc": base._utc_now(),
            "purpose": dict(purpose),
            "provider": "agy",
            "model": self.plan["model"],
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "runtime_identity_digest": base._digest(self.plan["runtime_identity"]),
            "agy_executable_digest": self.plan["runtime_identity"]["agy_executable_digest"],
            "prompt": prompt,
            "prompt_digest": base._digest(prompt),
            "system": system,
            "system_digest": base._digest(system),
            "timeout_seconds": self.config["llm_timeout_seconds"],
            "workdir_policy": "fresh-temporary-directory-per-call",
        }
        self._validate_call_request(request, expected=expected)
        base._write_json(request_path, request)
        started_at = base._utc_now()
        started_clock = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="spade-agy-replay-call-") as workdir:
                response = await self.dependencies.llm_call(
                    self.dependencies.client_or_bin,
                    str(self.plan["model"]),
                    prompt,
                    system=system,
                    provider="agy",
                    workdir=Path(workdir),
                    timeout_seconds=float(self.config["llm_timeout_seconds"]),
                )
            if not isinstance(response, str) or not response.strip():
                raise base.live.LiveEvalError("agy returned an empty response", 4)
            response = response.strip()
        except Exception as exc:
            error = str(exc)
            match = re.fullmatch(r"agy failed with exit (-?[0-9]+):.*", error, re.DOTALL)
            result = {
                "schema_version": CALL_RESULT_SCHEMA,
                "plan_digest": self.plan["plan_digest"],
                "call_id": call_id,
                "call_ordinal": request["call_ordinal"],
                "status": "error",
                "started_at_utc": started_at,
                "finished_at_utc": base._utc_now(),
                "duration_seconds": time.monotonic() - started_clock,
                "exit_status": int(match.group(1)) if match else None,
                "failure_category": self._failure_category(exc),
                "exception_type": type(exc).__name__,
                "error": error or type(exc).__name__,
                "response": None,
                "response_digest": None,
            }
            self._validate_call_result(result, request)
            base._write_json(result_path, result)
            raise _RecordedProviderFailure(
                call_id, str(result["failure_category"]), str(result["error"])
            ) from exc
        result = {
            "schema_version": CALL_RESULT_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "call_id": call_id,
            "call_ordinal": request["call_ordinal"],
            "status": "success",
            "started_at_utc": started_at,
            "finished_at_utc": base._utc_now(),
            "duration_seconds": time.monotonic() - started_clock,
            "exit_status": 0,
            "failure_category": None,
            "exception_type": None,
            "error": None,
            "response": response,
            "response_digest": base._digest(response),
        }
        self._validate_call_result(result, request)
        base._write_json(result_path, result)
        return response, call_id

    def _run_manifest_value(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA,
            "experiment_id": self.plan["experiment_id"],
            "plan_digest": self.plan["plan_digest"],
            "provider": "agy",
            "model": self.plan["model"],
            "route_authority": "requested-route-only",
            "backend_identity_attested": False,
            "stage": self.plan["stage"],
            "analysis_role": self.plan["analysis_role"],
            "protocol_id": self.plan["protocol_id"],
            "source_plan_digest": self.plan["source_evidence"]["source_plan_digest"],
            "source_cohort_digest": self.plan["source_evidence"]["source_cohort_digest"],
            "import_manifest_digest": self.plan["source_evidence"]["import_manifest_digest"],
            "total_call_cap": self.config["total_call_cap"],
            "computed_call_ceiling": self.config["computed_call_ceiling"],
            "source_revisions": self.plan["source_revisions"],
            "runtime_identity": self.plan["runtime_identity"],
            "replay_runner_digest": self.plan["replay_runner_digest"],
        }

    def initialize(self) -> None:
        if dict(self.dependencies.source_revisions) != self.plan["source_revisions"]:
            raise base.ExperimentIncomplete("current source revisions differ from replay plan")
        if dict(self.dependencies.runtime_identity) != self.plan["runtime_identity"]:
            raise base.ExperimentIncomplete("current runtime identity differs from replay plan")
        if _file_digest(Path(__file__).resolve()) != self.plan["replay_runner_digest"]:
            raise base.ExperimentIncomplete("outcome replay runner bytes differ from seal")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._validate_run_tree()
        base._write_immutable_bytes(self.run_dir / "plan.json", self.plan_bytes)
        base._write_json(
            self.run_dir / "run-manifest.json",
            self._run_manifest_value(),
        )

        import_root = self.run_dir / IMPORT_ROOT
        import_complete = False
        if import_root.exists() or import_root.is_symlink():
            import_complete = self._validate_import_manifest_subset(import_root)
        if import_complete:
            self._validate_import_manifest_bytes(import_root)
            source_plan = base.load_plan(import_root / "plan.json")
            source_cohort = base._read_json(import_root / "cohort-lock.json")
            snapshot = SourceSnapshot(
                run_dir=import_root,
                plan=source_plan,
                cohort=source_cohort,
                selections={},
                manifest=self.plan["source_evidence"]["import_manifest"],
            )
        else:
            # A partially published import is a validated immutable subset, not a
            # new authority. Revalidate the sealed source, then fill only missing
            # leaves so a crash between leaf publications is deterministically
            # resumable without ever mixing source snapshots.
            snapshot = _validate_source_snapshot(
                self.plan["source_evidence"]["source_run_dir"], dependencies=self.dependencies
            )
            if (
                snapshot.plan["plan_digest"] != self.plan["source_evidence"]["source_plan_digest"]
                or snapshot.cohort["cohort_digest"]
                != self.plan["source_evidence"]["source_cohort_digest"]
                or snapshot.manifest != self.plan["source_evidence"]["import_manifest"]
            ):
                raise base.ExperimentError("source v4 evidence differs from replay seal")
            import_root = self._copy_source_evidence(snapshot)
        imported_selections = self._validate_imported_cohort(import_root)
        self._source_snapshot = SourceSnapshot(
            run_dir=import_root,
            plan=snapshot.plan,
            cohort=snapshot.cohort,
            selections=imported_selections,
            manifest=snapshot.manifest,
        )
        self._rerun_offline_assurance(self._source_snapshot)
        cohort = self._cohort_value(imported_selections)
        base._write_json(self.run_dir / "cohort-lock.json", cohort)
        self._validate_cohort(cohort)

        calls_root = self.run_dir / "calls"
        if calls_root.is_dir():
            for directory in calls_root.iterdir():
                if directory.is_symlink() or not directory.is_dir():
                    raise base.ExperimentError(f"unsafe replay call directory: {directory}")
                inventory = {path.name for path in directory.iterdir()}
                if not inventory.issubset({"request.json", "result.json"}):
                    raise base.ExperimentError(f"unexpected replay call artifact: {directory}")
        requests = sorted(calls_root.glob("*/request.json")) if calls_root.is_dir() else []
        results = sorted(calls_root.glob("*/result.json")) if calls_root.is_dir() else []
        if {path.parent for path in results} - {path.parent for path in requests}:
            raise base.ExperimentError("replay call result exists without reservation")
        ordinals: list[int] = []
        for path in requests:
            request = self._validate_call_request(base._read_json(path))
            if path.parent.name != request["call_id"]:
                raise base.ExperimentError("replay call stored under a noncanonical directory")
            ordinals.append(int(request["call_ordinal"]))
            result_path = path.parent / "result.json"
            if not result_path.is_file():
                raise base.AmbiguousCall(
                    f"call {request['call_id']} has a reservation but no durable result"
                )
            self._validate_call_result(base._read_json(result_path), request)
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            raise base.ExperimentError("replay call ordinals are not one contiguous sequence")
        if len(ordinals) > int(self.config["total_call_cap"]):
            raise base.ExperimentError("persisted call reservations exceed the sealed cap")
        ordered_requests = sorted(
            (self._validate_call_request(base._read_json(path)) for path in requests),
            key=lambda item: int(item["call_ordinal"]),
        )
        prior_key: tuple[int, int, int, int] | None = None
        for request in ordered_requests:
            purpose = request["purpose"]
            pair = self._pair(str(purpose["pair_id"]))
            key = (
                int(purpose["pair_ordinal"]),
                int(purpose["pair_attempt"]),
                list(pair["arm_order"]).index(str(purpose["arm"])),
                int(purpose["turn"]),
            )
            if prior_key is not None and key <= prior_key:
                raise base.ExperimentError("replay calls violate the sealed pair/attempt/arm order")
            prior_key = key

    def _purpose(
        self,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        turn: int,
    ) -> dict[str, Any]:
        return {
            "phase": "actor-replay",
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "pair_attempt": pair_attempt,
            "outcome_id": logical["outcome_id"],
            "outcome_ordinal": logical["ordinal"],
            "cluster_id": logical["cluster_id"],
            "seed": logical["seed"],
            "arm": logical["arm"],
            "turn": turn,
            "horizon": logical["horizon"],
        }

    @staticmethod
    def _prompt(history: str, turn: int, horizon: int) -> str:
        return (
            "You are playing an interactive reasoning environment.\n"
            f"{history}\n\nTurn {turn}/{horizon}: reason about the state, then provide "
            "exactly one next action with the required answer format."
        )

    def _attempt_root(self, pair_id: str, pair_attempt: int) -> Path:
        return self.run_dir / "pair-attempts" / pair_id / f"attempt-{pair_attempt:02d}"

    def _arm_root(self, pair_id: str, pair_attempt: int, arm: str) -> Path:
        return self._attempt_root(pair_id, pair_attempt) / arm

    def _preflight_attempt_tree(self, pair: Mapping[str, Any], pair_attempt: int) -> None:
        root = self._attempt_root(str(pair["pair_id"]), pair_attempt)
        if not root.exists():
            return
        if root.is_symlink() or not root.is_dir():
            raise base.ExperimentError(f"unsafe pair attempt path: {root}")
        allowed_root = {"attempt.json", *ARMS}
        if any(entry.name not in allowed_root for entry in root.iterdir()):
            raise base.ExperimentError(f"unexpected pair attempt artifact: {root}")
        for arm in ARMS:
            arm_root = root / arm
            if not arm_root.exists():
                continue
            if arm_root.is_symlink() or not arm_root.is_dir():
                raise base.ExperimentError(f"unsafe pair arm path: {arm_root}")
            turn_paths: list[Path] = []
            for entry in arm_root.iterdir():
                if entry.is_symlink() or not entry.is_file():
                    raise base.ExperimentError(f"unsafe pair arm artifact: {entry}")
                if entry.name == "outcome.json":
                    continue
                if re.fullmatch(r"turn-[0-9]{2}\.json", entry.name) is None:
                    raise base.ExperimentError(f"unexpected pair arm artifact: {entry}")
                turn_paths.append(entry)
            indices = sorted(int(path.stem.removeprefix("turn-")) for path in turn_paths)
            if indices != list(range(1, len(indices) + 1)):
                raise base.ExperimentError("pair arm turns are not one contiguous prefix")
            if indices and indices[-1] > int(pair["horizon"]):
                raise base.ExperimentError("pair arm turns exceed the sealed horizon")

    def _validate_turn(
        self,
        leaf: Mapping[str, Any],
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        turn: int,
        cohort: Mapping[str, Any],
        expected_prompt: str,
    ) -> Mapping[str, Any]:
        where = f"pair {pair['pair_id']} attempt {pair_attempt} {logical['arm']} turn {turn}"
        base._validate_leaf(
            leaf,
            schema=TURN_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "cohort_digest",
                "pair_id",
                "pair_attempt",
                "outcome_id",
                "outcome_ordinal",
                "cluster_id",
                "seed",
                "arm",
                "horizon",
                "turn",
                "call_id",
                "raw_response",
                "raw_response_digest",
                "clean_action",
                "pre_observation",
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
                "pair_id": pair["pair_id"],
                "pair_attempt": pair_attempt,
                "outcome_id": logical["outcome_id"],
                "outcome_ordinal": logical["ordinal"],
                "cluster_id": logical["cluster_id"],
                "seed": logical["seed"],
                "arm": logical["arm"],
                "horizon": logical["horizon"],
                "turn": turn,
            },
        )
        raw = base._required_text(leaf["raw_response"], f"{where}.raw_response")
        if leaf["raw_response_digest"] != base._digest(raw):
            raise base.ExperimentError(f"{where} raw response digest mismatch")
        if leaf["clean_action"] != base.live.extract_clean_action(raw, "boxed"):
            raise base.ExperimentError(f"{where} clean action differs from response")
        for observation_field in ("pre_observation", "post_observation"):
            if not isinstance(leaf[observation_field], str):
                raise base.ExperimentError(f"{where} {observation_field} must be text")
            if leaf[f"{observation_field}_digest"] != base._digest(leaf[observation_field]):
                raise base.ExperimentError(f"{where} {observation_field} digest mismatch")
        base._finite_number(leaf["reward"], f"{where}.reward")
        if type(leaf["terminated"]) is not bool or type(leaf["truncated"]) is not bool:
            raise base.ExperimentError(f"{where} termination flags must be booleans")
        purpose = self._purpose(pair, logical, pair_attempt, turn)
        call_id = base._call_id(purpose)
        if leaf["call_id"] != call_id:
            raise base.ExperimentError(f"{where} call id mismatch")
        request_path = self.run_dir / "calls" / call_id / "request.json"
        request = self._validate_call_request(
            base._read_json(request_path),
            expected={
                "call_id": call_id,
                "purpose": purpose,
                "prompt": expected_prompt,
                "prompt_digest": base._digest(expected_prompt),
                "system": "",
                "system_digest": base._digest(""),
            },
        )
        result = self._validate_call_result(
            base._read_json(request_path.parent / "result.json"), request
        )
        if result["status"] != "success" or result["response"] != raw:
            raise base.ExperimentError(f"{where} does not link one successful response")
        body = {key: value for key, value in leaf.items() if key != "turn_digest"}
        if leaf["turn_digest"] != base._digest(body):
            raise base.ExperimentError(f"{where} digest mismatch")
        return leaf

    def _replay_attempt_outcome(
        self,
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        trajectory: Sequence[Mapping[str, Any]],
    ) -> None:
        selection = self._selection(str(logical["cluster_id"]))
        probe = selection["probes"][str(logical["seed"])]
        try:
            target = self.dependencies.target_factory(
                selection["code"],
                action_format="boxed",
                max_turns=int(logical["horizon"]),
                operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
            )
            env = target.instantiate()
        except Exception as exc:
            raise base.ExperimentIncomplete(
                f"cannot instantiate imported environment for replay: {exc}"
            ) from exc
        try:
            observation, _info = env.reset(seed=int(logical["seed"]))
            if str(observation) != probe["observation"]:
                raise base.ExperimentError("attempt replay reset differs from locked probe")
            for index, leaf in enumerate(trajectory, start=1):
                if leaf["pre_observation"] != str(observation):
                    raise base.ExperimentError("attempt replay observation chain diverged")
                result = env.step(str(leaf["clean_action"]))
                if not isinstance(result, tuple) or len(result) != 5:
                    raise base.ExperimentError("attempt replay requires a five-item Gym step")
                post, reward, terminated, truncated, _step_info = result
                expected = {
                    "post_observation": str(post),
                    "post_observation_digest": base._digest(str(post)),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                for key, value in expected.items():
                    if leaf[key] != value:
                        raise base.ExperimentError(f"attempt replay diverged at turn {index} {key}")
                observation = post
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _validate_attempt_outcome(
        self,
        outcome: Mapping[str, Any],
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        where = f"attempt outcome {logical['outcome_id']}/{pair_attempt}"
        base._validate_leaf(
            outcome,
            schema=ATTEMPT_OUTCOME_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "cohort_digest",
                "pair_id",
                "pair_attempt",
                "outcome_id",
                "outcome_ordinal",
                "cluster_id",
                "seed",
                "arm",
                "horizon",
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
                "pair_id": pair["pair_id"],
                "pair_attempt": pair_attempt,
                "outcome_id": logical["outcome_id"],
                "outcome_ordinal": logical["ordinal"],
                "cluster_id": logical["cluster_id"],
                "seed": logical["seed"],
                "arm": logical["arm"],
                "horizon": logical["horizon"],
                "status": "completed",
            },
        )
        base._finite_number(outcome["return"], f"{where}.return")
        if type(outcome["terminated"]) is not bool or type(outcome["truncated"]) is not bool:
            raise base.ExperimentError(f"{where} termination flags must be booleans")
        trajectory = outcome["trajectory"]
        if not isinstance(trajectory, list) or not 1 <= len(trajectory) <= logical["horizon"]:
            raise base.ExperimentError(f"{where} trajectory length is invalid")
        arm_root = self._arm_root(str(pair["pair_id"]), pair_attempt, str(logical["arm"]))
        expected_paths = [
            arm_root / f"turn-{index:02d}.json" for index in range(1, len(trajectory) + 1)
        ]
        actual_paths = sorted(
            arm_root.glob("turn-*.json"), key=lambda path: int(path.stem.removeprefix("turn-"))
        )
        if actual_paths != expected_paths:
            raise base.ExperimentError(f"{where} has missing or extra turns")
        selection = self._selection(str(logical["cluster_id"]), cohort)
        probe = selection["probes"][str(logical["seed"])]
        hint = (
            str(selection["hints"][str(logical["seed"])]["hint"])
            if logical["arm"] == "hinted"
            else ""
        )
        observation = str(probe["observation"])
        history = f"Initial observation: {observation}"
        if hint:
            history += f"\n\nPrivileged strategy hint:\n{hint}"
        prior_done = False
        for turn, embedded in enumerate(trajectory, start=1):
            if prior_done:
                raise base.ExperimentError(f"{where} contains a turn after completion")
            disk = base._read_json(arm_root / f"turn-{turn:02d}.json")
            if embedded != disk:
                raise base.ExperimentError(f"{where} embedded turn differs from disk")
            prompt = self._prompt(history, turn, int(logical["horizon"]))
            self._validate_turn(
                disk,
                pair=pair,
                logical=logical,
                pair_attempt=pair_attempt,
                turn=turn,
                cohort=cohort,
                expected_prompt=prompt,
            )
            if disk["pre_observation"] != observation:
                raise base.ExperimentError(f"{where} observation chain is broken")
            observation = str(disk["post_observation"])
            history += (
                f"\n\nAction at turn {turn}: {disk['clean_action']}\n"
                f"Environment response: {observation}"
            )
            prior_done = bool(disk["terminated"] or disk["truncated"])
        if not prior_done and len(trajectory) != logical["horizon"]:
            raise base.ExperimentError(f"{where} ended before completion or horizon")
        self._replay_attempt_outcome(
            pair=pair,
            logical=logical,
            pair_attempt=pair_attempt,
            trajectory=trajectory,
        )
        final = trajectory[-1]
        expected_return = float(final["reward"]) if final["terminated"] else 0.0
        if outcome["return"] != expected_return:
            raise base.ExperimentError(f"{where} return differs from final turn")
        if (
            outcome["terminated"] != final["terminated"]
            or outcome["truncated"] != final["truncated"]
        ):
            raise base.ExperimentError(f"{where} final flags differ from trajectory")
        body = {key: value for key, value in outcome.items() if key != "outcome_digest"}
        if outcome["outcome_digest"] != base._digest(body):
            raise base.ExperimentError(f"{where} digest mismatch")
        return outcome

    async def _run_arm(
        self,
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
    ) -> dict[str, Any]:
        arm_root = self._arm_root(str(pair["pair_id"]), pair_attempt, str(logical["arm"]))
        outcome_path = arm_root / "outcome.json"
        if outcome_path.is_file():
            return dict(
                self._validate_attempt_outcome(
                    base._read_json(outcome_path),
                    pair=pair,
                    logical=logical,
                    pair_attempt=pair_attempt,
                    cohort=cohort,
                )
            )
        selection = self._selection(str(logical["cluster_id"]), cohort)
        probe = selection["probes"][str(logical["seed"])]
        hint = (
            str(selection["hints"][str(logical["seed"])]["hint"])
            if logical["arm"] == "hinted"
            else ""
        )
        try:
            target = self.dependencies.target_factory(
                selection["code"],
                action_format="boxed",
                max_turns=int(logical["horizon"]),
                operation_timeout_seconds=float(self.config["qualification_timeout_seconds"]),
            )
            env = target.instantiate()
            observation, _info = env.reset(seed=int(logical["seed"]))
        except Exception as exc:
            raise _ArmExecutionFailure(
                f"environment reset failed: {type(exc).__name__}: {exc}"
            ) from exc
        trajectory: list[dict[str, Any]] = []
        try:
            if str(observation) != probe["observation"]:
                raise _ArmExecutionFailure("environment reset differs from locked probe")
            history = f"Initial observation: {observation}"
            if hint:
                history += f"\n\nPrivileged strategy hint:\n{hint}"
            terminated = False
            truncated = False
            last_reward = 0.0
            for turn in range(1, int(logical["horizon"]) + 1):
                if terminated or truncated:
                    break
                prompt = self._prompt(history, turn, int(logical["horizon"]))
                turn_path = arm_root / f"turn-{turn:02d}.json"
                if turn_path.is_file():
                    leaf = dict(
                        self._validate_turn(
                            base._read_json(turn_path),
                            pair=pair,
                            logical=logical,
                            pair_attempt=pair_attempt,
                            turn=turn,
                            cohort=cohort,
                            expected_prompt=prompt,
                        )
                    )
                    if leaf["pre_observation"] != str(observation):
                        raise base.ExperimentError("partial arm replay pre-state diverged")
                    action = str(leaf["clean_action"])
                else:
                    raw, call_id = await self._call_once(
                        purpose=self._purpose(pair, logical, pair_attempt, turn), prompt=prompt
                    )
                    action = base.live.extract_clean_action(raw, "boxed")
                    leaf = {
                        "schema_version": TURN_SCHEMA,
                        "plan_digest": self.plan["plan_digest"],
                        "cohort_digest": cohort["cohort_digest"],
                        "pair_id": pair["pair_id"],
                        "pair_attempt": pair_attempt,
                        "outcome_id": logical["outcome_id"],
                        "outcome_ordinal": logical["ordinal"],
                        "cluster_id": logical["cluster_id"],
                        "seed": logical["seed"],
                        "arm": logical["arm"],
                        "horizon": logical["horizon"],
                        "turn": turn,
                        "call_id": call_id,
                        "raw_response": raw,
                        "raw_response_digest": base._digest(raw),
                        "clean_action": action,
                        "pre_observation": str(observation),
                        "pre_observation_digest": base._digest(str(observation)),
                    }
                try:
                    result = env.step(action)
                except Exception as exc:
                    raise _ArmExecutionFailure(
                        f"environment step failed: {type(exc).__name__}: {exc}",
                        call_id=str(leaf["call_id"]),
                    ) from exc
                if not isinstance(result, tuple) or len(result) != 5:
                    raise _ArmExecutionFailure(
                        "environment returned a non-Gym five-item step",
                        call_id=str(leaf["call_id"]),
                    )
                post, reward, terminated, truncated, _step_info = result
                last_reward = base._finite_number(reward, "actor reward")
                replay_fields = {
                    "post_observation": str(post),
                    "post_observation_digest": base._digest(str(post)),
                    "reward": last_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                if turn_path.is_file():
                    for key, value in replay_fields.items():
                        if leaf[key] != value:
                            raise base.ExperimentError(f"partial arm replay diverged at {key}")
                else:
                    leaf.update(replay_fields)
                    leaf["turn_digest"] = base._digest(leaf)
                    self._validate_turn(
                        leaf,
                        pair=pair,
                        logical=logical,
                        pair_attempt=pair_attempt,
                        turn=turn,
                        cohort=cohort,
                        expected_prompt=prompt,
                    )
                    base._write_json(turn_path, leaf)
                trajectory.append(leaf)
                observation = post
                history += (
                    f"\n\nAction at turn {turn}: {action}\nEnvironment response: {observation}"
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        body = {
            "schema_version": ATTEMPT_OUTCOME_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "pair_id": pair["pair_id"],
            "pair_attempt": pair_attempt,
            "outcome_id": logical["outcome_id"],
            "outcome_ordinal": logical["ordinal"],
            "cluster_id": logical["cluster_id"],
            "seed": logical["seed"],
            "arm": logical["arm"],
            "horizon": logical["horizon"],
            "status": "completed",
            "return": last_reward if terminated else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "trajectory": trajectory,
        }
        outcome = {**body, "outcome_digest": base._digest(body)}
        self._validate_attempt_outcome(
            outcome,
            pair=pair,
            logical=logical,
            pair_attempt=pair_attempt,
            cohort=cohort,
        )
        base._write_json(outcome_path, outcome)
        return outcome

    def _attempt_call_ids(self, pair_id: str, pair_attempt: int) -> list[str]:
        calls: list[tuple[int, str]] = []
        root = self.run_dir / "calls"
        for path in root.glob("*/request.json") if root.is_dir() else ():
            request = self._validate_call_request(base._read_json(path))
            purpose = request["purpose"]
            if purpose["pair_id"] == pair_id and purpose["pair_attempt"] == pair_attempt:
                calls.append((int(request["call_ordinal"]), str(request["call_id"])))
        calls.sort()
        return [call_id for _ordinal, call_id in calls]

    def _attempt_outcomes(
        self,
        pair: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        outcomes: dict[str, dict[str, Any]] = {}
        for logical in pair["logical_outcomes"]:
            path = (
                self._arm_root(str(pair["pair_id"]), pair_attempt, str(logical["arm"]))
                / "outcome.json"
            )
            if path.is_file():
                outcomes[str(logical["arm"])] = dict(
                    self._validate_attempt_outcome(
                        base._read_json(path),
                        pair=pair,
                        logical=logical,
                        pair_attempt=pair_attempt,
                        cohort=cohort,
                    )
                )
        return outcomes

    def _partial_turn_call_ids(
        self,
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
    ) -> list[str]:
        """Validate and return every materialized turn in one possibly partial arm."""
        selection = self._selection(str(logical["cluster_id"]), cohort)
        probe = selection["probes"][str(logical["seed"])]
        hint = (
            str(selection["hints"][str(logical["seed"])]["hint"])
            if logical["arm"] == "hinted"
            else ""
        )
        observation = str(probe["observation"])
        history = f"Initial observation: {observation}"
        if hint:
            history += f"\n\nPrivileged strategy hint:\n{hint}"
        root = self._arm_root(str(pair["pair_id"]), pair_attempt, str(logical["arm"]))
        paths = sorted(
            root.glob("turn-*.json") if root.is_dir() else (),
            key=lambda path: int(path.stem.removeprefix("turn-")),
        )
        call_ids: list[str] = []
        trajectory: list[dict[str, Any]] = []
        prior_done = False
        for turn, path in enumerate(paths, start=1):
            if path.name != f"turn-{turn:02d}.json" or prior_done:
                raise base.ExperimentError("partial attempt arm has invalid turn ordering")
            prompt = self._prompt(history, turn, int(logical["horizon"]))
            leaf = dict(
                self._validate_turn(
                    base._read_json(path),
                    pair=pair,
                    logical=logical,
                    pair_attempt=pair_attempt,
                    turn=turn,
                    cohort=cohort,
                    expected_prompt=prompt,
                )
            )
            if leaf["pre_observation"] != observation:
                raise base.ExperimentError("partial attempt arm observation chain is broken")
            observation = str(leaf["post_observation"])
            history += (
                f"\n\nAction at turn {turn}: {leaf['clean_action']}\n"
                f"Environment response: {observation}"
            )
            prior_done = bool(leaf["terminated"] or leaf["truncated"])
            trajectory.append(leaf)
            call_ids.append(str(leaf["call_id"]))
        if trajectory:
            self._replay_attempt_outcome(
                pair=pair,
                logical=logical,
                pair_attempt=pair_attempt,
                trajectory=trajectory,
            )
        return call_ids

    def _attempt_summary_value(
        self,
        *,
        pair: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
        status: str,
        failure_category: str | None,
        error: str | None,
        failed_call_id: str | None,
    ) -> dict[str, Any]:
        outcomes = self._attempt_outcomes(pair, pair_attempt, cohort)
        body = {
            "schema_version": PAIR_ATTEMPT_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "pair_attempt": pair_attempt,
            "arm_order": pair["arm_order"],
            "horizon": pair["horizon"],
            "status": status,
            "failure_category": failure_category,
            "error": error,
            "failed_call_id": failed_call_id,
            "call_ids": self._attempt_call_ids(str(pair["pair_id"]), pair_attempt),
            "completed_outcome_digests": {
                arm: outcome["outcome_digest"] for arm, outcome in sorted(outcomes.items())
            },
        }
        return {**body, "attempt_digest": base._digest(body)}

    def _validate_attempt_summary(
        self,
        value: Mapping[str, Any],
        *,
        pair: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        where = f"pair attempt summary {pair['pair_id']}/{pair_attempt}"
        base._validate_leaf(
            value,
            schema=PAIR_ATTEMPT_SCHEMA,
            keys={
                "schema_version",
                "plan_digest",
                "cohort_digest",
                "pair_id",
                "pair_ordinal",
                "pair_attempt",
                "arm_order",
                "horizon",
                "status",
                "failure_category",
                "error",
                "failed_call_id",
                "call_ids",
                "completed_outcome_digests",
                "attempt_digest",
            },
            where=where,
            expected={
                "plan_digest": self.plan["plan_digest"],
                "cohort_digest": cohort["cohort_digest"],
                "pair_id": pair["pair_id"],
                "pair_ordinal": pair["pair_ordinal"],
                "pair_attempt": pair_attempt,
                "arm_order": pair["arm_order"],
                "horizon": pair["horizon"],
                "call_ids": self._attempt_call_ids(str(pair["pair_id"]), pair_attempt),
            },
        )
        outcomes = self._attempt_outcomes(pair, pair_attempt, cohort)
        expected_digests = {
            arm: outcome["outcome_digest"] for arm, outcome in sorted(outcomes.items())
        }
        if value["completed_outcome_digests"] != expected_digests:
            raise base.ExperimentError(f"{where} outcome digest map mismatch")
        turn_call_ids: list[str] = []
        for logical in pair["logical_outcomes"]:
            turn_call_ids.extend(
                self._partial_turn_call_ids(
                    pair=pair,
                    logical=logical,
                    pair_attempt=pair_attempt,
                    cohort=cohort,
                )
            )
        status = value["status"]
        if status == "complete":
            if (
                set(outcomes) != set(ARMS)
                or value["failure_category"] is not None
                or value["error"] is not None
                or value["failed_call_id"] is not None
            ):
                raise base.ExperimentError(f"{where} completion semantics are contradictory")
            linked_call_ids = turn_call_ids
        elif status in {"retryable_failure", "fatal_failure"}:
            category = base._required_text(value["failure_category"], f"{where}.failure_category")
            error = base._required_text(value["error"], f"{where}.error")
            call_id = value["failed_call_id"]
            if status == "retryable_failure" and category not in {
                "empty_response",
                "provider_timeout",
            }:
                raise base.ExperimentError(f"{where} has a noneligible retry category")
            if status == "fatal_failure" and category in {
                "empty_response",
                "provider_timeout",
            }:
                raise base.ExperimentError(f"{where} labels an eligible failure fatal")
            if set(outcomes) == set(ARMS):
                raise base.ExperimentError(f"{where} cannot fail after both arms completed")
            if call_id is not None:
                request_path = self.run_dir / "calls" / str(call_id) / "request.json"
                request = self._validate_call_request(base._read_json(request_path))
                result = self._validate_call_result(
                    base._read_json(request_path.parent / "result.json"), request
                )
                if category == "fatal_environment":
                    if result["status"] != "success" or call_id not in value["call_ids"]:
                        raise base.ExperimentError(
                            f"{where} environment failure must link a consumed response"
                        )
                    linked_call_ids = list(dict.fromkeys([*turn_call_ids, str(call_id)]))
                else:
                    if (
                        result["status"] != "error"
                        or result["failure_category"] != category
                        or result["error"] != error
                        or call_id not in value["call_ids"]
                    ):
                        raise base.ExperimentError(
                            f"{where} failure does not match its call result"
                        )
                    linked_call_ids = [*turn_call_ids, str(call_id)]
                failed_purpose = request["purpose"]
                failed_arm = str(failed_purpose["arm"])
                if failed_arm in outcomes:
                    raise base.ExperimentError(f"{where} failure follows a completed arm")
                failed_logical = next(
                    item for item in pair["logical_outcomes"] if item["arm"] == failed_arm
                )
                partial_ids = self._partial_turn_call_ids(
                    pair=pair,
                    logical=failed_logical,
                    pair_attempt=pair_attempt,
                    cohort=cohort,
                )
                if int(failed_purpose["turn"]) != len(partial_ids) + 1:
                    raise base.ExperimentError(
                        f"{where} failure is not the immediate next incomplete turn"
                    )
                if partial_ids:
                    last_turn = base._read_json(
                        self._arm_root(str(pair["pair_id"]), pair_attempt, failed_arm)
                        / f"turn-{len(partial_ids):02d}.json"
                    )
                    if last_turn["terminated"] or last_turn["truncated"]:
                        raise base.ExperimentError(f"{where} failure follows a terminal turn")
            elif category != "fatal_environment":
                raise base.ExperimentError(f"{where} missing failed_call_id is contradictory")
            else:
                linked_call_ids = turn_call_ids
        else:
            raise base.ExperimentError(f"{where} status is invalid")
        call_ordinals: dict[str, int] = {}
        for call_id in linked_call_ids:
            request = self._validate_call_request(
                base._read_json(self.run_dir / "calls" / call_id / "request.json")
            )
            call_ordinals[call_id] = int(request["call_ordinal"])
        linked_call_ids.sort(key=lambda call_id: call_ordinals[call_id])
        if value["call_ids"] != linked_call_ids or len(set(linked_call_ids)) != len(
            linked_call_ids
        ):
            raise base.ExperimentError(f"{where} contains unlinked or duplicate calls")
        body = {key: item for key, item in value.items() if key != "attempt_digest"}
        if value["attempt_digest"] != base._digest(body):
            raise base.ExperimentError(f"{where} digest mismatch")
        return value

    def _write_attempt_summary(
        self,
        *,
        pair: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
        status: str,
        failure_category: str | None = None,
        error: str | None = None,
        failed_call_id: str | None = None,
    ) -> dict[str, Any]:
        value = self._attempt_summary_value(
            pair=pair,
            pair_attempt=pair_attempt,
            cohort=cohort,
            status=status,
            failure_category=failure_category,
            error=error,
            failed_call_id=failed_call_id,
        )
        self._validate_attempt_summary(value, pair=pair, pair_attempt=pair_attempt, cohort=cohort)
        path = self._attempt_root(str(pair["pair_id"]), pair_attempt) / "attempt.json"
        base._write_json(path, value)
        return value

    def _load_attempt_summary(
        self, pair: Mapping[str, Any], pair_attempt: int, cohort: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        path = self._attempt_root(str(pair["pair_id"]), pair_attempt) / "attempt.json"
        if not path.is_file():
            return None
        return dict(
            self._validate_attempt_summary(
                base._read_json(path),
                pair=pair,
                pair_attempt=pair_attempt,
                cohort=cohort,
            )
        )

    def _selected_outcome_value(
        self,
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        attempt_outcome: Mapping[str, Any],
        cohort: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = {
            "schema_version": OUTCOME_REFERENCE_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "selected_pair_attempt": pair_attempt,
            "outcome_id": logical["outcome_id"],
            "ordinal": logical["ordinal"],
            "cluster_id": logical["cluster_id"],
            "seed": logical["seed"],
            "arm": logical["arm"],
            "horizon": logical["horizon"],
            "attempt_outcome_relative_path": (
                Path("pair-attempts")
                / str(pair["pair_id"])
                / f"attempt-{pair_attempt:02d}"
                / str(logical["arm"])
                / "outcome.json"
            ).as_posix(),
            "attempt_outcome_digest": attempt_outcome["outcome_digest"],
            "return": attempt_outcome["return"],
            "terminated": attempt_outcome["terminated"],
            "truncated": attempt_outcome["truncated"],
        }
        return {**body, "selected_outcome_digest": base._digest(body)}

    def _validate_selected_outcome(
        self,
        value: Mapping[str, Any],
        *,
        pair: Mapping[str, Any],
        logical: Mapping[str, Any],
        pair_attempt: int,
        cohort: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        relative = (
            Path("pair-attempts")
            / str(pair["pair_id"])
            / f"attempt-{pair_attempt:02d}"
            / str(logical["arm"])
            / "outcome.json"
        ).as_posix()
        attempt_outcome = self._validate_attempt_outcome(
            base._read_json(self.run_dir / relative),
            pair=pair,
            logical=logical,
            pair_attempt=pair_attempt,
            cohort=cohort,
        )
        expected = self._selected_outcome_value(
            pair=pair,
            logical=logical,
            pair_attempt=pair_attempt,
            attempt_outcome=attempt_outcome,
            cohort=cohort,
        )
        if value != expected:
            raise base.ExperimentError("selected outcome reference differs from pair attempt")
        return value

    def _resolution_value(
        self,
        *,
        pair: Mapping[str, Any],
        selected_attempt: int,
        cohort: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        selected_summary = self._load_attempt_summary(pair, selected_attempt, cohort)
        if selected_summary is None or selected_summary["status"] != "complete":
            raise base.ExperimentError("pair resolution requires one complete attempt")
        outcomes = self._attempt_outcomes(pair, selected_attempt, cohort)
        selected: list[dict[str, Any]] = []
        for logical in pair["logical_outcomes"]:
            selected.append(
                self._selected_outcome_value(
                    pair=pair,
                    logical=logical,
                    pair_attempt=selected_attempt,
                    attempt_outcome=outcomes[str(logical["arm"])],
                    cohort=cohort,
                )
            )
        excluded: list[dict[str, Any]] = []
        for attempt in range(1, selected_attempt):
            summary = self._load_attempt_summary(pair, attempt, cohort)
            if summary is None or summary["status"] != "retryable_failure":
                raise base.ExperimentError("pair resolution skips a nonretryable attempt")
            excluded.append(
                {
                    "pair_attempt": attempt,
                    "attempt_digest": summary["attempt_digest"],
                    "reason": "whole pair discarded after eligible pre-response failure",
                }
            )
        body = {
            "schema_version": PAIR_RESOLUTION_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "pair_id": pair["pair_id"],
            "pair_ordinal": pair["pair_ordinal"],
            "cluster_id": pair["cluster_id"],
            "seed": pair["seed"],
            "horizon": pair["horizon"],
            "arm_order": pair["arm_order"],
            "selection_rule": "first fully completed same-attempt pair independent of rewards",
            "selected_pair_attempt": selected_attempt,
            "selected_attempt_digest": selected_summary["attempt_digest"],
            "selected_outcome_digests": {
                item["arm"]: item["selected_outcome_digest"] for item in selected
            },
            "excluded_attempts": excluded,
        }
        return {**body, "resolution_digest": base._digest(body)}, selected

    def _validate_resolution(
        self,
        value: Mapping[str, Any],
        *,
        pair: Mapping[str, Any],
        cohort: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
        selected_attempt = base._positive_int(
            value.get("selected_pair_attempt"), "selected_pair_attempt"
        )
        if selected_attempt > PAIR_ATTEMPTS:
            raise base.ExperimentError("pair resolution exceeds attempt limit")
        expected, selected = self._resolution_value(
            pair=pair, selected_attempt=selected_attempt, cohort=cohort
        )
        if value != expected:
            raise base.ExperimentError("pair resolution differs from deterministic rule")
        for logical, reference in zip(pair["logical_outcomes"], selected):
            path = self.run_dir / "outcomes" / str(logical["outcome_id"]) / "outcome.json"
            if not path.is_file():
                raise base.ExperimentError("pair resolution lacks a selected outcome reference")
            self._validate_selected_outcome(
                base._read_json(path),
                pair=pair,
                logical=logical,
                pair_attempt=selected_attempt,
                cohort=cohort,
            )
            if base._read_json(path) != reference:
                raise base.ExperimentError("selected outcome reference digest mismatch")
        return value, selected

    def _write_resolution(
        self,
        *,
        pair: Mapping[str, Any],
        selected_attempt: int,
        cohort: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        value, selected = self._resolution_value(
            pair=pair, selected_attempt=selected_attempt, cohort=cohort
        )
        for logical, reference in zip(pair["logical_outcomes"], selected):
            base._write_json(
                self.run_dir / "outcomes" / str(logical["outcome_id"]) / "outcome.json",
                reference,
            )
        path = self.run_dir / "pair-resolutions" / f"{pair['pair_id']}.json"
        base._write_json(path, value)
        self._validate_resolution(value, pair=pair, cohort=cohort)
        return value, selected

    async def _run_pair(
        self, pair: Mapping[str, Any], cohort: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        resolution_path = self.run_dir / "pair-resolutions" / f"{pair['pair_id']}.json"
        if resolution_path.is_file():
            value, selected = self._validate_resolution(
                base._read_json(resolution_path), pair=pair, cohort=cohort
            )
            return dict(value), selected
        for pair_attempt in range(1, PAIR_ATTEMPTS + 1):
            self._preflight_attempt_tree(pair, pair_attempt)
            summary = self._load_attempt_summary(pair, pair_attempt, cohort)
            if summary is not None:
                if summary["status"] == "complete":
                    return self._write_resolution(
                        pair=pair, selected_attempt=pair_attempt, cohort=cohort
                    )
                if summary["status"] == "retryable_failure" and pair_attempt < PAIR_ATTEMPTS:
                    continue
                raise base.ExperimentIncomplete(
                    f"pair {pair['pair_id']} has terminal {summary['status']}"
                )
            try:
                for logical in pair["logical_outcomes"]:
                    await self._run_arm(
                        pair=pair,
                        logical=logical,
                        pair_attempt=pair_attempt,
                        cohort=cohort,
                    )
            except _RecordedProviderFailure as exc:
                status = "retryable_failure" if exc.retryable else "fatal_failure"
                self._write_attempt_summary(
                    pair=pair,
                    pair_attempt=pair_attempt,
                    cohort=cohort,
                    status=status,
                    failure_category=exc.category,
                    error=exc.error,
                    failed_call_id=exc.call_id,
                )
                if exc.retryable and pair_attempt < PAIR_ATTEMPTS:
                    continue
                raise base.ExperimentIncomplete(
                    f"pair {pair['pair_id']} exhausted after {exc.category}"
                ) from exc
            except _ArmExecutionFailure as exc:
                self._write_attempt_summary(
                    pair=pair,
                    pair_attempt=pair_attempt,
                    cohort=cohort,
                    status="fatal_failure",
                    failure_category="fatal_environment",
                    error=str(exc),
                    failed_call_id=exc.call_id,
                )
                raise
            summary = self._write_attempt_summary(
                pair=pair,
                pair_attempt=pair_attempt,
                cohort=cohort,
                status="complete",
            )
            if summary["status"] != "complete":  # pragma: no cover - validated above
                raise base.ExperimentError("pair completion summary is contradictory")
            return self._write_resolution(pair=pair, selected_attempt=pair_attempt, cohort=cohort)
        raise base.ExperimentIncomplete(f"pair {pair['pair_id']} did not resolve")

    def _resolution_manifest(
        self, resolutions: Sequence[Mapping[str, Any]], cohort: Mapping[str, Any]
    ) -> dict[str, Any]:
        body = {
            "schema_version": RESOLUTION_MANIFEST_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "source_plan_digest": self.plan["source_evidence"]["source_plan_digest"],
            "source_cohort_digest": self.plan["source_evidence"]["source_cohort_digest"],
            "import_manifest_digest": self.plan["source_evidence"]["import_manifest_digest"],
            "pair_schedule_digest": base._digest(self.plan["pair_schedule"]),
            "pair_count": 54,
            "selected_outcome_count": 108,
            "resolutions": [
                {
                    "pair_id": item["pair_id"],
                    "pair_ordinal": item["pair_ordinal"],
                    "selected_pair_attempt": item["selected_pair_attempt"],
                    "resolution_digest": item["resolution_digest"],
                }
                for item in resolutions
            ],
        }
        return {**body, "manifest_digest": base._digest(body)}

    def _validate_resolution_manifest(
        self, value: Mapping[str, Any], cohort: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if len(list((self.run_dir / "pair-resolutions").glob("*.json"))) != 54:
            raise base.ExperimentError("pair resolution inventory is not exactly 54 leaves")
        resolutions = []
        for pair in self.plan["pair_schedule"]:
            path = self.run_dir / "pair-resolutions" / f"{pair['pair_id']}.json"
            resolution, _selected = self._validate_resolution(
                base._read_json(path), pair=pair, cohort=cohort
            )
            resolutions.append(resolution)
        expected = self._resolution_manifest(resolutions, cohort)
        if value != expected:
            raise base.ExperimentError("pair resolution manifest differs from all pair leaves")
        outcome_paths = list((self.run_dir / "outcomes").glob("*/outcome.json"))
        if len(outcome_paths) != 108:
            raise base.ExperimentError("selected outcome inventory is not exactly 108 leaves")
        return value

    async def run_outcomes(
        self, cohort: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        resolutions: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        for pair in self.plan["pair_schedule"]:
            resolution, pair_outcomes = await self._run_pair(pair, cohort)
            resolutions.append(resolution)
            selected.extend(pair_outcomes)
        if len(resolutions) != 54 or len(selected) != 108:
            raise base.ExperimentIncomplete("the sealed pair schedule is incomplete")
        value = self._resolution_manifest(resolutions, cohort)
        path = self.run_dir / "pair-resolution-manifest.json"
        base._write_json(path, value)
        self._validate_resolution_manifest(value, cohort)
        selected.sort(key=lambda item: int(item["ordinal"]))
        return selected, value

    def _validate_progress(self, cohort: Mapping[str, Any]) -> None:
        """Reject out-of-order or out-of-schedule resume artifacts."""
        pair_ids = {str(pair["pair_id"]) for pair in self.plan["pair_schedule"]}
        outcome_ids = {str(item["outcome_id"]) for item in self.plan["outcome_schedule"]}
        attempts_root = self.run_dir / "pair-attempts"
        if attempts_root.exists():
            if attempts_root.is_symlink() or not attempts_root.is_dir():
                raise base.ExperimentError("pair-attempts root is unsafe")
            for pair_root in attempts_root.iterdir():
                if (
                    pair_root.is_symlink()
                    or not pair_root.is_dir()
                    or pair_root.name not in pair_ids
                ):
                    raise base.ExperimentError(f"out-of-schedule pair attempt path: {pair_root}")
                pair = self._pair(pair_root.name)
                for attempt_root in pair_root.iterdir():
                    match = re.fullmatch(r"attempt-([0-9]{2})", attempt_root.name)
                    if (
                        attempt_root.is_symlink()
                        or not attempt_root.is_dir()
                        or match is None
                        or int(match.group(1)) not in {1, 2}
                    ):
                        raise base.ExperimentError(
                            f"out-of-schedule pair attempt directory: {attempt_root}"
                        )
                    self._preflight_attempt_tree(pair, int(match.group(1)))
        resolutions_root = self.run_dir / "pair-resolutions"
        if resolutions_root.exists():
            if resolutions_root.is_symlink() or not resolutions_root.is_dir():
                raise base.ExperimentError("pair-resolutions root is unsafe")
            for path in resolutions_root.iterdir():
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix != ".json"
                    or path.stem not in pair_ids
                ):
                    raise base.ExperimentError(f"out-of-schedule pair resolution: {path}")
        outcomes_root = self.run_dir / "outcomes"
        if outcomes_root.exists():
            if outcomes_root.is_symlink() or not outcomes_root.is_dir():
                raise base.ExperimentError("selected outcomes root is unsafe")
            for directory in outcomes_root.iterdir():
                if (
                    directory.is_symlink()
                    or not directory.is_dir()
                    or directory.name not in outcome_ids
                    or {path.name for path in directory.iterdir()} != {"outcome.json"}
                    or any(path.is_symlink() or not path.is_file() for path in directory.iterdir())
                ):
                    raise base.ExperimentError(
                        f"out-of-schedule selected outcome path: {directory}"
                    )

        unresolved_ordinal: int | None = None
        for pair in self.plan["pair_schedule"]:
            pair_id = str(pair["pair_id"])
            resolution_path = resolutions_root / f"{pair_id}.json"
            if resolution_path.is_file():
                if unresolved_ordinal is not None:
                    raise base.ExperimentError("pair resolutions are not one contiguous prefix")
                resolution, _selected = self._validate_resolution(
                    base._read_json(resolution_path), pair=pair, cohort=cohort
                )
                selected_attempt = int(resolution["selected_pair_attempt"])
                if selected_attempt == 1 and self._attempt_root(pair_id, 2).exists():
                    raise base.ExperimentError("attempt 2 exists after attempt 1 completed")
                continue
            if unresolved_ordinal is None:
                unresolved_ordinal = int(pair["pair_ordinal"])
            pair_root = attempts_root / pair_id
            selected_roots = [
                self.run_dir / "outcomes" / str(item["outcome_id"])
                for item in pair["logical_outcomes"]
            ]
            if int(pair["pair_ordinal"]) > unresolved_ordinal and (
                pair_root.exists() or any(path.exists() for path in selected_roots)
            ):
                raise base.ExperimentError("future pair evidence exists before prior resolution")

        for pair in self.plan["pair_schedule"]:
            first = self._load_attempt_summary(pair, 1, cohort)
            second_root = self._attempt_root(str(pair["pair_id"]), 2)
            if second_root.exists() and (first is None or first["status"] != "retryable_failure"):
                raise base.ExperimentError("attempt 2 exists without an eligible attempt-1 failure")

        requests: list[dict[str, Any]] = []
        calls_root = self.run_dir / "calls"
        for path in calls_root.glob("*/request.json") if calls_root.is_dir() else ():
            requests.append(dict(self._validate_call_request(base._read_json(path))))
        requests.sort(key=lambda item: int(item["call_ordinal"]))
        request_attempts: dict[str, set[int]] = {}
        for request in requests:
            purpose = request["purpose"]
            request_attempts.setdefault(str(purpose["pair_id"]), set()).add(
                int(purpose["pair_attempt"])
            )
        for pair in self.plan["pair_schedule"]:
            pair_id = str(pair["pair_id"])
            if 2 in request_attempts.get(pair_id, set()):
                first = self._load_attempt_summary(pair, 1, cohort)
                if first is None or first["status"] != "retryable_failure":
                    raise base.ExperimentError(
                        "attempt-2 call exists without an eligible attempt-1 failure"
                    )
            resolution_path = self.run_dir / "pair-resolutions" / f"{pair_id}.json"
            if resolution_path.is_file():
                resolution = base._read_json(resolution_path)
                if resolution.get("selected_pair_attempt") == 1 and 2 in request_attempts.get(
                    pair_id, set()
                ):
                    raise base.ExperimentError("attempt-2 call exists after attempt 1 completed")
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for request in requests:
            purpose = request["purpose"]
            grouped.setdefault(
                (
                    str(purpose["pair_id"]),
                    int(purpose["pair_attempt"]),
                    str(purpose["arm"]),
                ),
                [],
            ).append(request)
        for (pair_id, pair_attempt, arm), arm_requests in grouped.items():
            turns = [int(request["purpose"]["turn"]) for request in arm_requests]
            if turns != list(range(1, len(turns) + 1)):
                raise base.ExperimentError("replay call turns are not a contiguous arm prefix")
            pair = self._pair(pair_id)
            arm_index = list(pair["arm_order"]).index(arm)
            if arm_index:
                prior_arm = str(pair["arm_order"][arm_index - 1])
                prior_outcome = self._arm_root(pair_id, pair_attempt, prior_arm) / "outcome.json"
                if not prior_outcome.is_file():
                    raise base.ExperimentError("later arm call exists before prior arm completion")
        for index, request in enumerate(requests):
            purpose = request["purpose"]
            if unresolved_ordinal is not None and int(purpose["pair_ordinal"]) > unresolved_ordinal:
                raise base.ExperimentError("future pair call exists before prior resolution")
            result = self._validate_call_result(
                base._read_json(self.run_dir / "calls" / str(request["call_id"]) / "result.json"),
                request,
            )
            pair = self._pair(str(purpose["pair_id"]))
            logical = self._logical_outcome(str(purpose["outcome_id"]))
            turn_path = (
                self._arm_root(
                    str(pair["pair_id"]),
                    int(purpose["pair_attempt"]),
                    str(logical["arm"]),
                )
                / f"turn-{int(purpose['turn']):02d}.json"
            )
            arm_outcome_path = (
                self._arm_root(
                    str(pair["pair_id"]),
                    int(purpose["pair_attempt"]),
                    str(logical["arm"]),
                )
                / "outcome.json"
            )
            if int(purpose["turn"]) > 1:
                prior_turn_path = turn_path.parent / f"turn-{int(purpose['turn']) - 1:02d}.json"
                if prior_turn_path.is_file():
                    prior_turn = base._read_json(prior_turn_path)
                    if prior_turn.get("terminated") is True or prior_turn.get("truncated") is True:
                        raise base.ExperimentError("call exists after a terminal turn")
            summary = self._load_attempt_summary(pair, int(purpose["pair_attempt"]), cohort)
            resolution_path = self.run_dir / "pair-resolutions" / f"{pair['pair_id']}.json"
            recoverable_latest = (
                index == len(requests) - 1
                and unresolved_ordinal == int(purpose["pair_ordinal"])
                and summary is None
                and not resolution_path.is_file()
            )
            terminal_consumed_response = (
                summary is not None
                and summary["status"] == "fatal_failure"
                and summary["failure_category"] == "fatal_environment"
                and summary["failed_call_id"] == request["call_id"]
            )
            if arm_outcome_path.is_file() and not turn_path.is_file():
                raise base.ExperimentError("call exists after its arm outcome completed")
            if (
                result["status"] == "success"
                and not turn_path.is_file()
                and not recoverable_latest
                and not terminal_consumed_response
            ):
                raise base.ExperimentError("successful call is not linked to a turn leaf")
            if result["status"] == "error":
                if summary is None and not recoverable_latest:
                    raise base.ExperimentError("failed call is not bound by an attempt summary")
                if summary is not None and summary["failed_call_id"] != request["call_id"]:
                    raise base.ExperimentError(
                        "failed call is not bound by its pair attempt summary"
                    )
                if summary is not None and summary["call_ids"][-1] != request["call_id"]:
                    raise base.ExperimentError("calls exist after the summarized failure")

    def _validate_full_tree(self) -> None:
        """Reject every unmodeled run artifact before calls or Assay."""
        self._validate_run_tree()
        allowed_top = {
            ".writer.lock",
            "plan.json",
            "run-manifest.json",
            "cohort-lock.json",
            IMPORT_ROOT,
            "qualification-revalidation",
            "horizon-viability",
            "calls",
            "pair-attempts",
            "pair-resolutions",
            "pair-resolution-manifest.json",
            "outcomes",
            "ledger-root.json",
            "assay-request.json",
            "assay-result.json",
            "assay",
        }
        for entry in self.run_dir.iterdir():
            if entry.name not in allowed_top:
                raise base.ExperimentError(f"unexpected top-level replay artifact: {entry}")
        if base._read_json(self.run_dir / "plan.json") != self.plan:
            raise base.ExperimentError("persisted replay plan differs from the in-memory seal")
        if base._read_json(self.run_dir / "run-manifest.json") != self._run_manifest_value():
            raise base.ExperimentError("persisted replay run manifest differs from the seal")
        self._load_cohort()
        cluster_files = {
            f"{cluster['cluster_id']}.json" for cluster in self.plan["cluster_schedule"]
        }
        for root_name in ("qualification-revalidation", "horizon-viability"):
            root = self.run_dir / root_name
            if not root.is_dir() or {path.name for path in root.iterdir()} != cluster_files:
                raise base.ExperimentError(f"{root_name} inventory differs from sealed clusters")
            if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
                raise base.ExperimentError(f"{root_name} contains an unsafe leaf")
        for record in self.plan["qualification_revalidations"]:
            if (
                base._read_json(
                    self.run_dir / "qualification-revalidation" / f"{record['cluster_id']}.json"
                )
                != record
            ):
                raise base.ExperimentError("qualification revalidation leaf differs from seal")
        for record in self.plan["horizon_viability"]:
            if (
                base._read_json(self.run_dir / "horizon-viability" / f"{record['cluster_id']}.json")
                != record
            ):
                raise base.ExperimentError("actor-horizon viability leaf differs from seal")
        self._validate_import_manifest_bytes(self.run_dir / IMPORT_ROOT)
        assay_root = self.run_dir / "assay"
        if assay_root.exists():
            if assay_root.is_symlink() or not assay_root.is_dir():
                raise base.ExperimentError("Assay output root is unsafe")
            for path in assay_root.rglob("*"):
                if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                    raise base.ExperimentError(f"unsafe Assay artifact: {path}")

    def _ledger_root(self) -> dict[str, Any]:
        roots = (
            "plan.json",
            "run-manifest.json",
            "cohort-lock.json",
            IMPORT_ROOT,
            "qualification-revalidation",
            "horizon-viability",
            "calls",
            "pair-attempts",
            "pair-resolutions",
            "pair-resolution-manifest.json",
            "outcomes",
        )
        leaves: list[dict[str, Any]] = []
        for name in roots:
            root = self.run_dir / name
            paths = (
                [root]
                if root.is_file()
                else sorted(path for path in root.rglob("*") if path.is_file())
                if root.is_dir()
                else []
            )
            for path in paths:
                if path.is_symlink() or not path.is_file():
                    raise base.ExperimentError(f"unsafe outcome replay ledger leaf: {path}")
                content = path.read_bytes()
                leaves.append(
                    {
                        "path": path.relative_to(self.run_dir).as_posix(),
                        "digest": base._bytes_digest(content),
                        "size_bytes": len(content),
                    }
                )
        leaves.sort(key=lambda item: item["path"])
        body = {
            "schema_version": base.LEDGER_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "leaf_count": len(leaves),
            "leaves": leaves,
        }
        value = {**body, "ledger_root_digest": base._digest(body)}
        base._write_json(self.run_dir / "ledger-root.json", value)
        return value

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
        resolution_manifest = base._read_json(self.run_dir / "pair-resolution-manifest.json")
        self._validate_resolution_manifest(resolution_manifest, cohort)
        horizons = {
            str(item["cluster_id"]): int(item["horizon"]) for item in self.plan["cluster_horizons"]
        }
        qualification = {
            str(item["cluster_id"]): str(item["record_digest"])
            for item in self.plan["qualification_revalidations"]
        }
        viability = {
            str(item["cluster_id"]): str(item["record_digest"])
            for item in self.plan["horizon_viability"]
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
                    "solution": base._canonical_json(solutions),
                    "max_turns": horizons[cluster_id],
                    "seed": min(self.plan["evaluation_seeds"]),
                    "metadata": {
                        "plan_digest": self.plan["plan_digest"],
                        "cohort_digest": cohort["cohort_digest"],
                        "experiment_ledger_root_digest": ledger["ledger_root_digest"],
                        "pair_resolution_manifest_digest": resolution_manifest["manifest_digest"],
                        "source_plan_digest": self.plan["source_evidence"]["source_plan_digest"],
                        "source_cohort_digest": self.plan["source_evidence"][
                            "source_cohort_digest"
                        ],
                        "source_import_manifest_digest": self.plan["source_evidence"][
                            "import_manifest_digest"
                        ],
                        "qualification_revalidation_digest": qualification[cluster_id],
                        "actor_horizon_viability_digest": viability[cluster_id],
                        "actor_horizon": horizons[cluster_id],
                        "solution_digests_by_seed": {
                            str(seed): selection["probes"][str(seed)]["solution_digest"]
                            for seed in self.plan["evaluation_seeds"]
                        },
                        "selected_candidate_id": selection["candidate_id"],
                        "proofpack_environment_digest": selection["environment_digest"],
                        "source_proofpack_qualification_digest": selection["qualification_digest"],
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

    @staticmethod
    def _reject_physical_model_lock(inventory: Sequence[Mapping[str, Any]]) -> None:
        if any(Path(str(item.get("path", ""))).name == "model.lock" for item in inventory):
            raise base.ExperimentIncomplete("Assay physically emitted a forbidden model.lock")

    def _validate_assay_result(
        self, result: Mapping[str, Any], request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        validated = super()._validate_assay_result(result, request)
        self._reject_physical_model_lock(result["assay_file_inventory"])
        return validated

    def write_assay(
        self,
        cohort: Mapping[str, Any],
        outcomes: Sequence[Mapping[str, Any]],
        resolution_manifest: Mapping[str, Any],
    ) -> Path:
        if len(outcomes) != 108:
            raise base.ExperimentIncomplete("Assay requires exactly 108 selected outcomes")
        self._validate_resolution_manifest(resolution_manifest, cohort)
        self._validate_full_tree()
        self._validate_progress(cohort)
        ledger = self._ledger_root()
        tasks_plain, clusters_plain = self._plain_assay_inputs(cohort, outcomes, ledger)
        request = {
            "schema_version": ASSAY_REQUEST_SCHEMA,
            "plan_digest": self.plan["plan_digest"],
            "cohort_digest": cohort["cohort_digest"],
            "source_plan_digest": self.plan["source_evidence"]["source_plan_digest"],
            "source_cohort_digest": self.plan["source_evidence"]["source_cohort_digest"],
            "source_import_manifest_digest": self.plan["source_evidence"]["import_manifest_digest"],
            "pair_resolution_manifest_digest": resolution_manifest["manifest_digest"],
            "ledger_root_digest": ledger["ledger_root_digest"],
            "tasks_digest": base._digest(tasks_plain),
            "clusters_digest": base._digest(clusters_plain),
            "task_count": 18,
            "cluster_count": 18,
            "pair_count": 54,
            "selected_outcome_count": 108,
        }
        request_path = self.run_dir / "assay-request.json"
        result_path = self.run_dir / "assay-result.json"
        existed = request_path.is_file()
        if result_path.exists() and not existed:
            raise base.ExperimentError("Assay result exists without a prior request")
        if (self.run_dir / "assay").exists() and not existed:
            raise base.ExperimentError("Assay output exists without a prior request")
        base._write_json(request_path, request)
        if result_path.is_file():
            self._validate_assay_result(base._read_json(result_path), request)
            return result_path
        if existed:
            raise base.ExperimentIncomplete(
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
                raise base.ExperimentIncomplete(
                    "Assay must explicitly refuse release authorization"
                )
            if not hasattr(artifact, "model_lock_path") or artifact.model_lock_path is not None:
                raise base.ExperimentIncomplete("Assay unexpectedly emitted a model.lock")
            evaluation_relative = self._assay_relative_path(
                getattr(artifact, "evaluation_path", None), "Assay evaluation_path"
            )
            certification_relative = self._assay_relative_path(
                getattr(artifact, "certification_path", None), "Assay certification_path"
            )
            assay_inventory = self._assay_file_inventory()
            self._reject_physical_model_lock(assay_inventory)
            value = {
                "schema_version": base.ASSAY_RESULT_SCHEMA,
                "plan_digest": self.plan["plan_digest"],
                "status": "complete",
                "assay_request_digest": base._digest(request),
                "cohort_digest": request["cohort_digest"],
                "ledger_root_digest": request["ledger_root_digest"],
                "bundle_digest": artifact.bundle_digest,
                "report": report,
                "release_authorized": False,
                "model_lock_path": None,
                "assay_file_inventory": assay_inventory,
                "evaluation_relative_path": evaluation_relative,
                "certification_relative_path": certification_relative,
            }
        except Exception as exc:
            value = {
                "schema_version": base.ASSAY_RESULT_SCHEMA,
                "plan_digest": self.plan["plan_digest"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            base._write_json(result_path, value)
            raise base.ExperimentIncomplete(f"aggregate Assay write failed: {exc}") from exc
        self._validate_assay_result(value, request)
        base._write_json(result_path, value)
        return result_path

    async def execute(self) -> base.RunResult:
        self.initialize()
        cohort = self._load_cohort()
        self._validate_full_tree()
        self._validate_progress(cohort)
        outcomes, resolution_manifest = await self.run_outcomes(cohort)
        assay_result_path = self.write_assay(cohort, outcomes, resolution_manifest)
        self._validate_full_tree()
        return base.RunResult(
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
    dependencies: base.RunnerDependencies | None = None,
) -> base.RunResult:
    """Validate by default; spend only after an exact cap acknowledgement."""
    plan = load_plan(plan_path)
    destination = derive_run_dir(output_root, plan)
    source_run = Path(plan["source_evidence"]["source_run_dir"])
    if (
        destination == source_run
        or destination.is_relative_to(source_run)
        or source_run.is_relative_to(destination)
    ):
        raise base.ExperimentError("v5 destination must not overlap the historical source run")
    if not execute:
        return base.RunResult(
            status="validated",
            plan_digest=plan["plan_digest"],
            run_dir=destination,
            call_count=0,
        )
    cap = int(plan["configuration"]["total_call_cap"])
    if acknowledged_call_cap != cap:
        raise base.ExperimentError(
            "--acknowledge-call-cap must exactly equal the sealed total_call_cap "
            f"({cap}) before --execute can spend calls"
        )
    if dependencies is None:
        current_sources = base._source_revisions()
        current_runtime = base._runtime_identity()
        resolved_dependencies = base._default_dependencies(
            plan,
            source_revisions=current_sources,
            runtime_identity=current_runtime,
        )
    else:
        resolved_dependencies = dependencies
    engine = _ReplayEngine(plan, base._pretty_json(plan), destination, resolved_dependencies)
    with base._single_writer(destination):
        return await engine.execute()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective outcome-only replay of a sealed v4 SPADE AGY cohort"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "outcome-replay-plan", help="validate/import v4 pre-outcome evidence and seal v5"
    )
    plan.add_argument("--source-run", required=True)
    plan.add_argument("--expected-source-plan-digest", required=True)
    plan.add_argument("--expected-source-cohort-digest", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--experiment-id", required=True)
    plan.add_argument("--total-call-cap", type=int, default=None)
    run = subparsers.add_parser("run", help="validate or explicitly execute a v5 plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--acknowledge-call-cap", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "outcome-replay-plan":
            plan = build_outcome_replay_plan(
                experiment_id=args.experiment_id,
                source_run_dir=args.source_run,
                run_output_root=args.output_root,
                expected_source_plan_digest=args.expected_source_plan_digest,
                expected_source_cohort_digest=args.expected_source_cohort_digest,
                total_call_cap=args.total_call_cap,
            )
            target = write_plan(args.output, plan)
            print(
                f"sealed outcome replay plan: {target} (54 pairs, 108 outcomes, "
                f"ceiling {plan['configuration']['computed_call_ceiling']}, "
                f"digest {plan['plan_digest']})"
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
    except base.ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
