"""Optional, fail-closed ProofPack qualification for generated SPADE games.

SPADE supports Python 3.10+, while ``proofpack-env`` requires Python 3.12+.
Consequently this module deliberately does not import ProofPack at module load
time or discover a sibling checkout by filesystem layout. Callers opt in to
qualification and the bridge then requires an importable, compatible
``proofpack_env`` installation. A requested qualification can never silently
degrade to native SPADE validation.

The isolation boundary covers qualification executions only. A positive
ProofPack receipt replaces SPADE's native generation-time smoke check, but later
training/playback still loads accepted source in the trainer process. This
bridge is a generation gate, not a sandbox for the training runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_ACTION_FORMATS = frozenset({"boxed", "tool_call"})
QUALIFICATION_SCHEMA = "proofpack-spade-qualification/v2"
EXPECTED_CLAUSES = frozenset(
    {
        "v0_syntax",
        "v1_sandbox_smoke",
        "v2_oracle_solvable",
        "v3_no_agent_unwinnable",
        "v4_mutation_robustness",
    }
)
EXPECTED_EXECUTION_BOUNDARY = "macos-sandbox-exec-worker/v1"

# Backwards-compatible test/embedding override. ``False`` forces the bridge to
# behave as unavailable; ``None`` and ``True`` both perform the real lazy import.
# Unlike the previous implementation, this value is not populated by probing a
# machine-specific sibling directory during module import.
HAS_PROOFPACK: bool | None = None


def _load_qualifier() -> tuple[Callable[..., Any] | None, str | None]:
    """Load ProofPack without coupling SPADE to a checkout location."""
    if HAS_PROOFPACK is False:
        return None, "ProofPack availability was explicitly disabled"

    try:
        module = importlib.import_module("proofpack_env.spade_qualification")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    qualifier = getattr(module, "qualify_spade_environment", None)
    if not callable(qualifier):
        return None, "proofpack_env.spade_qualification has no callable qualifier"

    # Passing the rollout horizon is required for sound multi-turn
    # qualification. Refuse an older ProofPack instead of certifying a game
    # under a different horizon than SPADE will use for training.
    try:
        parameters = inspect.signature(qualifier).parameters
    except (TypeError, ValueError) as exc:
        return None, f"could not inspect ProofPack qualifier: {exc}"
    required_parameters = {
        "game_code",
        "action_format",
        "seeds",
        "timeout_seconds",
        "max_turns",
    }
    missing = sorted(required_parameters.difference(parameters))
    if missing:
        return None, (
            "incompatible proofpack_env qualifier; missing parameter(s): "
            + ", ".join(missing)
        )
    return qualifier, None


def proofpack_available() -> tuple[bool, str]:
    """Return whether the required ProofPack SPADE interface is importable."""
    qualifier, error = _load_qualifier()
    if qualifier is not None:
        return True, "ProofPack SPADE qualification interface is available"
    return False, error or "ProofPack SPADE qualification interface is unavailable"


def _positive_report_error(
    report: Any,
    *,
    game_code: str,
    action_format: str,
    seeds: list[int],
    timeout_seconds: float,
    max_turns: int,
) -> str | None:
    """Validate the complete versioned receipt before trusting a positive bit."""
    if getattr(report, "schema_version", None) != QUALIFICATION_SCHEMA:
        return f"expected receipt schema {QUALIFICATION_SCHEMA!r}"
    expected_digest = "sha256:" + hashlib.sha256(
        game_code.encode("utf-8", errors="replace")
    ).hexdigest()
    if getattr(report, "environment_digest", None) != expected_digest:
        return "receipt environment digest does not match the candidate source"
    environment_name = getattr(report, "environment_name", None)
    if not isinstance(environment_name, str) or not environment_name.strip():
        return "receipt lacks a non-empty environment name"

    clauses = getattr(report, "clauses", None)
    if not isinstance(clauses, dict) or set(clauses) != EXPECTED_CLAUSES:
        return "receipt does not contain exactly the required V0-V4 clauses"
    for clause_id in sorted(EXPECTED_CLAUSES):
        clause = clauses[clause_id]
        if getattr(clause, "clause_id", None) != clause_id:
            return f"receipt clause {clause_id!r} has a mismatched identifier"
        if getattr(clause, "status", None) != "pass":
            return f"receipt clause {clause_id!r} is not marked pass"

    metadata = getattr(report, "metadata", None)
    if not isinstance(metadata, dict):
        return "receipt lacks qualification metadata"
    expected_metadata = {
        "action_format": action_format,
        "seeds": seeds,
        "max_turns": max_turns,
        "timeout_seconds": timeout_seconds,
        "execution_boundary": EXPECTED_EXECUTION_BOUNDARY,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            return f"receipt metadata {key!r} does not match the requested qualification"
    return None


def validate_positive_proofpack_receipt(
    report: Any,
    *,
    game_code: str,
    action_format: str,
    seeds: Sequence[int],
    timeout_seconds: float,
    max_turns: int,
) -> tuple[bool, str]:
    """Validate a claimed positive ProofPack receipt without running its qualifier.

    This is the reusable fail-closed boundary for persisted or externally supplied
    receipts. It binds the positive bit to the exact candidate source and requested
    qualification configuration. The seed sequence is order-sensitive.
    """
    if getattr(report, "passed", None) is not True:
        return False, "receipt is not marked passed"
    if not isinstance(game_code, str) or not game_code.strip():
        return False, "candidate source must be non-empty text"
    if not isinstance(action_format, str) or action_format not in SUPPORTED_ACTION_FORMATS:
        return False, "requested action_format is unsupported"
    try:
        normalized_seeds = list(seeds)
    except TypeError:
        return False, "requested seeds must be a non-empty sequence of integers"
    if not normalized_seeds or any(type(seed) is not int for seed in normalized_seeds):
        return False, "requested seeds must be a non-empty sequence of integers"
    try:
        normalized_timeout = float(timeout_seconds)
    except (TypeError, ValueError, OverflowError):
        normalized_timeout = math.nan
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(normalized_timeout)
        or normalized_timeout <= 0
    ):
        return False, "requested timeout_seconds must be a positive finite number"
    if type(max_turns) is not int or max_turns <= 0:
        return False, "requested max_turns must be positive"

    metadata = getattr(report, "metadata", None)
    if not isinstance(metadata, dict):
        return False, "receipt lacks qualification metadata"
    receipt_action_format = metadata.get("action_format")
    if not isinstance(receipt_action_format, str):
        return False, "receipt metadata 'action_format' must be text"
    receipt_seeds = metadata.get("seeds")
    if not isinstance(receipt_seeds, list) or any(
        type(seed) is not int for seed in receipt_seeds
    ):
        return False, "receipt metadata 'seeds' must be an ordered integer list"
    if type(metadata.get("max_turns")) is not int:
        return False, "receipt metadata 'max_turns' must be an integer"
    receipt_timeout = metadata.get("timeout_seconds")
    if isinstance(receipt_timeout, bool) or not isinstance(receipt_timeout, (int, float)):
        return False, "receipt metadata 'timeout_seconds' must be numeric"
    try:
        normalized_receipt_timeout = float(receipt_timeout)
    except (TypeError, ValueError, OverflowError):
        normalized_receipt_timeout = math.nan
    if not math.isfinite(normalized_receipt_timeout):
        return False, "receipt metadata 'timeout_seconds' must be finite"
    receipt_boundary = metadata.get("execution_boundary")
    if not isinstance(receipt_boundary, str):
        return False, "receipt metadata 'execution_boundary' must be text"

    report_error = _positive_report_error(
        report,
        game_code=game_code,
        action_format=action_format,
        seeds=normalized_seeds,
        timeout_seconds=normalized_timeout,
        max_turns=max_turns,
    )
    if report_error is not None:
        return False, report_error
    environment_name = getattr(report, "environment_name")
    return True, f"Valid ProofPack V0-V4 positive receipt ({environment_name})"


def validate_game_with_proofpack(
    game_code: str,
    action_format: str = "boxed",
    seeds: Sequence[int] | None = None,
    timeout_seconds: float = 5.0,
    max_turns: int = 20,
) -> tuple[bool, str]:
    """Run ProofPack's V0--V4 ladder for one generated environment.

    The function is always fail-closed: unavailable or incompatible ProofPack,
    invalid configuration, qualifier errors, and negative reports all return a
    false verdict. Whether to call this function is controlled by explicit
    SPADE configuration.
    """
    if not isinstance(game_code, str) or not game_code.strip():
        return False, "ProofPack qualification requires non-empty game code"
    if action_format not in SUPPORTED_ACTION_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_ACTION_FORMATS))
        return False, (
            f"ProofPack qualification does not support action_format={action_format!r}; "
            f"supported formats: {supported}"
        )
    try:
        normalized_timeout = float(timeout_seconds)
    except (TypeError, ValueError, OverflowError):
        normalized_timeout = math.nan
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(normalized_timeout)
        or normalized_timeout <= 0
    ):
        return False, "ProofPack timeout_seconds must be a positive finite number"
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
        return False, "ProofPack max_turns must be positive"

    try:
        qualification_seeds = [0, 1, 42] if seeds is None else list(seeds)
    except TypeError:
        return False, "ProofPack qualification seeds must be a non-empty sequence of integers"
    if not qualification_seeds:
        return False, "ProofPack qualification requires at least one seed"
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in qualification_seeds):
        return False, "ProofPack qualification seeds must all be integers"

    qualifier, import_error = _load_qualifier()
    if qualifier is None:
        reason = (
            "ProofPack qualification was enabled but a compatible proofpack_env "
            "installation is not available. Install/run ProofPack in a Python >=3.12 "
            f"environment. Detail: {import_error or 'unknown import error'}"
        )
        logger.error(reason)
        return False, reason

    try:
        report = qualifier(
            game_code=game_code,
            action_format=action_format,
            seeds=qualification_seeds,
            timeout_seconds=normalized_timeout,
            max_turns=max_turns,
        )
    except Exception as exc:  # ProofPack is an optional process boundary.
        reason = f"ProofPack qualification error: {type(exc).__name__}: {exc}"
        logger.error(reason)
        return False, reason

    if getattr(report, "passed", False) is True:
        report_error = _positive_report_error(
            report,
            game_code=game_code,
            action_format=action_format,
            seeds=qualification_seeds,
            timeout_seconds=normalized_timeout,
            max_turns=max_turns,
        )
        if report_error is not None:
            reason = f"ProofPack qualification returned an invalid positive receipt: {report_error}"
            logger.error(reason)
            return False, reason
        environment_name = getattr(report, "environment_name", "unknown") or "unknown"
        return True, f"Passed ProofPack V0-V4 qualification ({environment_name})"

    clauses = getattr(report, "clauses", {})
    failed_clauses = []
    if isinstance(clauses, dict):
        for clause_id, clause in clauses.items():
            if getattr(clause, "status", None) in {"fail", "error"}:
                failed_clauses.append(
                    f"{clause_id}: {getattr(clause, 'summary', 'no summary')}"
                )
    details = "; ".join(failed_clauses) or "qualifier returned a negative report"
    reason = "ProofPack qualification failed: " + details
    logger.info("SPADE environment rejected by ProofPack: %s", reason)
    return False, reason
