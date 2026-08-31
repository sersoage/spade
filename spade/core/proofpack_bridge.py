"""ProofPack qualification bridge for SPADE.

Enables SPADE to use ProofPack's V0-V4 qualification ladder (Oracle solvability,
No-Agent non-triviality, deterministic sandbox execution, and mutant testing)
during Environment Designer generation rejection sampling.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Attempt import directly, or via sibling repository discovery
try:
    from proofpack_env.spade_qualification import qualify_spade_environment
    HAS_PROOFPACK = True
except ImportError:
    sibling_path = Path(__file__).resolve().parents[3] / "proofpack" / "packages" / "env" / "src"
    if sibling_path.exists() and str(sibling_path) not in sys.path:
        sys.path.insert(0, str(sibling_path))
        try:
            from proofpack_env.spade_qualification import qualify_spade_environment
            HAS_PROOFPACK = True
        except ImportError:
            HAS_PROOFPACK = False
    else:
        HAS_PROOFPACK = False


def validate_game_with_proofpack(
    game_code: str,
    action_format: str = "boxed",
    seeds: list[int] | None = None,
) -> Tuple[bool, str]:
    """Validate a SPADE environment using ProofPack qualification ladder.

    Fail-closed: Returns (False, reason) if ProofPack is unavailable or rejects the code.

    Returns:
        (passed, reason): True if environment passed all mandatory V0-V4 clauses.
    """
    if not HAS_PROOFPACK:
        logger.error("proofpack_env is not available; failing closed on environment qualification")
        return False, "ProofPack qualification is required but proofpack_env is not available"

    try:
        report = qualify_spade_environment(
            game_code=game_code,
            action_format="boxed" if action_format == "boxed" else "tool_call",
            seeds=seeds or [0, 1],
        )

        if report.passed:
            return True, f"Passed ProofPack V0-V4 qualification ({report.environment_name})"

        failed_clauses = [
            f"{k}: {v.summary}"
            for k, v in report.clauses.items()
            if v.status in ("fail", "error")
        ]
        reason = "ProofPack qualification failed: " + "; ".join(failed_clauses)
        logger.info(f"SPADE environment rejected by ProofPack: {reason}")
        return False, reason
    except Exception as exc:
        logger.error(f"ProofPack qualification encountered unexpected error: {exc}")
        return False, f"ProofPack qualification error: {exc}"
