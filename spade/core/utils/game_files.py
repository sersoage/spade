"""Generated-game filesystem operations."""

import logging
from collections.abc import Sequence
from pathlib import Path

from spade.core.envs.synthetic_game_env import make_synthetic_env
from spade.core.proofpack_bridge import validate_game_with_proofpack

logger = logging.getLogger(__name__)


def validate_game_with_reason(
    game_file: Path,
    *,
    validate_runtime: bool = True,
    proofpack_enabled: bool = False,
    action_format: str = "boxed",
    proofpack_seeds: Sequence[int] | None = None,
    proofpack_timeout_seconds: float = 5.0,
    max_turns: int = 20,
) -> tuple[bool, str]:
    """Validate a generated game and retain the rejection reason.

    Native SPADE validation remains the default so the Python 3.10+ package has
    no mandatory dependency on ProofPack. If ``proofpack_enabled`` is true, the
    ProofPack gate replaces the native generation-time smoke check when enabled:
    its isolated V0--V4 replays are both stricter and protected by hard process
    deadlines. This avoids executing untrusted candidate source in the generator
    merely to validate it. Later training rollouts still use SPADE's native
    runtime and remain a separate trusted-code boundary.
    """
    env = None
    try:
        if not validate_runtime and not proofpack_enabled:
            return True, "Generated-game validation disabled"
        game_code = Path(game_file).read_text(encoding="utf-8")
        if proofpack_enabled:
            is_valid, reason = validate_game_with_proofpack(
                game_code,
                action_format=action_format,
                seeds=proofpack_seeds,
                timeout_seconds=proofpack_timeout_seconds,
                max_turns=max_turns,
            )
            if not is_valid:
                logger.warning("ProofPack validation failed for %s: %s", game_file, reason)
                return False, reason

            # Do not execute the same generated source again in-process. Besides
            # weakening the isolation boundary, a candidate that special-cases
            # public qualification probes could hang this unbounded native smoke
            # check after passing the bounded ProofPack ladder.
            return True, "ProofPack qualification passed"

        if not validate_runtime:
            return True, "Generated-game validation disabled"

        # Additional native interface/runtime check. ``respect_game_max_turns``
        # keeps validation and gameplay on the same multi-turn pacing contract.
        env = make_synthetic_env(
            str(game_file),
            max_turns=max_turns,
            respect_game_max_turns=True,
        )
        reset_seed = list(proofpack_seeds)[0] if proofpack_seeds else 0
        env.reset(seed=reset_seed)
        if action_format == "tool_call":
            actions = [
                '<tool_call>{"name":"__spade_validation_probe__","arguments":{}}</tool_call>',
                "<answer>__spade_validation_probe__</answer>",
            ]
        else:
            actions = [
                "\\boxed{__spade_validation_probe_1__}",
                "\\boxed{__spade_validation_probe_2__}",
                "\\boxed{__spade_validation_probe_3__}",
            ]
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return True, "Native SPADE runtime validation passed"
    except Exception as exc:
        logger.warning("Game validation failed for %s: %s", game_file, exc)
        return False, f"Native SPADE validation error: {type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.debug("Failed to close validation environment for %s", game_file)


def validate_game(
    game_file: Path,
    *,
    validate_runtime: bool = True,
    proofpack_enabled: bool = False,
    action_format: str = "boxed",
    proofpack_seeds: Sequence[int] | None = None,
    proofpack_timeout_seconds: float = 5.0,
    max_turns: int = 20,
) -> bool:
    """Return only the verdict from :func:`validate_game_with_reason`."""
    passed, _ = validate_game_with_reason(
        game_file,
        validate_runtime=validate_runtime,
        proofpack_enabled=proofpack_enabled,
        action_format=action_format,
        proofpack_seeds=proofpack_seeds,
        proofpack_timeout_seconds=proofpack_timeout_seconds,
        max_turns=max_turns,
    )
    return passed


def save_game_file(
    game_code: str,
    games_dir: Path,
    rollout_id: int,
    index: int,
    skill: str,
) -> Path:
    slug = skill.lower().replace(" ", "_")
    game_file = games_dir / f"game_{rollout_id:05d}_{index:03d}_{slug}.py"
    game_file.write_text('"""SPADE self-play generated game"""\n\n' + game_code)
    return game_file


def save_rejected_game(
    game_code: str,
    games_dir: Path,
    rollout_id: int,
    index: int,
    skill: str,
    reject_stage: str,
    reasoning: str,
) -> Path | None:
    try:
        rejects_dir = Path(games_dir).parent / "spade_games_rejected"
        rejects_dir.mkdir(parents=True, exist_ok=True)
        slug = skill.lower().replace(" ", "_")
        stem = f"reject_{rollout_id:05d}_{index:03d}_{slug}_{reject_stage}"
        game_file = rejects_dir / f"{stem}.py"
        game_file.write_text(
            f'"""SPADE self-play REJECTED game (stage={reject_stage})"""\n\n{game_code}'
        )
        (rejects_dir / f"{stem}.reason.txt").write_text(reasoning or "")
        return game_file
    except Exception:
        return None


def cleanup_old_games(game_files: list[Path]) -> None:
    for game_file in game_files:
        try:
            Path(game_file).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("[COLLECT] Failed to delete %s: %s", game_file, exc)
    logger.info("[COLLECT] Deleted %d old games", len(game_files))
