"""Public utility exports grouped behind lazy module boundaries."""

from spade._lazy import lazy_exports

_EXPORTS = {
    "LanguageGameReward": ("spade.core.utils.parsing", "LanguageGameReward"),
    "assign_env_rewards": ("spade.core.utils.env_rewards", "assign_env_rewards"),
    "recompute_delayed_env_rewards": (
        "spade.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards",
    ),
    "recompute_delayed_env_rewards_micro_lp": (
        "spade.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards_micro_lp",
    ),
    "recompute_delayed_env_rewards_regret": (
        "spade.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards_regret",
    ),
    "recompute_delayed_env_rewards_blend": (
        "spade.core.utils.delayed_env_rewards",
        "recompute_delayed_env_rewards_blend",
    ),
    "assign_env_rewards_regret": (
        "spade.core.utils.env_rewards",
        "assign_env_rewards_regret",
    ),
    "assign_trajectory_weights": (
        "spade.core.utils.trajectory_build",
        "assign_trajectory_weights",
    ),
    "build_actor_trajectory": (
        "spade.core.utils.trajectory_build",
        "build_actor_trajectory",
    ),
    "build_env_trajectory": (
        "spade.core.utils.trajectory_build",
        "build_env_trajectory",
    ),
    "cleanup_old_games": ("spade.core.utils.game_files", "cleanup_old_games"),
    "compute_env_reward_scale": (
        "spade.core.utils.rewards",
        "compute_env_reward_scale",
    ),
    "compute_format_reward": ("spade.core.utils.rewards", "compute_format_reward"),
    "compute_returns": ("spade.core.utils.rewards", "compute_returns"),
    "episode_reward": ("spade.core.utils.rewards", "episode_reward"),
    "extract_boxed_answer": ("spade.core.utils.parsing", "extract_boxed_answer"),
    "extract_game_code": ("spade.core.utils.parsing", "extract_game_code"),
    "format_error_response": ("spade.core.utils.parsing", "format_error_response"),
    "get_token_delta": ("spade.core.utils.token_utils", "get_token_delta"),
    "normalize_rewards_per_game": (
        "spade.core.utils.trajectory_build",
        "normalize_rewards_per_game",
    ),
    "parse_action": ("spade.core.utils.parsing", "parse_action"),
    "save_game_file": ("spade.core.utils.game_files", "save_game_file"),
    "save_rejected_game": ("spade.core.utils.game_files", "save_rejected_game"),
    "upsample_trajectories": (
        "spade.core.utils.trajectory_build",
        "upsample_trajectories",
    ),
    "validate_boxed_format": ("spade.core.utils.parsing", "validate_boxed_format"),
    "validate_game": ("spade.core.utils.game_files", "validate_game"),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
