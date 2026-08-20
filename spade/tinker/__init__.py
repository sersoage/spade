"""Tinker backend integration for SPADE."""

from spade._lazy import lazy_exports

_EXPORTS = {
    "TinkerModelAdapter": ("spade.tinker.model_adapter", "TinkerModelAdapter"),
    "get_game_policy": ("spade.tinker.rollout", "get_game_policy"),
    "get_learning_potentials": ("spade.tinker.rollout", "get_learning_potentials"),
    "spade_generate_rollout": ("spade.tinker.rollout", "spade_generate_rollout"),
    "spade_trajectory_to_tinker_trajectory": (
        "spade.tinker.trajectory_converter",
        "spade_trajectory_to_tinker_trajectory",
    ),
    "train_step": ("spade.tinker.train_step", "train_step"),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
