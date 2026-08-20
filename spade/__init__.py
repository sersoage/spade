"""SPADE: Self-Play in Adaptive Synthetic Executable Environments (package name retains the historical 'spade' module name)."""

from spade.__about__ import __version__
from spade._lazy import lazy_exports

_EXPORTS = {
    "LearningPotential": ("spade.core.learning_potential", "LearningPotential"),
    "SyntheticGameEnv": ("spade.core.envs.synthetic_game_env", "SyntheticGameEnv"),
    "SyntheticGameGenerator": ("spade.core.game_generator", "SyntheticGameGenerator"),
    "core": ("spade.core", None),
    "make_synthetic_env": ("spade.core.envs.synthetic_game_env", "make_synthetic_env"),
    "tinker": ("spade.tinker", None),
}

__all__ = ["__version__", *_EXPORTS]
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
