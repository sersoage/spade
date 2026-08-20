"""Framework-independent SPADE components."""

from spade._lazy import lazy_exports

_EXPORTS = {
    "EMA": ("spade.core.learning_potential", "EMA"),
    "ExploitabilityBasedPotential": (
        "spade.core.learning_potential",
        "ExploitabilityBasedPotential",
    ),
    "GameBaselineTracker": ("spade.core.learning_potential", "GameBaselineTracker"),
    "LearningPotential": ("spade.core.learning_potential", "LearningPotential"),
    "ModelAdapter": ("spade.core.model_adapter", "ModelAdapter"),
    "MultiAgentLearningPotential": (
        "spade.core.learning_potential",
        "MultiAgentLearningPotential",
    ),
    "SpadeConfig": ("spade.core.types", "SpadeConfig"),
    "SpadeOrchestrator": ("spade.core.orchestrator", "SpadeOrchestrator"),
    "SyntheticGameEnv": ("spade.core.envs.synthetic_game_env", "SyntheticGameEnv"),
    "SyntheticGameGenerator": ("spade.core.game_generator", "SyntheticGameGenerator"),
    "calculate_game_progress": (
        "spade.core.learning_potential",
        "calculate_game_progress",
    ),
    "make_synthetic_env": ("spade.core.envs.synthetic_game_env", "make_synthetic_env"),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
