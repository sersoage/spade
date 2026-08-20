"""Slime backend integration for SPADE.

Backend dependencies are loaded only when an exported object is accessed. This
keeps lightweight utilities such as the trajectory converter importable without
a complete Slime installation.
"""

from spade._lazy import lazy_exports

_EXPORTS = {
    "SlimeModelAdapter": ("spade.slime.model_adapter", "SlimeModelAdapter"),
    "trajectories_to_grouped_samples": (
        "spade.slime.trajectory_converter",
        "trajectories_to_grouped_samples",
    ),
    "trajectories_to_samples": (
        "spade.slime.trajectory_converter",
        "trajectories_to_samples",
    ),
    "trajectory_to_slime_sample": (
        "spade.slime.trajectory_converter",
        "trajectory_to_slime_sample",
    ),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
