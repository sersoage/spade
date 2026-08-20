"""Evaluation modules for SPADE training.

Evaluators have independent optional dependencies, so public exports are
resolved lazily instead of making every evaluation dependency mandatory.
"""

from spade._lazy import lazy_exports

_EXPORTS = {
    "DEFAULT_GEM_TASKS": ("spade.core.eval.gem_tasks", "DEFAULT_GEM_TASKS"),
    "GemEvalDefaults": ("spade.core.eval.gem_tasks", "GemEvalDefaults"),
    "GemEvalResult": ("spade.core.eval.gem_evaluator", "GemEvalResult"),
    "GemEvaluator": ("spade.core.eval.gem_evaluator", "GemEvaluator"),
    "GemTaskResult": ("spade.core.eval.gem_evaluator", "GemTaskResult"),
    "GemTaskSpec": ("spade.core.eval.gem_tasks", "GemTaskSpec"),
    "load_gem_eval_config": ("spade.core.eval.gem_tasks", "load_gem_eval_config"),
    "run_fixed_model_evaluation": (
        "spade.core.eval.fixed_model_eval",
        "run_fixed_model_evaluation",
    ),
}

__all__ = list(_EXPORTS)
__getattr__ = lazy_exports(__name__, globals(), _EXPORTS)
