"""Regression tests for optional dependency boundaries."""

import subprocess
import sys


def test_eval_submodule_does_not_import_gem() -> None:
    code = "import sys; import spade.core.eval.tau2_tasks; assert 'gem' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_slime_namespace_does_not_load_backend() -> None:
    code = (
        "import sys; import spade.slime; " "assert 'spade.slime.model_adapter' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_package_is_lightweight() -> None:
    code = (
        "import sys; import spade; "
        "assert 'spade.core.orchestrator' not in sys.modules; "
        "assert 'spade.core.game_generator' not in sys.modules; "
        "assert spade.LearningPotential.__name__ == 'LearningPotential'; "
        "assert 'spade.core.game_generator' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_exports_remain_available() -> None:
    code = (
        "from spade.core import LearningPotential, SpadeConfig; "
        "assert LearningPotential.__name__ == 'LearningPotential'; "
        "assert SpadeConfig.__name__ == 'SpadeConfig'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_token_utility_does_not_load_rollout_utilities() -> None:
    code = (
        "import sys; from spade.core.utils import get_token_delta; "
        "assert callable(get_token_delta); "
        "assert 'spade.core.utils.game_utils' not in sys.modules; "
        "assert 'spade.core.utils.env_rewards' not in sys.modules; "
        "assert 'spade.core.utils.delayed_env_rewards' not in sys.modules; "
        "assert 'spade.core.utils.trajectory_build' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
