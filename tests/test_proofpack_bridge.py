"""Focused tests for optional ProofPack qualification and core generation wiring."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from spade.core.game_policy import GamePolicy
from spade.core.proofpack_bridge import (
    proofpack_available,
    validate_game_with_proofpack,
    validate_positive_proofpack_receipt,
)
from spade.core.types import SpadeConfig
from spade.core.utils.game_files import validate_game, validate_game_with_reason


VALID_GAME = r"""
import re

class SimpleMathEnv:
    def __init__(self, max_turns=5, **kwargs):
        self.max_turns = max_turns
        self.turn = 0

    def reset(self, seed=None):
        self.turn = 0
        return "Calculate: 10 + 20. Format: \\boxed{answer}", {}

    def solution(self):
        return "30"

    def step(self, action):
        self.turn += 1
        m = re.search(r"\\boxed\{([^}]+)\}", action)
        ans = m.group(1).strip() if m else action.strip()
        if ans == "30":
            return "Correct", 1.0, True, False, {}
        return "Wrong", 0.0, False, self.turn >= self.max_turns, {}
"""


class _Clause:
    def __init__(self, status: str, summary: str, clause_id: str = ""):
        self.status = status
        self.summary = summary
        self.clause_id = clause_id


def _passing_report(
    *,
    game_code: str = VALID_GAME,
    action_format: str = "boxed",
    seeds: list[int] | None = None,
    timeout_seconds: float = 5.0,
    max_turns: int = 20,
):
    selected_seeds = [0, 1, 42] if seeds is None else seeds
    clause_ids = (
        "v0_syntax",
        "v1_sandbox_smoke",
        "v2_oracle_solvable",
        "v3_no_agent_unwinnable",
        "v4_mutation_robustness",
    )
    return SimpleNamespace(
        schema_version="proofpack-spade-qualification/v2",
        passed=True,
        environment_name="MultiTurnEnv",
        environment_digest="sha256:" + hashlib.sha256(game_code.encode()).hexdigest(),
        clauses={clause_id: _Clause("pass", "ok", clause_id) for clause_id in clause_ids},
        metadata={
            "action_format": action_format,
            "seeds": selected_seeds,
            "max_turns": max_turns,
            "timeout_seconds": timeout_seconds,
            "execution_boundary": "macos-sandbox-exec-worker/v1",
        },
    )


def _qualifier_returning(report, calls: list[dict]):
    def qualify_spade_environment(
        game_code,
        action_format="boxed",
        seeds=None,
        timeout_seconds=5.0,
        max_turns=20,
    ):
        calls.append(
            {
                "game_code": game_code,
                "action_format": action_format,
                "seeds": seeds,
                "timeout_seconds": timeout_seconds,
                "max_turns": max_turns,
            }
        )
        return report

    return qualify_spade_environment


def test_public_positive_receipt_validator_accepts_exact_binding() -> None:
    report = _passing_report(
        action_format="tool_call",
        seeds=[7, 11],
        timeout_seconds=1.25,
        max_turns=37,
    )

    passed, reason = validate_positive_proofpack_receipt(
        report,
        game_code=VALID_GAME,
        action_format="tool_call",
        seeds=[7, 11],
        timeout_seconds=1.25,
        max_turns=37,
    )

    assert passed is True
    assert reason == "Valid ProofPack V0-V4 positive receipt (MultiTurnEnv)"


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("passed", 1, "not marked passed"),
        ("schema_version", "proofpack-spade-qualification/v1", "schema"),
        ("environment_digest", "sha256:not-the-source", "source"),
    ],
)
def test_public_positive_receipt_validator_rejects_unbound_report_fields(
    attribute: str,
    value: object,
    message: str,
) -> None:
    report = _passing_report()
    setattr(report, attribute, value)

    passed, reason = validate_positive_proofpack_receipt(
        report,
        game_code=VALID_GAME,
        action_format="boxed",
        seeds=[0, 1, 42],
        timeout_seconds=5.0,
        max_turns=20,
    )

    assert passed is False
    assert message in reason


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "exactly the required V0-V4"),
        ("extra", "exactly the required V0-V4"),
        ("mismatched_id", "mismatched identifier"),
        ("non_pass", "not marked pass"),
    ],
)
def test_public_positive_receipt_validator_requires_exact_passing_clauses(
    mutation: str,
    message: str,
) -> None:
    report = _passing_report()
    if mutation == "missing":
        del report.clauses["v4_mutation_robustness"]
    elif mutation == "extra":
        report.clauses["v5_unrequested"] = _Clause("pass", "ok", "v5_unrequested")
    elif mutation == "mismatched_id":
        report.clauses["v2_oracle_solvable"].clause_id = "v3_no_agent_unwinnable"
    else:
        report.clauses["v2_oracle_solvable"].status = "fail"

    passed, reason = validate_positive_proofpack_receipt(
        report,
        game_code=VALID_GAME,
        action_format="boxed",
        seeds=[0, 1, 42],
        timeout_seconds=5.0,
        max_turns=20,
    )

    assert passed is False
    assert message in reason


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("action_format", "boxed", "action_format"),
        ("seeds", [11, 7], "seeds"),
        ("max_turns", 38, "max_turns"),
        ("timeout_seconds", 1.5, "timeout_seconds"),
        ("execution_boundary", "in-process", "execution_boundary"),
    ],
)
def test_public_positive_receipt_validator_rejects_metadata_mismatch(
    field: str,
    value: object,
    message: str,
) -> None:
    report = _passing_report(
        action_format="tool_call",
        seeds=[7, 11],
        timeout_seconds=1.25,
        max_turns=37,
    )
    report.metadata[field] = value

    passed, reason = validate_positive_proofpack_receipt(
        report,
        game_code=VALID_GAME,
        action_format="tool_call",
        seeds=[7, 11],
        timeout_seconds=1.25,
        max_turns=37,
    )

    assert passed is False
    assert message in reason


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", [True], "seeds"),
        ("max_turns", True, "max_turns"),
        ("timeout_seconds", True, "timeout_seconds"),
    ],
)
def test_public_positive_receipt_validator_rejects_boolean_number_spoofs(
    field: str,
    value: object,
    message: str,
) -> None:
    report = _passing_report(seeds=[1], timeout_seconds=1.0, max_turns=1)
    report.metadata[field] = value

    passed, reason = validate_positive_proofpack_receipt(
        report,
        game_code=VALID_GAME,
        action_format="boxed",
        seeds=[1],
        timeout_seconds=1.0,
        max_turns=1,
    )

    assert passed is False
    assert message in reason


def test_public_positive_receipt_validator_allows_backward_compatible_metadata() -> None:
    report = _passing_report()
    report.metadata["newer_schema_hint"] = "ignored by the v2 validator"

    passed, reason = validate_positive_proofpack_receipt(
        report,
        game_code=VALID_GAME,
        action_format="boxed",
        seeds=[0, 1, 42],
        timeout_seconds=5.0,
        max_turns=20,
    )

    assert passed is True
    assert reason == "Valid ProofPack V0-V4 positive receipt (MultiTurnEnv)"


def test_bridge_passes_action_seed_timeout_and_multiturn_horizon() -> None:
    calls: list[dict] = []
    report = _passing_report(
        action_format="tool_call",
        seeds=[7, 11],
        timeout_seconds=1.25,
        max_turns=37,
    )
    module = SimpleNamespace(
        qualify_spade_environment=_qualifier_returning(report, calls)
    )

    with patch(
        "spade.core.proofpack_bridge.importlib.import_module",
        return_value=module,
    ):
        passed, reason = validate_game_with_proofpack(
            VALID_GAME,
            action_format="tool_call",
            seeds=[7, 11],
            timeout_seconds=1.25,
            max_turns=37,
        )

    assert passed is True
    assert reason == "Passed ProofPack V0-V4 qualification (MultiTurnEnv)"
    assert calls == [
        {
            "game_code": VALID_GAME,
            "action_format": "tool_call",
            "seeds": [7, 11],
            "timeout_seconds": 1.25,
            "max_turns": 37,
        }
    ]


def test_bridge_reports_failed_clause() -> None:
    report = SimpleNamespace(
        passed=False,
        environment_name="BrokenEnv",
        clauses={"v2_oracle_solvable": _Clause("fail", "oracle did not win")},
    )
    module = SimpleNamespace(
        qualify_spade_environment=_qualifier_returning(report, [])
    )
    with patch(
        "spade.core.proofpack_bridge.importlib.import_module",
        return_value=module,
    ):
        passed, reason = validate_game_with_proofpack(VALID_GAME)

    assert passed is False
    assert "v2_oracle_solvable: oracle did not win" in reason


def test_bridge_rejects_spoofed_positive_receipt() -> None:
    report = _passing_report()
    report.schema_version = "proofpack-spade-qualification/v1"
    module = SimpleNamespace(
        qualify_spade_environment=_qualifier_returning(report, [])
    )
    with patch(
        "spade.core.proofpack_bridge.importlib.import_module",
        return_value=module,
    ):
        passed, reason = validate_game_with_proofpack(VALID_GAME)

    assert passed is False
    assert "invalid positive receipt" in reason
    assert "schema" in reason


def test_bridge_fails_closed_when_dependency_is_unavailable() -> None:
    with patch("spade.core.proofpack_bridge.HAS_PROOFPACK", False):
        passed, reason = validate_game_with_proofpack(VALID_GAME)

    assert passed is False
    assert "was enabled" in reason
    assert "Python >=3.12" in reason


def test_bridge_fails_closed_for_older_interface_without_max_turns() -> None:
    def legacy_qualifier(game_code, action_format="boxed", seeds=None, timeout_seconds=5.0):
        raise AssertionError("an incompatible qualifier must not run")

    module = SimpleNamespace(qualify_spade_environment=legacy_qualifier)
    with patch(
        "spade.core.proofpack_bridge.importlib.import_module",
        return_value=module,
    ):
        passed, reason = validate_game_with_proofpack(VALID_GAME)

    assert passed is False
    assert "incompatible proofpack_env qualifier" in reason
    assert "max_turns" in reason


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_turns": 0}, "max_turns"),
        ({"max_turns": True}, "max_turns"),
        ({"proofpack_timeout_seconds": 0}, "proofpack_timeout_seconds"),
        ({"proofpack_timeout_seconds": float("inf")}, "proofpack_timeout_seconds"),
        ({"proofpack_timeout_seconds": 10**1000}, "proofpack_timeout_seconds"),
        ({"proofpack_seeds": []}, "proofpack_seeds"),
        ({"proofpack_seeds": [0, True]}, "proofpack_seeds"),
        (
            {"use_proofpack_qualification": True, "action_format": "command"},
            "action_format",
        ),
    ],
)
def test_config_rejects_invalid_assurance_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SpadeConfig(**kwargs)


def test_config_normalizes_valid_seed_sequence() -> None:
    config = SpadeConfig(proofpack_seeds=(3, 5))  # type: ignore[arg-type]

    assert config.proofpack_seeds == [3, 5]


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 10**1000])
def test_bridge_rejects_nonfinite_timeout(timeout: float) -> None:
    passed, reason = validate_game_with_proofpack(VALID_GAME, timeout_seconds=timeout)

    assert passed is False
    assert "finite" in reason


def test_bridge_fails_closed_for_non_sequence_seeds() -> None:
    passed, reason = validate_game_with_proofpack(VALID_GAME, seeds=7)  # type: ignore[arg-type]

    assert passed is False
    assert "sequence" in reason


def test_native_validation_remains_available_without_proofpack(tmp_path: Path) -> None:
    game_file = tmp_path / "valid.py"
    game_file.write_text(VALID_GAME)

    with patch("spade.core.proofpack_bridge.HAS_PROOFPACK", False):
        assert validate_game(game_file) is True


def test_enabled_qualification_fails_before_native_execution(tmp_path: Path) -> None:
    game_file = tmp_path / "valid.py"
    game_file.write_text(VALID_GAME)

    with (
        patch("spade.core.proofpack_bridge.HAS_PROOFPACK", False),
        patch("spade.core.utils.game_files.make_synthetic_env") as make_env,
    ):
        passed, reason = validate_game_with_reason(
            game_file,
            proofpack_enabled=True,
        )

    assert passed is False
    assert "was enabled" in reason
    make_env.assert_not_called()


def test_enabled_qualification_replaces_native_execution(tmp_path: Path) -> None:
    game_file = tmp_path / "valid.py"
    game_file.write_text(VALID_GAME)

    with (
        patch(
            "spade.core.utils.game_files.validate_game_with_proofpack",
            return_value=(True, "qualified"),
        ),
        patch("spade.core.utils.game_files.make_synthetic_env") as make_env,
    ):
        passed, reason = validate_game_with_reason(
            game_file,
            proofpack_enabled=True,
            validate_runtime=True,
        )

    assert passed is True
    assert reason == "ProofPack qualification passed"
    make_env.assert_not_called()


def test_native_validation_uses_configured_seed_horizon_and_tool_actions(
    tmp_path: Path,
) -> None:
    game_file = tmp_path / "valid.py"
    game_file.write_text(VALID_GAME)
    stepped: list[str] = []

    class _Env:
        def reset(self, seed=None):
            assert seed == 7
            return "ready", {}

        def step(self, action):
            stepped.append(action)
            return "continue", 0.0, False, False, {}

        def close(self):
            return None

    with patch(
        "spade.core.utils.game_files.make_synthetic_env",
        return_value=_Env(),
    ) as make_env:
        passed, reason = validate_game_with_reason(
            game_file,
            action_format="tool_call",
            proofpack_seeds=[7, 11],
            max_turns=9,
        )

    assert passed is True
    assert reason == "Native SPADE runtime validation passed"
    make_env.assert_called_once_with(
        str(game_file),
        max_turns=9,
        respect_game_max_turns=True,
    )
    assert stepped == [
        '<tool_call>{"name":"__spade_validation_probe__","arguments":{}}</tool_call>',
        "<answer>__spade_validation_probe__</answer>",
    ]


class _GenerationModel:
    def apply_template(self, messages, **kwargs):
        return [1]

    async def generate_async(self, **kwargs):
        return [
            {
                "text": f"```python\n{VALID_GAME}\n```",
                "token_ids": [2],
                "logprobs": [-0.1],
            }
        ]


def _load_orchestrator_module():
    """Import the core orchestrator without optional Transformers/Torch."""
    import importlib

    openrouter_stub = types.ModuleType("spade.core.openrouter_adapter")
    openrouter_stub.OpenAIModelAdapter = object
    openrouter_stub.create_openai_adapter = lambda **kwargs: None
    with patch.dict(
        sys.modules,
        {"spade.core.openrouter_adapter": openrouter_stub},
    ):
        return importlib.import_module("spade.core.orchestrator")


def _load_orchestrator_class():
    module = _load_orchestrator_module()

    return module.SpadeOrchestrator


def test_async_validation_deadline_covers_every_proofpack_replay() -> None:
    orchestrator_module = _load_orchestrator_module()

    observed_timeouts: list[float] = []

    async def _capture_wait(awaitable, *, timeout):
        observed_timeouts.append(timeout)
        return await awaitable

    with (
        patch.object(
            orchestrator_module,
            "validate_game_with_reason",
            return_value=(True, "passed"),
        ),
        patch.object(orchestrator_module.asyncio, "wait_for", new=_capture_wait),
    ):
        result = asyncio.run(
            orchestrator_module.validate_game_async_with_reason(
                Path("candidate.py"),
                proofpack_enabled=True,
                proofpack_seeds=[0, 1],
                proofpack_timeout_seconds=2.0,
            )
        )

    assert result == (True, "passed")
    # 60s native allowance + (8 ProofPack operations * 2 seeds * 2s) + 10s.
    assert observed_timeouts == [102.0]


def test_async_validation_deadline_covers_default_proofpack_seeds() -> None:
    orchestrator_module = _load_orchestrator_module()

    observed_timeouts: list[float] = []

    async def _capture_wait(awaitable, *, timeout):
        observed_timeouts.append(timeout)
        return await awaitable

    with (
        patch.object(
            orchestrator_module,
            "validate_game_with_reason",
            return_value=(True, "passed"),
        ),
        patch.object(orchestrator_module.asyncio, "wait_for", new=_capture_wait),
    ):
        result = asyncio.run(
            orchestrator_module.validate_game_async_with_reason(
                Path("candidate.py"),
                proofpack_enabled=True,
                proofpack_timeout_seconds=2.0,
            )
        )

    assert result == (True, "passed")
    # 60s native allowance + (8 operations * 3 default seeds * 2s) + 10s.
    assert observed_timeouts == [118.0]


def test_orchestrator_fails_before_generation_when_assurance_is_unavailable() -> None:
    orchestrator_module = _load_orchestrator_module()
    SpadeOrchestrator = orchestrator_module.SpadeOrchestrator
    with (
        patch.object(
            orchestrator_module,
            "proofpack_available",
            return_value=(False, "missing max_turns capability"),
        ),
        pytest.raises(RuntimeError, match="refusing to generate environments"),
    ):
        SpadeOrchestrator(
            model=_GenerationModel(),
            config=SpadeConfig(use_proofpack_qualification=True),
            learning_potentials={},
            game_policy=GamePolicy(),
        )


def test_batched_generation_also_fails_before_model_call(tmp_path: Path) -> None:
    from spade.core.batched import generate_games_batched

    model = SimpleNamespace(
        generate_batch=lambda *args, **kwargs: pytest.fail("model must not be called")
    )
    with (
        patch(
            "spade.core.batched.proofpack_available",
            return_value=(False, "proofpack is missing"),
        ),
        pytest.raises(RuntimeError, match="refusing to generate environments"),
    ):
        generate_games_batched(
            model=model,
            config=SpadeConfig(use_proofpack_qualification=True),
            skills=["Pattern Recognition"],
            difficulty="medium",
            games_dir=tmp_path,
            num_games=1,
        )


def test_batched_playback_uses_the_qualified_horizon(tmp_path: Path) -> None:
    from spade.core.batched import play_games_batched

    class _Env:
        def reset(self):
            return "ready", {}

        def step(self, action):
            return "done", 1.0, True, False, {}

        def close(self):
            return None

    class _Model:
        @staticmethod
        def tokenizer(text, add_special_tokens=False):
            return {"input_ids": [1]}

        @staticmethod
        def generate_batch(messages, **kwargs):
            return [{"text": r"\boxed{ok}", "token_ids": [2], "logprobs": [0.0]}]

    trajectory = SimpleNamespace(turn_count=1, reward=1.0)
    game_file = tmp_path / "game.py"
    with (
        patch("spade.core.batched.make_synthetic_env", return_value=_Env()) as make_env,
        patch("spade.core.batched.build_actor_trajectory", return_value=trajectory),
    ):
        trajectories, _ = play_games_batched(
            model=_Model(),
            config=SpadeConfig(max_turns=7),
            game_files=[game_file],
        )

    assert trajectories == [trajectory]
    make_env.assert_called_once_with(
        str(game_file),
        max_turns=7,
        respect_game_max_turns=True,
    )


def test_fixed_synthetic_adapter_uses_the_qualified_horizon(tmp_path: Path) -> None:
    from spade.core.envs.synthetic_game_adapter import SyntheticGameAdapter

    game_file = tmp_path / "game_00001.py"
    game_file.write_text(VALID_GAME)
    wrapped_env = object()
    adapter = SyntheticGameAdapter(str(tmp_path), max_turns=7)

    with patch(
        "spade.core.envs.synthetic_game_adapter.make_synthetic_env",
        return_value=wrapped_env,
    ) as make_env:
        instance = adapter.create_instance(str(game_file))

    assert instance.env is wrapped_env
    assert instance.metadata["max_turns"] == 7
    make_env.assert_called_once_with(
        str(game_file),
        max_turns=7,
        respect_game_max_turns=True,
    )


def test_legacy_difficulty_probe_uses_qualified_horizon() -> None:
    from spade.core.generate_and_validate_games import test_game_difficulty

    class _Env:
        def reset(self, seed=None):
            return "ready", {}

        def step(self, action):
            return "done", 0.0, False, True, {}

        def close(self):
            return None

    with patch(
        "spade.core.generate_and_validate_games.make_synthetic_env",
        return_value=_Env(),
    ) as make_env:
        valid, success_rate = test_game_difficulty(
            "game.py",
            num_test_runs=1,
            max_turns=7,
        )

    assert valid is True
    assert success_rate == 0.0
    make_env.assert_called_once_with(
        "game.py",
        max_turns=7,
        respect_game_max_turns=True,
    )


def test_legacy_assured_difficulty_probe_stays_in_proofpack_proxy() -> None:
    from spade.core.generate_and_validate_games import test_game_difficulty

    closed: list[bool] = []

    class _Session:
        def reset(self, seed=None):
            return "ready", {}

        def step(self, action):
            return "done", 0.0, False, True, {}

        def close(self):
            closed.append(True)

    target = SimpleNamespace(instantiate=lambda: _Session())
    with patch(
        "spade.core.generate_and_validate_games.make_synthetic_env",
        side_effect=AssertionError("native loader must not run"),
    ):
        valid, success_rate = test_game_difficulty(
            "game.py",
            num_test_runs=2,
            max_turns=7,
            proofpack_target=target,
        )

    assert valid is True
    assert success_rate == 0.0
    assert closed == [True, True]


def test_actual_orchestrator_generation_path_enforces_explicit_assurance(
    tmp_path: Path,
) -> None:
    orchestrator_module = _load_orchestrator_module()
    SpadeOrchestrator = orchestrator_module.SpadeOrchestrator

    config = SpadeConfig(
        use_proofpack_qualification=True,
        proofpack_seeds=[7, 11],
        proofpack_timeout_seconds=1.25,
        max_turns=37,
        action_format="tool_call",
    )
    with patch.object(
        orchestrator_module,
        "proofpack_available",
        return_value=(True, "available"),
    ):
        orchestrator = SpadeOrchestrator(
            model=_GenerationModel(),
            config=config,
            learning_potentials={},
            game_policy=GamePolicy(),
        )

    with patch.object(
        orchestrator_module,
        "validate_game_with_reason",
        return_value=(False, "ProofPack qualification failed: planted rejection"),
    ) as validation:
        result = asyncio.run(
            orchestrator._generate_single_game_async(
                skill="Pattern Recognition",
                difficulty="medium",
                games_dir=tmp_path,
                rollout_id=4,
                index=2,
                validate=False,
                cache_step_dir=None,
                max_attempts=1,
            )
        )

    assert result is None
    validation.assert_called_once()
    kwargs = validation.call_args.kwargs
    assert kwargs["validate_runtime"] is False
    assert kwargs["proofpack_enabled"] is True
    assert kwargs["action_format"] == "tool_call"
    assert kwargs["proofpack_seeds"] == [7, 11]
    assert kwargs["proofpack_timeout_seconds"] == 1.25
    assert kwargs["max_turns"] == 37
    assert list(tmp_path.glob("game_*.py")) == []


@pytest.mark.skipif(not proofpack_available()[0], reason="compatible proofpack_env not installed")
def test_installed_proofpack_qualifies_a_real_game() -> None:
    passed, reason = validate_game_with_proofpack(
        VALID_GAME,
        seeds=[0, 1],
        max_turns=5,
    )
    assert passed is True, reason
