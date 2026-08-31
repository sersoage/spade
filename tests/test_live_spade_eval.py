"""Focused tests for the live SPADE/ProofPack/Assay runner."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tools import run_live_spade_eval as live


class _TwoTurnEnv:
    def __init__(self) -> None:
        self.turn = 0

    def reset(self, seed=None):
        self.turn = 0
        return f"seed={seed}; first then second", {}

    def step(self, action):
        self.turn += 1
        if self.turn == 1:
            assert action == r"\boxed{first}"
            return "now submit second", 0.25, False, False, {}
        assert action == r"\boxed{second}"
        return "complete", 1.0, True, False, {}

    def close(self) -> None:
        pass


class _Target:
    def instantiate(self):
        return _TwoTurnEnv()


def test_extract_python_code_and_clean_nested_boxed_action() -> None:
    assert live.extract_python_code("before\n```python\nprint('ok')\n```\nafter") == "print('ok')"
    response = r"Reasoning first. Final: \boxed{\frac{1}{2}}"
    assert live.extract_clean_action(response) == r"\boxed{\frac{1}{2}}"


def test_extract_clean_action_never_forwards_unboxed_reasoning() -> None:
    assert live.extract_clean_action("First I reason for a while. Final answer: 4") == (
        r"\boxed{__spade_invalid_action_format__}"
    )


def test_hint_leak_detection_rejects_answer_but_accepts_strategy() -> None:
    assert live.hint_reveals_solution("The answer is 32.", "32") is True
    assert live.hint_reveals_solution(r"Submit \boxed{32}.", "32") is True
    assert live.hint_reveals_solution("Work backward from the target sum.", "32") is False
    assert live.hint_reveals_solution("Try 32 after simplifying.", "32") is True
    assert live.hint_reveals_solution("Choose A.", "A", "Choose A or B.") is True
    assert (
        live.hint_reveals_solution(
            "Subtract the visible 32 from both sides.",
            "32",
            "Solve x + 32 = 40.",
        )
        is False
    )


def test_multi_turn_rollout_parses_actions_and_uses_terminal_reward(monkeypatch, tmp_path) -> None:
    responses = iter(
        [
            r"I should begin with the first operation. \boxed{first}",
            r"The next required operation is clear. \boxed{second}",
        ]
    )

    async def fake_call(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(live, "call_llm", fake_call)
    reward, trajectory = asyncio.run(
        live.run_multi_turn_rollout(
            object(),
            "model",
            _Target(),
            42,
            provider="agy",
            max_turns=5,
            workdir=tmp_path,
        )
    )
    assert reward == 1.0
    assert [row["clean_action"] for row in trajectory[1:]] == [
        r"\boxed{first}",
        r"\boxed{second}",
    ]


def test_unfinished_rollout_does_not_turn_partial_reward_into_success(monkeypatch, tmp_path) -> None:
    class PartialEnv(_TwoTurnEnv):
        def step(self, action):
            return "time", 0.75, False, True, {}

    class PartialTarget:
        def instantiate(self):
            return PartialEnv()

    async def fake_call(*args, **kwargs):
        return r"\boxed{wait}"

    monkeypatch.setattr(live, "call_llm", fake_call)
    reward, _ = asyncio.run(
        live.run_multi_turn_rollout(
            object(),
            "model",
            PartialTarget(),
            42,
            max_turns=1,
            workdir=tmp_path,
        )
    )
    assert reward == 0.0


def test_agy_environment_does_not_forward_unrelated_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AGY_TEST_SETTING", "kept")
    child_env = live._agy_environment()
    assert "OPENAI_API_KEY" not in child_env
    assert child_env["AGY_TEST_SETTING"] == "kept"


def test_llm_call_rejects_oversized_prompt_before_spawn(tmp_path: Path) -> None:
    async def invoke() -> None:
        await live.call_llm(
            "/missing/agy",
            "model",
            "x" * (live.MAX_LLM_PROMPT_BYTES + 1),
            provider="agy",
            workdir=tmp_path,
        )

    try:
        asyncio.run(invoke())
    except live.LiveEvalError as exc:
        assert "prompt" in str(exc)
        assert exc.exit_code == 4
    else:  # pragma: no cover - assertion form keeps Python 3.10 compatibility
        raise AssertionError("oversized prompt was accepted")


def test_stream_reader_rejects_oversized_child_output() -> None:
    async def invoke() -> None:
        stream = asyncio.StreamReader()
        stream.feed_data(b"x" * 9)
        stream.feed_eof()
        await live._read_stream_limited(stream, limit=8, label="stdout")

    try:
        asyncio.run(invoke())
    except live.LiveEvalError as exc:
        assert "stdout exceeded 8 bytes" in str(exc)
        assert exc.exit_code == 4
    else:  # pragma: no cover
        raise AssertionError("oversized child output was accepted")


def test_bounded_communication_times_out_after_pipe_eof() -> None:
    class _HungProcess:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.returncode = None
            self.killed = False
            self._done: asyncio.Future[int] | None = None

        async def wait(self) -> int:
            if self._done is None:
                self._done = asyncio.get_running_loop().create_future()
            return await asyncio.shield(self._done)

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            if self._done is not None and not self._done.done():
                self._done.set_result(self.returncode)

    async def invoke() -> _HungProcess:
        process = _HungProcess()
        try:
            await live._communicate_agy_bounded(process, timeout_seconds=0.01)  # type: ignore[arg-type]
        except asyncio.TimeoutError:
            return process
        raise AssertionError("closed pipes bypassed the child-process timeout")

    process = asyncio.run(invoke())
    assert process.killed is True


def test_explicit_env_file_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SPADE_LIVE_TEST_KEY", raising=False)
    env_file = tmp_path / "explicit.env"
    env_file.write_text("SPADE_LIVE_TEST_KEY=value\n", encoding="utf-8")
    live._load_explicit_env_file(env_file)
    assert live.os.environ["SPADE_LIVE_TEST_KEY"] == "value"


def test_cli_rejects_nonpositive_bounds() -> None:
    assert live.main(["--max-turns", "0"]) == 2
    assert live.main(["--llm-timeout", "0"]) == 2
    assert live.main(["--llm-timeout", "nan"]) == 2
    assert live.main(["--qualification-timeout", "inf"]) == 2
    assert live.main(["--minimum-certification-clusters", "3"]) == 2
