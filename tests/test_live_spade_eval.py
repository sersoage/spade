"""Focused tests for the live SPADE/ProofPack/Assay runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tools import run_live_spade_eval as live


_CONVERSATION_ID = "12345678-1234-4234-9234-123456789abc"
_FIXED_GEMINI_DIR = ".agy-gemini-00000000000000000000000000000000"
_POLICY_CONFIG = {
    "relative_path": "config/config.json",
    "exists": False,
    "digest": None,
    "size_bytes": 0,
}


def _ndjson(*events: dict) -> bytes:
    return b"".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for event in events
    )


def _structured_evidence(
    tmp_path: Path,
    *,
    model: str = "gemini-3.7-flash-high",
    prompt: str = "sealed prompt",
    response: str = r"\boxed{ok}",
    tool: str | None = None,
    include_response_id: bool = True,
) -> dict[str, bytes | str]:
    workdir = str(tmp_path.resolve())
    events = [
        {
            "event": "init",
            "conversation_id": _CONVERSATION_ID,
            "init": {
                "model": model,
                "cwd": workdir,
                "tools": ["RunCommand"],
                "permission_mode": "request-review",
            },
        }
    ]
    if tool is not None:
        events.append(
            {
                "event": "step_update",
                "conversation_id": _CONVERSATION_ID,
                "step_update": {
                    "conversation_id": _CONVERSATION_ID,
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "tool",
                    "tool_name": tool,
                    "tool_info": {"name": tool, "parameters": {"secret": "do-not-persist"}},
                },
            }
        )
    events.append(
        {
            "event": "result",
            "conversation_id": _CONVERSATION_ID,
            "result": {
                "conversation_id": _CONVERSATION_ID,
                "status": "SUCCESS",
                "response": response,
                "duration_seconds": 1.0,
                "num_turns": 1,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "thinking_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_tokens": 2,
                },
            },
        }
    )
    log_lines = [
        f"CLI app data directory: {tmp_path / _FIXED_GEMINI_DIR / 'antigravity-cli'}",
        "applyUserSettings: no shared config permissions from "
        f"{tmp_path / _FIXED_GEMINI_DIR / 'config' / 'config.json'}",
        f"Creating CLI server backend: product=antigravity workspaceDirs=[{workdir}]",
        f'Print mode: starting (promptLength=1, model="{model}", conversationID="")',
        f"Created conversation {_CONVERSATION_ID}",
        "PRIVATE historical allow rules and user@example.invalid must not persist",
    ]
    if include_response_id:
        log_lines.append("ResponseID: response-id-1")
    if tool is not None:
        log_lines.append(f'Print mode: soft-denying tool confirmation "{tool}" at step 2')
        log_lines.append(
            f"Tool confirmation for conversation {_CONVERSATION_ID} step 2 "
            "(type=tool approved=false)"
        )
    transcript_events = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": f"<USER_REQUEST>\n{prompt}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>",
        }
    ]
    if tool is not None:
        transcript_events.append(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "tool_calls": [{"name": tool, "args": {"x": 1}}],
            }
        )
    else:
        transcript_events.append(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": response,
                "thinking": "private chain of thought must not persist",
            }
        )
    return {
        "model": model,
        "prompt": prompt,
        "workdir": workdir,
        "stdout_ndjson": _ndjson(*events),
        "stderr": b"",
        "log": ("\n".join(log_lines) + "\n").encode(),
        "transcript": _ndjson(*transcript_events),
    }


def _analyze(fixture: dict[str, bytes | str], **overrides):
    arguments = {
        "requested_model": fixture["model"],
        "full_prompt": fixture["prompt"],
        "invocation_workdir": fixture["workdir"],
        "exit_status": 0,
        "timed_out": False,
        "capture_failures": (),
        "stdout_ndjson": fixture["stdout_ndjson"],
        "stderr": fixture["stderr"],
        "log": fixture["log"],
        "transcript": fixture["transcript"],
        "policy_config_identity": _POLICY_CONFIG,
    }
    arguments.update(overrides)
    return live.analyze_agy_evidence(**arguments)


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


def test_unfinished_rollout_does_not_turn_partial_reward_into_success(
    monkeypatch, tmp_path
) -> None:
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


def test_agy_environment_does_not_forward_credentials_or_conversation_context(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AGY_TEST_SETTING", "not-forwarded")
    monkeypatch.setenv("ANTIGRAVITY_CONVERSATION_ID", "not-forwarded")
    monkeypatch.setenv("HTTPS_PROXY", "not-forwarded")
    child_env = live._agy_environment()
    assert "OPENAI_API_KEY" not in child_env
    assert "AGY_TEST_SETTING" not in child_env
    assert "ANTIGRAVITY_CONVERSATION_ID" not in child_env
    assert "HTTPS_PROXY" not in child_env


def test_structured_agy_response_binds_route_prompt_workdir_and_transcript(
    tmp_path: Path,
) -> None:
    fixture = _structured_evidence(tmp_path)
    evidence = _analyze(fixture)
    assert evidence.disposition == "response"
    assert evidence.response == r"\boxed{ok}"
    assert evidence.reported_model == "gemini-3.7-flash-high"
    assert evidence.conversation_id == _CONVERSATION_ID


def test_structured_agy_tool_call_dominates_later_terminal_text(tmp_path: Path) -> None:
    fixture = _structured_evidence(
        tmp_path,
        tool="RunCommand",
        response=r"\boxed{fabricated-after-tool}",
    )
    evidence = _analyze(fixture)
    assert evidence.disposition == "tool_policy_no_action"
    assert evidence.response is None
    assert evidence.tool_call_names == ("RunCommand",)


def test_structured_agy_only_retries_explicit_pre_response_provider_failure(
    tmp_path: Path,
) -> None:
    fixture = _structured_evidence(tmp_path)
    evidence = _analyze(
        fixture,
        exit_status=1,
        stdout_ndjson=b"",
        stderr=b"HTTP status 429: provider temporarily unavailable\n",
        log=b"",
        transcript=b"",
    )
    assert evidence.disposition == "pre_response_provider_failure"
    assert evidence.response_ids == ()
    assert b"provider temporarily unavailable" not in evidence.stderr
    stderr_receipt = json.loads(evidence.stderr)
    assert stderr_receipt["schema_version"] == live.AGY_STDERR_RECEIPT_SCHEMA
    assert stderr_receipt["explicit_pre_response_failure_marker"] is True


def test_structured_agy_sanitized_receipts_preserve_failure_classification(
    tmp_path: Path,
) -> None:
    malformed = _structured_evidence(tmp_path)
    events = [json.loads(line) for line in malformed["stdout_ndjson"].splitlines()]
    events[0]["result"] = {"unexpected": True}
    malformed["stdout_ndjson"] = _ndjson(*events)
    first = _analyze(malformed)
    replayed = _analyze(
        malformed,
        stdout_ndjson=first.stdout_ndjson,
        stderr=first.stderr,
        log=first.log,
        transcript=first.transcript,
        policy_config_identity=None,
        sanitized_stream_receipt=True,
        sanitized_stderr_receipt=True,
        sanitized_log_receipt=True,
        sanitized_transcript_receipt=True,
    )
    assert (replayed.disposition, replayed.error) == (first.disposition, first.error)

    refused = _structured_evidence(tmp_path, include_response_id=False)
    terminal = [json.loads(line) for line in refused["stdout_ndjson"].splitlines()]
    terminal[-1]["result"]["status"] = "ERROR"
    terminal[-1]["result"]["response"] = ""
    terminal[-1]["result"]["error"] = "HTTP status 429: provider unavailable"
    refused["stdout_ndjson"] = _ndjson(*terminal)
    first = _analyze(refused, exit_status=1)
    replayed = _analyze(
        refused,
        exit_status=1,
        stdout_ndjson=first.stdout_ndjson,
        stderr=first.stderr,
        log=first.log,
        transcript=first.transcript,
        policy_config_identity=None,
        sanitized_stream_receipt=True,
        sanitized_stderr_receipt=True,
        sanitized_log_receipt=True,
        sanitized_transcript_receipt=True,
    )
    assert first.disposition == "pre_response_provider_failure"
    assert (replayed.disposition, replayed.error) == (first.disposition, first.error)


def test_structured_agy_blank_after_response_id_is_ambiguous(tmp_path: Path) -> None:
    fixture = _structured_evidence(tmp_path, response="")
    evidence = _analyze(fixture)
    assert evidence.disposition == "ambiguous_provider_disposition"
    assert "blank after a ResponseID" in str(evidence.error)


def test_structured_agy_route_or_trace_mismatch_fails_integrity(tmp_path: Path) -> None:
    wrong_route = _structured_evidence(tmp_path, model="gemini-wrong")
    assert (
        _analyze(wrong_route, requested_model="gemini-3.7-flash-high").disposition
        == "evidence_integrity_failure"
    )

    missing_tool_trace = _structured_evidence(tmp_path, tool="RunCommand")
    missing_tool_trace["transcript"] = _ndjson(
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": "<USER_REQUEST>\nsealed prompt\n</USER_REQUEST>",
        }
    )
    assert _analyze(missing_tool_trace).disposition == "evidence_integrity_failure"

    executed = _structured_evidence(tmp_path, tool="RunCommand")
    executed_events = [json.loads(line) for line in executed["stdout_ndjson"].splitlines()]
    executed_events[1]["step_update"]["tool_info"]["output"] = "command output"
    executed["stdout_ndjson"] = _ndjson(*executed_events)
    executed_evidence = _analyze(executed)
    assert executed_evidence.disposition == "evidence_integrity_failure"
    assert "may have executed" in str(executed_evidence.error)

    transcript_execution = _structured_evidence(tmp_path, tool="RunCommand")
    transcript_execution["transcript"] += _ndjson(
        {
            "step_index": 2,
            "source": "MODEL",
            "type": "RUN_COMMAND",
            "status": "DONE",
            "content": "command output",
        }
    )
    transcript_execution_evidence = _analyze(transcript_execution)
    assert transcript_execution_evidence.disposition == "evidence_integrity_failure"
    assert "may have executed" in str(transcript_execution_evidence.error)

    execution_with_retry_marker = _structured_evidence(
        tmp_path,
        tool="RunCommand",
        include_response_id=False,
    )
    execution_with_retry_marker["stderr"] = b"HTTP status 429: provider unavailable\n"
    execution_with_retry_marker["transcript"] += _ndjson(
        {
            "step_index": 2,
            "source": "MODEL",
            "type": "RUN_COMMAND",
            "status": "DONE",
            "content": "command output",
        }
    )
    assert _analyze(execution_with_retry_marker).disposition == "evidence_integrity_failure"


def test_structured_agy_requires_isolated_default_policy(tmp_path: Path) -> None:
    wrong_permission = _structured_evidence(tmp_path)
    events = [json.loads(line) for line in wrong_permission["stdout_ndjson"].splitlines()]
    events[0]["init"]["permission_mode"] = "proceed-in-sandbox"
    wrong_permission["stdout_ndjson"] = _ndjson(*events)
    assert _analyze(wrong_permission).disposition == "evidence_integrity_failure"

    cached_policy = _structured_evidence(tmp_path)
    cached_policy["log"] += b"Global config.json not found. Using cached permissions.\n"
    assert _analyze(cached_policy).disposition == "evidence_integrity_failure"


def test_structured_agy_subprocess_uses_stream_json_and_unique_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _structured_evidence(tmp_path)
    captured_command: tuple[str, ...] | None = None

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_data(fixture["stdout_ndjson"])
            self.stdout.feed_eof()
            self.stderr.feed_data(
                b"PRIVATE prompt, tool arguments, account@example.invalid, and local path"
            )
            self.stderr.feed_eof()
            self.returncode = 0

        async def wait(self) -> int:
            return 0

        def kill(self) -> None:
            raise AssertionError("completed fake process must not be killed")

    async def fake_spawn(*command: str, **kwargs):
        nonlocal captured_command
        captured_command = command
        assert kwargs["cwd"] == fixture["workdir"]
        assert kwargs["env"]["AGY_CLI_DISABLE_AUTO_UPDATE"] == "1"
        log_path = Path(command[command.index("--log-file") + 1])
        gemini_dir = Path(command[command.index("--gemini_dir") + 1])
        private_log = fixture["log"].replace(
            str(tmp_path / _FIXED_GEMINI_DIR).encode(),
            str(gemini_dir).encode(),
        )
        log_path.write_bytes(private_log)
        transcript_path = (
            gemini_dir
            / "antigravity-cli"
            / "brain"
            / _CONVERSATION_ID
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )
        transcript_path.parent.mkdir(parents=True)
        transcript_path.write_bytes(fixture["transcript"])
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    evidence = asyncio.run(
        live.call_llm_with_evidence(
            "/sealed/agy",
            str(fixture["model"]),
            str(fixture["prompt"]),
            workdir=tmp_path,
            evidence_log_path=tmp_path / "unique.log",
            process_environment={"AGY_CLI_DISABLE_AUTO_UPDATE": "1"},
        )
    )
    assert evidence.disposition == "response"
    assert captured_command is not None
    assert captured_command[captured_command.index("--output-format") + 1] == "stream-json"
    assert captured_command[captured_command.index("--log-file") + 1] == str(
        tmp_path / "unique.log"
    )
    isolated = Path(captured_command[captured_command.index("--gemini_dir") + 1])
    assert isolated.parent == tmp_path
    assert isolated.name.startswith(".agy-gemini-")
    assert not isolated.exists()
    assert captured_command[captured_command.index("--app_data_dir") + 1] == "antigravity-cli"
    assert not (tmp_path / "unique.log").exists()
    receipt = json.loads(evidence.log)
    assert receipt["schema_version"] == live.AGY_LOG_RECEIPT_SCHEMA
    assert receipt["raw_log_digest"].startswith("sha256:")
    assert b"historical allow rules" not in evidence.log
    assert b"PRIVATE prompt" not in evidence.stderr
    assert json.loads(evidence.stderr)["schema_version"] == live.AGY_STDERR_RECEIPT_SCHEMA
    assert b"private chain of thought" not in evidence.transcript
    assert b"do-not-persist" not in evidence.stdout_ndjson


def test_agy_environment_override_is_narrow_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGY_CLI_DISABLE_AUTO_UPDATE", "inherited-but-not-authorized")
    assert "AGY_CLI_DISABLE_AUTO_UPDATE" not in live._agy_environment()
    assert live._agy_environment({"AGY_CLI_DISABLE_AUTO_UPDATE": "1"})[
        "AGY_CLI_DISABLE_AUTO_UPDATE"
    ] == "1"
    with pytest.raises(live.LiveEvalError, match="unsupported override"):
        live._agy_environment({"AGY_CLI_DISABLE_AUTO_UPDATE": "0"})
    with pytest.raises(live.LiveEvalError, match="unsupported override"):
        live._agy_environment({"UNSEALED_SECRET": "value"})


def test_bounded_evidence_reader_rejects_symlink_and_oversize(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"12345")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert live._read_bounded_regular_file(link, limit=8, label="trace")[1] == "trace_symlink"
    assert (
        live._read_bounded_regular_file(target, limit=4, label="trace")[1] == "trace_limit_exceeded"
    )
    malicious_stream = _ndjson(
        {
            "event": "result",
            "conversation_id": "../../outside",
            "result": {"conversation_id": "../../outside"},
        }
    )
    log = f"CLI app data directory: {tmp_path}\n".encode()
    assert live._agy_transcript_path(malicious_stream, log) is None


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
