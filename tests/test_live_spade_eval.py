"""Focused tests for the live SPADE/ProofPack/Assay runner."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools import run_live_spade_eval as live


_CONVERSATION_ID = "12345678-1234-4234-9234-123456789abc"
_FIXED_GEMINI_DIR = ".agy-gemini-00000000000000000000000000000000"
_POLICY_CONFIG = {
    "relative_path": "config/config.json",
    "exists": True,
    "digest": live._bytes_digest(live.AGY_SEALED_POLICY_CONFIG),
    "size_bytes": len(live.AGY_SEALED_POLICY_CONFIG),
}
_POLICY_TRANSITION = live._unchanged_policy_config_transition()


def _sandbox_invocation_receipt(
    *,
    prompt: str = "sealed prompt",
    model: str = "gemini-3.7-flash-high",
) -> dict:
    body = {
        "schema_version": live.AGY_SANDBOX_INVOCATION_RECEIPT_SCHEMA,
        "write_protection_policy": live.AGY_POLICY_CONFIG_WRITE_PROTECTION,
        "sandbox_executable": str(live.AGY_SANDBOX_EXECUTABLE),
        "sandbox_executable_digest": live.EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST,
        "agy_executable_digest": "sha256:" + "1" * 64,
        "profile_digest": live.AGY_POLICY_SANDBOX_PROFILE_DIGEST,
        "parameter_names": [
            "WORKDIR",
            "GEMINI_DIR",
            "CONFIG_DIR",
            "CONFIG_FILE",
            "AGY_BIN",
        ],
        "parameter_relationships": {
            "cwd_equals_workdir": True,
            "gemini_dir_direct_child_of_workdir": True,
            "config_dir_direct_child_of_gemini_dir": True,
            "config_file_direct_child_of_config_dir": True,
            "tmp_dir_direct_child_of_workdir": True,
            "log_direct_child_of_workdir": True,
        },
        "argv_policy": "sandbox-exec-D-parameters-static-profile-then-agy",
        "agy_argv_exact": True,
        "prompt_digest": live._bytes_digest(prompt.encode("utf-8")),
        "requested_model": model,
        "print_timeout_seconds": 180,
        "environment_allowlist_exact": True,
        "auto_update_disabled": True,
        "profile_bytes_passed_exactly": True,
        "cwd_policy": "canonical-private-workdir",
        "tmpdir_policy": "fresh-private-direct-child-of-workdir",
        "stdin_policy": "devnull",
        "stdout_policy": "pipe",
        "stderr_policy": "pipe",
        "close_fds": True,
        "start_new_session": True,
        "private_paths_persisted": False,
    }
    return {**body, "receipt_digest": live._canonical_json_digest(body)}


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
        "policy_config_transition": _POLICY_TRANSITION,
        "sandbox_invocation_receipt": _sandbox_invocation_receipt(
            prompt=str(fixture["prompt"]),
            model=str(fixture["model"]),
        ),
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
    assert evidence.disposition == "response", evidence.summary()
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


def test_frozen_v1_log_stderr_and_summary_replay_without_shape_drift(tmp_path: Path) -> None:
    fixture = _structured_evidence(tmp_path)
    _, stream_receipt, stream_error, _, _ = live._stream_receipt(
        fixture["stdout_ndjson"],
        workdir=str(fixture["workdir"]),
        sanitized=False,
    )
    assert stream_error is None
    _, transcript_receipt, transcript_error, _, _ = live._transcript_receipt(
        fixture["transcript"],
        full_prompt=str(fixture["prompt"]),
        sanitized=False,
    )
    assert transcript_error is None
    stderr_body = {
        "schema_version": live.AGY_STDERR_RECEIPT_SCHEMA_V1,
        "raw_stderr_digest": live._bytes_digest(b""),
        "raw_stderr_size_bytes": 0,
        "explicit_pre_response_failure_marker": False,
        "shape_error": None,
    }
    stderr_receipt = live._canonical_json_bytes(
        {**stderr_body, "receipt_digest": live._canonical_json_digest(stderr_body)}
    )
    log_body = {
        "schema_version": live.AGY_LOG_RECEIPT_SCHEMA_V1,
        "raw_log_digest": live._bytes_digest(fixture["log"]),
        "raw_log_size_bytes": len(fixture["log"]),
        "response_ids": ["response-id-1"],
        "conversation_ids": [_CONVERSATION_ID],
        "reported_models": [fixture["model"]],
        "workspace_dirs": [fixture["workdir"]],
        "soft_denied_tools": [],
        "approved_tool_confirmation_ids": [],
        "denied_tool_confirmation_ids": [],
        "explicit_pre_response_failure_marker": False,
        "policy_config_disposition": "absent",
        "policy_config_path_relative": (
            f"{_FIXED_GEMINI_DIR}/config/config.json"
        ),
        "policy_config_identity": {
            "relative_path": "config/config.json",
            "exists": False,
            "digest": None,
            "size_bytes": 0,
        },
    }
    log_receipt = live._canonical_json_bytes(
        {**log_body, "receipt_digest": live._canonical_json_digest(log_body)}
    )
    evidence = live.analyze_agy_evidence(
        requested_model=str(fixture["model"]),
        full_prompt=str(fixture["prompt"]),
        invocation_workdir=str(fixture["workdir"]),
        exit_status=0,
        timed_out=False,
        capture_failures=(),
        stdout_ndjson=stream_receipt,
        stderr=stderr_receipt,
        log=log_receipt,
        transcript=transcript_receipt,
        sanitized_stream_receipt=True,
        sanitized_stderr_receipt=True,
        sanitized_log_receipt=True,
        sanitized_transcript_receipt=True,
    )
    summary = evidence.summary()
    assert evidence.disposition == "response"
    assert summary["schema_version"] == live.AGY_EVIDENCE_SCHEMA_V1
    assert summary["policy_config_identity"]["exists"] is False
    assert "policy_config_transition" not in summary
    assert "process_group_quiescent" not in summary


def test_frozen_v2_receipts_replay_without_v3_invocation_field(tmp_path: Path) -> None:
    fixture = _structured_evidence(tmp_path)
    original = live.analyze_agy_evidence(
        requested_model=str(fixture["model"]),
        full_prompt=str(fixture["prompt"]),
        invocation_workdir=str(fixture["workdir"]),
        exit_status=0,
        timed_out=False,
        process_group_descendant_detected=False,
        process_group_quiescent=True,
        capture_failures=(),
        stdout_ndjson=fixture["stdout_ndjson"],
        stderr=fixture["stderr"],
        log=fixture["log"],
        transcript=fixture["transcript"],
        policy_config_identity=_POLICY_CONFIG,
        policy_config_transition=live._legacy_unchanged_policy_config_transition(),
    )
    assert json.loads(original.log)["schema_version"] == live.AGY_LOG_RECEIPT_SCHEMA_V2
    assert original.summary()["schema_version"] == live.AGY_EVIDENCE_SCHEMA_V2
    assert "sandbox_invocation_receipt" not in original.summary()
    replayed = live.analyze_agy_evidence(
        requested_model=str(fixture["model"]),
        full_prompt=str(fixture["prompt"]),
        invocation_workdir=str(fixture["workdir"]),
        exit_status=0,
        timed_out=False,
        process_group_descendant_detected=False,
        process_group_quiescent=True,
        capture_failures=original.capture_failures,
        stdout_ndjson=original.stdout_ndjson,
        stderr=original.stderr,
        log=original.log,
        transcript=original.transcript,
        sanitized_stream_receipt=True,
        sanitized_stderr_receipt=True,
        sanitized_log_receipt=True,
        sanitized_transcript_receipt=True,
    )
    assert replayed.summary() == original.summary()


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
    spawn_frontier_receipts: list[dict] = []

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
        assert kwargs["env"]["TMPDIR"].startswith(str(tmp_path))
        assert kwargs["close_fds"] is True
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
        assert len(spawn_frontier_receipts) == 1
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

    async def fake_quiesce(process):
        assert isinstance(process, FakeProcess)
        return False, True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(live, "_quiesce_agy_process_group", fake_quiesce)
    evidence = asyncio.run(
        live.call_llm_with_evidence(
            "/usr/bin/true",
            str(fixture["model"]),
            str(fixture["prompt"]),
            workdir=tmp_path,
            evidence_log_path=tmp_path / "unique.log",
            process_environment={"AGY_CLI_DISABLE_AUTO_UPDATE": "1"},
            before_spawn=lambda receipt: spawn_frontier_receipts.append(dict(receipt)),
        )
    )
    assert evidence.disposition == "response", evidence.summary()
    assert captured_command is not None
    assert captured_command[0] == str(live.AGY_SANDBOX_EXECUTABLE)
    assert captured_command[captured_command.index("-p") + 1] == (
        live.AGY_POLICY_SANDBOX_PROFILE
    )
    assert f"WORKDIR={tmp_path}" in captured_command
    assert "AGY_BIN=/usr/bin/true" in captured_command
    assert any(item.startswith("CONFIG_FILE=") for item in captured_command)
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
    assert not list(tmp_path.glob(".agy-tmp-*"))
    receipt = json.loads(evidence.log)
    assert receipt["schema_version"] == live.AGY_LOG_RECEIPT_SCHEMA
    assert evidence.summary()["schema_version"] == live.AGY_EVIDENCE_SCHEMA
    assert receipt["sandbox_invocation_receipt"] == spawn_frontier_receipts[0]
    assert receipt["raw_log_digest"].startswith("sha256:")
    assert b"historical allow rules" not in evidence.log
    assert b"PRIVATE prompt" not in evidence.stderr
    assert json.loads(evidence.stderr)["schema_version"] == live.AGY_STDERR_RECEIPT_SCHEMA
    assert b"private chain of thought" not in evidence.transcript
    assert b"do-not-persist" not in evidence.stdout_ndjson
    assert evidence.policy_config_transition["transition_class"] == (
        "precreated-to-unchanged-empty-grants"
    )


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


def _sealed_policy_tree(tmp_path: Path):
    gemini_dir = tmp_path / ".agy-gemini-00000000000000000000000000000000"
    gemini_dir.mkdir(mode=0o700)
    baseline, failure = live._create_sealed_policy_config(gemini_dir)
    assert failure is None
    assert baseline is not None
    return gemini_dir, baseline


@pytest.mark.parametrize(
    ("content", "failure"),
    [
        (
            b'{"userSettings":{"globalPermissionGrants":{"allow":[],"allow":[],"deny":[],"ask":[]}}}',
            "policy_config_duplicate_key",
        ),
        (b"{", "policy_config_invalid_json"),
        (b"\xff", "policy_config_invalid_utf8"),
        (b"\xef\xbb\xbf{}", "policy_config_invalid_utf8"),
        (b'{"userSettings":{}\x00}', "policy_config_contains_nul"),
        (b"[]", "policy_config_nonobject_root"),
        (
            b'{"plugins":{},"userSettings":{"globalPermissionGrants":{"allow":[],"deny":[],"ask":[]}}}',
            "policy_config_unknown_root_field",
        ),
        (
            b'{"userSettings":{"autoExecutionPolicy":"OFF","globalPermissionGrants":{"allow":[],"deny":[],"ask":[]}}}',
            "policy_config_unknown_user_setting",
        ),
        (
            b'{"userSettings":{"globalPermissionGrants":{"allow":["RunCommand"],"deny":[],"ask":[]}}}',
            "policy_config_allow_grants_nonempty",
        ),
        (
            b'{"userSettings":{"globalPermissionGrants":{"allow":[],"deny":["RunCommand"],"ask":[]}}}',
            "policy_config_nonempty_grants",
        ),
        (
            b'{"userSettings":{"globalPermissionGrants":{"allow":[],"deny":[],"ask":[],"other":[]}}}',
            "policy_config_global_grants_unknown_bucket",
        ),
    ],
)
def test_policy_config_parser_rejects_malformed_permissive_and_unknown(
    content: bytes,
    failure: str,
) -> None:
    _, observed = live._parse_restrictive_policy_config(content)
    assert observed == failure


def test_policy_config_receipt_is_private_self_consistent_and_semantically_fail_closed() -> None:
    projection, failure = live._parse_restrictive_policy_config(
        live.AGY_SEALED_POLICY_CONFIG
    )
    assert failure is None
    assert projection["grant_counts"] == {"allow": 0, "deny": 0, "ask": 0}
    receipt = live._unchanged_policy_config_transition()
    assert live._validate_policy_config_transition(receipt) == receipt
    encoded = live._canonical_json_bytes(receipt)
    assert live.AGY_SEALED_POLICY_CONFIG.strip() not in encoded
    assert b"config.json" not in encoded
    assert b"RunCommand" not in encoded
    assert b"/private/" not in encoded
    tampered = json.loads(encoded)
    tampered["global_permission_grant_counts"]["deny"] = 1
    body = {key: value for key, value in tampered.items() if key != "receipt_digest"}
    tampered["receipt_digest"] = live._canonical_json_digest(body)
    assert live._validate_policy_config_transition(tampered) is None


def test_policy_config_capture_accepts_only_exact_unchanged_precreated_file(
    tmp_path: Path,
) -> None:
    gemini_dir, baseline = _sealed_policy_tree(tmp_path)
    transition, failure = live._capture_policy_config_transition(
        gemini_dir,
        baseline,
        write_protection_applied=True,
    )
    assert failure is None
    assert transition == live._unchanged_policy_config_transition()


def test_policy_config_capture_rejects_absent_created_symlink_oversize_and_replacement(
    tmp_path: Path,
) -> None:
    gemini_dir, baseline = _sealed_policy_tree(tmp_path)
    config_path = gemini_dir / "config" / "config.json"

    config_path.unlink()
    transition, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=True
    )
    assert failure == "policy_config_final_missing"
    assert transition["transition_class"] == "unsafe"

    config_path.write_bytes(live.AGY_SEALED_POLICY_CONFIG)
    config_path.chmod(0o600)
    transition, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=True
    )
    assert failure == "policy_config_final_identity_changed"
    assert transition["capture_failure"] == failure

    config_path.unlink()
    config_path.symlink_to(tmp_path / "outside")
    _, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=True
    )
    assert failure == "policy_config_final_symlink"

    config_path.unlink()
    config_path.write_bytes(b"x" * (live.MAX_AGY_POLICY_CONFIG_BYTES + 1))
    config_path.chmod(0o600)
    _, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=True
    )
    assert failure == "policy_config_final_limit_exceeded"

    created_without_baseline, failure = live._capture_policy_config_transition(
        gemini_dir, None, write_protection_applied=True
    )
    assert failure == "policy_config_initial_missing"
    assert created_without_baseline["no_explicit_grants"] is False


def test_policy_config_capture_rejects_hardlink_mode_and_unsealed_profile(
    tmp_path: Path,
) -> None:
    gemini_dir, baseline = _sealed_policy_tree(tmp_path)
    config_path = gemini_dir / "config" / "config.json"
    hardlink = tmp_path / "outside-hardlink"
    os.link(config_path, hardlink)
    _, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=True
    )
    assert failure == "policy_config_private_owner_invalid"
    hardlink.unlink()

    config_path.chmod(0o644)
    _, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=True
    )
    assert failure == "policy_config_private_owner_invalid"
    config_path.chmod(0o600)

    transition, failure = live._capture_policy_config_transition(
        gemini_dir, baseline, write_protection_applied=False
    )
    assert failure == "policy_config_write_protection_unsealed"
    assert transition["write_protection_applied"] is False


def test_policy_sandbox_profile_is_static_and_parameterized(tmp_path: Path) -> None:
    workdir = tmp_path
    gemini_dir = workdir / ".agy-gemini-00000000000000000000000000000000"
    profile = live._agy_policy_sandbox_profile(
        gemini_dir,
        workdir,
        Path("/usr/bin/true"),
    )
    assert profile == live.AGY_POLICY_SANDBOX_PROFILE
    assert live._bytes_digest(profile.encode()) == live.AGY_POLICY_SANDBOX_PROFILE_DIGEST
    assert "(deny process-fork)" in profile
    assert "(deny process-exec*)" in profile
    assert '(param "AGY_BIN")' in profile
    assert '(param "CONFIG_DIR")' in profile
    assert '(param "CONFIG_FILE")' in profile
    assert str(tmp_path) not in profile


@pytest.mark.skipif(
    not live.AGY_SANDBOX_EXECUTABLE.is_file(),
    reason="requires the pinned sandbox executable for its byte receipt",
)
def test_sandbox_invocation_receipt_is_derived_from_exact_argv_and_spawn_options(
    tmp_path: Path,
) -> None:
    if live._bytes_digest(live.AGY_SANDBOX_EXECUTABLE.read_bytes()) != (
        live.EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST
    ):
        pytest.skip("sandbox executable bytes differ on this host")
    workdir = tmp_path.resolve(strict=True)
    gemini = workdir / _FIXED_GEMINI_DIR
    config_dir = gemini / "config"
    config_file = config_dir / "config.json"
    tmp_dir = workdir / "tmp"
    log_path = workdir / "agy.log"
    agy = Path("/usr/bin/true")
    prompt = "sealed prompt"
    model = "gemini-3.7-flash-high"
    agy_tail = [
        str(agy),
        "-p",
        prompt,
        "--disable-slash-commands",
        "--sandbox",
        "--print-timeout",
        "180s",
        "--output-format",
        live.AGY_OUTPUT_FORMAT,
        "--log-file",
        str(log_path),
        "--gemini_dir",
        str(gemini),
        "--app_data_dir",
        "antigravity-cli",
        "--model",
        model,
    ]
    command = [
        str(live.AGY_SANDBOX_EXECUTABLE),
        "-D",
        f"WORKDIR={workdir}",
        "-D",
        f"GEMINI_DIR={gemini}",
        "-D",
        f"CONFIG_DIR={config_dir}",
        "-D",
        f"CONFIG_FILE={config_file}",
        "-D",
        f"AGY_BIN={agy}",
        "-p",
        live.AGY_POLICY_SANDBOX_PROFILE,
        *agy_tail,
    ]
    options = {
        "cwd": str(workdir),
        "env": {"TMPDIR": str(tmp_dir), "AGY_CLI_DISABLE_AUTO_UPDATE": "1"},
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "close_fds": True,
        "start_new_session": True,
    }

    def derive(selected_command, selected_options):
        return live._sandbox_invocation_receipt(
            command=selected_command,
            spawn_options=selected_options,
            workdir=workdir,
            gemini_dir=gemini,
            config_dir=config_dir,
            config_file=config_file,
            tmp_dir=tmp_dir,
            log_path=log_path,
            agy_executable=agy,
            sandbox_executable=live.AGY_SANDBOX_EXECUTABLE,
            model=model,
            full_prompt=prompt,
            timeout_seconds=180,
        )

    assert live._validate_sandbox_invocation_receipt(derive(command, options)) is not None
    mutations = []
    missing_parameter = list(command)
    del missing_parameter[7:9]
    mutations.append((missing_parameter, dict(options)))
    changed_profile = list(command)
    changed_profile[12] += "\n"
    mutations.append((changed_profile, dict(options)))
    missing_nested_sandbox = list(command)
    missing_nested_sandbox.remove("--sandbox")
    mutations.append((missing_nested_sandbox, dict(options)))
    for key, replacement in (
        ("cwd", str(workdir / "wrong")),
        ("stdin", asyncio.subprocess.PIPE),
        ("close_fds", False),
        ("start_new_session", False),
    ):
        changed_options = dict(options)
        changed_options[key] = replacement
        mutations.append((list(command), changed_options))
    changed_env = dict(options)
    changed_env["env"] = {"TMPDIR": str(workdir / "wrong")}
    mutations.append((list(command), changed_env))
    assert all(
        live._validate_sandbox_invocation_receipt(derive(cmd, opts)) is None
        for cmd, opts in mutations
    )

    prompt_is_dash_d = list(command)
    prompt_is_dash_d[prompt_is_dash_d.index(prompt)] = "-D"
    dash_receipt = live._sandbox_invocation_receipt(
        command=prompt_is_dash_d,
        spawn_options=options,
        workdir=workdir,
        gemini_dir=gemini,
        config_dir=config_dir,
        config_file=config_file,
        tmp_dir=tmp_dir,
        log_path=log_path,
        agy_executable=agy,
        sandbox_executable=live.AGY_SANDBOX_EXECUTABLE,
        model=model,
        full_prompt="-D",
        timeout_seconds=180,
    )
    assert live._validate_sandbox_invocation_receipt(dash_receipt) is not None


@pytest.mark.skipif(
    sys.platform != "darwin" or not live.AGY_SANDBOX_EXECUTABLE.is_file(),
    reason="requires the sealed macOS sandbox executable",
)
def test_policy_sandbox_blocks_config_escape_and_allows_appdata() -> None:
    script = r"""
import json
import os
import sys
from pathlib import Path

workdir, gemini, config, alias, outside_source, outside = map(Path, sys.argv[1:])

def denied(operation):
    try:
        operation()
    except OSError:
        return True
    return False

def fork_once():
    child = os.fork()
    if child == 0:
        os._exit(99)
    os.waitpid(child, 0)

results = {
    "mutate": denied(lambda: config.write_bytes(b"changed")),
    "unlink": denied(config.unlink),
    "config_create": denied(lambda: (config.parent / "new.json").write_bytes(b"x")),
    "config_chmod": denied(lambda: os.chmod(config, 0o644)),
    "config_rename": denied(lambda: os.rename(config.parent, workdir / "config-moved")),
    "root_rename": denied(lambda: os.rename(gemini, workdir / "gemini-moved")),
    "workdir_rename": denied(lambda: os.rename(workdir, outside.with_suffix(".moved"))),
    "outward_hardlink": denied(lambda: os.link(config, workdir / "hardlink")),
    "inward_hardlink": denied(lambda: os.link(outside_source, workdir / "hardlink-in")),
    "symlink_mutate": denied(lambda: alias.write_bytes(b"changed")),
    "outside_write": denied(lambda: outside.write_bytes(b"escape")),
    "fork": denied(fork_once),
    "child_exec": denied(lambda: os.execve("/usr/bin/true", ["true"], {})),
}
appdata = gemini / "antigravity-cli" / "brain"
try:
    appdata.mkdir(parents=True)
except OSError:
    results["appdata_allowed"] = False
else:
    results["appdata_allowed"] = True
try:
    (workdir / "tmp" / "allowed").write_bytes(b"ok")
except OSError:
    results["tmp_write_allowed"] = False
else:
    results["tmp_write_allowed"] = True
print(json.dumps(results, sort_keys=True))
"""
    with tempfile.TemporaryDirectory(
        prefix="spade-policy-sandbox-test-",
        dir="/private/tmp",
    ) as workdir_name:
        workdir = Path(workdir_name).resolve(strict=True)
        gemini_dir, _ = _sealed_policy_tree(workdir)
        config_path = gemini_dir / "config" / "config.json"
        alias = workdir / "config-alias"
        alias.symlink_to(config_path)
        (workdir / "tmp").mkdir(mode=0o700)
        outside = Path("/private/tmp") / f"{workdir.name}-escape"
        outside_source = Path("/private/tmp") / f"{workdir.name}-outside-source"
        outside_source.write_bytes(b"outside")
        outside_source.chmod(0o600)
        python = Path(sys.executable).resolve(strict=True)
        command = [
            str(live.AGY_SANDBOX_EXECUTABLE),
            "-D",
            f"WORKDIR={workdir}",
            "-D",
            f"GEMINI_DIR={gemini_dir}",
            "-D",
            f"CONFIG_DIR={config_path.parent}",
            "-D",
            f"CONFIG_FILE={config_path}",
            "-D",
            f"AGY_BIN={python}",
            "-p",
            live.AGY_POLICY_SANDBOX_PROFILE,
            str(python),
            "-c",
            script,
            str(workdir),
            str(gemini_dir),
            str(config_path),
            str(alias),
            str(outside_source),
            str(outside),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                timeout=10,
            )
        finally:
            outside_source.unlink(missing_ok=True)
        if completed.returncode == 71 and "sandbox_apply: Operation not permitted" in (
            completed.stderr
        ):
            pytest.skip("nested sandbox execution is unavailable in this test host")
        assert completed.returncode == 0, completed.stderr
        results = json.loads(completed.stdout)
        assert results == {
            "appdata_allowed": True,
            "child_exec": True,
            "config_chmod": True,
            "config_create": True,
            "config_rename": True,
            "fork": True,
            "inward_hardlink": True,
            "mutate": True,
            "outside_write": True,
            "outward_hardlink": True,
            "root_rename": True,
            "symlink_mutate": True,
            "tmp_write_allowed": True,
            "unlink": True,
            "workdir_rename": True,
        }
        assert config_path.read_bytes() == live.AGY_SEALED_POLICY_CONFIG
        assert not outside.exists()


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


def test_process_group_survivor_is_detected_even_when_cleanup_succeeds(monkeypatch) -> None:
    class Process:
        pid = 424242

    calls: list[int] = []

    def fake_killpg(process_group: int, selected_signal: int) -> None:
        assert process_group == Process.pid
        calls.append(selected_signal)
        if len(calls) == 3:
            raise ProcessLookupError

    monkeypatch.setattr(live.os, "killpg", fake_killpg)
    detected, quiescent = asyncio.run(live._quiesce_agy_process_group(Process()))
    assert detected is True
    assert quiescent is True
    assert calls[:2] == [0, live.signal.SIGKILL]


def test_process_group_survivor_forces_generic_integrity_failure(tmp_path: Path) -> None:
    fixture = _structured_evidence(tmp_path)
    evidence = _analyze(
        fixture,
        process_group_descendant_detected=True,
        process_group_quiescent=True,
    )
    assert evidence.disposition == "evidence_integrity_failure"
    assert "process_group_descendant_detected" in evidence.capture_failures


def test_evidence_capture_cancellation_kills_session_and_reaps(monkeypatch) -> None:
    class HungProcess:
        def __init__(self) -> None:
            self.pid = 434343
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = None
            self.done: asyncio.Future[int] | None = None

        async def wait(self) -> int:
            if self.done is None:
                self.done = asyncio.get_running_loop().create_future()
            return await asyncio.shield(self.done)

        def kill(self) -> None:
            self.returncode = -9
            if self.done is not None and not self.done.done():
                self.done.set_result(-9)

    async def invoke() -> HungProcess:
        process = HungProcess()

        def fake_killpg(process_group: int, selected_signal: int) -> None:
            assert process_group == process.pid
            assert selected_signal == live.signal.SIGKILL
            process.kill()

        monkeypatch.setattr(live.os, "killpg", fake_killpg)
        task = asyncio.create_task(
            live._communicate_agy_evidence(process, timeout_seconds=60.0)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return process

    process = asyncio.run(invoke())
    assert process.returncode == -9


@pytest.mark.skipif(
    not live.AGY_SANDBOX_EXECUTABLE.is_file(),
    reason="requires the pinned sandbox executable receipt",
)
def test_spawn_cancellation_after_frontier_cleans_private_policy_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frontier: list[dict] = []

    async def cancelled_spawn(*_args, **_kwargs):
        assert len(frontier) == 1
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cancelled_spawn)

    async def invoke() -> None:
        with pytest.raises(asyncio.CancelledError):
            await live.call_llm_with_evidence(
                "/usr/bin/true",
                "gemini-3.7-flash-high",
                "sealed prompt",
                workdir=tmp_path,
                process_environment={"AGY_CLI_DISABLE_AUTO_UPDATE": "1"},
                before_spawn=lambda receipt: frontier.append(dict(receipt)),
            )

    asyncio.run(invoke())
    assert len(frontier) == 1
    assert list(tmp_path.iterdir()) == []


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
