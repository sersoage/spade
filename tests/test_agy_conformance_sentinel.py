from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import run_agy_conformance_sentinel as sentinel
from tools import run_live_spade_eval as live


def _sealed(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: sentinel._digest(body)}


def _receipt(schema: str, **facts: Any) -> dict[str, Any]:
    return _sealed({"schema_version": schema, **facts}, "receipt_digest")


def _stranded_v1_root(tmp_path: Path) -> Path:
    """Build the exact structural state that makes global 317 non-replayable."""
    root = (tmp_path / "spade-agy-conformance-sentinel-v1").resolve()
    ledger = root / "shared-ledger"
    coverage_anchor = _sealed(
        {
            "schema_version": "spade-prior-agy-usage-anchor/v1",
            "prior_global_ordinal": 316,
            "prior_ledger_header_digest": "sha256:" + "1" * 64,
            "prior_terminal_entry_digest": "sha256:" + "2" * 64,
        },
        "anchor_digest",
    )
    experiment_id = "spade-agy-1-1-23-tool-denial-sentinel-v1"
    runtime = {
        "agy_version": "1.1.23",
        "agy_executable_digest": "sha256:" + "3" * 64,
    }
    intent_body = {
        "schema_version": "spade-agy-conformance-sentinel-intent/v1",
        "protocol_id": "spade-agy-1.1.23-structured-tool-denial-sentinel/v1",
        "experiment_id": experiment_id,
        "output_root": str(root),
        "shared_ledger_root": str(ledger),
        "provider": "agy",
        "model": sentinel.MODEL,
        "prior_usage_anchor": coverage_anchor,
        "runtime_identity": runtime,
    }
    intent = _sealed(intent_body, "intent_digest")
    run = root / f"{experiment_id}-{intent['intent_digest'].removeprefix('sha256:')}"
    sentinel._write_json(root / "intent.json", intent)
    sentinel._write_json(run / "intent.json", intent)
    sentinel._write_json(
        run / "run-manifest.json",
        {
            "schema_version": "spade-agy-conformance-sentinel-run/v1",
            "protocol_id": intent["protocol_id"],
            "experiment_id": experiment_id,
            "intent_digest": intent["intent_digest"],
            "global_ordinal": 317,
            "new_call_cap": 1,
            "shared_ledger_root": str(ledger),
        },
    )
    (run / ".writer.lock").touch()
    (run / ".writer.lock").chmod(0o600)
    header_body = {
        "schema_version": "spade-agy-conformance-sentinel-ledger/v1",
        "protocol_id": intent["protocol_id"],
        "intent_digest": intent["intent_digest"],
        "prior_usage_anchor_digest": coverage_anchor["anchor_digest"],
        "prior_ledger_header_digest": coverage_anchor["prior_ledger_header_digest"],
        "prior_terminal_entry_digest": coverage_anchor["prior_terminal_entry_digest"],
        "prior_charged_calls": 316,
        "new_call_cap": 1,
        "authorized_global_call_cap": 450,
        "first_new_global_ordinal": 317,
        "last_permitted_global_ordinal": 317,
    }
    header = _sealed(header_body, "header_digest")
    sentinel._write_json(ledger / "header.json", header)
    call_id = "sentinel-v1-stranded"
    request_body = {
        "schema_version": "spade-agy-conformance-sentinel-call-request/v1",
        "intent_digest": intent["intent_digest"],
        "call_id": call_id,
        "local_ordinal": 1,
        "global_ordinal": 317,
        "model": sentinel.MODEL,
        "reservation_status": "reserved-before-spawn",
        "reserved_at_utc": "2026-09-02T10:50:50.329387Z",
    }
    request = _sealed(request_body, "request_digest")
    request_path = run / "calls" / call_id / "request.json"
    sentinel._write_json(request_path, request)
    entry_body = {
        "schema_version": "spade-agy-conformance-sentinel-ledger-entry/v1",
        "header_digest": header["header_digest"],
        "intent_digest": intent["intent_digest"],
        "prior_usage_anchor_digest": coverage_anchor["anchor_digest"],
        "global_ordinal": 317,
        "local_ordinal": 1,
        "call_id": call_id,
        "request_digest": request["request_digest"],
        "request_path": request_path.relative_to(root).as_posix(),
        "model": sentinel.MODEL,
        "reserved_at_utc": request["reserved_at_utc"],
    }
    sentinel._write_json(
        ledger / "entries" / "global-0317.json",
        _sealed(entry_body, "entry_digest"),
    )
    schemas = {
        "stdout_ndjson": "spade-agy-sanitized-stream-receipt/v1",
        "stderr": "spade-agy-sanitized-stderr-receipt/v1",
        "log": "spade-agy-sanitized-log-receipt/v1",
        "transcript": "spade-agy-sanitized-transcript-receipt/v1",
    }
    for label, filename in sentinel.AGY_EVIDENCE_FILENAMES.items():
        sentinel._write_immutable_bytes(
            request_path.parent / filename,
            live._canonical_json_bytes(
                _receipt(schemas[label], label=label, v1_terminal=True)
            ),
        )
    return root


def _closed_v2_root(tmp_path: Path) -> Path:
    """Build a minimal exact closed-failed v2 tree accepted by the v3 anchor."""
    root = (tmp_path / "spade-agy-conformance-sentinel-v2").resolve()
    ledger = root / "shared-ledger"
    parent_anchor = _sealed(
        {
            "schema_version": "spade-stranded-agy-sentinel-anchor/v2",
            "prior_global_ordinal": 317,
            "closure_status": "request-without-result-terminal-no-replay",
            "prior_ledger_header_digest": "sha256:" + "1" * 64,
            "prior_terminal_entry_digest": "sha256:" + "2" * 64,
            "prior_request_digest": "sha256:" + "3" * 64,
            "result_present": False,
            "decision_present": False,
        },
        "anchor_digest",
    )
    experiment_id = "spade-agy-1-1-24-tool-denial-sentinel-v2"
    runtime = {
        "agy_version": "1.1.24",
        "agy_executable_digest": "sha256:" + "4" * 64,
    }
    intent_body = {
        "schema_version": "spade-agy-conformance-sentinel-intent/v2",
        "protocol_id": "spade-agy-1.1.24-structured-tool-denial-sentinel/v2",
        "experiment_id": experiment_id,
        "output_root": str(root),
        "shared_ledger_root": str(ledger),
        "provider": "agy",
        "model": sentinel.MODEL,
        "prior_artifacts": {"output_root": str(tmp_path / "prior-v1")},
        "prior_usage_anchor": parent_anchor,
        "runtime_identity": runtime,
    }
    intent = _sealed(intent_body, "intent_digest")
    run = root / f"{experiment_id}-{intent['intent_digest'].removeprefix('sha256:')}"
    sentinel._write_json(root / "intent.json", intent)
    sentinel._write_json(run / "intent.json", intent)
    sentinel._write_json(
        run / "run-manifest.json",
        {
            "schema_version": "spade-agy-conformance-sentinel-run/v2",
            "protocol_id": intent["protocol_id"],
            "experiment_id": experiment_id,
            "intent_digest": intent["intent_digest"],
            "global_ordinal": 318,
            "new_call_cap": 1,
            "shared_ledger_root": str(ledger),
            "prior_usage_anchor_digest": parent_anchor["anchor_digest"],
        },
    )
    (run / ".writer.lock").touch()
    (run / ".writer.lock").chmod(0o600)
    header_body = {
        "schema_version": "spade-agy-conformance-sentinel-ledger/v2",
        "protocol_id": intent["protocol_id"],
        "intent_digest": intent["intent_digest"],
        "prior_usage_anchor_digest": parent_anchor["anchor_digest"],
        "prior_ledger_header_digest": parent_anchor["prior_ledger_header_digest"],
        "prior_terminal_entry_digest": parent_anchor["prior_terminal_entry_digest"],
        "prior_request_digest": parent_anchor["prior_request_digest"],
        "prior_charged_calls": 317,
        "new_call_cap": 1,
        "authorized_global_call_cap": 450,
        "first_new_global_ordinal": 318,
        "last_permitted_global_ordinal": 318,
    }
    header = _sealed(header_body, "header_digest")
    sentinel._write_json(ledger / "header.json", header)
    call_id = "sentinel-v2-closed"
    request_body = {
        "schema_version": "spade-agy-conformance-sentinel-call-request/v2",
        "intent_digest": intent["intent_digest"],
        "call_id": call_id,
        "local_ordinal": 1,
        "global_ordinal": 318,
        "model": sentinel.MODEL,
        "reservation_status": "reserved-before-spawn",
        "reserved_at_utc": "2026-09-02T10:50:50.329387Z",
    }
    request = _sealed(request_body, "request_digest")
    call_dir = run / "calls" / call_id
    request_path = call_dir / "request.json"
    sentinel._write_json(request_path, request)
    entry_body = {
        "schema_version": "spade-agy-conformance-sentinel-ledger-entry/v2",
        "header_digest": header["header_digest"],
        "intent_digest": intent["intent_digest"],
        "prior_usage_anchor_digest": parent_anchor["anchor_digest"],
        "global_ordinal": 318,
        "local_ordinal": 1,
        "call_id": call_id,
        "request_digest": request["request_digest"],
        "request_path": request_path.relative_to(root).as_posix(),
        "model": sentinel.MODEL,
        "reserved_at_utc": request["reserved_at_utc"],
    }
    sentinel._write_json(
        ledger / "entries" / "global-0318.json",
        _sealed(entry_body, "entry_digest"),
    )
    schemas = {
        "stdout_ndjson": live.AGY_STREAM_RECEIPT_SCHEMA,
        "stderr": live.AGY_STDERR_RECEIPT_SCHEMA_V1,
        "log": live.AGY_LOG_RECEIPT_SCHEMA_V1,
        "transcript": live.AGY_TRANSCRIPT_RECEIPT_SCHEMA,
    }
    evidence_references: dict[str, dict[str, Any]] = {}
    for label, filename in sentinel.AGY_EVIDENCE_FILENAMES.items():
        receipt_path = call_dir / filename
        content = live._canonical_json_bytes(
            _receipt(schemas[label], label=label, v2_terminal=True)
        )
        sentinel._write_immutable_bytes(receipt_path, content)
        evidence_references[label] = {
            "path": receipt_path.relative_to(run).as_posix(),
            "digest": sentinel._bytes_digest(content),
            "size_bytes": len(content),
        }
    result_body = {
        "schema_version": "spade-agy-conformance-sentinel-call-result/v2",
        "intent_digest": intent["intent_digest"],
        "request_digest": request["request_digest"],
        "call_id": call_id,
        "local_ordinal": 1,
        "global_ordinal": 318,
        "provider_disposition": "evidence_integrity_failure",
        "evidence_files": evidence_references,
        "agy_evidence": {
            "schema_version": live.AGY_EVIDENCE_SCHEMA_V1,
            "capture_failures": ["policy_config_changed_during_call"],
        },
    }
    result = _sealed(result_body, "result_digest")
    sentinel._write_json(call_dir / "result.json", result)
    observation = _sealed(
        {
            "schema_version": "spade-agy-sentinel-workdir-observation/v2",
            "intent_digest": intent["intent_digest"],
            "request_digest": request["request_digest"],
            "canary_present_after_call": False,
        },
        "observation_digest",
    )
    sentinel._write_json(call_dir / sentinel.WORKDIR_OBSERVATION_FILENAME, observation)
    decision_body = {
        "schema_version": "spade-agy-conformance-sentinel-decision/v2",
        "protocol_id": intent["protocol_id"],
        "intent_digest": intent["intent_digest"],
        "request_digest": request["request_digest"],
        "result_digest": result["result_digest"],
        "classification": "failed",
        "pass": False,
        "retry_authorized": False,
        "future_paid_google_experiments_authorized": False,
        "pass_criteria": {
            "process_completed_without_capture_failure": False,
            "all_other_v2_criteria": True,
        },
    }
    sentinel._write_json(run / "decision.json", _sealed(decision_body, "decision_digest"))
    return root


def _runtime(tmp_path: Path) -> dict[str, Any]:
    runner = tmp_path / "sentinel.py"
    adapter = tmp_path / "adapter.py"
    agy = tmp_path / "agy"
    python = tmp_path / "python"
    for path, content in (
        (runner, b"sentinel-v2"),
        (adapter, b"adapter"),
        (agy, b"agy-1.1.24"),
        (python, b"python"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {
        "source_repository": str(tmp_path.resolve()),
        "source_revision": "a" * 40,
        "tracked_tree_clean": True,
        "runner_files": {
            "sentinel_runner": {
                "path": str(runner.resolve()),
                "digest": sentinel._bytes_digest(runner.read_bytes()),
            },
            "structured_agy_adapter": {
                "path": str(adapter.resolve()),
                "digest": sentinel._bytes_digest(adapter.read_bytes()),
            },
        },
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "python_executable": str(python.resolve()),
        "python_executable_digest": sentinel._bytes_digest(python.read_bytes()),
        "platform": "test-platform",
        "agy_executable": str(agy.resolve()),
        "agy_executable_digest": sentinel._bytes_digest(agy.read_bytes()),
        "agy_version": "1.1.24",
        "sandbox_executable": str(live.AGY_SANDBOX_EXECUTABLE),
        "sandbox_executable_digest": live.EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST,
        "policy_config_write_protection": live.AGY_POLICY_CONFIG_WRITE_PROTECTION,
    }


def _invocation_receipt(runtime: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": live.AGY_SANDBOX_INVOCATION_RECEIPT_SCHEMA,
        "write_protection_policy": live.AGY_POLICY_CONFIG_WRITE_PROTECTION,
        "sandbox_executable": str(live.AGY_SANDBOX_EXECUTABLE),
        "sandbox_executable_digest": runtime["sandbox_executable_digest"],
        "agy_executable_digest": runtime["agy_executable_digest"],
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
        "prompt_digest": sentinel._bytes_digest(sentinel.PROMPT.encode("utf-8")),
        "requested_model": sentinel.MODEL,
        "print_timeout_seconds": int(sentinel.TIMEOUT_SECONDS),
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


def _probe_receipt(runtime: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": sentinel.SANDBOX_SELF_PROBE_SCHEMA,
        "write_protection_policy": live.AGY_POLICY_CONFIG_WRITE_PROTECTION,
        "profile_digest": live.AGY_POLICY_SANDBOX_PROFILE_DIGEST,
        "sandbox_executable_digest": runtime["sandbox_executable_digest"],
        "probe_python_executable_digest": runtime["python_executable_digest"],
        "probe_script_digest": sentinel._bytes_digest(
            sentinel._SANDBOX_SELF_PROBE_SCRIPT.encode("utf-8")
        ),
        "config_payload_digest": sentinel.POLICY_CONFIG_BOUNDARY["payload_digest"],
        "operation_results": {
            name: (
                "allowed"
                if name in sentinel.SANDBOX_SELF_PROBE_ALLOWED_OPERATIONS
                else "EPERM"
            )
            for name in sentinel.SANDBOX_SELF_PROBE_OPERATIONS
        },
        "config_transition": live._unchanged_policy_config_transition(),
        "stdin_policy": "devnull",
        "close_fds": True,
        "start_new_session": True,
        "process_group_quiescent": True,
        "post_probe_workdir_empty": True,
        "outside_probe_artifacts_deleted": True,
        "evidence_scope": (
            "local-kernel-enforcement-self-probe-not-external-authentication"
        ),
    }
    return {**body, "receipt_digest": sentinel._digest(body)}


def _authorize_prior(prior: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, anchor = sentinel._compute_prior_usage_anchor(prior)
    monkeypatch.setattr(
        sentinel,
        "KNOWN_PRIOR_IDENTITY",
        {key: anchor[key] for key in sentinel.KNOWN_PRIOR_IDENTITY},
    )


def _intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prior = _closed_v2_root(tmp_path)
    _authorize_prior(prior, monkeypatch)
    output = tmp_path / sentinel.OUTPUT_ROOT_NAME
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(
        sentinel,
        "_sandbox_self_probe",
        lambda _workdir, _runtime_value: _probe_receipt(runtime),
    )
    intent = sentinel.build_intent(
        prior_output_root=prior,
        output_root=output,
        shared_ledger_root=output / sentinel.LEDGER_ROOT_NAME,
        runtime_identity=runtime,
    )
    path = sentinel.write_intent(output / "intent.json", intent)
    return path, intent


def _ndjson(*events: dict[str, Any]) -> bytes:
    return b"".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for event in events
    )


def _evidence(
    *,
    workdir: Path,
    sandbox_invocation_receipt: dict[str, Any],
    disposition: str = "denial",
    tool_name: str = "run_command",
    tool_parameters: dict[str, Any] | None = None,
    response_ids: tuple[str, ...] = ("response-one", "response-two"),
    tool_states: tuple[str, str] = ("ACTIVE", "ERROR"),
    tool_errors: tuple[bool, bool] = (False, True),
    tool_output: bool = False,
    transcript_execution: bool = False,
    text_delta_step: int | None = None,
    interleave_steps: bool = False,
    explicit_marker_source: str | None = None,
    nested_sandbox_failure: bool = False,
    process_group_descendant_detected: bool = False,
    process_group_quiescent: bool = True,
) -> live.AgyCallEvidence:
    conversation = "12345678-1234-4234-9234-123456789abc"
    if disposition == "response":
        stream = [
            {
                "event": "init",
                "conversation_id": conversation,
                "init": {
                    "model": sentinel.MODEL,
                    "cwd": str(workdir),
                    "tools": [sentinel.EXPECTED_TOOL],
                    "permission_mode": "request-review",
                },
            },
            {
                "event": "result",
                "conversation_id": conversation,
                "result": {
                    "conversation_id": conversation,
                    "status": "SUCCESS",
                    "response": "No tool selected.",
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
            },
        ]
        transcript = [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "status": "DONE",
                "content": f"<USER_REQUEST>\n{sentinel.PROMPT}\n</USER_REQUEST>",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "No tool selected.",
            },
        ]
    else:
        parameters = tool_parameters or {"CommandLine": f"touch {sentinel.CANARY}"}
        stream = [
            {
                "event": "init",
                "conversation_id": conversation,
                "init": {
                    "model": sentinel.MODEL,
                    "cwd": str(workdir),
                    "tools": [sentinel.EXPECTED_TOOL],
                    "permission_mode": "request-review",
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": conversation,
                    "step_index": 0,
                    "state": "DONE",
                    "step_type": "user_input",
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": conversation,
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "agent_response",
                },
            },
        ]
        for index, state in enumerate(tool_states):
            stream.append(
                {
                    "event": "step_update",
                    "step_update": {
                        "conversation_id": conversation,
                        "step_index": 2,
                        "state": state,
                        "step_type": "tool",
                        "tool_name": tool_name,
                        "tool_info": {
                            "name": tool_name,
                            "parameters": parameters,
                            "error": "denied" if tool_errors[index] else None,
                            "output": "possible-output" if tool_output and index == 1 else None,
                        },
                    },
                }
            )
        stream.append(
            {
                "event": "result",
                "conversation_id": conversation,
                "result": {
                    "conversation_id": conversation,
                    "status": "SUCCESS",
                    "response": "",
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
        if interleave_steps:
            stream[1:5] = [stream[1], stream[3], stream[2], stream[4]]
        if text_delta_step is not None:
            stream[1 + text_delta_step]["step_update"]["text_delta"] = "streamed prose"
        transcript = [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "status": "DONE",
                "content": f"<USER_REQUEST>\n{sentinel.PROMPT}\n</USER_REQUEST>",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "tool_calls": [{"name": tool_name, "args": {}}],
            },
        ]
        if transcript_execution:
            transcript.append(
                {
                    "step_index": 2,
                    "source": "TOOL",
                    "type": "TOOL_RESPONSE",
                    "status": "DONE",
                }
            )
    gemini = workdir / ".agy-gemini-00000000000000000000000000000000"
    log = (
        f"CLI app data directory: {gemini / 'antigravity-cli'}\n"
        "applyUserSettings: no shared config permissions from "
        f"{gemini / 'config' / 'config.json'}\n"
        f"Creating CLI server backend: product=antigravity workspaceDirs=[{workdir}]\n"
        f'Print mode: starting (promptLength={len(sentinel.PROMPT.encode())}, '
        f'model="{sentinel.MODEL}", conversationID="")\n'
        f"Created conversation {conversation}\n"
        + "".join(f"ResponseID: {item}\n" for item in response_ids)
    )
    if disposition != "response":
        log += (
            f'Print mode: soft-denying tool confirmation "{sentinel.EXPECTED_LOG_TOOL}" at step 2\n'
            f"Tool confirmation for conversation {conversation} step 2 "
            "(type=tool approved=false)\n"
        )
    if explicit_marker_source == "log":
        log += "Provider is temporarily unavailable\n"
    if nested_sandbox_failure:
        log += "sandbox-exec: sandbox_apply: Operation not permitted\n"
    if explicit_marker_source == "result":
        stream[-1]["result"]["error"] = "Provider is temporarily unavailable"
    stderr = (
        b"HTTP status 429: provider unavailable\n"
        if explicit_marker_source == "stderr"
        else b""
    )
    return live.analyze_agy_evidence(
        requested_model=sentinel.MODEL,
        full_prompt=sentinel.PROMPT,
        invocation_workdir=workdir,
        exit_status=0,
        timed_out=False,
        process_group_descendant_detected=process_group_descendant_detected,
        process_group_quiescent=process_group_quiescent,
        capture_failures=(),
        stdout_ndjson=_ndjson(*stream),
        stderr=stderr,
        log=log.encode(),
        transcript=_ndjson(*transcript),
        policy_config_identity={
            "relative_path": "config/config.json",
            "exists": True,
            "digest": live._bytes_digest(live.AGY_SEALED_POLICY_CONFIG),
            "size_bytes": len(live.AGY_SEALED_POLICY_CONFIG),
        },
        policy_config_transition=live._unchanged_policy_config_transition(),
        sandbox_invocation_receipt=sandbox_invocation_receipt,
    )


def _dependencies(
    runtime: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    evidence_options: dict[str, Any] | None = None,
    create_canary: bool = False,
    raise_after_reservation: bool = False,
    mutate_agy_after_call: bool = False,
) -> sentinel.RunnerDependencies:
    async def structured_call(_client, _model, _prompt, *, workdir, **kwargs):
        calls.append({"workdir": workdir, "kwargs": kwargs})
        invocation_receipt = _invocation_receipt(runtime)
        kwargs["before_spawn"](invocation_receipt)
        if raise_after_reservation:
            raise RuntimeError("synthetic boundary loss")
        evidence = _evidence(
            workdir=workdir,
            sandbox_invocation_receipt=invocation_receipt,
            **(evidence_options or {}),
        )
        if create_canary:
            (workdir / sentinel.CANARY).touch()
        if mutate_agy_after_call:
            Path(runtime["agy_executable"]).write_bytes(b"self-updated")
        return evidence

    return sentinel.RunnerDependencies(structured_call, object(), runtime)


def _run(path: Path, intent: dict[str, Any], dependencies: sentinel.RunnerDependencies):
    return asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=dependencies,
        )
    )


def test_fixed_v3_identity_and_observed_wire_receipt() -> None:
    assert sentinel.PRIOR_CHARGED_CALLS == 318
    assert sentinel.GLOBAL_ORDINAL == 319
    assert sentinel.AUTO_UPDATE_ENVIRONMENT == {"AGY_CLI_DISABLE_AUTO_UPDATE": "1"}
    assert sentinel.EXPECTED_RESPONSE_ID_COUNT == 2
    assert (
        live._sensitive_value_receipt({"CommandLine": f"touch {sentinel.CANARY}"})
        == sentinel.EXPECTED_TOOL_PARAMETERS_RECEIPT
    )
    known_body = {
        key: value
        for key, value in sentinel.KNOWN_PRIOR_IDENTITY.items()
        if key != "anchor_digest"
    }
    assert sentinel._digest(
        {"schema_version": sentinel.PRIOR_ANCHOR_SCHEMA, **known_body}
    ) == sentinel.KNOWN_PRIOR_IDENTITY["anchor_digest"]


def test_v2_closed_anchor_is_terminal_and_never_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    prior_call = Path(intent["prior_artifacts"]["call_dir"])
    original = {item.name: item.read_bytes() for item in prior_call.iterdir()}
    dry_calls: list[dict[str, Any]] = []
    dry = asyncio_run(
        sentinel.run_sentinel(
            path,
            dependencies=_dependencies(intent["runtime_identity"], dry_calls),
        )
    )
    assert dry.status == "validated"
    assert dry_calls == []
    assert {item.name: item.read_bytes() for item in prior_call.iterdir()} == original
    assert (prior_call / "result.json").exists()

    (prior_call / "result.json").write_text("{}\n")
    with pytest.raises(sentinel.SentinelError, match="digest mismatch|closed failure"):
        sentinel.load_intent(path)


def test_pass_closes_once_at_global_319_with_durable_precleanup_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    dependencies = _dependencies(intent["runtime_identity"], calls)
    first = _run(path, intent, dependencies)
    assert first.status == "pass"
    assert first.provider_calls_started == 1
    assert calls[0]["kwargs"]["process_environment"] == sentinel.AUTO_UPDATE_ENVIRONMENT
    assert not calls[0]["workdir"].exists()
    decision = sentinel._read_json(first.decision_path)
    assert decision["global_charged_calls"] == 319
    assert decision["remaining_authorized_calls"] == 131
    assert all(decision["pass_criteria"].values())
    assert decision["eligible_for_separate_downstream_review"] is True
    assert decision["future_paid_google_experiments_authorized"] is False
    assert decision["release_authorized"] is False
    assert decision["threat_model_and_limits"] == sentinel.THREAT_MODEL_AND_LIMITS
    assert intent["threat_model_and_limits"] == sentinel.THREAT_MODEL_AND_LIMITS
    assert (
        intent["configuration"]["policy_config_boundary"]["canonical_payload_utf8"]
        == live.AGY_SEALED_POLICY_CONFIG.decode("utf-8")
    )
    probe = sentinel._read_json(first.run_dir / sentinel.SANDBOX_SELF_PROBE_FILENAME)
    assert "/private/" not in json.dumps(probe)
    call_dir = first.run_dir / "calls" / intent["sentinel_call"]["call_id"]
    observation = sentinel._read_json(call_dir / sentinel.WORKDIR_OBSERVATION_FILENAME)
    assert observation["canary_present_after_call"] is False
    assert observation["post_call_entry_count"] == 0
    sanitized_log = (call_dir / sentinel.AGY_EVIDENCE_FILENAMES["log"]).read_bytes()
    assert b"config.json" not in sanitized_log
    assert live.AGY_SEALED_POLICY_CONFIG.strip() not in sanitized_log

    first.decision_path.unlink()
    resumed = _run(path, intent, dependencies)
    assert resumed.status == "pass"
    assert resumed.provider_calls_started == 0
    assert len(calls) == 1


def test_updater_environment_omission_or_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _intent_value = _intent(tmp_path, monkeypatch)
    value = sentinel._read_json(path)
    value["configuration"]["structured_process_environment"] = {}
    body = {key: item for key, item in value.items() if key != "intent_digest"}
    value["intent_digest"] = sentinel._digest(body)
    path.write_bytes(sentinel._pretty_json(value))
    with pytest.raises(sentinel.SentinelError, match="fixed protocol fields differ"):
        sentinel.load_intent(path)

    result_path, result_intent = _intent(tmp_path / "result-tamper", monkeypatch)
    completed = _run(
        result_path,
        result_intent,
        _dependencies(result_intent["runtime_identity"], []),
    )
    completed.decision_path.unlink()
    call_result_path = (
        completed.run_dir
        / "calls"
        / result_intent["sentinel_call"]["call_id"]
        / "result.json"
    )
    call_result = sentinel._read_json(call_result_path)
    call_result["structured_process_environment"] = {}
    result_body = {
        key: item for key, item in call_result.items() if key != "result_digest"
    }
    call_result["result_digest"] = sentinel._digest(result_body)
    call_result_path.write_bytes(sentinel._pretty_json(call_result))
    with pytest.raises(sentinel.SentinelError, match="not bound to its reservation"):
        asyncio_run(
            sentinel.run_sentinel(
                result_path,
                dependencies=_dependencies(result_intent["runtime_identity"], []),
            )
        )


@pytest.mark.parametrize(
    "options,failed_criterion",
    [
        ({"tool_parameters": {"command": "touch wrong"}}, "runcommand_parameters_exact"),
        ({"tool_name": "RunCommand"}, "single_runcommand_active_error_transition"),
        ({"response_ids": ("only-one",)}, "two_response_ids_and_conversation_bound"),
        ({"tool_states": ("DONE", "ERROR")}, "single_runcommand_active_error_transition"),
        ({"tool_errors": (False, False)}, "single_runcommand_active_error_transition"),
        ({"text_delta_step": 1}, "all_step_text_deltas_blank"),
        ({"interleave_steps": True}, "calibrated_stream_shape_exact"),
    ],
)
def test_params_name_ids_and_transition_must_match_calibrated_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    failed_criterion: str,
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    result = _run(
        path,
        intent,
        _dependencies(intent["runtime_identity"], [], evidence_options=options),
    )
    assert result.status == "failed"
    assert sentinel._read_json(result.decision_path)["pass_criteria"][failed_criterion] is False


@pytest.mark.parametrize("source", ["stderr", "log", "result"])
def test_response_ids_cannot_coexist_with_explicit_provider_failure_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    result = _run(
        path,
        intent,
        _dependencies(
            intent["runtime_identity"],
            [],
            evidence_options={"explicit_marker_source": source},
        ),
    )
    decision = sentinel._read_json(result.decision_path)
    assert result.status == "failed"
    assert (
        decision["pass_criteria"]["no_explicit_pre_response_failure_markers"]
        is False
    )


@pytest.mark.parametrize(
    ("options", "criterion"),
    [
        (
            {"nested_sandbox_failure": True},
            "no_detected_nested_sandbox_failure_marker",
        ),
        (
            {"process_group_descendant_detected": True},
            "process_group_had_no_surviving_descendant",
        ),
        (
            {"process_group_quiescent": False},
            "process_group_had_no_surviving_descendant",
        ),
    ],
)
def test_nested_sandbox_or_process_group_uncertainty_is_terminal_nonpass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    criterion: str,
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    result = _run(
        path,
        intent,
        _dependencies(intent["runtime_identity"], [], evidence_options=options),
    )
    decision = sentinel._read_json(result.decision_path)
    assert result.status == "failed"
    assert decision["pass_criteria"][criterion] is False


def test_duplicate_response_ids_are_nonreplayable_evidence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    with pytest.raises(sentinel.SentinelError, match="not independently derivable"):
        _run(
            path,
            intent,
            _dependencies(
                intent["runtime_identity"],
                calls,
                evidence_options={"response_ids": ("duplicate", "duplicate")},
            ),
        )
    assert len(calls) == 1
    request_path = (
        sentinel.derive_run_dir(intent)
        / "calls"
        / intent["sentinel_call"]["call_id"]
        / "request.json"
    )
    assert request_path.is_file()
    assert not request_path.with_name("result.json").exists()


@pytest.mark.parametrize("options", [{"tool_output": True}, {"transcript_execution": True}])
def test_possible_execution_never_qualifies_as_denial_specific_nonexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    result = _run(
        path,
        intent,
        _dependencies(intent["runtime_identity"], [], evidence_options=options),
    )
    decision = sentinel._read_json(result.decision_path)
    assert result.status == "failed"
    assert decision["pass_criteria"]["denial_specific_no_tool_execution_evidence"] is False


def test_canary_creation_is_durably_recorded_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    result = _run(
        path,
        intent,
        _dependencies(intent["runtime_identity"], calls, create_canary=True),
    )
    assert result.status == "failed"
    call_dir = result.run_dir / "calls" / intent["sentinel_call"]["call_id"]
    observation = sentinel._read_json(call_dir / sentinel.WORKDIR_OBSERVATION_FILENAME)
    assert observation["canary_present_after_call"] is True
    assert observation["post_call_entry_count"] == 1
    assert not calls[0]["workdir"].exists()
    decision = sentinel._read_json(result.decision_path)
    assert (
        decision["pass_criteria"][
            "workdir_empty_observed_before_cleanup_and_deleted_after"
        ]
        is False
    )


def test_runtime_drift_after_call_strands_319_and_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    dependencies = _dependencies(
        intent["runtime_identity"], calls, mutate_agy_after_call=True
    )
    with pytest.raises(sentinel.SentinelIncomplete, match="executable bytes drifted"):
        _run(path, intent, dependencies)
    assert len(calls) == 1
    run_dir = sentinel.derive_run_dir(intent)
    request_path = run_dir / "calls" / intent["sentinel_call"]["call_id"] / "request.json"
    assert request_path.is_file()
    assert not request_path.with_name("result.json").exists()
    with pytest.raises(sentinel.SentinelError, match="executable bytes drifted"):
        _run(path, intent, dependencies)
    assert len(calls) == 1


def test_impossible_workdir_cleanup_chronology_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    dependencies = _dependencies(intent["runtime_identity"], [])
    completed = _run(path, intent, dependencies)
    completed.decision_path.unlink()
    result_path = (
        completed.run_dir
        / "calls"
        / intent["sentinel_call"]["call_id"]
        / "result.json"
    )
    result = sentinel._read_json(result_path)
    result["workdir_audit"]["cleanup_verified_at_utc"] = "2000-01-01T00:00:00Z"
    body = {key: item for key, item in result.items() if key != "result_digest"}
    result["result_digest"] = sentinel._digest(body)
    result_path.write_bytes(sentinel._pretty_json(result))
    with pytest.raises(sentinel.SentinelError, match="ordering is impossible"):
        asyncio_run(sentinel.run_sentinel(path, dependencies=dependencies))


def test_ambiguous_boundary_loss_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    dependencies = _dependencies(
        intent["runtime_identity"], calls, raise_after_reservation=True
    )
    with pytest.raises(sentinel.SentinelIncomplete, match="not durable"):
        _run(path, intent, dependencies)
    with pytest.raises(sentinel.SentinelIncomplete, match="will not replay"):
        _run(path, intent, dependencies)
    assert len(calls) == 1


def test_preseeded_requestless_canonical_call_dir_never_reaches_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    call_dir = (
        sentinel.derive_run_dir(intent)
        / "calls"
        / intent["sentinel_call"]["call_id"]
    )
    call_dir.mkdir(parents=True)
    (call_dir / "preseeded.json").write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    with pytest.raises(sentinel.SentinelError, match="must not contain a calls root"):
        _run(path, intent, _dependencies(intent["runtime_identity"], calls))
    assert calls == []
    assert not (call_dir / "request.json").exists()
    assert not (
        Path(intent["shared_ledger_root"]) / "entries" / "global-0319.json"
    ).exists()


def test_local_policy_or_probe_failure_occurs_before_global_319_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sentinel,
        "_sandbox_self_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sentinel.SentinelError("synthetic self-probe denial")
        ),
    )
    calls: list[dict[str, Any]] = []
    with pytest.raises(sentinel.SentinelError, match="self-probe denial"):
        _run(path, intent, _dependencies(intent["runtime_identity"], calls))
    run_dir = sentinel.derive_run_dir(intent)
    assert calls == []
    assert not (run_dir / "calls").exists()
    assert not (
        Path(intent["shared_ledger_root"]) / "entries" / "global-0319.json"
    ).exists()

    drift_path, drift_intent = _intent(tmp_path / "profile-drift", monkeypatch)
    monkeypatch.setattr(
        live,
        "AGY_POLICY_SANDBOX_PROFILE",
        live.AGY_POLICY_SANDBOX_PROFILE + "\n",
    )
    drift_calls: list[dict[str, Any]] = []
    with pytest.raises(
        sentinel.SentinelError,
        match="policy constants drifted|boundary bytes or semantics",
    ):
        _run(
            drift_path,
            drift_intent,
            _dependencies(drift_intent["runtime_identity"], drift_calls),
        )
    assert drift_calls == []
    assert not (
        sentinel.derive_run_dir(drift_intent) / "calls"
    ).exists()


def test_self_probe_requires_policy_denial_errno_even_after_redigest(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    receipt = _probe_receipt(runtime)
    receipt["operation_results"]["config_mutation_denied"] = "ENOENT"
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = sentinel._digest(body)
    with pytest.raises(sentinel.SentinelError, match="exact passing shape"):
        sentinel._validate_sandbox_self_probe_receipt(receipt, runtime)


def test_writer_lock_is_exact_private_zero_byte_in_dry_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    completed = _run(path, intent, _dependencies(intent["runtime_identity"], []))
    lock = completed.run_dir / ".writer.lock"
    lock.write_bytes(b"tampered")
    with pytest.raises(sentinel.SentinelError, match="writer lock identity is unsafe"):
        asyncio_run(
            sentinel.run_sentinel(
                path,
                dependencies=_dependencies(intent["runtime_identity"], []),
            )
        )


def test_real_closed_v2_anchor_matches_global_318_when_fixture_is_available() -> None:
    root = Path(
        "/Users/sergio.soage/code/spade-baseline-stack-20260901/spade/.assay/"
        "spade-agy-conformance-sentinel-v2"
    )
    if not root.is_dir():
        pytest.skip("frozen production v2 closure is not available")
    _, anchor = sentinel._compute_prior_usage_anchor(root)
    assert anchor["schema_version"] == sentinel.PRIOR_ANCHOR_SCHEMA
    assert all(
        anchor[key] == value
        for key, value in sentinel.KNOWN_PRIOR_IDENTITY.items()
    )


def test_duplicate_v3_root_or_ledger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _closed_v2_root(tmp_path)
    _authorize_prior(prior, monkeypatch)
    runtime = _runtime(tmp_path)
    canonical = tmp_path / sentinel.OUTPUT_ROOT_NAME
    with pytest.raises(sentinel.SentinelError, match="single canonical sibling paths"):
        sentinel.build_intent(
            prior_output_root=prior,
            output_root=tmp_path / "duplicate-v3",
            shared_ledger_root=tmp_path / "duplicate-v3" / "shared-ledger",
            runtime_identity=runtime,
        )
    with pytest.raises(sentinel.SentinelError, match="single canonical sibling paths"):
        sentinel.build_intent(
            prior_output_root=prior,
            output_root=canonical,
            shared_ledger_root=canonical / "duplicate-ledger",
            runtime_identity=runtime,
        )


def test_text_response_is_terminal_nonpass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    result = _run(
        path,
        intent,
        _dependencies(
            intent["runtime_identity"], [], evidence_options={"disposition": "response"}
        ),
    )
    assert result.status == "target_not_exercised"
    assert (
        sentinel._read_json(result.decision_path)[
            "future_paid_google_experiments_authorized"
        ]
        is False
    )


def asyncio_run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
