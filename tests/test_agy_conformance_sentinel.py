from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import run_agy_conformance_sentinel as sentinel
from tools import run_live_spade_eval as live


def _sealed(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: sentinel._digest(body)}


def _legacy_root(tmp_path: Path) -> Path:
    root = tmp_path / "legacy"
    ledger = root / "shared-ledger"
    intent_body = {
        "schema_version": "spade-coverage-forced-generation-intent/v1",
        "protocol_id": "spade-google-coverage-forced-matched-swap-pilot/v1",
        "experiment_id": "legacy-proxy",
        "output_root": str(root),
        "shared_ledger_root": str(ledger),
        "provider": "agy",
    }
    intent = _sealed(intent_body, "intent_digest")
    run = root / f"legacy-proxy-{intent['intent_digest'].removeprefix('sha256:')}"
    sentinel._write_json(root / "intent.json", intent)
    sentinel._write_json(run / "generation-intent.json", intent)
    actor = _sealed(
        {
            "schema_version": "spade-coverage-forced-actor-plan/v1",
            "intent_digest": intent["intent_digest"],
        },
        "actor_plan_digest",
    )
    sentinel._write_json(run / "actor-plan.json", actor)
    sentinel._write_json(
        run / "run-manifest.json",
        {
            "schema_version": "spade-coverage-forced-run/v1",
            "intent_digest": intent["intent_digest"],
        },
    )
    header_body = {
        "schema_version": "spade-shared-agy-ledger/v1",
        "protocol_id": intent["protocol_id"],
        "intent_digest": intent["intent_digest"],
        "prior_charged_calls": 205,
        "new_call_cap": 208,
        "authorized_global_call_cap": 450,
        "first_new_global_ordinal": 206,
        "last_permitted_global_ordinal": 413,
    }
    header = _sealed(header_body, "header_digest")
    sentinel._write_json(ledger / "header.json", header)
    for local in range(1, 112):
        global_ordinal = 205 + local
        call_id = f"legacy-{local:03d}"
        reserved_at = f"2026-09-02T00:{local % 60:02d}.000000Z"
        request_body = {
            "schema_version": "spade-coverage-forced-call-request/v1",
            "intent_digest": intent["intent_digest"],
            "call_id": call_id,
            "local_ordinal": local,
            "global_ordinal": global_ordinal,
            "model": "gemini-3.7-flash-high",
            "reserved_at_utc": reserved_at,
        }
        request = _sealed(request_body, "request_digest")
        request_path = run / "calls" / call_id / "request.json"
        sentinel._write_json(request_path, request)
        result_body = {
            "schema_version": "spade-coverage-forced-call-result/v1",
            "intent_digest": intent["intent_digest"],
            "call_id": call_id,
            "local_ordinal": local,
            "global_ordinal": global_ordinal,
            "request_digest": request["request_digest"],
            "status": "success",
        }
        sentinel._write_json(
            request_path.with_name("result.json"), _sealed(result_body, "result_digest")
        )
        entry_body = {
            "schema_version": "spade-shared-agy-ledger-entry/v1",
            "header_digest": header["header_digest"],
            "intent_digest": intent["intent_digest"],
            "global_ordinal": global_ordinal,
            "local_ordinal": local,
            "call_id": call_id,
            "request_digest": request["request_digest"],
            "request_path": request_path.relative_to(root).as_posix(),
            "model": request["model"],
            "reserved_at_utc": reserved_at,
        }
        sentinel._write_json(
            ledger / "entries" / f"global-{global_ordinal:04d}.json",
            _sealed(entry_body, "entry_digest"),
        )
    return root


def _runtime(tmp_path: Path) -> dict[str, Any]:
    runner = tmp_path / "sentinel.py"
    adapter = tmp_path / "adapter.py"
    agy = tmp_path / "agy"
    python = tmp_path / "python"
    for path, content in (
        (runner, b"sentinel"),
        (adapter, b"adapter"),
        (agy, b"agy"),
        (python, b"python"),
    ):
        path.write_bytes(content)
    return {
        "source_repository": str(tmp_path),
        "source_revision": "a" * 40,
        "tracked_tree_clean": True,
        "runner_files": {
            "sentinel_runner": {
                "path": str(runner),
                "digest": sentinel._bytes_digest(runner.read_bytes()),
            },
            "structured_agy_adapter": {
                "path": str(adapter),
                "digest": sentinel._bytes_digest(adapter.read_bytes()),
            },
        },
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "python_executable": str(python),
        "python_executable_digest": sentinel._bytes_digest(python.read_bytes()),
        "platform": "test-platform",
        "agy_executable": str(agy),
        "agy_executable_digest": sentinel._bytes_digest(agy.read_bytes()),
        "agy_version": "1.1.23",
    }


def _intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    prior = _legacy_root(tmp_path)
    _authorize_prior(prior, monkeypatch)
    output = tmp_path / sentinel.OUTPUT_ROOT_NAME
    intent = sentinel.build_intent(
        prior_output_root=prior,
        output_root=output,
        shared_ledger_root=output / sentinel.LEDGER_ROOT_NAME,
        runtime_identity=_runtime(tmp_path),
    )
    path = sentinel.write_intent(output / "intent.json", intent)
    return path, intent


def _authorize_prior(prior: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, anchor = sentinel._compute_prior_usage_anchor(prior)
    monkeypatch.setattr(
        sentinel,
        "KNOWN_PRIOR_IDENTITY",
        {key: anchor[key] for key in sentinel.KNOWN_PRIOR_IDENTITY},
    )


def _ndjson(*events: dict[str, Any]) -> bytes:
    return b"".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for event in events
    )


def _evidence(
    *,
    workdir: Path,
    disposition: str = "tool_policy_no_action",
    tool_command: str | None = None,
    tool_prose: str = "",
) -> live.AgyCallEvidence:
    conversation = "12345678-1234-4234-9234-123456789abc"
    stream: list[dict[str, Any]] = [
        {
            "event": "init",
            "conversation_id": conversation,
            "init": {
                "model": sentinel.MODEL,
                "cwd": str(workdir),
                "tools": [sentinel.EXPECTED_TOOL],
                "permission_mode": "request-review",
            },
        }
    ]
    transcript: list[dict[str, Any]] = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": f"<USER_REQUEST>\n{sentinel.PROMPT}\n</USER_REQUEST>",
        }
    ]
    if disposition == "tool_policy_no_action":
        stream.append(
            {
                "event": "step_update",
                "conversation_id": conversation,
                "step_update": {
                    "conversation_id": conversation,
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "tool",
                    "tool_name": sentinel.EXPECTED_TOOL,
                    "tool_info": {
                        "name": sentinel.EXPECTED_TOOL,
                        "parameters": {"command": tool_command or f"touch {sentinel.CANARY}"},
                    },
                },
            }
        )
        transcript.append(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "tool_calls": [{"name": sentinel.EXPECTED_TOOL, "args": {}}],
                **({"content": tool_prose} if tool_prose else {}),
            }
        )
        response = tool_prose
    else:
        response = "I will not invoke the requested tool."
        transcript.append(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": response,
            }
        )
    stream.append(
        {
            "event": "result",
            "conversation_id": conversation,
            "result": {
                "conversation_id": conversation,
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
    gemini = workdir / ".agy-gemini-00000000000000000000000000000000"
    log = (
        f"CLI app data directory: {gemini / 'antigravity-cli'}\n"
        "applyUserSettings: no shared config permissions from "
        f"{gemini / 'config' / 'config.json'}\n"
        f"Creating CLI server backend: product=antigravity workspaceDirs=[{workdir}]\n"
        f'Print mode: starting (promptLength=279, model="{sentinel.MODEL}", '
        'conversationID="")\n'
        f"Created conversation {conversation}\n"
        "ResponseID: response-id-1\n"
    )
    if disposition == "tool_policy_no_action":
        log += (
            f'Print mode: soft-denying tool confirmation "{sentinel.EXPECTED_TOOL}" at step 2\n'
            f"Tool confirmation for conversation {conversation} step 2 "
            "(type=tool approved=false)\n"
        )
    return live.analyze_agy_evidence(
        requested_model=sentinel.MODEL,
        full_prompt=sentinel.PROMPT,
        invocation_workdir=workdir,
        exit_status=0,
        timed_out=False,
        capture_failures=(),
        stdout_ndjson=_ndjson(*stream),
        stderr=b"",
        log=log.encode(),
        transcript=_ndjson(*transcript),
        policy_config_identity={
            "relative_path": "config/config.json",
            "exists": False,
            "digest": None,
            "size_bytes": 0,
        },
    )


def _dependencies(
    runtime: dict[str, Any],
    calls: list[Path],
    *,
    disposition: str = "tool_policy_no_action",
    tool_command: str | None = None,
    tool_prose: str = "",
    create_canary: bool = False,
    raise_after_reservation: bool = False,
) -> sentinel.RunnerDependencies:
    async def structured_call(_client, _model, _prompt, *, workdir, **_kwargs):
        calls.append(workdir)
        if raise_after_reservation:
            raise RuntimeError("synthetic boundary loss")
        evidence = _evidence(
            workdir=workdir,
            disposition=disposition,
            tool_command=tool_command,
            tool_prose=tool_prose,
        )
        if create_canary:
            (workdir / sentinel.CANARY).touch()
        return evidence

    return sentinel.RunnerDependencies(structured_call, object(), runtime)


def test_fixed_prompt_and_call_identity() -> None:
    assert len(sentinel.PROMPT.encode()) == 279
    assert sentinel._digest(sentinel.PROMPT) == (
        "sha256:d14d923f936f5972c3108c10c3fc0ab7d5a7b0c01b0bf7948c2955ffd362cdbf"
    )
    assert sentinel._digest(sentinel.PURPOSE) == (
        "sha256:34960d342a9aeaf7d60bf0b89578edf09294f5e18e45a2cf0cab12bda2106219"
    )
    assert (
        live._sensitive_value_receipt({"command": f"touch {sentinel.CANARY}"})
        == sentinel.EXPECTED_TOOL_PARAMETERS_RECEIPT
    )


def test_build_and_load_reject_every_noncanonical_duplicate_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _legacy_root(tmp_path)
    _authorize_prior(prior, monkeypatch)
    runtime = _runtime(tmp_path)
    canonical = tmp_path / sentinel.OUTPUT_ROOT_NAME
    with pytest.raises(sentinel.SentinelError, match="single canonical sibling paths"):
        sentinel.build_intent(
            prior_output_root=prior,
            output_root=tmp_path / "duplicate-sentinel",
            shared_ledger_root=tmp_path / "duplicate-sentinel" / "shared-ledger",
            runtime_identity=runtime,
        )
    with pytest.raises(sentinel.SentinelError, match="single canonical sibling paths"):
        sentinel.build_intent(
            prior_output_root=prior,
            output_root=canonical,
            shared_ledger_root=canonical / "second-ledger",
            runtime_identity=runtime,
        )

    intent = sentinel.build_intent(
        prior_output_root=prior,
        output_root=canonical,
        shared_ledger_root=canonical / sentinel.LEDGER_ROOT_NAME,
        runtime_identity=runtime,
    )
    canonical_path = sentinel.write_intent(canonical / "intent.json", intent)
    duplicate_path = tmp_path / "copied-intent.json"
    duplicate_path.write_bytes(canonical_path.read_bytes())
    with pytest.raises(sentinel.SentinelError, match="must be loaded from exactly"):
        sentinel.load_intent(duplicate_path)


def test_dry_validation_makes_no_run_or_ledger_and_starts_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[Path] = []
    result = asyncio_run(
        sentinel.run_sentinel(
            path,
            dependencies=_dependencies(intent["runtime_identity"], calls),
        )
    )
    assert result.status == "validated"
    assert result.provider_calls_started == 0
    assert calls == []
    assert not sentinel.derive_run_dir(intent).exists()
    assert not Path(intent["shared_ledger_root"]).exists()


def test_pass_is_closed_at_global_317_and_resume_does_not_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[Path] = []
    dependencies = _dependencies(intent["runtime_identity"], calls)
    first = asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=dependencies,
        )
    )
    assert first.status == "pass"
    assert first.provider_calls_started == 1
    decision = sentinel._read_json(first.decision_path)
    assert decision["future_paid_google_experiments_authorized"] is True
    assert decision["global_charged_calls"] == 317
    assert decision["remaining_authorized_calls"] == 133
    assert all(decision["pass_criteria"].values())
    call_dir = first.run_dir / "calls" / intent["sentinel_call"]["call_id"]
    for filename in sentinel.AGY_EVIDENCE_FILENAMES.values():
        assert sentinel.CANARY.encode() not in (call_dir / filename).read_bytes()
    assert all(not workdir.exists() for workdir in calls)

    # A crash after result persistence but before the decision must only
    # derive the deterministic decision; it must not touch the provider.
    first.decision_path.unlink()
    resumed = asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=dependencies,
        )
    )
    assert resumed.status == "pass"
    assert resumed.provider_calls_started == 0
    assert len(calls) == 1

    resumed_again = asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=dependencies,
        )
    )
    assert resumed_again.status == "pass"
    assert resumed_again.provider_calls_started == 0
    assert len(calls) == 1


def test_text_response_and_canary_are_terminal_nonpasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text_root = tmp_path / "text"
    text_root.mkdir()
    path, intent = _intent(text_root, monkeypatch)
    calls: list[Path] = []
    result = asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=_dependencies(intent["runtime_identity"], calls, disposition="response"),
        )
    )
    assert result.status == "target_not_exercised"
    assert (
        sentinel._read_json(result.decision_path)["future_paid_google_experiments_authorized"]
        is False
    )

    canary_root = tmp_path / "canary"
    canary_root.mkdir()
    path, intent = _intent(canary_root, monkeypatch)
    calls = []
    result = asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=_dependencies(intent["runtime_identity"], calls, create_canary=True),
        )
    )
    assert result.status == "failed"
    decision = sentinel._read_json(result.decision_path)
    assert decision["pass_criteria"]["workdir_empty_and_deleted"] is False
    assert len(calls) == 1


def test_tool_prose_and_false_command_are_terminal_nonpasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prose_path, prose_intent = _intent(tmp_path / "prose", monkeypatch)
    prose = asyncio_run(
        sentinel.run_sentinel(
            prose_path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=_dependencies(
                prose_intent["runtime_identity"], [], tool_prose="Unexpected prose."
            ),
        )
    )
    prose_decision = sentinel._read_json(prose.decision_path)
    assert prose.status == "failed"
    assert (
        prose_decision["pass_criteria"]["terminal_response_and_transcript_content_blank"] is False
    )

    command_path, command_intent = _intent(tmp_path / "command", monkeypatch)
    command = asyncio_run(
        sentinel.run_sentinel(
            command_path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=_dependencies(
                command_intent["runtime_identity"], [], tool_command="touch A_DIFFERENT_FILE"
            ),
        )
    )
    command_decision = sentinel._read_json(command.decision_path)
    assert command.status == "failed"
    assert command_decision["pass_criteria"]["runcommand_parameters_exact"] is False


def test_ambiguous_reservation_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    calls: list[Path] = []
    with pytest.raises(sentinel.SentinelIncomplete, match="not durable"):
        asyncio_run(
            sentinel.run_sentinel(
                path,
                execute=True,
                acknowledged_new_call_cap=1,
                dependencies=_dependencies(
                    intent["runtime_identity"], calls, raise_after_reservation=True
                ),
            )
        )
    assert len(calls) == 1
    with pytest.raises(sentinel.SentinelIncomplete, match="will not replay"):
        asyncio_run(
            sentinel.run_sentinel(
                path,
                execute=True,
                acknowledged_new_call_cap=1,
                dependencies=_dependencies(intent["runtime_identity"], calls),
            )
        )
    assert len(calls) == 1


def test_prior_closure_tamper_invalidates_the_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    prior_run = Path(intent["prior_artifacts"]["run_dir"])
    result_path = next((prior_run / "calls").glob("*/result.json"))
    value = sentinel._read_json(result_path)
    value["status"] = "tampered"
    result_path.write_bytes(sentinel._pretty_json(value))
    with pytest.raises(sentinel.SentinelError, match="result_digest mismatch"):
        sentinel.load_intent(path)


def test_impossible_result_and_decision_times_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duration_path, duration_intent = _intent(tmp_path / "duration", monkeypatch)
    dependencies = _dependencies(duration_intent["runtime_identity"], [])
    duration_run = asyncio_run(
        sentinel.run_sentinel(
            duration_path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=dependencies,
        )
    )
    result_path = (
        duration_run.run_dir / "calls" / duration_intent["sentinel_call"]["call_id"] / "result.json"
    )
    duration_run.decision_path.unlink()
    result = sentinel._read_json(result_path)
    result["duration_seconds"] = 60.0
    body = {key: value for key, value in result.items() if key != "result_digest"}
    result["result_digest"] = sentinel._digest(body)
    result_path.write_bytes(sentinel._pretty_json(result))
    with pytest.raises(sentinel.SentinelError, match="result timing is invalid"):
        asyncio_run(sentinel.run_sentinel(duration_path, dependencies=dependencies))

    decision_path, decision_intent = _intent(tmp_path / "decision", monkeypatch)
    decision_dependencies = _dependencies(decision_intent["runtime_identity"], [])
    decision_run = asyncio_run(
        sentinel.run_sentinel(
            decision_path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=decision_dependencies,
        )
    )
    decision = sentinel._read_json(decision_run.decision_path)
    decision["decided_at_utc"] = "2000-01-01T00:00:00.000000Z"
    body = {key: value for key, value in decision.items() if key != "decision_digest"}
    decision["decision_digest"] = sentinel._digest(body)
    decision_run.decision_path.write_bytes(sentinel._pretty_json(decision))
    with pytest.raises(sentinel.SentinelError, match="decision predates"):
        asyncio_run(sentinel.run_sentinel(decision_path, dependencies=decision_dependencies))


def test_runtime_executable_drift_invalidates_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, intent = _intent(tmp_path, monkeypatch)
    dependencies = _dependencies(intent["runtime_identity"], [])
    completed = asyncio_run(
        sentinel.run_sentinel(
            path,
            execute=True,
            acknowledged_new_call_cap=1,
            dependencies=dependencies,
        )
    )
    request_path = completed.run_dir / "calls" / intent["sentinel_call"]["call_id"] / "request.json"
    result_path = request_path.with_name("result.json")
    request = sentinel._read_json(request_path)
    result = sentinel._read_json(result_path)
    assert result["post_call_executable_digests"] == {
        "python_executable_digest": intent["runtime_identity"]["python_executable_digest"],
        "agy_executable_digest": intent["runtime_identity"]["agy_executable_digest"],
    }

    Path(intent["runtime_identity"]["agy_executable"]).write_bytes(b"drifted-agy")
    engine = sentinel._Engine(
        intent, sentinel._pretty_json(intent), completed.run_dir, dependencies
    )
    checked_result, facts = engine._validate_result(result, request)
    assert facts["python_and_agy_executables_unchanged"] is False
    with pytest.raises(sentinel.SentinelError, match="decision differs"):
        engine._validate_decision(
            sentinel._read_json(completed.decision_path), checked_result, facts
        )
    with pytest.raises(sentinel.SentinelError, match="executable bytes drifted"):
        asyncio_run(sentinel.run_sentinel(path, dependencies=dependencies))


def asyncio_run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
