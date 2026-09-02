from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spade.core.witness_qd_proxy import (
    COVERAGE_CHALLENGER_DESCRIPTOR,
    STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR,
    ProxyCandidate,
    PairedOutcome,
    counterbalanced_pairs,
    lock_all_portfolios,
    portfolio_quality_diagnostics,
)
from tools import run_spade_witness_qd_proxy as pilot


def _sha(index: int) -> str:
    return f"sha256:{index:064x}"


def _engine(tmp_path: Path) -> pilot._Engine:
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("provider boundary was crossed")

    intent = {
        "intent_digest": _sha(1),
        "output_root": str(tmp_path),
        "shared_ledger_root": str(tmp_path / "ledger"),
        "runtime_identity": {"agy_executable_digest": _sha(2)},
        "configuration": {
            "llm_timeout_seconds": 10.0,
            "qualification_timeout_seconds": 1.0,
            "maximum_stratum_absolute_quality_gap": 0.125,
            "maximum_mean_absolute_quality_gap": 0.0625,
        },
    }
    dependencies = SimpleNamespace(llm_call=forbidden, client_or_bin=tmp_path / "agy")
    return pilot._Engine(intent, b"{}", tmp_path / "run", dependencies)


def _request(
    engine: pilot._Engine,
    *,
    purpose: dict | None = None,
    prompt: str = "sealed prompt",
    system: str = "sealed system",
) -> dict:
    selected_purpose = purpose or {
        "phase": "challenger-design",
        "stratum_id": "c001",
        "candidate_id": "c001--challenger",
        "attempt": 1,
    }
    body = {
        "schema_version": pilot.CALL_REQUEST_SCHEMA,
        "intent_digest": engine.intent["intent_digest"],
        "actor_plan_digest": None,
        "call_id": pilot.base._call_id(selected_purpose),
        "local_ordinal": 1,
        "global_ordinal": pilot.PRIOR_CHARGED_CALLS + 1,
        "purpose": selected_purpose,
        "provider": "agy",
        "model": pilot.DESIGN_MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "runtime_identity_digest": pilot._digest(engine.intent["runtime_identity"]),
        "agy_executable_digest": engine.intent["runtime_identity"]["agy_executable_digest"],
        "prompt": prompt,
        "prompt_digest": pilot._digest(prompt),
        "system": system,
        "system_digest": pilot._digest(system),
        "timeout_seconds": 10.0,
        "workdir_policy": "fresh-temporary-directory-per-call",
        "reservation_status": "reserved-before-spawn",
        "reserved_at_utc": pilot.base._utc_now(),
    }
    return {**body, "request_digest": pilot._digest(body)}


def _structured_request(engine: pilot._Engine, **kwargs) -> dict:
    legacy = _request(engine, **kwargs)
    body = {key: value for key, value in legacy.items() if key != "request_digest"}
    body.update(
        {
            "schema_version": pilot.CALL_REQUEST_SCHEMA_V2,
            "output_format": pilot.live.AGY_OUTPUT_FORMAT,
            "log_policy": pilot.live.AGY_LOG_POLICY,
            "evidence_policy": (
                "bounded-sanitized-stream-stderr-log-transcript-receipts-with-exact-digests"
            ),
        }
    )
    return {**body, "request_digest": pilot._digest(body)}


def _ndjson(*events: dict) -> bytes:
    return b"".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for event in events
    )


def _agy_evidence_for_request(
    request: dict,
    *,
    disposition: str = "response",
    response: str = r"\boxed{ok}",
    workdir: Path,
) -> pilot.live.AgyCallEvidence:
    conversation_id = "12345678-1234-4234-9234-123456789abc"
    tool = "RunCommand" if disposition == "tool_policy_no_action" else None
    stream = [
        {
            "event": "init",
            "conversation_id": conversation_id,
            "init": {
                "model": request["model"],
                "cwd": str(workdir),
                "tools": ["RunCommand"],
                "permission_mode": "request-review",
            },
        }
    ]
    if tool:
        stream.append(
            {
                "event": "step_update",
                "conversation_id": conversation_id,
                "step_update": {
                    "conversation_id": conversation_id,
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "tool",
                    "tool_name": tool,
                    "tool_info": {"name": tool, "parameters": {"secret": "do-not-persist"}},
                },
            }
        )
    terminal_response = "" if disposition == "ambiguous_provider_disposition" else response
    stream.append(
        {
            "event": "result",
            "conversation_id": conversation_id,
            "result": {
                "conversation_id": conversation_id,
                "status": "SUCCESS",
                "response": terminal_response,
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
    full_prompt = (
        f"{request['system']}\n\n{request['prompt']}" if request["system"] else request["prompt"]
    )
    transcript = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": f"<USER_REQUEST>\n{full_prompt}\n</USER_REQUEST>",
        }
    ]
    if tool:
        transcript.append(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "tool_calls": [{"name": tool, "args": {}}],
            }
        )
    elif terminal_response:
        transcript.append(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": terminal_response,
            }
        )
    gemini_dir = Path(workdir) / ".agy-gemini-00000000000000000000000000000000"
    log = (
        f"CLI app data directory: {gemini_dir / 'antigravity-cli'}\n"
        "applyUserSettings: no shared config permissions from "
        f"{gemini_dir / 'config' / 'config.json'}\n"
        f"Creating CLI server backend: product=antigravity workspaceDirs=[{workdir}]\n"
        f'Print mode: starting (promptLength=1, model="{request["model"]}", '
        'conversationID="")\n'
        f"Created conversation {conversation_id}\n"
        "ResponseID: response-id-1\n"
        + (
            f'Print mode: soft-denying tool confirmation "{tool}" at step 2\n'
            f"Tool confirmation for conversation {conversation_id} step 2 "
            "(type=tool approved=false)\n"
            if tool
            else ""
        )
    ).encode()
    return pilot.live.analyze_agy_evidence(
        requested_model=request["model"],
        full_prompt=full_prompt,
        invocation_workdir=workdir,
        exit_status=0,
        timed_out=False,
        capture_failures=(),
        stdout_ndjson=_ndjson(*stream),
        stderr=b"",
        log=log,
        transcript=_ndjson(*transcript),
        policy_config_identity={
            "relative_path": "config/config.json",
            "exists": False,
            "digest": None,
            "size_bytes": 0,
        },
    )


def _structured_result(
    engine: pilot._Engine,
    request: dict,
    *,
    disposition: str = "response",
) -> dict:
    workdir = Path(f"/private/tmp/spade-coverage-forced-agy-{request['call_id']}-fixture")
    evidence = _agy_evidence_for_request(
        request,
        disposition=disposition,
        workdir=workdir,
    )
    call_dir = engine.run_dir / "calls" / request["call_id"]
    payloads = {
        "stdout_ndjson": evidence.stdout_ndjson,
        "stderr": evidence.stderr,
        "log": evidence.log,
        "transcript": evidence.transcript,
    }
    files = {}
    for label, filename in pilot.AGY_EVIDENCE_FILENAMES.items():
        path = call_dir / filename
        pilot.base._write_immutable_bytes(path, payloads[label])
        files[label] = {
            "path": path.relative_to(engine.run_dir).as_posix(),
            "digest": pilot._bytes_digest(payloads[label]),
            "size_bytes": len(payloads[label]),
        }
    status_by_disposition = {
        "response": ("success", None, evidence.response, None),
        "tool_policy_no_action": (
            "model_behavior",
            "tool_policy_no_action",
            None,
            evidence.error,
        ),
        "ambiguous_provider_disposition": (
            "error",
            "ambiguous_provider_disposition",
            None,
            evidence.error,
        ),
    }
    status, category, selected_response, error = status_by_disposition[disposition]
    body = {
        "schema_version": pilot.CALL_RESULT_SCHEMA_V2,
        "intent_digest": engine.intent["intent_digest"],
        "call_id": request["call_id"],
        "local_ordinal": request["local_ordinal"],
        "global_ordinal": request["global_ordinal"],
        "request_digest": request["request_digest"],
        "status": status,
        "failure_category": category,
        "provider_disposition": disposition,
        "exception_type": None,
        "error": error,
        "exit_status": evidence.exit_status,
        "response": selected_response,
        "response_digest": pilot._digest(selected_response) if selected_response else None,
        "agy_evidence": evidence.summary(),
        "evidence_files": files,
        "started_at_utc": request["reserved_at_utc"],
        "finished_at_utc": request["reserved_at_utc"],
        "duration_seconds": 0.0,
    }
    return {**body, "result_digest": pilot._digest(body)}


def test_preactor_request_remains_valid_after_actor_plan_seal(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _request(engine)
    assert engine._validate_call_request(request) == request

    engine.run_dir.mkdir(parents=True)
    pilot.base._write_json(engine.run_dir / "actor-plan.json", {"deliberately": "opaque"})
    assert engine._validate_call_request(request) == request


def test_ambiguous_request_is_never_replayed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _request(engine)
    request_path = engine.run_dir / "calls" / request["call_id"] / "request.json"
    pilot.base._write_json(request_path, request)

    with pytest.raises(pilot.AmbiguousProviderCall, match="will not replay"):
        asyncio.run(
            engine.call(
                purpose=request["purpose"],
                prompt=request["prompt"],
                system=request["system"],
            )
        )


def test_structured_call_artifacts_recompute_tool_disposition_and_reject_tamper(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    request = _structured_request(engine)
    result = _structured_result(engine, request, disposition="tool_policy_no_action")
    assert engine._validate_call_result(result, request)["status"] == "model_behavior"
    assert result["failure_category"] == "tool_policy_no_action"

    transcript_entry = result["evidence_files"]["transcript"]
    transcript_path = engine.run_dir / transcript_entry["path"]
    transcript_path.write_bytes(transcript_path.read_bytes() + b"{}\n")
    with pytest.raises(pilot.ProxyExperimentError, match="evidence bytes differ"):
        engine._validate_call_result(result, request)


def test_structured_call_rejects_noncanonical_path_symlink_size_and_route(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    request = _structured_request(engine)

    bad_path = _structured_result(engine, request)
    bad_path["evidence_files"]["stderr"]["path"] = "../stderr"
    bad_path_body = {key: value for key, value in bad_path.items() if key != "result_digest"}
    bad_path["result_digest"] = pilot._digest(bad_path_body)
    with pytest.raises(pilot.ProxyExperimentError, match="noncanonical"):
        engine._validate_call_result(bad_path, request)

    engine = _engine(tmp_path / "symlink")
    request = _structured_request(engine)
    symlinked = _structured_result(engine, request)
    stderr_entry = symlinked["evidence_files"]["stderr"]
    stderr_path = engine.run_dir / stderr_entry["path"]
    stderr_path.unlink()
    target = tmp_path / "outside-stderr"
    target.write_bytes(b"")
    stderr_path.symlink_to(target)
    with pytest.raises(pilot.ProxyExperimentError, match="symlink"):
        engine._validate_call_result(symlinked, request)

    engine = _engine(tmp_path / "size")
    request = _structured_request(engine)
    oversized = _structured_result(engine, request)
    oversized["evidence_files"]["log"]["size_bytes"] = pilot.live.MAX_AGY_LOG_BYTES + 1
    oversized_body = {key: value for key, value in oversized.items() if key != "result_digest"}
    oversized["result_digest"] = pilot._digest(oversized_body)
    with pytest.raises(pilot.ProxyExperimentError, match="evidence bytes differ"):
        engine._validate_call_result(oversized, request)

    engine = _engine(tmp_path / "route")
    request = _structured_request(engine)
    route = _structured_result(engine, request)
    stdout_entry = route["evidence_files"]["stdout_ndjson"]
    stdout_path = engine.run_dir / stdout_entry["path"]
    receipt = json.loads(stdout_path.read_text())
    receipt["events"][0]["init"]["model"] = "gemini-wrong-route"
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = pilot.live._canonical_json_digest(body)
    changed = pilot.live._canonical_json_bytes(receipt)
    stdout_path.write_bytes(changed)
    stdout_entry["digest"] = pilot._bytes_digest(changed)
    stdout_entry["size_bytes"] = len(changed)
    route_body = {key: value for key, value in route.items() if key != "result_digest"}
    route["result_digest"] = pilot._digest(route_body)
    with pytest.raises(pilot.ProxyExperimentError, match="summary is not derivable"):
        engine._validate_call_result(route, request)


def test_structured_unknown_blank_is_fatal_not_retryable(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _structured_request(engine)
    result = _structured_result(engine, request, disposition="ambiguous_provider_disposition")
    validated = engine._validate_call_result(result, request)
    failure = pilot._RecordedCallFailure(
        request["call_id"],
        str(validated["failure_category"]),
        str(validated["error"]),
    )
    assert failure.retryable is False


def test_structured_engine_tool_is_terminal_and_only_pre_response_failure_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "_validate_runtime_identity", lambda _value: None)
    purpose = {
        "phase": "challenger-design",
        "stratum_id": "c001",
        "candidate_id": "c001--challenger",
        "attempt": 1,
    }

    def ready_engine(name: str, disposition: str) -> pilot._Engine:
        engine = _engine(tmp_path / name)
        engine.run_dir.mkdir(parents=True)
        engine.ledger_root.mkdir(parents=True)
        engine._verify_provider_executable = lambda: None  # type: ignore[method-assign]
        engine._validate_tree = lambda: None  # type: ignore[method-assign]

        async def structured(
            _client,
            model,
            prompt,
            *,
            system,
            workdir,
            **_kwargs,
        ):
            request = {"model": model, "prompt": prompt, "system": system}
            if disposition == "pre_response_provider_failure":
                return pilot.live.analyze_agy_evidence(
                    requested_model=model,
                    full_prompt=f"{system}\n\n{prompt}" if system else prompt,
                    invocation_workdir=workdir,
                    exit_status=1,
                    timed_out=False,
                    capture_failures=(),
                    stdout_ndjson=b"",
                    stderr=b"HTTP status 429: provider temporarily unavailable\n",
                    log=b"",
                    transcript=b"",
                    policy_config_identity={
                        "relative_path": "config/config.json",
                        "exists": False,
                        "digest": None,
                        "size_bytes": 0,
                    },
                )
            return _agy_evidence_for_request(
                request,
                disposition=disposition,
                workdir=workdir,
            )

        engine.dependencies.structured_llm_call = structured
        return engine

    tool_engine = ready_engine("tool", "tool_policy_no_action")
    output, _call_id = asyncio.run(
        tool_engine.call(purpose=purpose, prompt="sealed", system="system")
    )
    assert output == pilot._ModelCallOutput("", "tool_policy_no_action")
    result_paths = list((tool_engine.run_dir / "calls").glob("*/result.json"))
    assert len(result_paths) == 1
    assert pilot.base._read_json(result_paths[0])["status"] == "model_behavior"

    retry_engine = ready_engine("retry", "pre_response_provider_failure")
    with pytest.raises(pilot._RecordedCallFailure) as retry:
        asyncio.run(retry_engine.call(purpose=purpose, prompt="sealed", system="system"))
    assert retry.value.retryable is True

    blank_engine = ready_engine("blank", "ambiguous_provider_disposition")
    with pytest.raises(pilot._RecordedCallFailure) as blank:
        asyncio.run(blank_engine.call(purpose=purpose, prompt="sealed", system="system"))
    assert blank.value.retryable is False


def test_actor_tool_policy_outcome_is_zero_and_does_not_retry(tmp_path: Path) -> None:
    class Env:
        def reset(self, seed=None):
            return "observation", {}

        def step(self, action):
            assert action == r"\boxed{__spade_invalid_action_format__}"
            return "format feedback", 0.0, False, False, {}

    class Target:
        def instantiate(self):
            return Env()

    actor = object.__new__(pilot._ActorEngine)
    actor.run_dir = tmp_path
    actor.intent = {"intent_digest": _sha(1)}
    actor.actor_plan = {"actor_plan_digest": _sha(2)}
    actor.config = {"qualification_timeout_seconds": 1.0}
    actor.dependencies = SimpleNamespace(target_factory=lambda *_args, **_kwargs: Target())
    actor._candidate_by_id = {
        "c001--challenger": {
            "candidate_id": "c001--challenger",
            "code": "sealed",
            "probes": {"0": {"observation": "observation"}},
            "hints": {"0": {"hint": "strategy"}},
        }
    }
    call_count = 0

    async def tool_call(**_kwargs):
        nonlocal call_count
        call_count += 1
        return pilot._ModelCallOutput("", "tool_policy_no_action"), f"call-{call_count}"

    actor.call = tool_call
    actor._proofpack_call = lambda operation, *args, **kwargs: operation(*args, **kwargs)
    actor._validate_arm = lambda value, **_kwargs: value
    actor._validate_attempt = lambda value, *_args: value
    pair = {
        "pair_id": "pair-001",
        "pair_ordinal": 0,
        "stratum_id": "c001",
        "candidate_id": "c001--challenger",
        "seed": 0,
        "arm_order": ["unhinted", "hinted"],
    }
    attempt = asyncio.run(actor._run_pair_attempt(pair, 1))
    assert attempt["status"] == "completed"
    assert attempt["failure_category"] is None
    assert call_count == 2
    for reference in attempt["arms"]:
        arm = pilot.base._read_json(tmp_path / reference["path"])
        assert arm["model_disposition"] == "tool_policy_no_action"
        assert arm["parser_miss"] is True
        assert arm["binary_return"] == 0.0


def test_total_208_cap_fails_before_provider_spawn(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    for ordinal in range(1, pilot.NEW_CALL_CAP + 1):
        path = engine.run_dir / "calls" / f"prior-{ordinal:03d}" / "request.json"
        pilot.base._write_json(path, {"charged": ordinal})

    with pytest.raises(pilot.CallCapExceeded, match="208"):
        asyncio.run(
            engine.call(
                purpose={
                    "phase": "challenger-design",
                    "stratum_id": "c001",
                    "candidate_id": "c001--challenger",
                    "attempt": 1,
                },
                prompt="no spawn",
            )
        )


def test_sealed_budget_arithmetic_has_37_call_headroom() -> None:
    assert pilot.PAIR_COUNT == 36
    assert pilot.PRE_ACTOR_CALL_CEILING == 36
    assert pilot.ACTOR_CALL_CEILING == 172
    assert pilot.NEW_CALL_CAP == 208
    assert pilot.PRIOR_CHARGED_CALLS + pilot.NEW_CALL_CAP == 413
    assert pilot.AUTHORIZED_GLOBAL_CALL_CAP - 413 == 37


def test_extended_runtime_identity_validates_legacy_base_schema(tmp_path: Path) -> None:
    runtime_file = tmp_path / "sealed-runtime.py"
    runtime_file.write_text("# sealed\n", encoding="utf-8")
    runtime_digest = pilot._bytes_digest(runtime_file.read_bytes())
    base_identity = {
        "runner_digest": _sha(1),
        "python_implementation": "CPython",
        "python_version": "3.12.0",
        "python_executable": "/sealed/python",
        "python_executable_digest": _sha(2),
        "platform": "sealed-platform",
        "agy_executable": "/sealed/agy",
        "agy_executable_digest": _sha(3),
        "agy_version": "sealed-version",
        "imported_sources": {
            name: {"path": f"/sealed/{name}.py", "digest": _sha(index)}
            for index, name in enumerate(
                (
                    "spade_live_runner",
                    "proofpack_qualifier",
                    "proofpack_receipt_validator",
                    "assay_writer",
                ),
                start=10,
            )
        },
    }
    extended = {
        **base_identity,
        "coverage_forced_files": {
            name: {"path": str(runtime_file), "digest": runtime_digest}
            for name in (
                "coverage_forced_runner",
                "coverage_forced_core",
                "counterfactual_witness_core",
                "witness_archive_core",
                "counterfactual_witness_runner",
                "outcome_replay_runner",
                "proofpack_spade_target",
                "proofpack_spade_launcher",
                "proofpack_spade_worker",
                "proofpack_sandbox_executable",
            )
        },
    }

    assert pilot._validate_runtime_identity(extended) == extended


def test_historical_and_challenger_hints_keep_distinct_lineage_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot.live, "hint_reveals_solution", lambda *_args: False)
    probe = {
        "observation": "sealed observation",
        "observation_digest": pilot._digest("sealed observation"),
        "solution": ["\\boxed{sealed}"],
        "solution_digest": pilot._digest(["\\boxed{sealed}"]),
    }
    historical_body = {
        "schema_version": "spade-agy-hint-attempt/v1",
        "plan_digest": _sha(1),
        "cluster_id": "c001",
        "candidate_id": "c001-primary",
        "seed": 0,
        "attempt": 1,
        "call_id": "hint-historical",
        "status": "accepted",
        "reason": "accepted",
        "hint": "Track the invariant without revealing the answer.",
        "hint_digest": pilot._digest("Track the invariant without revealing the answer."),
        "feedback_for_next_attempt": "",
    }
    historical = {
        **historical_body,
        "attempt_digest": pilot._digest(historical_body),
    }
    pilot._Engine._validate_candidate_hint(
        historical,
        probe=probe,
        seed_text="0",
        source_arm="v3",
    )

    challenger_body = {
        "schema_version": "spade-coverage-forced-hint-attempt/v1",
        "intent_digest": _sha(2),
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
        "call_id": "hint-challenger",
        "status": "accepted",
        "reason": "nonleaking",
        "hint": "Track the invariant without revealing the answer.",
        "hint_digest": pilot._digest("Track the invariant without revealing the answer."),
        "observation_digest": probe["observation_digest"],
        "solution_digest": probe["solution_digest"],
        "feedback_for_next_attempt": "",
    }
    challenger = {
        **challenger_body,
        "attempt_digest": pilot._digest(challenger_body),
    }
    pilot._Engine._validate_candidate_hint(
        challenger,
        probe=probe,
        seed_text="0",
        source_arm="challenger",
    )

    missing_lineage_body = {
        key: value for key, value in challenger_body.items() if key != "solution_digest"
    }
    missing_lineage = {
        **missing_lineage_body,
        "attempt_digest": pilot._digest(missing_lineage_body),
    }
    with pytest.raises(pilot.ProxyExperimentError, match="invalid digests"):
        pilot._Engine._validate_candidate_hint(
            missing_lineage,
            probe=probe,
            seed_text="0",
            source_arm="challenger",
        )


def test_failure_category_is_recomputed_not_trusted(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _request(engine)
    timestamp = request["reserved_at_utc"]
    body = {
        "schema_version": pilot.CALL_RESULT_SCHEMA,
        "intent_digest": engine.intent["intent_digest"],
        "call_id": request["call_id"],
        "local_ordinal": request["local_ordinal"],
        "global_ordinal": request["global_ordinal"],
        "request_digest": request["request_digest"],
        "status": "error",
        "failure_category": "fatal_transport",
        "exception_type": "LiveEvalError",
        "error": "agy returned an empty response",
        "exit_status": None,
        "response": None,
        "response_digest": None,
        "started_at_utc": timestamp,
        "finished_at_utc": timestamp,
        "duration_seconds": 0.0,
    }
    tampered = {**body, "result_digest": pilot._digest(body)}
    with pytest.raises(pilot.ProxyExperimentError, match="not derivable"):
        engine._validate_call_result(tampered, request)


def test_artifact_tree_rejects_symlink(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.run_dir.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (engine.run_dir / "escape.json").symlink_to(target)
    with pytest.raises(pilot.ProxyExperimentError, match="symlink"):
        engine._validate_tree()


def _actor_plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    strata = (1, 3, 4, 5, 6, 7)
    intent = {
        "intent_digest": _sha(10),
        "strata": [{"stratum_id": f"c{index:03d}"} for index in strata],
        "configuration": {
            "maximum_stratum_absolute_quality_gap": 0.125,
            "maximum_mean_absolute_quality_gap": 0.0625,
        },
        "analysis_gates": {"sealed": True},
    }
    candidates = []
    references = []
    grouped = {}
    for stratum in strata:
        stratum_id = f"c{stratum:03d}"
        grouped[stratum_id] = []
        for offset, (arm, quality) in enumerate(
            (("v3", 0.75), ("v4", 1.0), ("challenger", 1.0)),
            start=1,
        ):
            candidate_id = f"{stratum_id}--{arm}"
            grouped[stratum_id].append(candidate_id)
            serial = stratum * 10 + offset
            evidence = {
                "intent_digest": intent["intent_digest"],
                "stratum_id": stratum_id,
                "candidate_id": candidate_id,
                "source_arm": arm,
                "environment_digest": _sha(serial),
                "evidence_digest": _sha(1_000 + serial),
                "qualification": {"qualification_digest": _sha(2_000 + serial)},
                "cwa": {
                    "evidence_digest": _sha(3_000 + serial),
                    "quality_score": quality,
                    "descriptor": dict(
                        COVERAGE_CHALLENGER_DESCRIPTOR
                        if arm == "challenger"
                        else STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR
                    ),
                },
                "one_turn_viability": {"viability_digest": _sha(4_000 + serial)},
                "hints": {"0": {"hint": "h0"}, "42": {"hint": "h42"}},
            }
            relative = f"candidate-evidence/{candidate_id}.json"
            pilot.base._write_json(tmp_path / relative, evidence)
            references.append(
                {
                    "stratum_id": stratum_id,
                    "candidate_id": candidate_id,
                    "source_arm": arm,
                    "path": relative,
                    "evidence_digest": evidence["evidence_digest"],
                    "qualification_digest": evidence["qualification"]["qualification_digest"],
                    "cwa_evidence_digest": evidence["cwa"]["evidence_digest"],
                    "viability_digest": evidence["one_turn_viability"]["viability_digest"],
                    "hint_digests": {
                        seed: pilot._digest(evidence["hints"][seed]) for seed in ("0", "42")
                    },
                }
            )
            candidates.append(
                ProxyCandidate.create(
                    candidate_id=candidate_id,
                    stratum_id=stratum_id,
                    source_arm=arm,
                    quality_score=quality,
                    descriptor=evidence["cwa"]["descriptor"],
                    environment_digest=evidence["environment_digest"],
                    evidence_digest=evidence["cwa"]["evidence_digest"],
                )
            )
    monkeypatch.setattr(
        pilot._Engine,
        "_validate_candidate_evidence",
        staticmethod(lambda value: value),
    )
    schedule = list(
        counterbalanced_pairs({key: tuple(value) for key, value in grouped.items()}, pilot.SEEDS)
    )
    locks = lock_all_portfolios(candidates)
    portfolios = [item.to_dict() for item in locks]
    body = {
        "schema_version": pilot.ACTOR_PLAN_SCHEMA,
        "protocol_id": pilot.PROTOCOL_ID,
        "intent_digest": intent["intent_digest"],
        "chronology": "sealed-after-candidates-cwa-hints-before-actor",
        "provider": "agy",
        "model": pilot.ACTOR_MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "candidate_evidence": references,
        "portfolios": portfolios,
        "portfolio_quality_diagnostics": portfolio_quality_diagnostics(locks),
        "pair_schedule": schedule,
        "pair_schedule_digest": pilot._digest(schedule),
        "attempt_policy": {
            "whole_pair_retries": True,
            "waves_1_and_2": "all-unresolved",
            "wave_3": "only-if-unresolved-after-wave-2-at-most-14",
            "retryable_failure_categories": ["pre_response_provider_failure"],
            "model_tool_calls_are_zero_reward_and_not_retried": True,
            "nonempty_parser_misses_are_zero_reward_and_not_retried": True,
            "unknown_or_post_response_failures_are_fatal": True,
            "environment_runtime_integrity_failures_are_fatal": True,
        },
        "actor_call_ceiling": pilot.ACTOR_CALL_CEILING,
        "success_rule": "reward-positive-even-if-simultaneously-truncated",
        "analysis_gates": intent["analysis_gates"],
        "analysis_interpretation": (
            "quality-matched coverage-forced portfolio-swap association; exact 3!^6 "
            "label-permutation and 2^6 sign-flip sensitivity analyses under strong "
            "exchangeability/symmetry assumptions; neither is design-based"
        ),
    }
    return intent, {**body, "actor_plan_digest": pilot._digest(body)}


def test_actor_plan_recomputes_portfolios_from_candidate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, actor_plan = _actor_plan_fixture(tmp_path, monkeypatch)
    assert pilot.validate_actor_plan(actor_plan, intent, run_dir=tmp_path) == actor_plan

    tampered = deepcopy(actor_plan)
    tampered["portfolios"][0] = {
        **tampered["portfolios"][0],
        "coverage_forced": tampered["portfolios"][0]["redundant_historical"],
    }
    body = {key: value for key, value in tampered.items() if key != "actor_plan_digest"}
    tampered["actor_plan_digest"] = pilot._digest(body)
    with pytest.raises(pilot.ProxyExperimentError, match="portfolios differ"):
        pilot.validate_actor_plan(tampered, intent, run_dir=tmp_path)


def test_actor_plan_and_retry_leaf_preserve_exact_legacy_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, actor_plan = _actor_plan_fixture(tmp_path, monkeypatch)
    legacy_plan = deepcopy(actor_plan)
    legacy_plan["attempt_policy"] = deepcopy(pilot.LEGACY_ACTOR_ATTEMPT_POLICY)
    body = {key: value for key, value in legacy_plan.items() if key != "actor_plan_digest"}
    legacy_plan["actor_plan_digest"] = pilot._digest(body)
    with pytest.raises(pilot.ProxyExperimentError, match="attempt policy"):
        pilot.validate_actor_plan(legacy_plan, intent, run_dir=tmp_path)
    monkeypatch.setattr(pilot, "LEGACY_INTENT_DIGEST", intent["intent_digest"])
    monkeypatch.setattr(
        pilot,
        "LEGACY_ACTOR_PLAN_DIGEST",
        legacy_plan["actor_plan_digest"],
    )
    assert pilot.validate_actor_plan(legacy_plan, intent, run_dir=tmp_path) == legacy_plan

    pair = legacy_plan["pair_schedule"][0]
    engine = object.__new__(pilot._ActorEngine)
    engine.intent = {"intent_digest": intent["intent_digest"]}
    engine.actor_plan = legacy_plan
    engine.run_dir = tmp_path
    engine._candidate_by_id = {
        pair["candidate_id"]: {
            "probes": {str(pair["seed"]): {"observation": "locked observation"}},
            "hints": {str(pair["seed"]): {"hint": "locked strategy"}},
        }
    }
    failed_arm = pair["arm_order"][0]
    purpose = {
        "phase": "actor",
        "actor_plan_digest": legacy_plan["actor_plan_digest"],
        "pair_id": pair["pair_id"],
        "pair_ordinal": pair["pair_ordinal"],
        "pair_attempt": 1,
        "stratum_id": pair["stratum_id"],
        "candidate_id": pair["candidate_id"],
        "seed": pair["seed"],
        "arm": failed_arm,
        "turn": 1,
        "horizon": 1,
    }
    call_id = pilot.base._call_id(purpose)
    call_dir = tmp_path / "calls" / call_id
    pilot.base._write_json(call_dir / "request.json", {"call_id": call_id})
    pilot.base._write_json(call_dir / "result.json", {"call_id": call_id})
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda _value, _request: {
        "status": "error",
        "failure_category": "empty_response",
    }
    attempt_body = {
        "schema_version": pilot.PAIR_ATTEMPT_SCHEMA,
        "intent_digest": intent["intent_digest"],
        "actor_plan_digest": legacy_plan["actor_plan_digest"],
        "pair_id": pair["pair_id"],
        "pair_ordinal": pair["pair_ordinal"],
        "pair_attempt": 1,
        "status": "retryable_failure",
        "failure_category": "empty_response",
        "failed_call_id": call_id,
        "arms": [],
    }
    attempt = {**attempt_body, "attempt_digest": pilot._digest(attempt_body)}
    assert engine._validate_attempt(attempt, pair, 1) == attempt


def test_persisted_actor_reward_is_replayed_not_trusted(tmp_path: Path) -> None:
    class Env:
        def reset(self, seed=None):
            return "locked observation", {}

        def step(self, _action):
            return "post", 0.0, False, False, {}

    class Target:
        def instantiate(self):
            return Env()

    engine = object.__new__(pilot._ActorEngine)
    engine.intent = {"intent_digest": _sha(20)}
    engine.actor_plan = {"actor_plan_digest": _sha(21)}
    engine.run_dir = tmp_path
    engine.config = {"qualification_timeout_seconds": 1.0}
    engine.dependencies = SimpleNamespace(target_factory=lambda *_args, **_kwargs: Target())
    candidate_id = "c001--v3"
    engine._candidate_by_id = {
        candidate_id: {
            "code": "class FakeEnv: pass",
            "probes": {"0": {"observation": "locked observation"}},
            "hints": {"0": {"hint": "strategy"}},
        }
    }
    pair = {
        "pair_id": "pair-001",
        "pair_ordinal": 0,
        "stratum_id": "c001",
        "candidate_id": candidate_id,
        "seed": 0,
        "arm_order": ["unhinted", "hinted"],
    }
    purpose = {
        "phase": "actor",
        "actor_plan_digest": engine.actor_plan["actor_plan_digest"],
        "pair_id": pair["pair_id"],
        "pair_ordinal": pair["pair_ordinal"],
        "pair_attempt": 1,
        "stratum_id": pair["stratum_id"],
        "candidate_id": pair["candidate_id"],
        "seed": pair["seed"],
        "arm": "unhinted",
        "turn": 1,
        "horizon": 1,
    }
    call_id = pilot.base._call_id(purpose)
    pilot.base._write_json(tmp_path / "calls" / call_id / "request.json", {})
    pilot.base._write_json(tmp_path / "calls" / call_id / "result.json", {})
    raw = "\\boxed{answer}"
    clean = pilot.live.extract_clean_action(raw, "boxed")
    body = {
        "schema_version": "spade-coverage-forced-arm/v1",
        "intent_digest": engine.intent["intent_digest"],
        "actor_plan_digest": engine.actor_plan["actor_plan_digest"],
        "pair_id": pair["pair_id"],
        "pair_ordinal": pair["pair_ordinal"],
        "pair_attempt": 1,
        "stratum_id": pair["stratum_id"],
        "candidate_id": pair["candidate_id"],
        "seed": 0,
        "arm": "unhinted",
        "call_id": call_id,
        "raw_response": raw,
        "raw_response_digest": pilot._digest(raw),
        "parser_miss": False,
        "clean_action": clean,
        "pre_observation": "locked observation",
        "pre_observation_digest": pilot._digest("locked observation"),
        "post_observation": "post",
        "post_observation_digest": pilot._digest("post"),
        "raw_reward": 0.0,
        "binary_return": 0.0,
        "terminated": False,
        "truncated": False,
        "success_rule": "reward-positive-even-if-simultaneously-truncated",
    }
    good = {**body, "arm_digest": pilot._digest(body)}
    engine._validate_call_request = lambda _value, _expected=None: {}
    engine._validate_call_result = lambda _value, _request: {
        "status": "success",
        "response": raw,
    }
    engine._proofpack_call = lambda operation, *args, **kwargs: operation(*args, **kwargs)
    assert engine._validate_arm(good, pair=pair, attempt=1, arm="unhinted") == good

    changed_body = {**body, "raw_reward": 1.0, "binary_return": 1.0}
    changed = {**changed_body, "arm_digest": pilot._digest(changed_body)}
    with pytest.raises(pilot.ProxyExperimentError, match="deterministic replay"):
        engine._validate_arm(changed, pair=pair, attempt=1, arm="unhinted")


def test_design_prompt_matches_v4_difficulty_clause_and_bounds_output(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    medium = {
        "stratum_id": "c001",
        "challenger_candidate_id": "c001--challenger",
        "skill": "logic",
        "difficulty": "medium",
    }
    hard = {**medium, "difficulty": "hard"}
    assert (
        "Medium difficulty: require 2-3 dependent reasoning steps and one unique boxed answer."
        in engine._design_prompt(medium, "")
    )
    assert (
        "Hard difficulty: require 4-6 dependent reasoning steps, plausible distractors, and one "
        "unique boxed answer." in engine._design_prompt(hard, "")
    )
    raw = "```python\n" + ("# x\n" * 121) + "```"
    assessment = engine._assess_design_response(raw, "c001--challenger", "c001")
    assert assessment["status"] == "rejected"
    assert assessment["source_nonblank_line_count"] == 121
    assert "120 nonblank lines" in assessment["reason"]


def test_design_attempt_is_terminal_only_after_exact_cell_and_quality_match(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    historical = tuple(
        ProxyCandidate.create(
            candidate_id=f"c001--{arm}",
            stratum_id="c001",
            source_arm=arm,
            quality_score=0.75,
            descriptor=STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR,
            environment_digest=_sha(index),
            evidence_digest=_sha(100 + index),
        )
        for index, arm in enumerate(("v3", "v4"), start=1)
    )
    engine._historical_proxy_candidates = lambda _stratum_id: historical
    qualification = {
        "environment_digest": _sha(3),
        "qualification_digest": _sha(103),
    }
    probes = {"0": {"observation": "a"}, "42": {"observation": "b"}}
    viability = {"viability_digest": _sha(203)}
    engine._qualification = lambda *_args: qualification
    engine._probe_candidate = lambda *_args: probes
    engine._one_turn_viability = lambda *_args: viability

    descriptor = dict(STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR)

    def cwa_evidence(_candidate):
        return {
            "descriptor": descriptor,
            "quality_score": 1.0,
            "evidence_digest": _sha(303),
        }

    engine._cwa_evidence = cwa_evidence
    raw = "```python\nclass PuzzleEnv:\n    pass\n```"
    rejected = engine._assess_design_response(raw, "c001--challenger", "c001")
    assert rejected["status"] == "rejected"
    assert rejected["scientific_preview"] is None
    assert "descriptor" in rejected["feedback_for_next_attempt"]

    descriptor = dict(COVERAGE_CHALLENGER_DESCRIPTOR)
    accepted = engine._assess_design_response(raw, "c001--challenger", "c001")
    assert accepted["status"] == "coverage_eligible"
    assert accepted["scientific_preview"]["absolute_quality_gap"] == 0.125


def test_portfolio_identity_uses_stable_cwa_digest_not_later_hint_bytes(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    evidence = {
        "candidate_id": "c001--challenger",
        "stratum_id": "c001",
        "source_arm": "challenger",
        "environment_digest": _sha(3),
        "evidence_digest": _sha(900),
        "cwa": {
            "quality_score": 1.0,
            "descriptor": dict(COVERAGE_CHALLENGER_DESCRIPTOR),
            "evidence_digest": _sha(303),
        },
    }
    before = engine._proxy_candidate(evidence)
    after = engine._proxy_candidate({**evidence, "evidence_digest": _sha(1)})
    assert before == after
    assert before.evidence_digest == _sha(303)


def test_design_phase_resume_does_not_require_prior_challenger_hints(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    strata = (1, 3, 4, 5, 6, 7)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in strata
    ]
    for index in strata:
        for arm in ("v3", "v4"):
            pilot.base._write_json(engine._candidate_path(f"c{index:03d}--{arm}"), {"ok": True})
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda value, _request: value
    engine._validate_design_attempt = lambda value, **_kwargs: value
    for ordinal, index in enumerate((1, 3), start=1):
        purpose = {
            "phase": "challenger-design",
            "stratum_id": f"c{index:03d}",
            "candidate_id": f"c{index:03d}--challenger",
            "attempt": 1,
        }
        call_id = pilot.base._call_id(purpose)
        pilot.base._write_json(
            engine.run_dir / "calls" / call_id / "request.json",
            {"call_id": call_id, "local_ordinal": ordinal, "purpose": purpose},
        )
        pilot.base._write_json(engine.run_dir / "calls" / call_id / "result.json", {})
        pilot.base._write_json(
            engine.run_dir
            / "challenger-generation"
            / f"c{index:03d}--challenger"
            / "attempt-01.json",
            {
                "status": "coverage_eligible",
                "call_id": call_id,
                "feedback_for_next_attempt": "",
            },
        )
    engine._validate_pre_actor_call_inventory(require_closed=False)


def test_any_hint_request_first_revalidates_complete_global_quality_gate(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in (1, 3, 4, 5, 6, 7)
    ]
    purpose = {
        "phase": "challenger-hint",
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
    }
    call_id = pilot.base._call_id(purpose)
    pilot.base._write_json(
        engine.run_dir / "calls" / call_id / "request.json",
        {"call_id": call_id, "local_ordinal": 1, "purpose": purpose},
    )
    pilot.base._write_json(engine.run_dir / "calls" / call_id / "result.json", {})
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda value, _request: value

    def quality_gate():
        raise RuntimeError("complete quality gate invoked")

    engine._validate_complete_pre_hint_quality_gate = quality_gate
    with pytest.raises(RuntimeError, match="complete quality gate invoked"):
        engine._validate_pre_actor_call_inventory(require_closed=False)


def test_terminal_design_and_hint_attempts_reject_later_leaves(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    stratum = {
        "stratum_id": "c001",
        "challenger_candidate_id": "c001--challenger",
        "skill": "logic",
        "difficulty": "medium",
    }
    design_root = engine.run_dir / "challenger-generation" / "c001--challenger"
    pilot.base._write_json(
        design_root / "attempt-01.json", {"status": "coverage_eligible", "code": "x"}
    )
    pilot.base._write_json(design_root / "attempt-02.json", {"status": "rejected"})
    engine._validate_design_attempt = lambda value, **_kwargs: value
    with pytest.raises(pilot.ProxyExperimentError, match="after terminal eligibility"):
        asyncio.run(engine._generate_challenger(stratum))

    hint_root = engine.run_dir / "challenger-hints" / "c001--challenger" / "0"
    accepted = {"status": "accepted", "hint": "strategy"}
    pilot.base._write_json(hint_root / "attempt-01.json", accepted)
    pilot.base._write_json(hint_root / "attempt-02.json", {"status": "leaked"})
    engine._validate_hint_attempt = lambda value, **_kwargs: value
    with pytest.raises(pilot.ProxyExperimentError, match="after terminal acceptance"):
        asyncio.run(
            engine._lock_hint(
                candidate_id="c001--challenger",
                seed=0,
                observation="obs",
                solution="answer",
            )
        )


def test_hint_attempt_binds_locked_probe_digests(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    purpose = {
        "phase": "challenger-hint",
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
    }
    body = {
        "schema_version": "spade-coverage-forced-hint-attempt/v1",
        "intent_digest": engine.intent["intent_digest"],
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
        "call_id": pilot.base._call_id(purpose),
        "status": "accepted",
        "reason": "nonleaking",
        "hint": "strategy",
        "hint_digest": pilot._digest("strategy"),
        "observation_digest": pilot._digest("different observation"),
        "solution_digest": pilot._digest("answer"),
        "feedback_for_next_attempt": "",
    }
    leaf = {**body, "attempt_digest": pilot._digest(body)}
    with pytest.raises(pilot.ProxyExperimentError, match="probe binding"):
        engine._validate_hint_attempt(
            leaf,
            candidate_id="c001--challenger",
            seed=0,
            observation="locked observation",
            solution="answer",
            attempt=1,
            feedback="",
        )


def test_preactor_inventory_rejects_arbitrary_attempt_paths(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in (1, 3, 4, 5, 6, 7)
    ]
    pilot.base._write_json(
        engine.run_dir / "challenger-generation" / "junk" / "attempt-99.json",
        {"call_id": "laundered"},
    )
    with pytest.raises(pilot.ProxyExperimentError, match="noncanonical candidate"):
        engine._validate_pre_actor_call_inventory(require_closed=False)


def test_proofpack_boundary_revalidates_before_and_after_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    checks = 0

    def check(_identity):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise pilot.ProxyExperimentError("runtime drift")

    monkeypatch.setattr(pilot, "_validate_runtime_identity", check)
    operation_calls = 0

    def operation():
        nonlocal operation_calls
        operation_calls += 1
        return "result"

    with pytest.raises(pilot.ProxyExperimentError, match="runtime drift"):
        engine._proofpack_call(operation)
    assert operation_calls == 1
    assert checks == 2


def test_actor_preflight_rejects_calls_after_unmaterialized_terminal_result(
    tmp_path: Path,
) -> None:
    engine = object.__new__(pilot._ActorEngine)
    engine.run_dir = tmp_path
    engine.actor_plan = {
        "actor_plan_digest": _sha(31),
        "pair_schedule": [
            {
                "pair_id": "pair-001",
                "pair_ordinal": 0,
                "stratum_id": "c001",
                "candidate_id": "c001--v3",
                "seed": 0,
                "arm_order": ["unhinted", "hinted"],
            }
        ],
    }
    engine._candidate_by_id = {
        "c001--v3": {
            "probes": {"0": {"observation": "obs"}},
            "hints": {"0": {"hint": "strategy"}},
        }
    }
    engine._scientific_reaudit_complete = True
    engine._validate_pre_actor_call_inventory = lambda **_kwargs: None
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda value, _request: value
    for ordinal, arm in enumerate(("unhinted", "hinted"), start=1):
        purpose = {
            "phase": "actor",
            "actor_plan_digest": engine.actor_plan["actor_plan_digest"],
            "pair_id": "pair-001",
            "pair_ordinal": 0,
            "pair_attempt": 1,
            "stratum_id": "c001",
            "candidate_id": "c001--v3",
            "seed": 0,
            "arm": arm,
            "turn": 1,
            "horizon": 1,
        }
        call_id = pilot.base._call_id(purpose)
        pilot.base._write_json(
            tmp_path / "calls" / call_id / "request.json",
            {"call_id": call_id, "local_ordinal": ordinal, "purpose": purpose},
        )
        pilot.base._write_json(
            tmp_path / "calls" / call_id / "result.json",
            {
                "status": "error",
                "failure_category": "empty_response",
            },
        )
    with pytest.raises(pilot.ProxyExperimentError, match="multiple actor calls"):
        engine._preflight_actor_state()


def test_actor_preflight_rejects_postseal_orphaned_preactor_call(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in (1, 3, 4, 5, 6, 7)
    ]
    for source_arm in ("v3", "v4"):
        pilot.base._write_json(
            engine._candidate_path(f"c001--{source_arm}"), {"sealed": source_arm}
        )
    request = _request(engine)
    request_path = engine.run_dir / "calls" / request["call_id"] / "request.json"
    pilot.base._write_json(request_path, request)
    timestamp = request["reserved_at_utc"]
    result_body = {
        "schema_version": pilot.CALL_RESULT_SCHEMA,
        "intent_digest": engine.intent["intent_digest"],
        "call_id": request["call_id"],
        "local_ordinal": request["local_ordinal"],
        "global_ordinal": request["global_ordinal"],
        "request_digest": request["request_digest"],
        "status": "success",
        "failure_category": None,
        "exception_type": None,
        "error": None,
        "exit_status": 0,
        "response": "```python\nclass PuzzleEnv: pass\n```",
        "response_digest": pilot._digest("```python\nclass PuzzleEnv: pass\n```"),
        "started_at_utc": timestamp,
        "finished_at_utc": timestamp,
        "duration_seconds": 0.0,
    }
    pilot.base._write_json(
        request_path.parent / "result.json",
        {**result_body, "result_digest": pilot._digest(result_body)},
    )

    actor = object.__new__(pilot._ActorEngine)
    actor.intent = engine.intent
    actor.config = engine.config
    actor.run_dir = engine.run_dir
    actor.dependencies = engine.dependencies
    actor.actor_plan = {"actor_plan_digest": _sha(40), "pair_schedule": []}
    actor._candidate_by_id = {}
    actor._scientific_reaudit_complete = True
    with pytest.raises(pilot.ProxyExperimentError, match="orphaned pre-actor calls"):
        actor._preflight_actor_state()


def test_final_inventory_roots_nested_aggregate_and_rejects_model_lock(tmp_path: Path) -> None:
    engine = object.__new__(pilot._ActorEngine)
    engine.run_dir = tmp_path / "run"
    engine.intent = {"shared_ledger_root": str(tmp_path / "ledger")}
    engine.run_dir.mkdir(parents=True)
    engine.ledger_root.mkdir(parents=True)
    nested = engine.run_dir / "nested" / "aggregate.json"
    pilot.base._write_json(nested, {"must": "be rooted"})
    inventory = engine._complete_evidence_inventory()
    assert any(item["path"] == "nested/aggregate.json" for item in inventory)

    (engine.run_dir / "model.lock").write_text("forbidden", encoding="utf-8")
    with pytest.raises(pilot.ProxyExperimentError, match="model.lock"):
        engine._complete_evidence_inventory()
    (engine.run_dir / "model.lock").unlink()
    (engine.ledger_root / "model.lock").write_text("also forbidden", encoding="utf-8")
    with pytest.raises(pilot.ProxyExperimentError, match="model.lock"):
        engine._complete_evidence_inventory()


def test_aggregate_recomputes_outcomes_and_analysis_and_refuses_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_intent, actor_plan = _actor_plan_fixture(tmp_path, monkeypatch)
    engine = object.__new__(pilot._ActorEngine)
    engine.run_dir = tmp_path
    engine.actor_plan = actor_plan
    gates = {
        "pooled_unhinted_min": 0.10,
        "pooled_unhinted_max": 0.90,
        "minimum_discordant_pairs": 8,
        "minimum_coverage_forced_delta": 0.10,
        "one_sided_label_permutation_alpha": 0.05,
        "maximum_first_attempt_exogenous_failure_rate": 0.15,
    }
    engine.intent = {
        "intent_digest": fixture_intent["intent_digest"],
        "shared_ledger_root": str(tmp_path / "ledger"),
        "analysis_role": "exploratory",
        "claim_exclusions": ["no-causal-claim"],
        "analysis_gates": gates,
        "budget": {"governance_scope": "external-context"},
    }
    outcomes: list[PairedOutcome] = []
    for portfolio in actor_plan["portfolios"]:
        for candidate_id in portfolio["candidate_ids"]:
            for seed in (0, 42):
                if candidate_id == portfolio["challenger_id"]:
                    unhinted, hinted = 0.0, 1.0
                elif candidate_id == portfolio["displaced_historical_id"]:
                    unhinted, hinted = 1.0, 0.0
                else:
                    unhinted = hinted = 0.0
                outcomes.append(
                    PairedOutcome(
                        candidate_id=candidate_id,
                        stratum_id=portfolio["stratum_id"],
                        seed=seed,
                        unhinted=unhinted,
                        hinted=hinted,
                    )
                )
    engine._paired_outcomes = lambda _resolutions: outcomes
    engine._complete_evidence_inventory = lambda: []
    locks = tuple(
        pilot.LockedPortfolios(
            stratum_id=item["stratum_id"],
            candidate_ids=tuple(item["candidate_ids"]),
            coverage_forced=tuple(item["coverage_forced"]),
            redundant_historical=tuple(item["redundant_historical"]),
            challenger_id=item["challenger_id"],
            retained_historical_id=item["retained_historical_id"],
            displaced_historical_id=item["displaced_historical_id"],
            coverage_forced_quality=item["coverage_forced_quality"],
            redundant_historical_quality=item["redundant_historical_quality"],
            signed_quality_gap=item["signed_quality_gap"],
            absolute_quality_gap=item["absolute_quality_gap"],
        )
        for item in actor_plan["portfolios"]
    )
    analysis = pilot.analyze_proxy_pilot(locks, outcomes).to_dict()
    resolutions = [{"resolution_digest": _sha(index)} for index in range(pilot.PAIR_COUNT)]
    body = {
        "schema_version": pilot.AGGREGATE_SCHEMA,
        "protocol_id": pilot.PROTOCOL_ID,
        "intent_digest": engine.intent["intent_digest"],
        "actor_plan_digest": actor_plan["actor_plan_digest"],
        "analysis_role": engine.intent["analysis_role"],
        "claim_exclusions": engine.intent["claim_exclusions"],
        "provider": "agy",
        "model": pilot.ACTOR_MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "estimand": (
            "association over locked realized environment plus source-specific hint packages; "
            "environment effects are not isolated from hint quality, epoch, or source arm"
        ),
        "resolution_digests": [item["resolution_digest"] for item in resolutions],
        "portfolios": actor_plan["portfolios"],
        "portfolio_quality_diagnostics": actor_plan["portfolio_quality_diagnostics"],
        "outcomes": [pilot._plain(item) for item in outcomes],
        "analysis": analysis,
        "analysis_interpretation": (
            "exploratory quality-matched coverage-forced portfolio-swap association; exact "
            "label-permutation and sign-flip sensitivity analyses under strong assumptions; "
            "neither is design-based and there is no causal or learner-improvement claim"
        ),
        "new_charged_calls": 0,
        "global_charged_calls": pilot.PRIOR_CHARGED_CALLS,
        "authorized_global_call_cap": pilot.AUTHORIZED_GLOBAL_CALL_CAP,
        "global_budget_governance_scope": "external-context",
        "evidence_inventory": [],
        "evidence_root_digest": pilot._digest([]),
        "assay_decision": "not-run-not-applicable",
        "release_authorized": False,
        "model_lock_status": "absent",
    }
    value = {**body, "aggregate_digest": pilot._digest(body)}
    engine._validate_aggregate(value, resolutions)

    tampered = deepcopy(value)
    tampered["analysis"]["coverage_forced_delta"] = -1.0
    tampered_body = {key: item for key, item in tampered.items() if key != "aggregate_digest"}
    tampered["aggregate_digest"] = pilot._digest(tampered_body)
    with pytest.raises(pilot.ProxyExperimentError, match="deterministic recomputation"):
        engine._validate_aggregate(tampered, resolutions)

    extra = {**deepcopy(value), "promotion_authorized": True}
    extra_body = {key: item for key, item in extra.items() if key != "aggregate_digest"}
    extra["aggregate_digest"] = pilot._digest(extra_body)
    with pytest.raises(pilot.ProxyExperimentError, match="fields/schema"):
        engine._validate_aggregate(extra, resolutions)

    released = deepcopy(value)
    released["release_authorized"] = True
    released_body = {key: item for key, item in released.items() if key != "aggregate_digest"}
    released["aggregate_digest"] = pilot._digest(released_body)
    with pytest.raises(pilot.ProxyExperimentError, match="identity/release boundary"):
        engine._validate_aggregate(released, resolutions)
