#!/usr/bin/env python3
"""Run a live SPADE -> ProofPack -> Assay integration smoke test.

The live command is deliberately an evidence-producing smoke test, not a model
promotion ceremony. It generates one environment, qualifies it with ProofPack,
performs paired hinted/unhinted rollouts, and asks Assay to persist the manifest
and statistical decision. Assay must reject promotion when the single live task
does not meet its independent-cluster requirement.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import shutil
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Editable sibling checkouts are a development convenience. Installed use is
# supported by importing the packages normally before considering siblings.
PROOFPACK_CORE_PATH = ROOT_DIR.parent / "proofpack" / "packages" / "core" / "src"
PROOFPACK_ENV_PATH = ROOT_DIR.parent / "proofpack" / "packages" / "env" / "src"
ASSAY_PATH = ROOT_DIR.parent / "assay"
for source_path in (PROOFPACK_CORE_PATH, PROOFPACK_ENV_PATH, ASSAY_PATH):
    if source_path.exists() and str(source_path) not in sys.path:
        sys.path.append(str(source_path))

PROOFPACK_IMPORT_ERROR: Exception | None = None
try:
    from proofpack_env.spade_qualification import qualify_spade_environment
    from proofpack_env.spade_target import SpadeEnvironmentTarget
except (ImportError, SyntaxError) as exc:  # pragma: no cover - CLI dependency path
    PROOFPACK_IMPORT_ERROR = exc

ASSAY_IMPORT_ERROR: Exception | None = None
try:
    from assay.bench.spade_emitter import SpadeTaskPayload
    from assay.certify.spade_artifacts import SpadeRunMetadata, write_spade_evaluation
    from assay.certify.spade_evaluator import SpadeClusterData
except (ImportError, SyntaxError) as exc:  # pragma: no cover - CLI dependency path
    ASSAY_IMPORT_ERROR = exc

from spade.core.utils.parsing import extract_boxed_answer, parse_action


QUALIFIED_SEEDS = (0, 1, 42)
MAX_LLM_PROMPT_BYTES = 64 * 1024
MAX_LLM_RESPONSE_BYTES = 1024 * 1024
MAX_LLM_STDERR_BYTES = 256 * 1024
MAX_AGY_LOG_BYTES = 2 * 1024 * 1024
MAX_AGY_TRANSCRIPT_BYTES = 2 * 1024 * 1024
MAX_AGY_POLICY_CONFIG_BYTES = 4 * 1024
AGY_EVIDENCE_SCHEMA_V1 = "spade-agy-structured-evidence/v1"
AGY_EVIDENCE_SCHEMA_V2 = "spade-agy-structured-evidence/v2"
AGY_EVIDENCE_SCHEMA = "spade-agy-structured-evidence/v3"
AGY_OUTPUT_FORMAT = "stream-json"
AGY_LOG_POLICY = "isolated-gemini-dir-unique-private-log-sanitized-receipt"
AGY_LOG_RECEIPT_SCHEMA_V1 = "spade-agy-sanitized-log-receipt/v1"
AGY_LOG_RECEIPT_SCHEMA_V2 = "spade-agy-sanitized-log-receipt/v2"
AGY_LOG_RECEIPT_SCHEMA = "spade-agy-sanitized-log-receipt/v3"
AGY_STREAM_RECEIPT_SCHEMA = "spade-agy-sanitized-stream-receipt/v1"
AGY_STDERR_RECEIPT_SCHEMA_V1 = "spade-agy-sanitized-stderr-receipt/v1"
AGY_STDERR_RECEIPT_SCHEMA = "spade-agy-sanitized-stderr-receipt/v2"
AGY_TRANSCRIPT_RECEIPT_SCHEMA = "spade-agy-sanitized-transcript-receipt/v1"
AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1 = "spade-agy-policy-config-transition/v1"
AGY_POLICY_CONFIG_TRANSITION_SCHEMA = "spade-agy-policy-config-transition/v2"
AGY_SANDBOX_INVOCATION_RECEIPT_SCHEMA = "spade-agy-sandbox-invocation-receipt/v1"
MAX_AGY_POLICY_GRANT_ENTRIES = 256
MAX_AGY_POLICY_GRANT_BYTES = 512
AGY_POLICY_CONFIG_WRITE_PROTECTION = "darwin-sandbox-exec-agy-workdir-jail/v1"
AGY_SANDBOX_EXECUTABLE = Path("/usr/bin/sandbox-exec")
EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST = (
    "sha256:7fa7df193f26e32cc740e38d55eae13b31a9a98165b9fd9d03473a96e5b37284"
)
AGY_POLICY_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal (param "AGY_BIN")))
(deny file-write*)
(allow file-write* (subpath (param "WORKDIR")))
(allow file-write* (literal "/dev/null"))
(deny file-write* (literal "/"))
(deny file-write* (literal "/private"))
(deny file-write* (literal "/private/tmp"))
(deny file-write* (literal (param "WORKDIR")))
(deny file-write* (literal (param "GEMINI_DIR")))
(deny file-write* (literal (param "CONFIG_DIR")))
(deny file-write* (literal (param "CONFIG_FILE")))
(deny file-write* (subpath (param "CONFIG_DIR")))
"""
AGY_POLICY_SANDBOX_PROFILE_DIGEST = (
    "sha256:e5e5567b476d9186847ad8f14e066912ab8d0c68af3d8cb89d8b0a564f6f37c9"
)
AGY_SEALED_POLICY_CONFIG = (
    b'{"userSettings":{"globalPermissionGrants":{"allow":[],"deny":[],"ask":[]}}}\n'
)
_AGY_RESULT_STATUSES = {
    "UNKNOWN",
    "ERROR",
    "SUCCESS",
    "INVALID",
    "CANCELED",
    "WAITING",
    "INTERRUPTED",
    "RUNNING",
}
_AGY_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cache_read_tokens",
    "total_tokens",
}
_AGY_STEP_STATES = {"ACTIVE", "DONE", "ERROR"}
_AGY_STEP_TYPES = {
    "agent_response",
    "user_input",
    "finish",
    "tool",
    "unknown",
    "subagent",
    "system_message",
    "checkpoint",
    "error_message",
}

_AGY_RESPONSE_ID = re.compile(r"\bResponseID:\s*([A-Za-z0-9_-]+)")
_AGY_CONVERSATION_ID = re.compile(
    r"\b(?:Created conversation|Print mode: conversation=|Streaming conversation|"
    r"Stream completed for)\s*"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
_AGY_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_AGY_APP_DATA_DIR = re.compile(r"CLI app data directory:\s*([^\r\n]+)")
_AGY_PRINT_MODEL = re.compile(r'Print mode: starting \([^\r\n]*model="([^"]+)"')
_AGY_WORKSPACE = re.compile(r"workspaceDirs=\[([^\]\r\n]+)\]")
_AGY_SOFT_DENIAL = re.compile(r'soft-denying tool confirmation "([^"]+)"')
_AGY_TOOL_CONFIRMATION = re.compile(
    r"Tool confirmation for conversation\s+"
    r"([0-9a-f-]{36})[^\r\n]*?approved=(true|false)",
    re.IGNORECASE,
)
_AGY_SHARED_CONFIG = re.compile(r"stored shared config permissions:[^\r\n]*?\sfrom\s([^\r\n]+)")
_AGY_NO_SHARED_CONFIG = re.compile(
    r"applyUserSettings: no shared config permissions from\s+([^\r\n]+)"
)
_AGY_UNSAFE_CONFIG_FALLBACK = re.compile(
    r"(?:Global config\.json not found|Failed to reload config\.json)[^\r\n]*"
    r"Using cached permissions",
    re.IGNORECASE,
)
_AGY_NESTED_SANDBOX_FAILURE = re.compile(
    r"(?i)(?:failed to (?:start|initialize|create|enter)[^\r\n]{0,80}sandbox|"
    r"sandbox (?:initialization|setup|startup)[^\r\n]{0,40}(?:failed|error)|"
    r"(?:sandbox-exec|sandbox_apply):[^\r\n]{0,120}(?:error|deny|operation not permitted|"
    r"eperm))"
)
_AGY_EXPLICIT_PRE_RESPONSE_FAILURE = re.compile(
    r"(?ix)(?:"
    r"\b(?:http(?:\s+status)?|status(?:\s+code)?|code)\s*[:=]?\s*(?:429|5[0-9]{2})\b|"
    r"\bresource[_ -]?exhausted\b|"
    r"\brate[_ -]?limit(?:ed|ing)?\b|"
    r"\b(?:service|provider|backend|model)\s+(?:is\s+)?(?:temporarily\s+)?unavailable\b|"
    r"\b(?:provider|backend|model)\s+(?:request\s+)?refus(?:ed|al)\b|"
    r"\boverload(?:ed)?\b"
    r")"
)

DESIGNER_PROMPT = """You are an expert Environment Designer. Create a clean,
self-contained Python environment for a cognitive reasoning puzzle in the skill
area: {skill}.

Security and interface requirements:
1. Import only modules explicitly allowed by ProofPack: math, random, re,
   heapq, collections, itertools, typing, dataclasses, json, functools, string,
   and copy. Never access files, the network, processes, environment variables,
   frames, globals, or Python object internals.
2. Do not use any ProofPack-disallowed identifier anywhere, even as an
   ordinary variable, parameter, import alias, class/instance attribute, or
   indirect capability name. These words may appear in user-facing string
   literals, but never as Python identifiers. In particular, never use:
   system, modules, breakpoint, compile, delattr, eval, exec, getattr, globals,
   hasattr, input, locals, memoryview, open, setattr, vars, __import__, builtins,
   currentframe, f_back, f_builtins, f_code, f_globals, f_locals, fork, forkpty,
   importlib, inspect, os, pathlib, popen, subprocess, sys, tb_frame, types.
3. The class name must end in Env.
4. Implement __init__(self, max_turns=10, **kwargs).
5. Implement reset(self, seed=None) -> (observation: str, info: dict).
6. Implement solution() returning one answer or a list of turn-by-turn actions.
7. Implement step(action) -> (observation, reward, terminated, truncated, info).
8. Every seed must be deterministic. A correct completed episode returns 1.0;
   incorrect actions return 0.0 and must not terminate immediately.
9. Tell the player to respond with \\boxed{{action}}.
10. Keep the complete source at or below 120 nonblank lines and 8,000
    characters, including comments and docstrings. Implement only the required
    puzzle and interface; do not add a framework, tutorial, tests, or prose.

Return only executable Python inside one ```python ... ``` block."""

HINT_PROMPT = """You are an expert tutor. Give a high-level strategy for the
puzzle observation below. Do not provide the final answer, a boxed answer, an
exact action sequence, or values not visible in the observation. Keep the hint
under 80 words.

Observation:
{observation}

Return only the hint text."""


class LiveEvalError(RuntimeError):
    """Expected live-evaluation failure with a stable CLI exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class AgyCallEvidence:
    """Bounded raw evidence and a classification derived only from that evidence."""

    evidence_schema_version: str
    disposition: str
    response: str | None
    error: str | None
    requested_model: str
    reported_model: str | None
    invocation_workdir: str
    exit_status: int | None
    timed_out: bool
    process_group_descendant_detected: bool
    process_group_quiescent: bool
    capture_failures: tuple[str, ...]
    response_ids: tuple[str, ...]
    conversation_id: str | None
    stream_event_count: int
    terminal_event_digest: str | None
    tool_call_names: tuple[str, ...]
    policy_config_identity: Mapping[str, Any]
    policy_config_transition: Mapping[str, Any]
    sandbox_invocation_receipt: Mapping[str, Any]
    raw_log_digest: str
    raw_log_size_bytes: int
    raw_stream_digest: str
    raw_stream_size_bytes: int
    raw_stderr_digest: str
    raw_stderr_size_bytes: int
    raw_transcript_digest: str
    raw_transcript_size_bytes: int
    stdout_ndjson: bytes
    stderr: bytes
    log: bytes
    transcript: bytes

    def summary(self) -> dict[str, Any]:
        """Return the exact JSON-safe classification receipt."""
        result = {
            "schema_version": self.evidence_schema_version,
            "output_format": AGY_OUTPUT_FORMAT,
            "log_policy": AGY_LOG_POLICY,
            "disposition": self.disposition,
            "requested_model": self.requested_model,
            "reported_model": self.reported_model,
            "invocation_workdir": self.invocation_workdir,
            "exit_status": self.exit_status,
            "timed_out": self.timed_out,
            "capture_failures": list(self.capture_failures),
            "response_ids": list(self.response_ids),
            "conversation_id": self.conversation_id,
            "stream_event_count": self.stream_event_count,
            "terminal_event_digest": self.terminal_event_digest,
            "tool_call_names": list(self.tool_call_names),
            "raw_log_digest": self.raw_log_digest,
            "raw_log_size_bytes": self.raw_log_size_bytes,
            "raw_stream_digest": self.raw_stream_digest,
            "raw_stream_size_bytes": self.raw_stream_size_bytes,
            "raw_stderr_digest": self.raw_stderr_digest,
            "raw_stderr_size_bytes": self.raw_stderr_size_bytes,
            "raw_transcript_digest": self.raw_transcript_digest,
            "raw_transcript_size_bytes": self.raw_transcript_size_bytes,
        }
        if self.evidence_schema_version == AGY_EVIDENCE_SCHEMA_V1:
            result["policy_config_identity"] = dict(self.policy_config_identity)
        elif self.evidence_schema_version == AGY_EVIDENCE_SCHEMA_V2:
            result["policy_config_transition"] = dict(self.policy_config_transition)
            result["process_group_descendant_detected"] = (
                self.process_group_descendant_detected
            )
            result["process_group_quiescent"] = self.process_group_quiescent
        else:
            result["policy_config_transition"] = dict(self.policy_config_transition)
            result["sandbox_invocation_receipt"] = dict(
                self.sandbox_invocation_receipt
            )
            result["process_group_descendant_detected"] = (
                self.process_group_descendant_detected
            )
            result["process_group_quiescent"] = self.process_group_quiescent
        return result


class _AgyStreamLimit(RuntimeError):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def _canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _event_kind(event: Mapping[str, Any]) -> str | None:
    value = event.get("event")
    if value not in {"init", "step_update", "result"}:
        return None
    return str(value)


def _event_payload(event: Mapping[str, Any], kind: str | None) -> Mapping[str, Any] | None:
    if kind is None:
        return None
    value = event.get(kind)
    return value if isinstance(value, dict) else None


def _parse_ndjson(raw: bytes, label: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], f"{label}_invalid_utf8"
    if "\x00" in text:
        return [], f"{label}_contains_nul"
    events: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            return [], f"{label}_blank_line_{index}"
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return [], f"{label}_invalid_json_{index}"
        if not isinstance(value, dict):
            return [], f"{label}_nonobject_{index}"
        events.append(value)
    if not events:
        return [], f"{label}_empty"
    return events, None


def _terminal_response(event: Mapping[str, Any]) -> str | None:
    for key in ("response", "result", "output"):
        value = event.get(key)
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            for nested_key in ("response", "result", "output", "content", "text"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return None


def _terminal_error(event: Mapping[str, Any]) -> str:
    fragments: list[str] = []
    for key in ("error", "message", "status"):
        value = event.get(key)
        if isinstance(value, str) and value:
            fragments.append(value)
        elif isinstance(value, dict):
            fragments.extend(str(item) for item in value.values() if isinstance(item, str))
    return " ".join(fragments)


def _tool_name(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _stream_tool_names(events: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for event in events:
        if _event_kind(event) != "step_update":
            continue
        payload = _event_payload(event, "step_update")
        if payload is None:
            continue
        info = payload.get("tool_info")
        if isinstance(info, dict):
            name = _tool_name(info.get("name"))
            if name:
                names.add(name)
        name = _tool_name(payload.get("tool_name"))
        if name:
            names.add(name)
    return tuple(sorted(names, key=str.casefold))


def _stream_tool_execution_observed(events: Sequence[Mapping[str, Any]]) -> bool:
    for event in events:
        if _event_kind(event) != "step_update":
            continue
        payload = _event_payload(event, "step_update")
        info = payload.get("tool_info") if payload is not None else None
        if payload is not None and payload.get("subagent_info") not in (None, "", [], {}):
            return True
        if not isinstance(info, dict):
            continue
        if info.get("output") not in (None, "", [], {}) or info.get("error") not in (
            None,
            "",
            [],
            {},
        ):
            return True
    return False


def _normalized_tool_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reported_stream_models(events: Sequence[Mapping[str, Any]]) -> set[str]:
    models: set[str] = set()
    for event in events:
        if _event_kind(event) != "init":
            continue
        payload = _event_payload(event, "init")
        if payload is None:
            continue
        value = payload.get("model")
        if isinstance(value, str) and value:
            models.add(value)
        elif isinstance(value, dict):
            for key in ("id", "model", "name"):
                nested = value.get(key)
                if isinstance(nested, str) and nested:
                    models.add(nested)
    return models


def _stream_conversation_ids(events: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[str] = []
    for event in events:
        kind = _event_kind(event)
        payload = _event_payload(event, kind)
        for candidate in (
            event.get("conversation_id"),
            payload.get("conversation_id") if payload is not None else None,
        ):
            if isinstance(candidate, str) and candidate:
                values.append(candidate)
    return tuple(dict.fromkeys(values))


def _init_tool_names(payload: Mapping[str, Any]) -> tuple[str, ...] | None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return None
    names: list[str] = []
    for item in tools:
        if isinstance(item, str):
            name = _tool_name(item)
        elif isinstance(item, dict):
            name = _tool_name(item.get("name") or item.get("tool_name"))
        else:
            name = None
        if name is None:
            return None
        names.append(name)
    if len(set(names)) != len(names):
        return None
    return tuple(names)


def _stream_shape_error(events: Sequence[Mapping[str, Any]]) -> str | None:
    for index, event in enumerate(events, start=1):
        kind = _event_kind(event)
        payload = _event_payload(event, kind)
        if kind is None or payload is None:
            return f"stream_event_{index}_missing_discriminated_payload"
        if event.get("command") is not None or any(
            event.get(other) is not None for other in {"init", "step_update", "result"} - {kind}
        ):
            return f"stream_event_{index}_has_conflicting_payload"
        envelope_id = event.get("conversation_id")
        if envelope_id is not None and (
            not isinstance(envelope_id, str) or _AGY_UUID.fullmatch(envelope_id) is None
        ):
            return f"stream_event_{index}_conversation_id_invalid"
        payload_id = payload.get("conversation_id")
        if kind in {"step_update", "result"} and (
            not isinstance(payload_id, str)
            or _AGY_UUID.fullmatch(payload_id) is None
            or envelope_id is not None
            and payload_id != envelope_id
        ):
            return f"stream_event_{index}_payload_conversation_id_invalid"
        if kind == "init" and (
            not isinstance(payload.get("model"), str)
            or not isinstance(payload.get("cwd"), str)
            or payload.get("permission_mode") != "request-review"
            or _init_tool_names(payload) is None
        ):
            return "stream_init_route_or_cwd_invalid"
        if kind == "step_update":
            step_index = payload.get("step_index")
            if (
                isinstance(step_index, bool)
                or not isinstance(step_index, int)
                or step_index < 0
                or not isinstance(payload.get("state"), str)
                or not payload["state"]
                or payload["state"] not in _AGY_STEP_STATES
                or payload.get("step_type") not in _AGY_STEP_TYPES
            ):
                return f"stream_event_{index}_step_fields_invalid"
            tool_shape_present = (
                any(payload.get(key) is not None for key in ("tool_name", "tool_info"))
                or "tool" in str(payload.get("step_type", "")).casefold()
            )
            if tool_shape_present and not _stream_tool_names((event,)):
                return f"stream_event_{index}_tool_shape_unclassified"
        if kind == "result":
            duration = payload.get("duration_seconds")
            turns = payload.get("num_turns")
            usage = payload.get("usage")
            if (
                not isinstance(payload.get("status"), str)
                or not isinstance(payload.get("response"), str)
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) < 0
                or isinstance(turns, bool)
                or not isinstance(turns, int)
                or turns < 0
                or payload.get("status") not in _AGY_RESULT_STATUSES
                or not isinstance(usage, dict)
                or set(usage) != _AGY_USAGE_FIELDS
                or any(
                    isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or token_count < 0
                    for token_count in usage.values()
                )
            ):
                return "stream_result_fields_invalid"
    return None


def _sensitive_value_receipt(value: object) -> dict[str, Any]:
    encoded = _canonical_json_bytes(value)
    return {"digest": _bytes_digest(encoded), "size_bytes": len(encoded)}


def _sensitive_receipt_is_valid(value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == {"digest", "size_bytes"}
        and isinstance(value.get("digest"), str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value["digest"]) is not None
        and not isinstance(value.get("size_bytes"), bool)
        and isinstance(value.get("size_bytes"), int)
        and 0 <= value["size_bytes"] <= MAX_LLM_RESPONSE_BYTES
    )


def _sanitize_stream_event(event: Mapping[str, Any]) -> dict[str, Any]:
    kind = _event_kind(event)
    payload = _event_payload(event, kind)
    if kind is None or payload is None:
        raise ValueError("undiscriminated stream event")
    envelope = {
        "event": kind,
        "conversation_id": event.get("conversation_id"),
        "raw_event_digest": _canonical_json_digest(event),
    }
    if kind == "init":
        tools = _init_tool_names(payload)
        if tools is None:
            raise ValueError("invalid init tool inventory")
        envelope["init"] = {
            "model": payload.get("model"),
            "cwd_digest": _bytes_digest(str(payload.get("cwd", "")).encode()),
            "permission_mode": payload.get("permission_mode"),
            "tools": list(tools),
        }
    elif kind == "step_update":
        raw_info = payload.get("tool_info")
        info: Mapping[str, Any] = raw_info if isinstance(raw_info, dict) else {}
        text_delta = payload.get("text_delta")
        envelope["step_update"] = {
            "conversation_id": payload.get("conversation_id"),
            "step_index": payload.get("step_index"),
            "state": payload.get("state"),
            "step_type": payload.get("step_type"),
            "tool_name": payload.get("tool_name"),
            "tool_info_present": isinstance(payload.get("tool_info"), dict),
            "tool_info_name": info.get("name"),
            "tool_parameters_receipt": _sensitive_value_receipt(info.get("parameters")),
            "tool_output_present": info.get("output") not in (None, "", [], {}),
            "tool_error_present": info.get("error") not in (None, "", [], {}),
            "text_delta_receipt": _sensitive_value_receipt(text_delta),
            "subagent_info_present": payload.get("subagent_info") not in (None, "", [], {}),
        }
    else:
        envelope["result"] = {
            "conversation_id": payload.get("conversation_id"),
            "status": payload.get("status"),
            "response": payload.get("response"),
            "duration_seconds": payload.get("duration_seconds"),
            "num_turns": payload.get("num_turns"),
            "usage": payload.get("usage"),
            "explicit_pre_response_failure_marker": bool(
                _AGY_EXPLICIT_PRE_RESPONSE_FAILURE.search(_terminal_error(payload))
            ),
        }
    return envelope


def _restore_stream_event(value: Mapping[str, Any], workdir: str) -> dict[str, Any]:
    kind = value.get("event")
    common = {
        "event": kind,
        "conversation_id": value.get("conversation_id"),
        "_raw_event_digest": value.get("raw_event_digest"),
    }
    if kind == "init":
        payload = value.get("init")
        if not isinstance(payload, dict) or set(payload) != {
            "model",
            "cwd_digest",
            "permission_mode",
            "tools",
        }:
            raise ValueError("sanitized init fields invalid")
        if payload.get("cwd_digest") != _bytes_digest(workdir.encode()):
            raise ValueError("sanitized init cwd digest invalid")
        common["init"] = {
            "model": payload.get("model"),
            "cwd": workdir,
            "permission_mode": payload.get("permission_mode"),
            "tools": payload.get("tools"),
        }
    elif kind == "step_update":
        payload = value.get("step_update")
        required = {
            "conversation_id",
            "step_index",
            "state",
            "step_type",
            "tool_name",
            "tool_info_present",
            "tool_info_name",
            "tool_parameters_receipt",
            "tool_output_present",
            "tool_error_present",
            "text_delta_receipt",
            "subagent_info_present",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("sanitized step fields invalid")
        for key in (
            "tool_info_present",
            "tool_output_present",
            "tool_error_present",
            "subagent_info_present",
        ):
            if type(payload.get(key)) is not bool:
                raise ValueError("sanitized step presence flag invalid")
        if not _sensitive_receipt_is_valid(
            payload.get("tool_parameters_receipt")
        ) or not _sensitive_receipt_is_valid(payload.get("text_delta_receipt")):
            raise ValueError("sanitized sensitive-value receipt invalid")
        info = None
        if payload.get("tool_info_present"):
            info = {
                "name": payload.get("tool_info_name"),
                "parameters": None,
                "output": "present" if payload.get("tool_output_present") else None,
                "error": "present" if payload.get("tool_error_present") else None,
            }
        common["step_update"] = {
            "conversation_id": payload.get("conversation_id"),
            "step_index": payload.get("step_index"),
            "state": payload.get("state"),
            "step_type": payload.get("step_type"),
            "tool_name": payload.get("tool_name"),
            "tool_info": info,
            "subagent_info": ({"present": True} if payload.get("subagent_info_present") else None),
        }
    elif kind == "result":
        payload = value.get("result")
        if not isinstance(payload, dict) or set(payload) != {
            "conversation_id",
            "status",
            "response",
            "duration_seconds",
            "num_turns",
            "usage",
            "explicit_pre_response_failure_marker",
        }:
            raise ValueError("sanitized result fields invalid")
        if type(payload.get("explicit_pre_response_failure_marker")) is not bool:
            raise ValueError("sanitized result failure marker invalid")
        result_payload = {
            key: item
            for key, item in payload.items()
            if key != "explicit_pre_response_failure_marker"
        }
        if payload["explicit_pre_response_failure_marker"]:
            result_payload["error"] = "provider request refusal"
        common["result"] = result_payload
    else:
        raise ValueError("sanitized event discriminator invalid")
    expected = {"event", "conversation_id", "raw_event_digest", str(kind)}
    if set(value) != expected:
        raise ValueError("sanitized stream event keys invalid")
    if (
        not isinstance(value.get("raw_event_digest"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("raw_event_digest"))) is None
    ):
        raise ValueError("sanitized raw event digest invalid")
    return common


def _stream_receipt(
    raw: bytes,
    *,
    workdir: str,
    sanitized: bool,
) -> tuple[list[dict[str, Any]], bytes, str | None, str, int]:
    if sanitized:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return [], b"", "sanitized_stream_receipt_invalid_json", _bytes_digest(raw), len(raw)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "raw_stream_digest",
            "raw_stream_size_bytes",
            "events",
            "shape_error",
            "receipt_digest",
        }:
            return [], b"", "sanitized_stream_receipt_schema_invalid", _bytes_digest(raw), len(raw)
        body = {key: item for key, item in value.items() if key != "receipt_digest"}
        if (
            value.get("schema_version") != AGY_STREAM_RECEIPT_SCHEMA
            or value.get("receipt_digest") != _canonical_json_digest(body)
            or raw != _canonical_json_bytes(value)
            or not isinstance(value.get("events"), list)
            or not isinstance(value.get("raw_stream_size_bytes"), int)
            or value["raw_stream_size_bytes"] < 0
            or value["raw_stream_size_bytes"] > MAX_LLM_RESPONSE_BYTES
            or not isinstance(value.get("raw_stream_digest"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["raw_stream_digest"]) is None
            or (
                value.get("shape_error") is not None
                and not isinstance(value.get("shape_error"), str)
            )
        ):
            return [], b"", "sanitized_stream_receipt_digest_invalid", _bytes_digest(raw), len(raw)
        try:
            events = [_restore_stream_event(item, workdir) for item in value["events"]]
        except (TypeError, ValueError):
            return [], b"", "sanitized_stream_event_invalid", _bytes_digest(raw), len(raw)
        reconstructed_error = _stream_shape_error(events)
        sealed_shape_error = value.get("shape_error")
        if sealed_shape_error is None and (not events or reconstructed_error is not None):
            return (
                [],
                b"",
                "sanitized_stream_receipt_shape_invalid",
                _bytes_digest(raw),
                len(raw),
            )
        return (
            events,
            raw,
            sealed_shape_error,
            str(value["raw_stream_digest"]),
            int(value["raw_stream_size_bytes"]),
        )

    events, parse_error = _parse_ndjson(raw, "stdout_ndjson")
    shape_error = _stream_shape_error(events) if parse_error is None else None
    sanitized_events: list[dict[str, Any]] = []
    if parse_error is None:
        try:
            sanitized_events = [_sanitize_stream_event(event) for event in events]
        except ValueError as exc:
            shape_error = f"stream_sanitization_failed:{exc}"
    body = {
        "schema_version": AGY_STREAM_RECEIPT_SCHEMA,
        "raw_stream_digest": _bytes_digest(raw),
        "raw_stream_size_bytes": len(raw),
        "events": sanitized_events,
        "shape_error": parse_error or shape_error,
    }
    value = {**body, "receipt_digest": _canonical_json_digest(body)}
    return (
        events,
        _canonical_json_bytes(value),
        parse_error or shape_error,
        body["raw_stream_digest"],
        body["raw_stream_size_bytes"],
    )


def _stderr_receipt(
    raw: bytes,
    *,
    sanitized: bool,
) -> tuple[dict[str, Any], bytes, str | None, str, int]:
    """Retain stderr classification facts without persisting its sensitive text."""
    legacy_required = {
        "schema_version",
        "raw_stderr_digest",
        "raw_stderr_size_bytes",
        "explicit_pre_response_failure_marker",
        "shape_error",
        "receipt_digest",
    }
    required = {*legacy_required, "nested_sandbox_failure_marker"}
    if sanitized:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {}, b"", "sanitized_stderr_receipt_invalid_json", _bytes_digest(raw), len(raw)
        body = (
            {key: item for key, item in value.items() if key != "receipt_digest"}
            if isinstance(value, dict)
            else {}
        )
        if (
            not isinstance(value, dict)
            or (
                value.get("schema_version") == AGY_STDERR_RECEIPT_SCHEMA_V1
                and set(value) != legacy_required
            )
            or (
                value.get("schema_version") == AGY_STDERR_RECEIPT_SCHEMA
                and set(value) != required
            )
            or value.get("schema_version")
            not in {AGY_STDERR_RECEIPT_SCHEMA_V1, AGY_STDERR_RECEIPT_SCHEMA}
            or value.get("receipt_digest") != _canonical_json_digest(body)
            or raw != _canonical_json_bytes(value)
            or not isinstance(value.get("raw_stderr_digest"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["raw_stderr_digest"]) is None
            or isinstance(value.get("raw_stderr_size_bytes"), bool)
            or not isinstance(value.get("raw_stderr_size_bytes"), int)
            or not 0 <= value["raw_stderr_size_bytes"] <= MAX_LLM_STDERR_BYTES
            or type(value.get("explicit_pre_response_failure_marker")) is not bool
            or (
                value.get("schema_version") == AGY_STDERR_RECEIPT_SCHEMA
                and type(value.get("nested_sandbox_failure_marker")) is not bool
            )
            or value.get("shape_error") not in {None, "raw_stderr_invalid_utf8"}
        ):
            return {}, b"", "sanitized_stderr_receipt_schema_invalid", _bytes_digest(raw), len(raw)
        return (
            body,
            raw,
            value.get("shape_error"),
            str(value["raw_stderr_digest"]),
            int(value["raw_stderr_size_bytes"]),
        )

    shape_error: str | None = None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
        shape_error = "raw_stderr_invalid_utf8"
    body = {
        "schema_version": AGY_STDERR_RECEIPT_SCHEMA,
        "raw_stderr_digest": _bytes_digest(raw),
        "raw_stderr_size_bytes": len(raw),
        "explicit_pre_response_failure_marker": bool(
            _AGY_EXPLICIT_PRE_RESPONSE_FAILURE.search(text)
        ),
        "nested_sandbox_failure_marker": bool(
            _AGY_NESTED_SANDBOX_FAILURE.search(text)
        ),
        "shape_error": shape_error,
    }
    value = {**body, "receipt_digest": _canonical_json_digest(body)}
    return (
        body,
        _canonical_json_bytes(value),
        shape_error,
        body["raw_stderr_digest"],
        body["raw_stderr_size_bytes"],
    )


def _transcript_receipt(
    raw: bytes,
    *,
    full_prompt: str,
    sanitized: bool,
) -> tuple[dict[str, Any], bytes, str | None, str, int]:
    expected_prompt_digest = _bytes_digest(full_prompt.encode())
    if sanitized:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return (
                {},
                b"",
                "sanitized_transcript_receipt_invalid_json",
                _bytes_digest(raw),
                len(raw),
            )
        required = {
            "schema_version",
            "raw_transcript_digest",
            "raw_transcript_size_bytes",
            "prompt_digest",
            "explicit_user_inputs",
            "prompt_matches",
            "model_content_digests",
            "tool_call_names",
            "tool_execution_observed",
            "shape_error",
            "receipt_digest",
        }
        body = (
            {key: item for key, item in value.items() if key != "receipt_digest"}
            if isinstance(value, dict)
            else {}
        )
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("schema_version") != AGY_TRANSCRIPT_RECEIPT_SCHEMA
            or value.get("receipt_digest") != _canonical_json_digest(body)
            or raw != _canonical_json_bytes(value)
            or value.get("prompt_digest") != expected_prompt_digest
            or isinstance(value.get("raw_transcript_size_bytes"), bool)
            or not isinstance(value.get("raw_transcript_size_bytes"), int)
            or value["raw_transcript_size_bytes"] < 0
            or value["raw_transcript_size_bytes"] > MAX_AGY_TRANSCRIPT_BYTES
            or not isinstance(value.get("raw_transcript_digest"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["raw_transcript_digest"]) is None
            or type(value.get("tool_execution_observed")) is not bool
            or (
                value.get("shape_error") is not None
                and not isinstance(value.get("shape_error"), str)
            )
            or any(
                isinstance(value.get(key), bool)
                or not isinstance(value.get(key), int)
                or value[key] < 0
                for key in ("explicit_user_inputs", "prompt_matches")
            )
            or not isinstance(value.get("model_content_digests"), list)
            or not all(
                isinstance(item, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item)
                for item in value.get("model_content_digests", [])
            )
            or not isinstance(value.get("tool_call_names"), list)
            or not all(isinstance(item, str) and item for item in value.get("tool_call_names", []))
        ):
            return (
                {},
                b"",
                "sanitized_transcript_receipt_schema_invalid",
                _bytes_digest(raw),
                len(raw),
            )
        return (
            value,
            raw,
            value.get("shape_error"),
            str(value["raw_transcript_digest"]),
            int(value["raw_transcript_size_bytes"]),
        )

    events, parse_error = _parse_ndjson(raw, "transcript")
    explicit_user_inputs = 0
    prompt_matches = 0
    model_content_digests: list[str] = []
    tool_names: set[str] = set()
    tool_execution = False
    shape_error: str | None = parse_error
    if parse_error is None:
        for event in events:
            source = event.get("source")
            event_type = event.get("type")
            status = event.get("status")
            if source == "USER_EXPLICIT" and event_type == "USER_INPUT":
                explicit_user_inputs += 1
                if status != "DONE":
                    shape_error = "full_transcript_user_input_not_done"
                content = event.get("content")
                if isinstance(content, str):
                    match = re.search(r"<USER_REQUEST>\n(.*?)\n</USER_REQUEST>", content, re.DOTALL)
                    if match and match.group(1) == full_prompt:
                        prompt_matches += 1
            if source == "TOOL" or "TOOL" in str(event_type or "").upper():
                tool_execution = True
            if source != "MODEL":
                continue
            if event_type != "PLANNER_RESPONSE":
                tool_execution = True
            content = event.get("content")
            if (
                event_type == "PLANNER_RESPONSE"
                and status == "DONE"
                and isinstance(content, str)
                and content.strip()
            ):
                model_content_digests.append(_bytes_digest(content.strip().encode()))
            if event.get("truncated_fields") is not None:
                shape_error = "full_transcript_contains_truncated_fields"
            tool_calls = event.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                shape_error = "full_transcript_tool_calls_invalid"
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    name = _tool_name(call.get("name")) if isinstance(call, dict) else None
                    if name is None:
                        shape_error = "full_transcript_tool_call_invalid"
                    else:
                        tool_names.add(name)
    body = {
        "schema_version": AGY_TRANSCRIPT_RECEIPT_SCHEMA,
        "raw_transcript_digest": _bytes_digest(raw),
        "raw_transcript_size_bytes": len(raw),
        "prompt_digest": expected_prompt_digest,
        "explicit_user_inputs": explicit_user_inputs,
        "prompt_matches": prompt_matches,
        "model_content_digests": model_content_digests,
        "tool_call_names": sorted(tool_names, key=str.casefold),
        "tool_execution_observed": tool_execution,
        "shape_error": shape_error,
    }
    value = {**body, "receipt_digest": _canonical_json_digest(body)}
    return (
        body,
        _canonical_json_bytes(value),
        shape_error,
        body["raw_transcript_digest"],
        body["raw_transcript_size_bytes"],
    )


def _validate_policy_config_identity(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "relative_path",
        "exists",
        "digest",
        "size_bytes",
    }:
        return None
    if value.get("relative_path") != "config/config.json":
        return None
    exists = value.get("exists")
    size = value.get("size_bytes")
    digest = value.get("digest")
    if type(exists) is not bool or isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return None
    if exists:
        if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            return None
    elif digest is not None or size != 0:
        return None
    return dict(value)


class _PolicyConfigDuplicateKey(ValueError):
    """A duplicate key in policy config; the key itself is deliberately not retained."""


class _PolicyConfigConstant(ValueError):
    """A non-finite JSON constant in policy config."""


_POLICY_CONFIG_FAILURES = {
    "policy_config_initial_missing",
    "policy_config_initial_invalid",
    "policy_config_root_unsafe",
    "policy_config_private_root_invalid",
    "policy_config_private_owner_invalid",
    "policy_config_final_symlink",
    "policy_config_final_missing",
    "policy_config_final_not_regular",
    "policy_config_final_limit_exceeded",
    "policy_config_final_read_failed",
    "policy_config_final_changed_during_read",
    "policy_config_final_identity_changed",
    "policy_config_final_content_changed",
    "policy_config_write_protection_unsealed",
    "policy_config_invalid_utf8",
    "policy_config_contains_nul",
    "policy_config_duplicate_key",
    "policy_config_invalid_json",
    "policy_config_nonobject_root",
    "policy_config_unknown_root_field",
    "policy_config_plugins_nonempty",
    "policy_config_user_settings_missing",
    "policy_config_user_settings_nonobject",
    "policy_config_unknown_user_setting",
    "policy_config_privileged_field_present",
    "policy_config_global_grants_nonobject",
    "policy_config_global_grants_unknown_bucket",
    "policy_config_invalid_grant_list",
    "policy_config_allow_grants_nonempty",
    "policy_config_nonempty_grants",
    "policy_config_invalid_legacy_commands",
    "policy_config_legacy_allowed_commands_nonempty",
    "policy_config_agent_environment_invalid",
    "policy_config_agent_environment_nonempty",
    "policy_config_permissive_setting",
    "policy_config_cosmetic_setting_invalid",
}


def _policy_config_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _PolicyConfigDuplicateKey
        result[key] = value
    return result


def _reject_policy_config_constant(value: str) -> Any:
    del value
    raise _PolicyConfigConstant


def _policy_config_transition_body(
    *,
    transition_class: str,
    initial_config_present: bool,
    final_config_present: bool,
    private_regular_file: bool,
    strict_json_object: bool,
    schema_allowlisted: bool,
    no_explicit_grants: bool,
    write_protection_applied: bool,
    write_protection_parameters_bound: bool,
    same_file_identity: bool,
    exact_precreated_payload: bool,
    grant_counts: Mapping[str, int | None] | None = None,
    legacy_counts: Mapping[str, int | None] | None = None,
    agent_environment_entry_count: int | None = None,
    unknown_root_field_count: int = 0,
    unknown_user_setting_count: int = 0,
    capture_failure: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": AGY_POLICY_CONFIG_TRANSITION_SCHEMA,
        "transition_class": transition_class,
        "initial_config_present": initial_config_present,
        "final_config_present": final_config_present,
        "private_regular_file": private_regular_file,
        "strict_json_object": strict_json_object,
        "schema_allowlisted": schema_allowlisted,
        "no_explicit_grants": no_explicit_grants,
        "write_protection_policy": AGY_POLICY_CONFIG_WRITE_PROTECTION,
        "write_protection_profile_digest": AGY_POLICY_SANDBOX_PROFILE_DIGEST,
        "write_protection_applied": write_protection_applied,
        "write_protection_parameters_bound": write_protection_parameters_bound,
        "same_file_identity": same_file_identity,
        "exact_precreated_payload": exact_precreated_payload,
        "global_permission_grant_counts": dict(
            grant_counts or {"allow": None, "deny": None, "ask": None}
        ),
        "legacy_command_counts": dict(legacy_counts or {"allow": None, "deny": None}),
        "agent_environment_entry_count": agent_environment_entry_count,
        "unknown_root_field_count": unknown_root_field_count,
        "unknown_user_setting_count": unknown_user_setting_count,
        "raw_config_persisted": False,
        "scalar_values_persisted": False,
        "config_path_persisted": False,
        "raw_config_digest_persisted": False,
        "capture_failure": capture_failure,
    }


def _seal_policy_config_transition(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "receipt_digest": _canonical_json_digest(value)}


def _unchanged_policy_config_transition() -> dict[str, Any]:
    return _seal_policy_config_transition(
        _policy_config_transition_body(
            transition_class="precreated-to-unchanged-empty-grants",
            initial_config_present=True,
            final_config_present=True,
            private_regular_file=True,
            strict_json_object=True,
            schema_allowlisted=True,
            no_explicit_grants=True,
            write_protection_applied=True,
            write_protection_parameters_bound=True,
            same_file_identity=True,
            exact_precreated_payload=True,
            grant_counts={"allow": 0, "deny": 0, "ask": 0},
            legacy_counts={"allow": 0, "deny": 0},
            agent_environment_entry_count=0,
        )
    )


def _unsafe_policy_config_transition(
    failure: str,
    *,
    initial_config_present: bool = True,
    final_config_present: bool = True,
    private_regular_file: bool = False,
    strict_json_object: bool = False,
    schema_allowlisted: bool = False,
    write_protection_applied: bool = True,
    write_protection_parameters_bound: bool = True,
    same_file_identity: bool = False,
    exact_precreated_payload: bool = False,
    unknown_root_field_count: int = 0,
    unknown_user_setting_count: int = 0,
) -> dict[str, Any]:
    if failure not in _POLICY_CONFIG_FAILURES:
        raise ValueError("unregistered policy-config failure")
    return _seal_policy_config_transition(
        _policy_config_transition_body(
            transition_class="unsafe",
            initial_config_present=initial_config_present,
            final_config_present=final_config_present,
            private_regular_file=private_regular_file,
            strict_json_object=strict_json_object,
            schema_allowlisted=schema_allowlisted,
            no_explicit_grants=False,
            write_protection_applied=write_protection_applied,
            write_protection_parameters_bound=write_protection_parameters_bound,
            same_file_identity=same_file_identity,
            exact_precreated_payload=exact_precreated_payload,
            unknown_root_field_count=unknown_root_field_count,
            unknown_user_setting_count=unknown_user_setting_count,
            capture_failure=failure,
        )
    )


def _legacy_unchanged_policy_config_transition() -> dict[str, Any]:
    """Return the frozen prospective-v2 transition shape for replay tests/readers."""
    current = _policy_config_transition_body(
        transition_class="precreated-to-unchanged-empty-grants",
        initial_config_present=True,
        final_config_present=True,
        private_regular_file=True,
        strict_json_object=True,
        schema_allowlisted=True,
        no_explicit_grants=True,
        write_protection_applied=True,
        write_protection_parameters_bound=True,
        same_file_identity=True,
        exact_precreated_payload=True,
        grant_counts={"allow": 0, "deny": 0, "ask": 0},
        legacy_counts={"allow": 0, "deny": 0},
        agent_environment_entry_count=0,
    )
    current["schema_version"] = AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1
    current["transition_class"] = "precreated-to-unchanged-restrictive"
    current["restrictive_permissions"] = current.pop("no_explicit_grants")
    return _seal_policy_config_transition(current)


def _validate_policy_config_transition(value: object) -> dict[str, Any] | None:
    common_keys = {
        "schema_version",
        "transition_class",
        "initial_config_present",
        "final_config_present",
        "private_regular_file",
        "strict_json_object",
        "schema_allowlisted",
        "write_protection_policy",
        "write_protection_profile_digest",
        "write_protection_applied",
        "write_protection_parameters_bound",
        "same_file_identity",
        "exact_precreated_payload",
        "global_permission_grant_counts",
        "legacy_command_counts",
        "agent_environment_entry_count",
        "unknown_root_field_count",
        "unknown_user_setting_count",
        "raw_config_persisted",
        "scalar_values_persisted",
        "config_path_persisted",
        "raw_config_digest_persisted",
        "capture_failure",
        "receipt_digest",
    }
    schema = value.get("schema_version") if isinstance(value, dict) else None
    semantic_field = (
        "restrictive_permissions"
        if schema == AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1
        else "no_explicit_grants"
    )
    expected_keys = {*common_keys, semantic_field}
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    body = {key: item for key, item in value.items() if key != "receipt_digest"}
    if (
        value.get("schema_version")
        not in {AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1, AGY_POLICY_CONFIG_TRANSITION_SCHEMA}
        or value.get("receipt_digest") != _canonical_json_digest(body)
    ):
        return None
    boolean_fields = {
        "initial_config_present",
        "final_config_present",
        "private_regular_file",
        "strict_json_object",
        "schema_allowlisted",
        semantic_field,
        "write_protection_applied",
        "write_protection_parameters_bound",
        "same_file_identity",
        "exact_precreated_payload",
        "raw_config_persisted",
        "scalar_values_persisted",
        "config_path_persisted",
        "raw_config_digest_persisted",
    }
    if (
        value.get("write_protection_policy") != AGY_POLICY_CONFIG_WRITE_PROTECTION
        or value.get("write_protection_profile_digest")
        != AGY_POLICY_SANDBOX_PROFILE_DIGEST
        or any(type(value.get(key)) is not bool for key in boolean_fields)
        or any(
        value[key] is not False
        for key in {
            "raw_config_persisted",
            "scalar_values_persisted",
            "config_path_persisted",
            "raw_config_digest_persisted",
        }
        )
    ):
        return None
    grants = value.get("global_permission_grant_counts")
    legacy = value.get("legacy_command_counts")
    if not isinstance(grants, dict) or set(grants) != {"allow", "deny", "ask"}:
        return None
    if not isinstance(legacy, dict) or set(legacy) != {"allow", "deny"}:
        return None
    for count in (*grants.values(), *legacy.values(), value.get("agent_environment_entry_count")):
        if count is not None and (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= MAX_AGY_POLICY_GRANT_ENTRIES
        ):
            return None
    for key in ("unknown_root_field_count", "unknown_user_setting_count"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 1024:
            return None
    transition_class = value.get("transition_class")
    failure = value.get("capture_failure")
    passing_class = (
        "precreated-to-unchanged-restrictive"
        if schema == AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1
        else "precreated-to-unchanged-empty-grants"
    )
    if transition_class == passing_class:
        if not (
            value["initial_config_present"] is True
            and value["final_config_present"] is True
            and value["private_regular_file"] is True
            and value["strict_json_object"] is True
            and value["schema_allowlisted"] is True
            and value[semantic_field] is True
            and value["write_protection_applied"] is True
            and value["write_protection_parameters_bound"] is True
            and value["same_file_identity"] is True
            and value["exact_precreated_payload"] is True
            and all(isinstance(item, int) and not isinstance(item, bool) for item in grants.values())
            and all(isinstance(item, int) and not isinstance(item, bool) for item in legacy.values())
            and isinstance(value["agent_environment_entry_count"], int)
            and not isinstance(value["agent_environment_entry_count"], bool)
            and grants == {"allow": 0, "deny": 0, "ask": 0}
            and legacy["allow"] == 0
            and legacy["deny"] == 0
            and value["agent_environment_entry_count"] == 0
            and value["unknown_root_field_count"] == 0
            and value["unknown_user_setting_count"] == 0
            and failure is None
        ):
            return None
    elif transition_class == "unsafe":
        if failure not in _POLICY_CONFIG_FAILURES or value[semantic_field] is not False:
            return None
    else:
        return None
    return dict(value)


def _policy_transition_has_no_explicit_grants(value: object) -> bool:
    checked = _validate_policy_config_transition(value)
    if checked is None:
        return False
    if checked["schema_version"] == AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1:
        semantic_ok = (
            checked["transition_class"] == "precreated-to-unchanged-restrictive"
            and checked["restrictive_permissions"] is True
        )
    else:
        semantic_ok = (
            checked["transition_class"] == "precreated-to-unchanged-empty-grants"
            and checked["no_explicit_grants"] is True
        )
    return bool(
        semantic_ok
        and checked["global_permission_grant_counts"]
        == {"allow": 0, "deny": 0, "ask": 0}
    )


def _sandbox_invocation_receipt(
    *,
    command: Sequence[str],
    spawn_options: Mapping[str, Any],
    workdir: Path,
    gemini_dir: Path,
    config_dir: Path,
    config_file: Path,
    tmp_dir: Path,
    log_path: Path,
    agy_executable: Path,
    sandbox_executable: Path,
    model: str,
    full_prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Derive a privacy-safe receipt by inspecting actual argv/spawn options."""
    outer_prefix = [
        str(sandbox_executable),
        "-D",
        f"WORKDIR={workdir}",
        "-D",
        f"GEMINI_DIR={gemini_dir}",
        "-D",
        f"CONFIG_DIR={config_dir}",
        "-D",
        f"CONFIG_FILE={config_file}",
        "-D",
        f"AGY_BIN={agy_executable}",
        "-p",
        AGY_POLICY_SANDBOX_PROFILE,
    ]
    agy_tail = [
        str(agy_executable),
        "-p",
        full_prompt,
        "--disable-slash-commands",
        "--sandbox",
        "--print-timeout",
        f"{max(1, int(timeout_seconds))}s",
        "--output-format",
        AGY_OUTPUT_FORMAT,
        "--log-file",
        str(log_path),
        "--gemini_dir",
        str(gemini_dir),
        "--app_data_dir",
        "antigravity-cli",
    ]
    if model != "agy-subscription":
        agy_tail.extend(["--model", model])
    parameter_indexes = (1, 3, 5, 7, 9)
    parameters = [
        command[index + 1].partition("=")[0]
        for index in parameter_indexes
        if len(command) > index + 1 and command[index] == "-D"
    ]
    environment = spawn_options.get("env")
    environment_keys = {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TERM",
        "AGY_CLI_DISABLE_AUTO_UPDATE",
    }
    agy_argv_exact = list(command[len(outer_prefix) :]) == agy_tail
    relationships = {
        "cwd_equals_workdir": spawn_options.get("cwd") == str(workdir),
        "gemini_dir_direct_child_of_workdir": (
            gemini_dir.parent == workdir
            and f"GEMINI_DIR={gemini_dir}" in command
        ),
        "config_dir_direct_child_of_gemini_dir": (
            config_dir.parent == gemini_dir
            and f"CONFIG_DIR={config_dir}" in command
        ),
        "config_file_direct_child_of_config_dir": (
            config_file.parent == config_dir
            and f"CONFIG_FILE={config_file}" in command
        ),
        "tmp_dir_direct_child_of_workdir": (
            tmp_dir.parent == workdir
            and isinstance(environment, Mapping)
            and environment.get("TMPDIR") == str(tmp_dir)
        ),
        "log_direct_child_of_workdir": (
            log_path.parent == workdir and str(log_path) in command
        ),
    }
    body = {
        "schema_version": AGY_SANDBOX_INVOCATION_RECEIPT_SCHEMA,
        "write_protection_policy": AGY_POLICY_CONFIG_WRITE_PROTECTION,
        "sandbox_executable": command[0] if command else "",
        "sandbox_executable_digest": _bytes_digest(sandbox_executable.read_bytes()),
        "agy_executable_digest": _bytes_digest(agy_executable.read_bytes()),
        "profile_digest": _bytes_digest(
            str(command[12]).encode("utf-8") if len(command) > 12 else b""
        ),
        "parameter_names": parameters,
        "parameter_relationships": relationships,
        "argv_policy": (
            "sandbox-exec-D-parameters-static-profile-then-agy"
            if list(command[: len(outer_prefix)]) == outer_prefix and agy_argv_exact
            else "unverified"
        ),
        "agy_argv_exact": agy_argv_exact,
        "prompt_digest": _bytes_digest(full_prompt.encode("utf-8")),
        "requested_model": model,
        "print_timeout_seconds": max(1, int(timeout_seconds)),
        "environment_allowlist_exact": (
            isinstance(environment, Mapping)
            and set(environment).issubset(environment_keys)
        ),
        "auto_update_disabled": (
            isinstance(environment, Mapping)
            and environment.get("AGY_CLI_DISABLE_AUTO_UPDATE") == "1"
        ),
        "profile_bytes_passed_exactly": (
            len(command) > 12
            and command[11] == "-p"
            and command[12] == AGY_POLICY_SANDBOX_PROFILE
        ),
        "cwd_policy": (
            "canonical-private-workdir"
            if spawn_options.get("cwd") == str(workdir)
            else "unverified"
        ),
        "tmpdir_policy": (
            "fresh-private-direct-child-of-workdir"
            if relationships["tmp_dir_direct_child_of_workdir"]
            else "unverified"
        ),
        "stdin_policy": (
            "devnull"
            if spawn_options.get("stdin") == asyncio.subprocess.DEVNULL
            else "unverified"
        ),
        "stdout_policy": (
            "pipe"
            if spawn_options.get("stdout") == asyncio.subprocess.PIPE
            else "unverified"
        ),
        "stderr_policy": (
            "pipe"
            if spawn_options.get("stderr") == asyncio.subprocess.PIPE
            else "unverified"
        ),
        "close_fds": spawn_options.get("close_fds") is True,
        "start_new_session": spawn_options.get("start_new_session") is True,
        "private_paths_persisted": False,
    }
    return {**body, "receipt_digest": _canonical_json_digest(body)}


def _validate_sandbox_invocation_receipt(value: object) -> dict[str, Any] | None:
    expected_keys = {
        "schema_version",
        "write_protection_policy",
        "sandbox_executable",
        "sandbox_executable_digest",
        "agy_executable_digest",
        "profile_digest",
        "parameter_names",
        "parameter_relationships",
        "argv_policy",
        "agy_argv_exact",
        "prompt_digest",
        "requested_model",
        "print_timeout_seconds",
        "environment_allowlist_exact",
        "auto_update_disabled",
        "profile_bytes_passed_exactly",
        "cwd_policy",
        "tmpdir_policy",
        "stdin_policy",
        "stdout_policy",
        "stderr_policy",
        "close_fds",
        "start_new_session",
        "private_paths_persisted",
        "receipt_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    body = {key: item for key, item in value.items() if key != "receipt_digest"}
    relations = value.get("parameter_relationships")
    expected_relations = {
        "cwd_equals_workdir",
        "gemini_dir_direct_child_of_workdir",
        "config_dir_direct_child_of_gemini_dir",
        "config_file_direct_child_of_config_dir",
        "tmp_dir_direct_child_of_workdir",
        "log_direct_child_of_workdir",
    }
    if (
        value.get("schema_version") != AGY_SANDBOX_INVOCATION_RECEIPT_SCHEMA
        or value.get("receipt_digest") != _canonical_json_digest(body)
        or value.get("write_protection_policy") != AGY_POLICY_CONFIG_WRITE_PROTECTION
        or value.get("sandbox_executable") != str(AGY_SANDBOX_EXECUTABLE)
        or value.get("sandbox_executable_digest")
        != EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST
        or value.get("profile_digest") != AGY_POLICY_SANDBOX_PROFILE_DIGEST
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("agy_executable_digest")))
        is None
        or value.get("parameter_names")
        != ["WORKDIR", "GEMINI_DIR", "CONFIG_DIR", "CONFIG_FILE", "AGY_BIN"]
        or not isinstance(relations, dict)
        or set(relations) != expected_relations
        or any(relations[key] is not True for key in expected_relations)
        or value.get("argv_policy")
        != "sandbox-exec-D-parameters-static-profile-then-agy"
        or value.get("agy_argv_exact") is not True
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("prompt_digest")))
        is None
        or not isinstance(value.get("requested_model"), str)
        or not value["requested_model"]
        or isinstance(value.get("print_timeout_seconds"), bool)
        or not isinstance(value.get("print_timeout_seconds"), int)
        or value["print_timeout_seconds"] < 1
        or value.get("environment_allowlist_exact") is not True
        or type(value.get("auto_update_disabled")) is not bool
        or value.get("profile_bytes_passed_exactly") is not True
        or value.get("cwd_policy") != "canonical-private-workdir"
        or value.get("tmpdir_policy") != "fresh-private-direct-child-of-workdir"
        or value.get("stdin_policy") != "devnull"
        or value.get("stdout_policy") != "pipe"
        or value.get("stderr_policy") != "pipe"
        or value.get("close_fds") is not True
        or value.get("start_new_session") is not True
        or value.get("private_paths_persisted") is not False
    ):
        return None
    return dict(value)


def _legacy_sanitized_log_receipt(
    value: object, raw: bytes
) -> tuple[dict[str, Any], bytes, str | None]:
    """Validate the frozen v1 receipt shape without permitting new v1 writes."""
    expected_keys = {
        "schema_version",
        "raw_log_digest",
        "raw_log_size_bytes",
        "response_ids",
        "conversation_ids",
        "reported_models",
        "workspace_dirs",
        "soft_denied_tools",
        "approved_tool_confirmation_ids",
        "denied_tool_confirmation_ids",
        "explicit_pre_response_failure_marker",
        "policy_config_disposition",
        "policy_config_path_relative",
        "policy_config_identity",
        "receipt_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return {}, b"", "sanitized_log_receipt_schema_invalid"
    body = {key: item for key, item in value.items() if key != "receipt_digest"}
    if (
        value.get("schema_version") != AGY_LOG_RECEIPT_SCHEMA_V1
        or value.get("receipt_digest") != _canonical_json_digest(body)
        or raw != _canonical_json_bytes(value)
    ):
        return {}, b"", "sanitized_log_receipt_digest_invalid"
    config_identity = _validate_policy_config_identity(value.get("policy_config_identity"))
    string_lists = (
        "response_ids",
        "conversation_ids",
        "reported_models",
        "workspace_dirs",
        "soft_denied_tools",
        "approved_tool_confirmation_ids",
        "denied_tool_confirmation_ids",
    )
    if (
        config_identity is None
        or any(
            not isinstance(value.get(key), list)
            or not all(isinstance(item, str) and item for item in value[key])
            for key in string_lists
        )
        or type(value.get("explicit_pre_response_failure_marker")) is not bool
        or value.get("policy_config_disposition")
        not in {"absent", "loaded", "unsafe_cached", "unverified"}
        or not isinstance(value.get("raw_log_size_bytes"), int)
        or value["raw_log_size_bytes"] < 0
        or not isinstance(value.get("raw_log_digest"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["raw_log_digest"]) is None
        or value["raw_log_size_bytes"] > MAX_AGY_LOG_BYTES
        or any(value[key] != list(dict.fromkeys(value[key])) for key in string_lists)
        or any(_AGY_UUID.fullmatch(item) is None for item in value["conversation_ids"])
        or any(
            _AGY_UUID.fullmatch(item) is None
            for key in ("approved_tool_confirmation_ids", "denied_tool_confirmation_ids")
            for item in value[key]
        )
        or any(not Path(item).is_absolute() for item in value["workspace_dirs"])
    ):
        return {}, b"", "sanitized_log_receipt_fields_invalid"
    relative = value.get("policy_config_path_relative")
    disposition = value["policy_config_disposition"]
    if disposition in {"absent", "loaded"}:
        if (
            not isinstance(relative, str)
            or re.fullmatch(r"\.agy-gemini-[0-9a-f]{32}/config/config\.json", relative) is None
        ):
            return {}, b"", "sanitized_log_receipt_config_path_invalid"
    elif relative is not None:
        return {}, b"", "sanitized_log_receipt_config_path_invalid"
    return dict(value), raw, None


def _log_receipt(
    raw: bytes,
    *,
    invocation_workdir: str,
    policy_config_identity: Mapping[str, Any] | None,
    policy_config_transition: Mapping[str, Any] | None,
    sandbox_invocation_receipt: Mapping[str, Any] | None,
    sanitized: bool,
) -> tuple[dict[str, Any], bytes, str | None]:
    if sanitized:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {}, b"", "sanitized_log_receipt_invalid_json"
        if isinstance(value, dict) and value.get("schema_version") == AGY_LOG_RECEIPT_SCHEMA_V1:
            return _legacy_sanitized_log_receipt(value, raw)
        v2_keys = {
            "schema_version",
            "raw_log_digest",
            "raw_log_size_bytes",
            "response_ids",
            "conversation_ids",
            "reported_models",
            "workspace_dirs",
            "soft_denied_tools",
            "approved_tool_confirmation_ids",
            "denied_tool_confirmation_ids",
            "explicit_pre_response_failure_marker",
            "nested_sandbox_failure_marker",
            "policy_config_disposition",
            "policy_config_transition",
            "receipt_digest",
        }
        schema = value.get("schema_version") if isinstance(value, dict) else None
        expected_keys = (
            {*v2_keys, "sandbox_invocation_receipt"}
            if schema == AGY_LOG_RECEIPT_SCHEMA
            else v2_keys
        )
        if (
            not isinstance(value, dict)
            or schema not in {AGY_LOG_RECEIPT_SCHEMA_V2, AGY_LOG_RECEIPT_SCHEMA}
            or set(value) != expected_keys
        ):
            return {}, b"", "sanitized_log_receipt_schema_invalid"
        body = {key: item for key, item in value.items() if key != "receipt_digest"}
        if (
            value.get("receipt_digest") != _canonical_json_digest(body)
            or raw != _canonical_json_bytes(value)
        ):
            return {}, b"", "sanitized_log_receipt_digest_invalid"
        config_transition = _validate_policy_config_transition(
            value.get("policy_config_transition")
        )
        invocation_receipt = (
            _validate_sandbox_invocation_receipt(
                value.get("sandbox_invocation_receipt")
            )
            if schema == AGY_LOG_RECEIPT_SCHEMA
            else None
        )
        string_lists = (
            "response_ids",
            "conversation_ids",
            "reported_models",
            "workspace_dirs",
            "soft_denied_tools",
            "approved_tool_confirmation_ids",
            "denied_tool_confirmation_ids",
        )
        if (
            config_transition is None
            or (
                schema == AGY_LOG_RECEIPT_SCHEMA_V2
                and config_transition.get("schema_version")
                != AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1
            )
            or (
                schema == AGY_LOG_RECEIPT_SCHEMA
                and config_transition.get("schema_version")
                != AGY_POLICY_CONFIG_TRANSITION_SCHEMA
            )
            or (schema == AGY_LOG_RECEIPT_SCHEMA and invocation_receipt is None)
            or any(
                not isinstance(value.get(key), list)
                or not all(isinstance(item, str) and item for item in value[key])
                for key in string_lists
            )
            or type(value.get("explicit_pre_response_failure_marker")) is not bool
            or type(value.get("nested_sandbox_failure_marker")) is not bool
            or value.get("policy_config_disposition")
            not in {"absent", "loaded", "loaded_empty", "unsafe_cached", "unverified"}
            or not isinstance(value.get("raw_log_size_bytes"), int)
            or value["raw_log_size_bytes"] < 0
            or not isinstance(value.get("raw_log_digest"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["raw_log_digest"]) is None
            or value["raw_log_size_bytes"] > MAX_AGY_LOG_BYTES
            or any(value[key] != list(dict.fromkeys(value[key])) for key in string_lists)
            or any(_AGY_UUID.fullmatch(item) is None for item in value["conversation_ids"])
            or any(
                _AGY_UUID.fullmatch(item) is None
                for key in ("approved_tool_confirmation_ids", "denied_tool_confirmation_ids")
                for item in value[key]
            )
            or any(not Path(item).is_absolute() for item in value["workspace_dirs"])
        ):
            return {}, b"", "sanitized_log_receipt_fields_invalid"
        return value, raw, None

    config_identity = _validate_policy_config_identity(policy_config_identity)
    if config_identity is None:
        return {}, b"", "policy_config_identity_invalid"
    # Compatibility-only raw dispatch for callers that still construct frozen
    # v1/v2 evidence.  The live subprocess path always supplies v3 receipts.
    legacy_write = policy_config_transition is None
    config_transition = _validate_policy_config_transition(policy_config_transition)
    invocation_receipt = _validate_sandbox_invocation_receipt(
        sandbox_invocation_receipt
    )
    if not legacy_write and config_transition is None:
        return {}, b"", "policy_config_transition_invalid"
    v2_write = bool(
        config_transition is not None
        and config_transition.get("schema_version")
        == AGY_POLICY_CONFIG_TRANSITION_SCHEMA_V1
        and sandbox_invocation_receipt is None
    )
    if (
        not legacy_write
        and not v2_write
        and (
            invocation_receipt is None
            or config_transition is None
            or config_transition.get("schema_version")
            != AGY_POLICY_CONFIG_TRANSITION_SCHEMA
        )
    ):
        return {}, b"", "sandbox_invocation_receipt_invalid"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {}, b"", "raw_log_invalid_utf8"
    confirmations = _AGY_TOOL_CONFIRMATION.findall(text)
    loaded_config_paths = tuple(
        dict.fromkeys(item.strip() for item in _AGY_SHARED_CONFIG.findall(text))
    )
    absent_config_paths = tuple(
        dict.fromkeys(item.strip() for item in _AGY_NO_SHARED_CONFIG.findall(text))
    )

    def relative_isolated_config(path_text: str) -> str | None:
        candidate = Path(path_text)
        root = Path(invocation_workdir)
        if (
            not candidate.is_absolute()
            or candidate.resolve(strict=False) != candidate
            or not root.is_absolute()
            or root.resolve(strict=False) != root
        ):
            return None
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            return None
        if re.fullmatch(r"\.agy-gemini-[0-9a-f]{32}/config/config\.json", relative) is None:
            return None
        return relative

    config_disposition = "unverified"
    config_relative: str | None = None
    if _AGY_UNSAFE_CONFIG_FALLBACK.search(text):
        config_disposition = "unsafe_cached"
    elif (
        config_identity["exists"] is False
        and len(absent_config_paths) == 1
        and not loaded_config_paths
    ):
        config_relative = relative_isolated_config(absent_config_paths[0])
        if config_relative is not None:
            config_disposition = "absent"
    elif (
        config_identity["exists"] is True
        and config_transition is not None
        and _policy_transition_has_no_explicit_grants(config_transition)
        and len(absent_config_paths) == 1
        and not loaded_config_paths
    ):
        config_relative = relative_isolated_config(absent_config_paths[0])
        if config_relative is not None:
            config_disposition = "loaded_empty"
    elif (
        config_identity["exists"] is True
        and len(loaded_config_paths) == 1
        and not absent_config_paths
    ):
        config_relative = relative_isolated_config(loaded_config_paths[0])
        if config_relative is not None:
            config_disposition = "loaded"
    body = {
        "schema_version": (
            AGY_LOG_RECEIPT_SCHEMA_V1
            if legacy_write
            else AGY_LOG_RECEIPT_SCHEMA_V2
            if v2_write
            else AGY_LOG_RECEIPT_SCHEMA
        ),
        "raw_log_digest": _bytes_digest(raw),
        "raw_log_size_bytes": len(raw),
        "response_ids": list(_AGY_RESPONSE_ID.findall(text)),
        "conversation_ids": list(dict.fromkeys(_AGY_CONVERSATION_ID.findall(text))),
        "reported_models": list(dict.fromkeys(_AGY_PRINT_MODEL.findall(text))),
        "workspace_dirs": [item.strip() for item in _AGY_WORKSPACE.findall(text)],
        "soft_denied_tools": list(dict.fromkeys(_AGY_SOFT_DENIAL.findall(text))),
        "approved_tool_confirmation_ids": list(
            dict.fromkeys(item[0] for item in confirmations if item[1].casefold() == "true")
        ),
        "denied_tool_confirmation_ids": list(
            dict.fromkeys(item[0] for item in confirmations if item[1].casefold() == "false")
        ),
        "explicit_pre_response_failure_marker": bool(
            _AGY_EXPLICIT_PRE_RESPONSE_FAILURE.search(text)
        ),
        "policy_config_disposition": config_disposition,
    }
    if legacy_write:
        body["policy_config_path_relative"] = config_relative
        body["policy_config_identity"] = config_identity
    else:
        body["nested_sandbox_failure_marker"] = bool(
            _AGY_NESTED_SANDBOX_FAILURE.search(text)
        )
        body["policy_config_transition"] = config_transition
        if not v2_write:
            body["sandbox_invocation_receipt"] = invocation_receipt
    value = {**body, "receipt_digest": _canonical_json_digest(body)}
    return value, _canonical_json_bytes(value), None


def analyze_agy_evidence(
    *,
    requested_model: str,
    full_prompt: str,
    invocation_workdir: Path | str,
    exit_status: int | None,
    timed_out: bool,
    process_group_descendant_detected: bool = False,
    process_group_quiescent: bool = True,
    capture_failures: Sequence[str],
    stdout_ndjson: bytes,
    stderr: bytes,
    log: bytes,
    transcript: bytes,
    policy_config_identity: Mapping[str, Any] | None = None,
    policy_config_transition: Mapping[str, Any] | None = None,
    sandbox_invocation_receipt: Mapping[str, Any] | None = None,
    sanitized_stream_receipt: bool = False,
    sanitized_stderr_receipt: bool = False,
    sanitized_log_receipt: bool = False,
    sanitized_transcript_receipt: bool = False,
) -> AgyCallEvidence:
    """Classify a runtime-bound structured call; every uncertainty fails closed."""
    workdir = str(invocation_workdir)
    failures = tuple(str(item) for item in capture_failures)
    (
        stderr_facts,
        persisted_stderr,
        stderr_error,
        raw_stderr_digest,
        raw_stderr_size,
    ) = _stderr_receipt(stderr, sanitized=sanitized_stderr_receipt)
    if stderr_error is not None:
        failures = (*failures, stderr_error)
    log_facts, persisted_log, log_error = _log_receipt(
        log,
        invocation_workdir=workdir,
        policy_config_identity=policy_config_identity,
        policy_config_transition=policy_config_transition,
        sandbox_invocation_receipt=sandbox_invocation_receipt,
        sanitized=sanitized_log_receipt,
    )
    if log_error is not None:
        failures = (*failures, log_error)
    invocation_context = _validate_sandbox_invocation_receipt(
        log_facts.get("sandbox_invocation_receipt")
    )
    if (
        log_facts.get("schema_version") == AGY_LOG_RECEIPT_SCHEMA
        and (
            invocation_context is None
            or invocation_context.get("requested_model") != requested_model
            or invocation_context.get("prompt_digest")
            != _bytes_digest(full_prompt.encode("utf-8"))
        )
    ):
        failures = (*failures, "sandbox_invocation_context_mismatch")
    if (
        log_facts.get("schema_version")
        in {AGY_LOG_RECEIPT_SCHEMA_V2, AGY_LOG_RECEIPT_SCHEMA}
        and process_group_descendant_detected is True
        and "process_group_descendant_detected" not in failures
    ):
        failures = (*failures, "process_group_descendant_detected")
    if (
        log_facts.get("schema_version")
        in {AGY_LOG_RECEIPT_SCHEMA_V2, AGY_LOG_RECEIPT_SCHEMA}
        and process_group_quiescent is not True
        and "process_group_quiescence_unproven" not in failures
    ):
        failures = (*failures, "process_group_quiescence_unproven")
    sealed_transition = _validate_policy_config_transition(
        log_facts.get("policy_config_transition")
    )
    transition_failure = (
        sealed_transition.get("capture_failure")
        if sealed_transition is not None
        else None
    )
    if isinstance(transition_failure, str) and transition_failure not in failures:
        failures = (*failures, transition_failure)
    response_ids = tuple(log_facts.get("response_ids", ()))
    logged_conversations = tuple(log_facts.get("conversation_ids", ()))
    logged_models = tuple(log_facts.get("reported_models", ()))

    (
        events,
        persisted_stream,
        stream_error,
        raw_stream_digest,
        raw_stream_size,
    ) = _stream_receipt(
        stdout_ndjson,
        workdir=workdir,
        sanitized=sanitized_stream_receipt,
    )
    kinds = tuple(_event_kind(item) for item in events)
    terminal = events[-1] if events and kinds[-1] == "result" else None
    terminal_payload = _event_payload(terminal or {}, "result")
    terminal_digest = (
        str(terminal.get("_raw_event_digest"))
        if terminal is not None and terminal.get("_raw_event_digest") is not None
        else _canonical_json_digest(terminal)
        if terminal is not None
        else None
    )
    response = _terminal_response(terminal_payload or {})
    stream_tools = _stream_tool_names(events)
    stream_models = _reported_stream_models(events)
    stream_conversations = _stream_conversation_ids(events)
    conversation_id = stream_conversations[0] if len(stream_conversations) == 1 else None
    reported_model = next(iter(stream_models)) if len(stream_models) == 1 else None

    (
        transcript_facts,
        persisted_transcript,
        transcript_error,
        raw_transcript_digest,
        raw_transcript_size,
    ) = _transcript_receipt(
        transcript,
        full_prompt=full_prompt,
        sanitized=sanitized_transcript_receipt,
    )
    transcript_tools = set(transcript_facts.get("tool_call_names", ()))
    model_content_digests = tuple(transcript_facts.get("model_content_digests", ()))
    prompt_matches = transcript_facts.get("prompt_matches", 0)
    explicit_user_inputs = transcript_facts.get("explicit_user_inputs", 0)
    transcript_tool_execution = transcript_facts.get("tool_execution_observed") is True

    all_tools = tuple(sorted(set(stream_tools) | transcript_tools, key=str.casefold))
    terminal_error_text = _terminal_error(terminal_payload or {})
    explicit_pre_response_failure = bool(
        not response_ids
        and (
            stderr_facts.get("explicit_pre_response_failure_marker") is True
            or _AGY_EXPLICIT_PRE_RESPONSE_FAILURE.search(terminal_error_text)
            or log_facts.get("explicit_pre_response_failure_marker") is True
        )
        and (exit_status not in (None, 0) or terminal is not None)
    )

    def finish(
        disposition: str, error: str | None, selected_response: str | None = None
    ) -> AgyCallEvidence:
        legacy_receipt = log_facts.get("schema_version") == AGY_LOG_RECEIPT_SCHEMA_V1
        v2_receipt = log_facts.get("schema_version") == AGY_LOG_RECEIPT_SCHEMA_V2
        sealed_identity = log_facts.get("policy_config_identity")
        sealed_transition_value = log_facts.get("policy_config_transition")
        sealed_invocation_value = log_facts.get("sandbox_invocation_receipt")
        return AgyCallEvidence(
            evidence_schema_version=(
                AGY_EVIDENCE_SCHEMA_V1
                if legacy_receipt
                else AGY_EVIDENCE_SCHEMA_V2
                if v2_receipt
                else AGY_EVIDENCE_SCHEMA
            ),
            disposition=disposition,
            response=selected_response,
            error=error,
            requested_model=requested_model,
            reported_model=reported_model,
            invocation_workdir=workdir,
            exit_status=exit_status,
            timed_out=timed_out,
            process_group_descendant_detected=process_group_descendant_detected,
            process_group_quiescent=process_group_quiescent,
            capture_failures=failures,
            response_ids=response_ids,
            conversation_id=conversation_id,
            stream_event_count=len(events),
            terminal_event_digest=terminal_digest,
            tool_call_names=all_tools,
            policy_config_identity=(
                dict(sealed_identity) if isinstance(sealed_identity, dict) else {}
            ),
            policy_config_transition=(
                dict(sealed_transition_value)
                if isinstance(sealed_transition_value, dict)
                else {}
            ),
            sandbox_invocation_receipt=(
                dict(sealed_invocation_value)
                if isinstance(sealed_invocation_value, dict)
                else {}
            ),
            raw_log_digest=str(log_facts.get("raw_log_digest", _bytes_digest(log))),
            raw_log_size_bytes=int(log_facts.get("raw_log_size_bytes", len(log))),
            raw_stream_digest=raw_stream_digest,
            raw_stream_size_bytes=raw_stream_size,
            raw_stderr_digest=raw_stderr_digest,
            raw_stderr_size_bytes=raw_stderr_size,
            raw_transcript_digest=raw_transcript_digest,
            raw_transcript_size_bytes=raw_transcript_size,
            stdout_ndjson=persisted_stream,
            stderr=persisted_stderr,
            log=persisted_log,
            transcript=persisted_transcript,
        )

    if (
        _stream_tool_execution_observed(events)
        or transcript_tool_execution
        or log_facts.get("approved_tool_confirmation_ids")
    ):
        return finish(
            "evidence_integrity_failure",
            "agy evidence indicates that a model-selected tool may have executed",
        )
    if (
        log_facts.get("nested_sandbox_failure_marker") is True
        or stderr_facts.get("nested_sandbox_failure_marker") is True
    ):
        return finish(
            "evidence_integrity_failure",
            "agy evidence indicates nested sandbox initialization failure",
        )
    if timed_out:
        return finish("ambiguous_provider_disposition", "agy timed out with unknown disposition")
    if any(item.startswith("spawn_failed:") for item in failures):
        return finish("fatal_transport", "agy could not start")
    if failures:
        return finish(
            "evidence_integrity_failure",
            "agy evidence capture was incomplete: " + ", ".join(failures),
        )
    if (raw_stream_size > 0 and stream_error is not None) or (
        raw_transcript_size > 0 and transcript_error is not None
    ):
        return finish(
            "evidence_integrity_failure",
            "agy emitted malformed evidence before a provider disposition was established",
        )
    if explicit_pre_response_failure:
        return finish(
            "pre_response_provider_failure",
            "agy explicitly refused or could not serve the request before any ResponseID",
        )
    legacy_config_identity = _validate_policy_config_identity(
        log_facts.get("policy_config_identity")
    )
    config_transition = _validate_policy_config_transition(log_facts.get("policy_config_transition"))
    legacy_config_ok = (
        log_facts.get("schema_version") == AGY_LOG_RECEIPT_SCHEMA_V1
        and legacy_config_identity is not None
        and legacy_config_identity["exists"] is False
    )
    transition_config_ok = (
        config_transition is not None
        and _policy_transition_has_no_explicit_grants(config_transition)
    )
    if (
        not (legacy_config_ok or transition_config_ok)
        or log_facts.get("policy_config_disposition")
        != ("absent" if legacy_config_ok else "loaded_empty")
    ):
        return finish(
            "evidence_integrity_failure",
            "agy inherited policy config is not exactly bound to the call",
        )
    if not response_ids:
        return finish(
            "ambiguous_provider_disposition",
            "agy produced no ResponseID and no explicit pre-response provider failure",
        )
    if (
        stream_error is not None
        or not events
        or kinds[0] != "init"
        or kinds[-1] != "result"
        or any(kind not in {"init", "step_update", "result"} for kind in kinds)
        or kinds.count("init") != 1
        or kinds.count("result") != 1
    ):
        return finish(
            "evidence_integrity_failure",
            stream_error or "agy stream shape is invalid",
        )
    payloads_are_complete = all(
        _event_payload(event, kind) is not None for event, kind in zip(events, kinds)
    )
    if not payloads_are_complete or conversation_id is None:
        return finish(
            "evidence_integrity_failure",
            "agy stream payload or conversation identity is invalid",
        )
    terminal_conversations = {
        item
        for item in (
            terminal.get("conversation_id") if terminal is not None else None,
            terminal_payload.get("conversation_id") if terminal_payload is not None else None,
        )
        if isinstance(item, str) and item
    }
    if terminal_conversations != {conversation_id}:
        return finish(
            "evidence_integrity_failure",
            "agy terminal result does not bind the stream conversation",
        )
    if len(logged_conversations) != 1 or logged_conversations[0] != conversation_id:
        return finish("evidence_integrity_failure", "agy log/stream conversation mismatch")
    if transcript_error is not None or explicit_user_inputs != 1 or prompt_matches != 1:
        return finish(
            "evidence_integrity_failure",
            transcript_error or "agy transcript does not bind the exact prompt/conversation",
        )
    route_matches = (
        reported_model is not None
        and len(logged_models) == 1
        and (
            logged_models[0] == reported_model
            if requested_model == "agy-subscription"
            else logged_models[0] == reported_model == requested_model
        )
    )
    if not route_matches:
        return finish("evidence_integrity_failure", "agy requested/reported route mismatch")
    init_payload = _event_payload(events[0], "init")
    if init_payload is None or init_payload.get("cwd") != workdir:
        return finish("evidence_integrity_failure", "agy init does not bind the workdir")
    available_tools = _init_tool_names(init_payload)
    if available_tools is None or not {_normalized_tool_name(item) for item in all_tools}.issubset(
        {_normalized_tool_name(item) for item in available_tools}
    ):
        return finish(
            "evidence_integrity_failure",
            "agy selected a tool absent from its sealed init inventory",
        )
    workspace_matches = tuple(log_facts.get("workspace_dirs", ()))
    if len(workspace_matches) != 1 or workspace_matches[0].strip() != workdir:
        return finish("evidence_integrity_failure", "agy log does not bind the invocation workdir")
    if {_normalized_tool_name(item) for item in transcript_tools} != {
        _normalized_tool_name(item) for item in stream_tools
    }:
        return finish("evidence_integrity_failure", "agy stream/transcript tool traces differ")
    if all_tools:
        denied = {_normalized_tool_name(item) for item in log_facts.get("soft_denied_tools", ())}
        selected = {_normalized_tool_name(item) for item in all_tools}
        if not selected.issubset(denied):
            return finish(
                "evidence_integrity_failure", "agy tool selection lacks exact soft denial"
            )
        if set(log_facts.get("denied_tool_confirmation_ids", ())) != {conversation_id}:
            return finish(
                "evidence_integrity_failure",
                "agy soft denial is not bound to the stream conversation",
            )
        return finish(
            "tool_policy_no_action",
            "model selected a tool that print mode soft-denied; no action was produced",
        )
    if exit_status != 0:
        return finish(
            "ambiguous_provider_disposition",
            f"agy exited {exit_status} after a ResponseID without a complete model response",
        )
    if response is None:
        return finish(
            "ambiguous_provider_disposition",
            "agy terminal result was blank after a ResponseID",
        )
    terminal_status = terminal_payload.get("status") if terminal_payload is not None else None
    if terminal_status != "SUCCESS":
        return finish(
            "ambiguous_provider_disposition",
            "agy terminal result status is not a confirmed success",
        )
    if not model_content_digests or model_content_digests[-1] != _bytes_digest(response.encode()):
        return finish("evidence_integrity_failure", "agy stream/transcript response mismatch")
    return finish("response", None, response)


def _load_explicit_env_file(path: Path) -> None:
    """Load an explicitly requested dotenv file without reading ~/.env implicitly."""
    if not path.is_file():
        raise LiveEvalError(f"Environment file does not exist: {path}", 2)
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _require_integrations() -> None:
    if PROOFPACK_IMPORT_ERROR is not None:
        raise LiveEvalError(
            "ProofPack is required for live qualification but could not be imported: "
            f"{PROOFPACK_IMPORT_ERROR}. Run this command in a Python 3.12 environment "
            "that contains ProofPack and all of its runtime dependencies; a sibling "
            "source checkout alone is not sufficient.",
            2,
        )
    if ASSAY_IMPORT_ERROR is not None:
        raise LiveEvalError(
            "Assay is required for live evidence persistence but could not be imported: "
            f"{ASSAY_IMPORT_ERROR}. Run this command in a Python 3.12 environment "
            "that contains Assay and all of its runtime dependencies; a sibling source "
            "checkout alone is not sufficient.",
            2,
        )


def get_llm_client(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 180.0,
) -> tuple[Any, str]:
    """Resolve a subscription CLI or OpenAI-compatible asynchronous client."""
    if provider == "agy":
        agy_bin = shutil.which("agy")
        if not agy_bin:
            raise LiveEvalError("agy CLI is not available on PATH", 2)
        return agy_bin, model or "agy-subscription"

    from openai import AsyncOpenAI

    if provider in ("google", "gemini"):
        url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        resolved_model = model or "gemini-2.5-flash"
        env_hint = "GEMINI_API_KEY or GOOGLE_API_KEY"
    elif provider == "openrouter":
        url = base_url or "https://openrouter.ai/api/v1"
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        resolved_model = model or "deepseek/deepseek-chat"
        env_hint = "OPENROUTER_API_KEY"
    elif provider == "openai":
        url = base_url or "https://api.openai.com/v1"
        key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_model = model or "gpt-4o-mini"
        env_hint = "OPENAI_API_KEY"
    else:
        url = base_url or "http://localhost:11434/v1"
        key = api_key or "local-key"
        resolved_model = model or "qwen2.5:7b"
        env_hint = "--api-key"

    if not key:
        raise LiveEvalError(
            f"API key missing for provider {provider!r}; set {env_hint} or pass --api-key",
            2,
        )
    return AsyncOpenAI(base_url=url, api_key=key, timeout=timeout_seconds), resolved_model


def _agy_environment(
    process_environment: Mapping[str, str] | None = None,
    *,
    private_tmpdir: Path | None = None,
) -> dict[str, str]:
    """Pass inert startup variables plus an explicitly allowed AGY guard."""
    exact = {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TERM",
    }
    environment = {key: value for key, value in os.environ.items() if key in exact}
    overrides = dict(process_environment or {})
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in overrides.items()):
        raise LiveEvalError("agy process environment must contain only string pairs", 4)
    if overrides not in ({}, {"AGY_CLI_DISABLE_AUTO_UPDATE": "1"}):
        raise LiveEvalError("agy process environment contains an unsupported override", 4)
    environment.update(overrides)
    if private_tmpdir is not None:
        if (
            not private_tmpdir.is_absolute()
            or private_tmpdir.is_symlink()
            or not private_tmpdir.is_dir()
            or private_tmpdir.resolve(strict=True) != private_tmpdir
        ):
            raise LiveEvalError("agy private TMPDIR is unsafe", 4)
        environment["TMPDIR"] = str(private_tmpdir)
    return environment


async def _read_stream_limited(
    stream: asyncio.StreamReader,
    *,
    limit: int,
    label: str,
) -> bytes:
    """Drain one child stream while enforcing a hard in-memory byte bound."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise LiveEvalError(f"agy {label} exceeded {limit} bytes", 4)
        chunks.append(chunk)


async def _communicate_agy_bounded(
    proc: asyncio.subprocess.Process,
    *,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    """Collect bounded child output and always reap the process on cancellation."""
    if proc.stdout is None or proc.stderr is None:  # pragma: no cover - constructed with pipes
        raise LiveEvalError("agy subprocess pipes were not created", 4)
    stdout_task = asyncio.create_task(
        _read_stream_limited(proc.stdout, limit=MAX_LLM_RESPONSE_BYTES, label="stdout")
    )
    stderr_task = asyncio.create_task(
        _read_stream_limited(proc.stderr, limit=MAX_LLM_STDERR_BYTES, label="stderr")
    )
    wait_task = asyncio.create_task(proc.wait())
    tasks = (stdout_task, stderr_task, wait_task)
    try:
        stdout, stderr, _returncode = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=timeout_seconds,
        )
        return stdout, stderr
    except BaseException:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _read_stream_capture(
    stream: asyncio.StreamReader,
    *,
    limit: int,
    label: str,
    destination: bytearray,
) -> None:
    """Capture at most ``limit`` bytes, retaining the bounded prefix on failure."""
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return
        remaining = limit - len(destination)
        if remaining > 0:
            destination.extend(chunk[:remaining])
        if len(chunk) > remaining:
            raise _AgyStreamLimit(label)


async def _communicate_agy_evidence(
    proc: asyncio.subprocess.Process,
    *,
    timeout_seconds: float,
) -> tuple[bytes, bytes, bool, tuple[str, ...]]:
    """Reap AGY while preserving bounded stdout/stderr prefixes on every outcome."""
    if proc.stdout is None or proc.stderr is None:  # pragma: no cover - constructed with pipes
        return b"", b"", False, ("subprocess_pipes_missing",)
    stdout = bytearray()
    stderr = bytearray()
    stdout_task = asyncio.create_task(
        _read_stream_capture(
            proc.stdout,
            limit=MAX_LLM_RESPONSE_BYTES,
            label="stdout_limit_exceeded",
            destination=stdout,
        )
    )
    stderr_task = asyncio.create_task(
        _read_stream_capture(
            proc.stderr,
            limit=MAX_LLM_STDERR_BYTES,
            label="stderr_limit_exceeded",
            destination=stderr,
        )
    )
    wait_task = asyncio.create_task(proc.wait())
    tasks = (stdout_task, stderr_task, wait_task)
    timed_out = False
    failures: list[str] = []
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
    except asyncio.CancelledError:
        failures.append("capture_cancelled")
        raise
    except asyncio.TimeoutError:
        timed_out = True
    except _AgyStreamLimit as exc:
        failures.append(exc.label)
    except Exception as exc:  # pragma: no cover - defensive subprocess boundary
        failures.append(f"capture_failed:{type(exc).__name__}")
    finally:
        if (timed_out or failures) and proc.returncode is None:
            try:
                process_id = getattr(proc, "pid", None)
                if (
                    isinstance(process_id, int)
                    and process_id > 1
                    and process_id != os.getpgrp()
                ):
                    os.killpg(process_id, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        try:
            await proc.wait()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return bytes(stdout), bytes(stderr), timed_out, tuple(failures)


async def _quiesce_agy_process_group(
    proc: asyncio.subprocess.Process,
) -> tuple[bool, bool]:
    """Kill any surviving descendant in the fresh session and prove the group is gone."""
    process_id = getattr(proc, "pid", None)
    if (
        not isinstance(process_id, int)
        or process_id <= 1
        or process_id == os.getpgrp()
    ):
        return False, False
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False, True
    except OSError:
        return False, False
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        return True, True
    except OSError:
        return True, False
    for _ in range(100):
        await asyncio.sleep(0.01)
        try:
            os.killpg(process_id, 0)
        except ProcessLookupError:
            return True, True
        except OSError:
            return True, False
    return True, False


def _read_bounded_regular_file(path: Path, *, limit: int, label: str) -> tuple[bytes, str | None]:
    """Read one exact regular file without following a symlink or exceeding its cap."""
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        return b"", f"{label}_symlink"
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
    except OSError:
        return b"", f"{label}_missing"
    try:
        if not stat.S_ISREG(before.st_mode):
            return b"", f"{label}_not_regular"
        if before.st_size > limit:
            return b"", f"{label}_limit_exceeded"
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError:
        return b"", f"{label}_read_failed"
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(content) > limit:
        return b"", f"{label}_limit_exceeded"
    if (
        len(content) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        return b"", f"{label}_changed_during_read"
    return content, None


def _policy_config_identity(gemini_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not gemini_dir.is_absolute() or gemini_dir.resolve(strict=False) != gemini_dir:
        return None, "policy_config_root_unsafe"
    path = gemini_dir / "config" / "config.json"
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        return None, "policy_config_symlink"
    if not path.exists() and not path.is_symlink():
        return {
            "relative_path": "config/config.json",
            "exists": False,
            "digest": None,
            "size_bytes": 0,
        }, None
    content, failure = _read_bounded_regular_file(
        path,
        limit=MAX_AGY_POLICY_CONFIG_BYTES,
        label="policy_config",
    )
    if failure is not None:
        return None, failure
    return {
        "relative_path": "config/config.json",
        "exists": True,
        "digest": _bytes_digest(content),
        "size_bytes": len(content),
    }, None


def _bounded_policy_string_list(value: object) -> list[str] | None:
    if (
        not isinstance(value, list)
        or len(value) > MAX_AGY_POLICY_GRANT_ENTRIES
        or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > MAX_AGY_POLICY_GRANT_BYTES
            for item in value
        )
    ):
        return None
    return value


def _parse_restrictive_policy_config(content: bytes) -> tuple[dict[str, Any], str | None]:
    """Project config bytes to bounded facts, never returning a raw key value."""
    if content.startswith(b"\xef\xbb\xbf"):
        return {}, "policy_config_invalid_utf8"
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {}, "policy_config_invalid_utf8"
    if "\x00" in text:
        return {}, "policy_config_contains_nul"
    try:
        root = json.loads(
            text,
            object_pairs_hook=_policy_config_object,
            parse_constant=_reject_policy_config_constant,
        )
    except _PolicyConfigDuplicateKey:
        return {}, "policy_config_duplicate_key"
    except (json.JSONDecodeError, _PolicyConfigConstant, RecursionError, ValueError):
        return {}, "policy_config_invalid_json"
    if not isinstance(root, dict):
        return {}, "policy_config_nonobject_root"
    unknown_root = set(root) - {"userSettings"}
    if unknown_root:
        return {
            "unknown_root_field_count": len(unknown_root),
        }, "policy_config_unknown_root_field"
    if set(root) != {"userSettings"}:
        return {}, "policy_config_user_settings_missing"
    settings = root["userSettings"]
    if not isinstance(settings, dict):
        return {}, "policy_config_user_settings_nonobject"
    unknown_settings = set(settings) - {"globalPermissionGrants"}
    if unknown_settings:
        return {
            "unknown_user_setting_count": len(unknown_settings),
        }, "policy_config_unknown_user_setting"
    if set(settings) != {"globalPermissionGrants"}:
        return {}, "policy_config_global_grants_nonobject"

    grant_counts = {"allow": 0, "deny": 0, "ask": 0}
    grants = settings["globalPermissionGrants"]
    if not isinstance(grants, dict):
        return {}, "policy_config_global_grants_nonobject"
    if set(grants) - set(grant_counts):
        return {}, "policy_config_global_grants_unknown_bucket"
    if set(grants) != set(grant_counts):
        return {}, "policy_config_global_grants_unknown_bucket"
    for bucket in grant_counts:
        entries = _bounded_policy_string_list(grants.get(bucket, []))
        if entries is None:
            return {}, "policy_config_invalid_grant_list"
        grant_counts[bucket] = len(entries)
    if grant_counts["allow"] != 0:
        return {}, "policy_config_allow_grants_nonempty"
    if grant_counts != {"allow": 0, "deny": 0, "ask": 0}:
        return {}, "policy_config_nonempty_grants"

    return {
        "grant_counts": grant_counts,
        "legacy_counts": {"allow": 0, "deny": 0},
        "agent_environment_entry_count": 0,
        "unknown_root_field_count": 0,
        "unknown_user_setting_count": 0,
    }, None


_POLICY_CONFIG_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _policy_stat_identity(value: os.stat_result) -> dict[str, int]:
    return {field: int(getattr(value, field)) for field in _POLICY_CONFIG_IDENTITY_FIELDS}


def _create_sealed_policy_config(gemini_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Create the exact fixed empty-grants file and return private baseline facts."""
    if not gemini_dir.is_absolute() or gemini_dir.resolve(strict=False) != gemini_dir:
        return None, "policy_config_root_unsafe"
    config_dir = gemini_dir / "config"
    path = config_dir / "config.json"
    descriptor: int | None = None
    try:
        root_stat = gemini_dir.lstat()
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or gemini_dir.is_symlink()
            or root_stat.st_uid != os.geteuid()
            or stat.S_IMODE(root_stat.st_mode) != 0o700
        ):
            return None, "policy_config_private_root_invalid"
        config_dir.mkdir(mode=0o700)
        directory_stat = config_dir.lstat()
        if (
            config_dir.is_symlink()
            or not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            return None, "policy_config_private_root_invalid"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(AGY_SEALED_POLICY_CONFIG):
            written = os.write(descriptor, AGY_SEALED_POLICY_CONFIG[offset:])
            if written <= 0:
                return None, "policy_config_initial_invalid"
            offset += written
        os.fsync(descriptor)
        file_stat = os.fstat(descriptor)
    except OSError:
        return None, "policy_config_initial_invalid"
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        directory_stat = config_dir.lstat()
        file_stat = path.lstat()
    except OSError:
        return None, "policy_config_initial_invalid"
    projection, parse_failure = _parse_restrictive_policy_config(AGY_SEALED_POLICY_CONFIG)
    if parse_failure is not None or projection.get("grant_counts") != {
        "allow": 0,
        "deny": 0,
        "ask": 0,
    }:
        return None, "policy_config_initial_invalid"
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != os.geteuid()
        or file_stat.st_nlink != 1
        or stat.S_IMODE(file_stat.st_mode) != 0o600
        or file_stat.st_size != len(AGY_SEALED_POLICY_CONFIG)
    ):
        return None, "policy_config_initial_invalid"
    return {
        "file": _policy_stat_identity(file_stat),
        "directory": _policy_stat_identity(directory_stat),
        "content_digest": _bytes_digest(AGY_SEALED_POLICY_CONFIG),
    }, None


def _agy_policy_sandbox_profile(
    gemini_dir: Path,
    workdir: Path,
    agy_executable: Path,
) -> str:
    """Return the canonical no-fork, pinned-exec, workdir-only write jail."""
    if gemini_dir.parent != workdir or not agy_executable.is_absolute():
        raise LiveEvalError("policy sandbox parameters are noncanonical", 4)
    if _bytes_digest(AGY_POLICY_SANDBOX_PROFILE.encode("utf-8")) != (
        AGY_POLICY_SANDBOX_PROFILE_DIGEST
    ):
        raise LiveEvalError("policy sandbox profile bytes drifted", 4)
    return AGY_POLICY_SANDBOX_PROFILE


def _capture_policy_config_transition(
    gemini_dir: Path,
    baseline: Mapping[str, Any] | None,
    *,
    write_protection_applied: bool,
) -> tuple[dict[str, Any], str | None]:
    """Prove the exact precreated config and its containing directory were unchanged."""
    if baseline is None:
        failure = "policy_config_initial_missing"
        return _unsafe_policy_config_transition(
            failure,
            initial_config_present=False,
            final_config_present=False,
            write_protection_applied=write_protection_applied,
        ), failure
    if not write_protection_applied:
        failure = "policy_config_write_protection_unsealed"
        return _unsafe_policy_config_transition(
            failure,
            write_protection_applied=False,
            write_protection_parameters_bound=False,
        ), failure
    config_dir = gemini_dir / "config"
    path = config_dir / "config.json"
    if path.is_symlink() or config_dir.is_symlink():
        failure = "policy_config_final_symlink"
        return _unsafe_policy_config_transition(failure), failure
    try:
        directory_stat = config_dir.lstat()
        before = path.lstat()
    except OSError:
        failure = "policy_config_final_missing"
        return _unsafe_policy_config_transition(
            failure,
            final_config_present=False,
        ), failure
    if not stat.S_ISDIR(directory_stat.st_mode) or not stat.S_ISREG(before.st_mode):
        failure = "policy_config_final_not_regular"
        return _unsafe_policy_config_transition(failure), failure
    if (
        directory_stat.st_uid != os.geteuid()
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        failure = "policy_config_private_owner_invalid"
        return _unsafe_policy_config_transition(failure), failure
    content, read_failure = _read_bounded_regular_file(
        path,
        limit=MAX_AGY_POLICY_CONFIG_BYTES,
        label="policy_config_final",
    )
    if read_failure is not None:
        failure = read_failure
        if failure not in _POLICY_CONFIG_FAILURES:
            failure = "policy_config_final_read_failed"
        return _unsafe_policy_config_transition(failure), failure
    try:
        after = path.lstat()
    except OSError:
        failure = "policy_config_final_missing"
        return _unsafe_policy_config_transition(failure, final_config_present=False), failure
    if any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        failure = "policy_config_final_changed_during_read"
        return _unsafe_policy_config_transition(failure), failure
    if (
        baseline.get("file") != _policy_stat_identity(after)
        or baseline.get("directory") != _policy_stat_identity(directory_stat)
    ):
        failure = "policy_config_final_identity_changed"
        return _unsafe_policy_config_transition(
            failure,
            private_regular_file=True,
        ), failure
    if (
        content != AGY_SEALED_POLICY_CONFIG
        or baseline.get("content_digest") != _bytes_digest(content)
    ):
        failure = "policy_config_final_content_changed"
        return _unsafe_policy_config_transition(
            failure,
            private_regular_file=True,
            same_file_identity=True,
        ), failure
    projection, parse_failure = _parse_restrictive_policy_config(content)
    if parse_failure is not None:
        return _unsafe_policy_config_transition(
            parse_failure,
            private_regular_file=True,
            same_file_identity=True,
            exact_precreated_payload=True,
        ), parse_failure
    return _unchanged_policy_config_transition(), None


def _delete_private_gemini_dir(gemini_dir: Path, workdir: Path) -> str | None:
    """Delete only the exact per-call private tree created by this module."""
    if (
        gemini_dir.parent != workdir
        or re.fullmatch(r"\.agy-gemini-[0-9a-f]{32}", gemini_dir.name) is None
        or gemini_dir.is_symlink()
    ):
        return "private_gemini_dir_identity_invalid"
    try:
        if gemini_dir.exists():
            if not gemini_dir.is_dir():
                return "private_gemini_dir_identity_invalid"
            shutil.rmtree(gemini_dir)
    except OSError:
        return "private_gemini_dir_deletion_failed"
    if gemini_dir.exists() or gemini_dir.is_symlink():
        return "private_gemini_dir_deletion_failed"
    return None


def _delete_private_tmpdir(tmp_dir: Path, workdir: Path) -> str | None:
    if (
        tmp_dir.parent != workdir
        or tmp_dir.name != "tmp"
        and re.fullmatch(r"\.agy-tmp-[0-9a-f]{32}", tmp_dir.name) is None
        or tmp_dir.is_symlink()
    ):
        return "private_tmpdir_identity_invalid"
    try:
        if tmp_dir.exists():
            if not tmp_dir.is_dir():
                return "private_tmpdir_identity_invalid"
            shutil.rmtree(tmp_dir)
    except OSError:
        return "private_tmpdir_deletion_failed"
    if tmp_dir.exists() or tmp_dir.is_symlink():
        return "private_tmpdir_deletion_failed"
    return None


def _agy_transcript_path(
    stdout_ndjson: bytes,
    log: bytes,
    *,
    expected_app_data_dir: Path | None = None,
) -> Path | None:
    try:
        text = log.decode("utf-8")
    except UnicodeDecodeError:
        return None
    app_dirs = tuple(dict.fromkeys(item.strip() for item in _AGY_APP_DATA_DIR.findall(text)))
    events, stream_error = _parse_ndjson(stdout_ndjson, "stdout_ndjson")
    conversations = _stream_conversation_ids(events) if stream_error is None else ()
    if (
        len(app_dirs) != 1
        or len(conversations) != 1
        or _AGY_UUID.fullmatch(conversations[0]) is None
    ):
        return None
    app_dir = Path(app_dirs[0])
    if not app_dir.is_absolute() or app_dir.resolve(strict=False) != app_dir:
        return None
    if expected_app_data_dir is not None and app_dir != expected_app_data_dir:
        return None
    brain_root = app_dir / "brain"
    transcript = (
        app_dir
        / "brain"
        / conversations[0]
        / ".system_generated"
        / "logs"
        / "transcript_full.jsonl"
    )
    if transcript.resolve(strict=False).parent.parent.parent != brain_root / conversations[0]:
        return None
    return transcript


async def call_llm_with_evidence(
    client_or_bin: Any,
    model: str,
    prompt: str,
    *,
    system: str = "",
    provider: str = "agy",
    workdir: Path,
    timeout_seconds: float = 180.0,
    evidence_log_path: Path | None = None,
    process_environment: Mapping[str, str] | None = None,
    pinned_policy_workdir: bool = False,
    before_spawn: Callable[[Mapping[str, Any]], None] | None = None,
) -> AgyCallEvidence:
    """Run one runtime-bound AGY call and return a bounded, self-classifying receipt."""
    if provider != "agy":
        raise LiveEvalError("structured evidence is currently supported only for agy", 4)
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    prompt_size = len(full_prompt.encode("utf-8"))
    if prompt_size > MAX_LLM_PROMPT_BYTES:
        raise LiveEvalError(
            f"LLM prompt is {prompt_size} bytes; limit is {MAX_LLM_PROMPT_BYTES}",
            4,
        )
    supplied_workdir = Path(workdir)
    if (
        not supplied_workdir.is_absolute()
        or any(candidate.is_symlink() for candidate in (supplied_workdir, *supplied_workdir.parents))
    ):
        raise LiveEvalError("agy structured workdir must already be canonical", 4)
    workdir = supplied_workdir.resolve(strict=True)
    if workdir != supplied_workdir:
        raise LiveEvalError("agy structured workdir must already be canonical", 4)
    workdir_stat = workdir.lstat()
    if (
        not workdir.is_dir()
        or workdir.is_symlink()
        or workdir_stat.st_uid != os.geteuid()
        or workdir_stat.st_nlink != 2
        or stat.S_IMODE(workdir_stat.st_mode) != 0o700
        or any(workdir.iterdir())
    ):
        raise LiveEvalError("agy structured workdir must be a real directory", 4)
    if pinned_policy_workdir and workdir.parent != Path("/private/tmp"):
        raise LiveEvalError("agy pinned policy workdir must be under /private/tmp", 4)
    try:
        agy_executable = Path(client_or_bin)
    except TypeError as exc:
        raise LiveEvalError("agy executable path is invalid", 4) from exc
    if (
        not agy_executable.is_absolute()
        or agy_executable.is_symlink()
        or not agy_executable.is_file()
        or agy_executable.resolve(strict=True) != agy_executable
    ):
        raise LiveEvalError("agy executable must be one canonical regular file", 4)
    call_nonce = uuid.uuid4().hex
    log_path = evidence_log_path or workdir / f"agy-{call_nonce}.log"
    if (
        not log_path.is_absolute()
        or log_path.parent.resolve(strict=True) != workdir
        or any(candidate.is_symlink() for candidate in (log_path, *log_path.parents))
        or log_path.exists()
        or log_path.is_symlink()
    ):
        raise LiveEvalError("agy evidence log path must be unique and inside workdir", 4)
    gemini_dir = workdir / f".agy-gemini-{call_nonce}"
    gemini_dir.mkdir(mode=0o700)
    tmp_dir = workdir / ("tmp" if pinned_policy_workdir else f".agy-tmp-{call_nonce}")
    if tmp_dir.exists() or tmp_dir.is_symlink():
        _delete_private_gemini_dir(gemini_dir, workdir)
        raise LiveEvalError("agy private TMPDIR path must be fresh", 4)
    tmp_dir.mkdir(mode=0o700)
    tmp_stat = tmp_dir.lstat()
    if (
        tmp_dir.is_symlink()
        or not stat.S_ISDIR(tmp_stat.st_mode)
        or tmp_stat.st_uid != os.geteuid()
        or tmp_stat.st_nlink != 2
        or stat.S_IMODE(tmp_stat.st_mode) != 0o700
    ):
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise LiveEvalError("agy private TMPDIR identity is unsafe", 4)
    app_data_dir = gemini_dir / "antigravity-cli"
    policy_baseline, policy_failure = _create_sealed_policy_config(gemini_dir)
    if policy_failure is not None or policy_baseline is None:
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise LiveEvalError(
            f"agy fixed empty-grants config cannot be created: {policy_failure}",
            4,
        )
    policy_config, policy_failure = _policy_config_identity(gemini_dir)
    if (
        policy_failure is not None
        or policy_config is None
        or policy_config.get("exists") is not True
    ):
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise LiveEvalError(
            f"agy inherited policy config cannot be sealed: {policy_failure}",
            4,
        )
    sandbox_executable = AGY_SANDBOX_EXECUTABLE
    if (
        not sandbox_executable.is_absolute()
        or sandbox_executable.is_symlink()
        or not sandbox_executable.is_file()
        or _bytes_digest(sandbox_executable.read_bytes())
        != EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST
    ):
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise LiveEvalError("sealed policy-config sandbox executable is unavailable", 4)
    sandbox_digest_before = EXPECTED_AGY_SANDBOX_EXECUTABLE_DIGEST
    config_dir = gemini_dir / "config"
    config_file = config_dir / "config.json"
    agy_cmd = [
        str(agy_executable),
        "-p",
        full_prompt,
        "--disable-slash-commands",
        "--sandbox",
        "--print-timeout",
        f"{max(1, int(timeout_seconds))}s",
        "--output-format",
        AGY_OUTPUT_FORMAT,
        "--log-file",
        str(log_path),
        "--gemini_dir",
        str(gemini_dir),
        "--app_data_dir",
        "antigravity-cli",
    ]
    if model != "agy-subscription":
        agy_cmd.extend(["--model", model])
    cmd = [
        str(sandbox_executable),
        "-D",
        f"WORKDIR={workdir}",
        "-D",
        f"GEMINI_DIR={gemini_dir}",
        "-D",
        f"CONFIG_DIR={config_dir}",
        "-D",
        f"CONFIG_FILE={config_file}",
        "-D",
        f"AGY_BIN={agy_executable}",
        "-p",
        _agy_policy_sandbox_profile(gemini_dir, workdir, agy_executable),
        *agy_cmd,
    ]
    spawn_options = {
        "cwd": str(workdir),
        "env": _agy_environment(process_environment, private_tmpdir=tmp_dir),
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "stdin": asyncio.subprocess.DEVNULL,
        "close_fds": True,
        "start_new_session": True,
    }
    invocation_receipt = _sandbox_invocation_receipt(
        command=cmd,
        spawn_options=spawn_options,
        workdir=workdir,
        gemini_dir=gemini_dir,
        config_dir=config_dir,
        config_file=config_file,
        tmp_dir=tmp_dir,
        log_path=log_path,
        agy_executable=agy_executable,
        sandbox_executable=sandbox_executable,
        model=model,
        full_prompt=full_prompt,
        timeout_seconds=timeout_seconds,
    )
    if _validate_sandbox_invocation_receipt(invocation_receipt) is None:
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise LiveEvalError("agy sandbox invocation receipt is invalid", 4)
    try:
        if before_spawn is not None:
            before_spawn(invocation_receipt)
    except BaseException:
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            **spawn_options,
        )
    except (OSError, ValueError) as exc:
        policy_config_transition, _ = _capture_policy_config_transition(
            gemini_dir,
            policy_baseline,
            write_protection_applied=True,
        )
        cleanup_failure = _delete_private_gemini_dir(gemini_dir, workdir)
        spawn_failures = [f"spawn_failed:{type(exc).__name__}"]
        if cleanup_failure is not None:
            spawn_failures.append(cleanup_failure)
        tmp_cleanup_failure = _delete_private_tmpdir(tmp_dir, workdir)
        if tmp_cleanup_failure is not None:
            spawn_failures.append(tmp_cleanup_failure)
        return analyze_agy_evidence(
            requested_model=model,
            full_prompt=full_prompt,
            invocation_workdir=workdir,
            exit_status=None,
            timed_out=False,
            process_group_descendant_detected=False,
            process_group_quiescent=False,
            capture_failures=tuple(spawn_failures),
            stdout_ndjson=b"",
            stderr=str(exc).encode("utf-8", errors="replace")[:MAX_LLM_STDERR_BYTES],
            log=b"",
            transcript=b"",
            policy_config_identity=policy_config,
            policy_config_transition=policy_config_transition,
            sandbox_invocation_receipt=invocation_receipt,
        )
    except BaseException:
        _delete_private_gemini_dir(gemini_dir, workdir)
        _delete_private_tmpdir(tmp_dir, workdir)
        raise
    stdout, stderr, timed_out, capture_failures = await _communicate_agy_evidence(
        proc,
        timeout_seconds=timeout_seconds + 5.0,
    )
    (
        process_group_descendant_detected,
        process_group_quiescent,
    ) = await _quiesce_agy_process_group(proc)
    log, log_failure = _read_bounded_regular_file(
        log_path,
        limit=MAX_AGY_LOG_BYTES,
        label="log",
    )
    failures = list(capture_failures)
    if process_group_descendant_detected:
        failures.append("process_group_descendant_detected")
    if not process_group_quiescent:
        failures.append("process_group_quiescence_unproven")
    if log_failure:
        failures.append(log_failure)
    transcript = b""
    transcript_path = _agy_transcript_path(
        stdout,
        log,
        expected_app_data_dir=app_data_dir,
    )
    if transcript_path is None:
        failures.append("transcript_identity_missing")
    else:
        transcript, transcript_failure = _read_bounded_regular_file(
            transcript_path,
            limit=MAX_AGY_TRANSCRIPT_BYTES,
            label="transcript",
        )
        if transcript_failure:
            failures.append(transcript_failure)
    policy_config_transition, _ = _capture_policy_config_transition(
        gemini_dir,
        policy_baseline,
        write_protection_applied=True,
    )
    if (
        not sandbox_executable.is_file()
        or sandbox_executable.is_symlink()
        or _bytes_digest(sandbox_executable.read_bytes()) != sandbox_digest_before
    ):
        failures.append("sandbox_executable_changed_during_call")
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        failures.append("private_log_deletion_failed")
    if log_path.exists() or log_path.is_symlink():
        failures.append("private_log_deletion_failed")
    cleanup_failure = _delete_private_gemini_dir(gemini_dir, workdir)
    if cleanup_failure is not None:
        failures.append(cleanup_failure)
    tmp_cleanup_failure = _delete_private_tmpdir(tmp_dir, workdir)
    if tmp_cleanup_failure is not None:
        failures.append(tmp_cleanup_failure)
    return analyze_agy_evidence(
        requested_model=model,
        full_prompt=full_prompt,
        invocation_workdir=workdir,
        exit_status=proc.returncode,
        timed_out=timed_out,
        process_group_descendant_detected=process_group_descendant_detected,
        process_group_quiescent=process_group_quiescent,
        capture_failures=tuple(failures),
        stdout_ndjson=stdout,
        stderr=stderr,
        log=log,
        transcript=transcript,
        policy_config_identity=policy_config,
        policy_config_transition=policy_config_transition,
        sandbox_invocation_receipt=invocation_receipt,
    )


async def call_llm(
    client_or_bin: Any,
    model: str,
    prompt: str,
    *,
    system: str = "",
    provider: str = "agy",
    workdir: Path | None = None,
    timeout_seconds: float = 180.0,
) -> str:
    """Make one bounded LLM call without granting the CLI repository access."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    prompt_size = len(full_prompt.encode("utf-8"))
    if prompt_size > MAX_LLM_PROMPT_BYTES:
        raise LiveEvalError(
            f"LLM prompt is {prompt_size} bytes; limit is {MAX_LLM_PROMPT_BYTES}",
            4,
        )
    if provider == "agy":
        selected_workdir = (workdir or Path.cwd()).resolve(strict=True)
        evidence = await call_llm_with_evidence(
            client_or_bin,
            model,
            prompt,
            system=system,
            provider=provider,
            workdir=selected_workdir,
            timeout_seconds=timeout_seconds,
        )
        if evidence.disposition != "response" or evidence.response is None:
            raise LiveEvalError(
                f"agy structured call ended as {evidence.disposition}: {evidence.error}",
                4,
            )
        response = evidence.response
    else:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            result = await asyncio.wait_for(
                client_or_bin.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                ),
                timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise LiveEvalError(
                f"{provider} call timed out after {timeout_seconds:.0f}s", 4
            ) from exc
        response = result.choices[0].message.content or ""

    if not response.strip():
        raise LiveEvalError(f"{provider} returned an empty response", 4)
    response_size = len(response.encode("utf-8"))
    if response_size > MAX_LLM_RESPONSE_BYTES:
        raise LiveEvalError(
            f"{provider} response exceeded {MAX_LLM_RESPONSE_BYTES} bytes",
            4,
        )
    return response.strip()


def extract_python_code(text: str) -> str:
    """Extract a non-empty Python code block from a model response."""
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = (match.group(1) if match else text).strip()
    if not code:
        raise LiveEvalError("Designer returned no environment code", 4)
    return code


def extract_clean_action(response: str, action_format: str = "boxed") -> str:
    """Normalize an LLM response to the action contract SPADE uses."""
    if action_format == "boxed" and extract_boxed_answer(response) is None:
        # Never send an entire reasoning transcript to an environment. A malformed
        # response becomes one explicit invalid action so the environment can return
        # its normal format feedback and the model can try again on the next turn.
        return r"\boxed{__spade_invalid_action_format__}"
    return parse_action(response, action_format)


def _solution_values(solution: Any) -> list[str]:
    raw_values: Sequence[Any] = solution if isinstance(solution, (list, tuple)) else (solution,)
    values: list[str] = []
    for raw in raw_values:
        text = str(raw).strip()
        values.append(extract_boxed_answer(text) or text)
    return [value for value in values if value]


def hint_reveals_solution(
    hint: str,
    solution: Any,
    observation: str = "",
) -> bool:
    """Detect obvious exact-answer leakage before a hint reaches the actor."""
    hint_lower = " ".join(hint.lower().split())
    observation_lower = " ".join(observation.lower().split())
    boxed_hint = extract_boxed_answer(hint)
    for value in _solution_values(solution):
        normalized = " ".join(value.lower().split())
        if boxed_hint and " ".join(boxed_hint.lower().split()) == normalized:
            return True
        answer_pattern = re.compile(
            rf"(?:answer|solution|result|submit|choose|select|pick|enter|use|return|send)"
            rf"\s*(?:is|=|:|as)?\s*[`'\"]?{re.escape(normalized)}(?:\b|[`'\"])",
            re.IGNORECASE,
        )
        if answer_pattern.search(hint_lower):
            return True
        # A hidden value appearing verbatim anywhere in the hint is leakage,
        # including short numeric answers such as 4 or 32. Values already visible
        # in the observation still use the answer-phrase checks above, since a
        # strategy may legitimately refer to puzzle inputs.
        token_pattern = re.compile(
            rf"(?<![\w.]){re.escape(normalized)}(?![\w.])",
            re.IGNORECASE,
        )
        if normalized not in observation_lower and token_pattern.search(hint_lower):
            return True
    return False


async def generate_nonleaking_hint(
    client_or_bin: Any,
    model: str,
    observation: str,
    solution: Any,
    *,
    provider: str,
    workdir: Path,
    timeout_seconds: float,
    attempts: int = 2,
) -> str:
    """Generate and check a hint; fail rather than use leaked privileged values."""
    feedback = ""
    for attempt in range(1, attempts + 1):
        prompt = HINT_PROMPT.format(observation=observation) + feedback
        hint = await call_llm(
            client_or_bin,
            model,
            prompt,
            system="Provide strategy only; never solve the puzzle for the player.",
            provider=provider,
            workdir=workdir,
            timeout_seconds=timeout_seconds,
        )
        if not hint_reveals_solution(hint, solution, observation):
            return hint
        feedback = (
            "\n\nYour previous hint exposed the final answer. Rewrite it using only general "
            "strategy and no exact answer values."
        )
        if attempt == attempts:
            break
    raise LiveEvalError("Hint generation repeatedly exposed the oracle solution", 6)


async def run_multi_turn_rollout(
    client_or_bin: Any,
    model: str,
    target: Any,
    seed: int,
    *,
    hint_text: str = "",
    provider: str = "agy",
    max_turns: int = 5,
    action_format: str = "boxed",
    workdir: Path | None = None,
    timeout_seconds: float = 180.0,
) -> tuple[float, list[dict[str, Any]]]:
    """Execute a paired-compatible, bounded, multi-turn environment episode."""
    env = target.instantiate()
    try:
        observation, _info = env.reset(seed=seed)
        trajectory: list[dict[str, Any]] = [
            {"role": "environment", "observation": observation, "seed": seed}
        ]
        history = f"Initial observation: {observation}"
        if hint_text:
            history += f"\n\nPrivileged strategy hint:\n{hint_text}"

        terminated = False
        truncated = False
        last_reward = 0.0
        turn = 0
        while not (terminated or truncated) and turn < max_turns:
            turn += 1
            prompt = (
                "You are playing an interactive reasoning environment.\n"
                f"{history}\n\nTurn {turn}/{max_turns}: reason about the state, then provide "
                "exactly one next action with the required answer format."
            )
            raw_response = await call_llm(
                client_or_bin,
                model,
                prompt,
                provider=provider,
                workdir=workdir,
                timeout_seconds=timeout_seconds,
            )
            action = extract_clean_action(raw_response, action_format)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, _step_info = step_result
            elif len(step_result) == 4:
                next_observation, reward, terminated, _step_info = step_result
                truncated = False
            else:
                raise LiveEvalError(
                    f"Environment step returned {len(step_result)} values instead of 4 or 5",
                    7,
                )
            last_reward = float(reward)
            terminated = bool(terminated)
            truncated = bool(truncated)
            trajectory.append(
                {
                    "turn": turn,
                    "raw_response": raw_response,
                    "clean_action": action,
                    "observation": next_observation,
                    "reward": last_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            history += (
                f"\n\nAction at turn {turn}: {action}\nEnvironment response: {next_observation}"
            )

        # Match SPADE's outcome-only episode aggregation: an unfinished episode
        # is not a success merely because it received partial progress reward.
        return (last_reward if terminated else 0.0), trajectory
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _qualification_reason(report: Any) -> str:
    failures = [
        f"{clause_id}: {clause.summary}"
        for clause_id, clause in report.clauses.items()
        if clause.status != "pass"
    ]
    return "; ".join(failures) or "unknown qualification failure"


async def generate_qualified_environment(
    client_or_bin: Any,
    model: str,
    skill: str,
    *,
    provider: str,
    workdir: Path,
    llm_timeout: float,
    qualification_timeout: float,
    max_turns: int,
    attempts: int,
) -> tuple[str, Any]:
    """Generate until ProofPack accepts, then return code and its receipt."""
    feedback = ""
    last_reason = "designer produced no candidate"
    for attempt in range(1, attempts + 1):
        prompt = DESIGNER_PROMPT.format(skill=skill) + feedback
        raw_response = await call_llm(
            client_or_bin,
            model,
            prompt,
            system="Return secure, deterministic environment source code only.",
            provider=provider,
            workdir=workdir,
            timeout_seconds=llm_timeout,
        )
        code = extract_python_code(raw_response)
        report = qualify_spade_environment(
            code,
            seeds=list(QUALIFIED_SEEDS),
            timeout_seconds=qualification_timeout,
            max_turns=max_turns,
        )
        if report.passed:
            return code, report
        last_reason = _qualification_reason(report)
        print(f"   Attempt {attempt}/{attempts} rejected: {last_reason}")
        feedback = (
            "\n\nThe previous candidate failed formal qualification:\n"
            f"{last_reason}\nReturn a complete corrected environment, not a patch."
        )
    raise LiveEvalError(
        f"No environment passed ProofPack after {attempts} attempts: {last_reason}",
        5,
    )


async def run_live_eval(args: argparse.Namespace) -> Path:
    """Run the complete live smoke and return its artifact directory."""
    _require_integrations()
    if args.env_file:
        _load_explicit_env_file(Path(args.env_file).expanduser())

    client_or_bin, model = get_llm_client(
        args.provider,
        args.model,
        args.base_url,
        args.api_key,
        timeout_seconds=args.llm_timeout,
    )

    print("\n" + "=" * 76)
    print("LIVE SPADE + PROOFPACK + ASSAY VERIFICATION")
    print(f"Provider: {args.provider} | Model: {model} | Skill: {args.skill}")
    print("=" * 76)

    with tempfile.TemporaryDirectory(prefix="spade-live-llm-") as llm_tmp:
        llm_workdir = Path(llm_tmp)

        print("\n[1/5] Generating and formally qualifying an environment")
        env_code, report = await generate_qualified_environment(
            client_or_bin,
            model,
            args.skill,
            provider=args.provider,
            workdir=llm_workdir,
            llm_timeout=args.llm_timeout,
            qualification_timeout=args.qualification_timeout,
            max_turns=args.max_turns,
            attempts=args.design_attempts,
        )
        print(f"   Qualified {report.environment_name}: {report.environment_digest}")

        print("\n[2/5] Opening the qualified replay-backed environment session")
        target = SpadeEnvironmentTarget(
            env_code,
            action_format="boxed",
            max_turns=args.max_turns,
            operation_timeout_seconds=args.qualification_timeout,
        )
        probe = target.instantiate()
        try:
            play_seed = QUALIFIED_SEEDS[-1]
            observation, _ = probe.reset(seed=play_seed)
            oracle_solution = probe.solution()
        finally:
            close = getattr(probe, "close", None)
            if callable(close):
                close()
        print(f"   Using qualified seed {play_seed}; observation length={len(observation)}")

        print("\n[3/5] Generating a checked, observation-only strategy hint")
        hint = await generate_nonleaking_hint(
            client_or_bin,
            model,
            observation,
            oracle_solution,
            provider=args.provider,
            workdir=llm_workdir,
            timeout_seconds=args.llm_timeout,
        )
        print(
            f"   Hint accepted ({len(hint)} characters; explicit-answer leakage heuristic passed)"
        )

        print("\n[4/5] Running paired unhinted and hinted multi-turn episodes")
        unhinted_return, unhinted_trajectory = await run_multi_turn_rollout(
            client_or_bin,
            model,
            target,
            play_seed,
            provider=args.provider,
            max_turns=args.max_turns,
            action_format="boxed",
            workdir=llm_workdir,
            timeout_seconds=args.llm_timeout,
        )
        hinted_return, hinted_trajectory = await run_multi_turn_rollout(
            client_or_bin,
            model,
            target,
            play_seed,
            hint_text=hint,
            provider=args.provider,
            max_turns=args.max_turns,
            action_format="boxed",
            workdir=llm_workdir,
            timeout_seconds=args.llm_timeout,
        )
        regret = max(0.0, hinted_return - unhinted_return)
        print(
            f"   returns: unhinted={unhinted_return:.3f}, "
            f"hinted={hinted_return:.3f}, regret={regret:.3f}"
        )

    digest_suffix = report.environment_digest.split(":")[-1][:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"spade-live-{timestamp}-{digest_suffix}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    qualification_bytes = (report.to_json() + "\n").encode("utf-8")
    trace_bytes = (
        json.dumps(
            {
                "schema_version": "spade-live-trace/v1",
                "run_id": run_id,
                "provider": args.provider,
                "model": model,
                "skill": args.skill,
                "environment_digest": report.environment_digest,
                "seed": play_seed,
                "hint": hint,
                "unhinted_return": unhinted_return,
                "hinted_return": hinted_return,
                "regret": regret,
                "unhinted_trajectory": unhinted_trajectory,
                "hinted_trajectory": hinted_trajectory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (run_dir / "environment.py").write_text(env_code, encoding="utf-8")
    (run_dir / "proofpack-qualification.json").write_bytes(qualification_bytes)
    (run_dir / "live-trace.json").write_bytes(trace_bytes)

    print("\n[5/5] Writing native Assay evidence and certification artifacts")
    task = SpadeTaskPayload(
        task_id=f"{run_id}-task",
        environment_name=report.environment_name,
        skill=args.skill,
        code=env_code,
        solution=str(oracle_solution),
        max_turns=args.max_turns,
        seed=play_seed,
        metadata={
            "proofpack_environment_digest": report.environment_digest,
            "proofpack_qualification_receipt_digest": (
                "sha256:" + hashlib.sha256(qualification_bytes).hexdigest()
            ),
            "live_trace_digest": "sha256:" + hashlib.sha256(trace_bytes).hexdigest(),
        },
    )
    cluster = SpadeClusterData(
        cluster_id=f"{run_id}-task",
        candidate_returns=(hinted_return,),
        base_returns=(unhinted_return,),
        hinted_returns=(hinted_return,),
        regret=regret,
    )
    try:
        artifact_result = write_spade_evaluation(
            output_dir=run_dir / "assay",
            curriculum_id=run_id,
            tasks=(task,),
            clusters=(cluster,),
            candidate_arm=f"{model}:hinted",
            base_arm=f"{model}:unhinted",
            run_metadata=SpadeRunMetadata(run_id=run_id),
            minimum_clusters=args.minimum_certification_clusters,
        )
    except Exception as exc:
        raise LiveEvalError(
            f"Assay could not persist the live evidence: {type(exc).__name__}: {exc}",
            8,
        ) from exc
    print(f"   Assay decision: {artifact_result.report.rationale}")
    print(f"   Statistical signal: {artifact_result.report.promoted}")
    print(f"   Release authorized: {artifact_result.report.release_authorized}")
    print(f"   Model lock: {artifact_result.model_lock_path or 'not emitted'}")
    print(f"\nArtifacts: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live SPADE + ProofPack + Assay smoke test")
    parser.add_argument(
        "--provider",
        default="agy",
        choices=("agy", "google", "gemini", "openrouter", "openai", "local", "ollama", "vllm"),
    )
    parser.add_argument("--model", default=None, help="Provider model; omit for provider default")
    parser.add_argument("--skill", default="Graph Theory and Shortest Path")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--env-file",
        default=None,
        help="Explicit dotenv path for API providers; ~/.env is never loaded automatically",
    )
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--design-attempts", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--qualification-timeout", type=float, default=5.0)
    parser.add_argument("--minimum-certification-clusters", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / ".assay" / "spade-live"),
        help="Parent directory for unique, digest-bound per-run artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.max_turns < 1
        or args.design_attempts < 1
        or not math.isfinite(args.llm_timeout)
        or args.llm_timeout <= 0
        or not math.isfinite(args.qualification_timeout)
        or args.qualification_timeout <= 0
        or args.minimum_certification_clusters < 4
    ):
        print(
            "error: turn/attempt/timeouts must be positive and "
            "--minimum-certification-clusters must be at least 4",
            file=sys.stderr,
        )
        return 2
    try:
        asyncio.run(run_live_eval(args))
    except LiveEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
