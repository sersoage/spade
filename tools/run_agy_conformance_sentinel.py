#!/usr/bin/env python3
"""Seal, run, and audit the prospective AGY 1.1.24 conformance sentinel.

The v2 sentinel is anchored to the terminal v1 reservation at global ordinal
317.  V1 has a request, ledger entry, and four evidence receipts, but no result;
it is charged and is never replayed.  V2 exercises the observed AGY 1.1.24
``run_command`` ACTIVE-to-ERROR soft-denial wire shape with auto-update disabled.
It is not an evaluation, learner experiment, or backend-identity attestation.
A durable request reservation consumes the single-call allowance; an unclosed
reservation is never replayed.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import run_live_spade_eval as live  # noqa: E402


INTENT_SCHEMA = "spade-agy-conformance-sentinel-intent/v2"
RUN_SCHEMA = "spade-agy-conformance-sentinel-run/v2"
CALL_REQUEST_SCHEMA = "spade-agy-conformance-sentinel-call-request/v2"
CALL_RESULT_SCHEMA = "spade-agy-conformance-sentinel-call-result/v2"
DECISION_SCHEMA = "spade-agy-conformance-sentinel-decision/v2"
PRIOR_ANCHOR_SCHEMA = "spade-stranded-agy-sentinel-anchor/v2"
LEDGER_HEADER_SCHEMA = "spade-agy-conformance-sentinel-ledger/v2"
LEDGER_ENTRY_SCHEMA = "spade-agy-conformance-sentinel-ledger-entry/v2"
WORKDIR_OBSERVATION_SCHEMA = "spade-agy-sentinel-workdir-observation/v2"
PROTOCOL_ID = "spade-agy-1.1.24-structured-tool-denial-sentinel/v2"
EXPERIMENT_ID = "spade-agy-1-1-24-tool-denial-sentinel-v2"
OUTPUT_ROOT_NAME = "spade-agy-conformance-sentinel-v2"
LEDGER_ROOT_NAME = "shared-ledger"
MODEL = "gemini-3.7-flash-high"
EXPECTED_AGY_VERSION = "1.1.24"
EXPECTED_AGY_EXECUTABLE_DIGEST = (
    "sha256:4d1138b2dbde56127969fd307281494d4a7dcc22759ce9adb44d36247df86151"
)
EXPECTED_TOOL = "run_command"
EXPECTED_LOG_TOOL = "RunCommand"
EXPECTED_GENERIC_DISPOSITION = "evidence_integrity_failure"
EXPECTED_GENERIC_ERROR = "agy evidence indicates that a model-selected tool may have executed"
EXPECTED_RESPONSE_ID_COUNT = 2
AUTO_UPDATE_ENVIRONMENT = {"AGY_CLI_DISABLE_AUTO_UPDATE": "1"}
CANARY = "SPADE_AGY_SENTINEL_TOOL_MUST_NOT_EXECUTE"
PROMPT = (
    "This is a non-production AGY 1.1.24 adapter conformance sentinel in a disposable "
    "empty directory.\n"
    "Invoke the RunCommand tool exactly once with this harmless command:\n"
    f"touch {CANARY}\n"
    "Do not answer in prose, do not use another tool, and do not retry."
)
SYSTEM = ""
PURPOSE = {
    "phase": "sentinel",
    "check": "agy-1.1.24-structured-tool-soft-denial",
    "attempt": 2,
}
TIMEOUT_SECONDS = 180.0
RESULT_WALL_DURATION_TOLERANCE_SECONDS = 5.0
PRIOR_CHARGED_CALLS = 317
NEW_CALL_CAP = 1
AUTHORIZED_GLOBAL_CALL_CAP = 450
GLOBAL_ORDINAL = 318
EXPECTED_TOOL_PARAMETERS_RECEIPT = {
    "digest": "sha256:c0521fbb71b76cca29c8e3f43eab7c1e4558832e950ba06f40f6a29c41472017",
    "size_bytes": 65,
}
EXPECTED_NULL_VALUE_RECEIPT = {
    "digest": "sha256:38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed",
    "size_bytes": 5,
}
EVIDENCE_POLICY = "bounded-sanitized-stream-stderr-log-transcript-receipts-with-exact-digests"
WORKDIR_POLICY = "fresh-empty-temporary-directory-deleted-after-call"
PRIVACY_POLICY = (
    "no-raw-log-transcript-or-private-gemini-directory-survives;"
    "only-bounded-sanitized-digest-bound-receipts"
)
AGY_EVIDENCE_FILENAMES = {
    "stdout_ndjson": "agy.stream-receipt.json",
    "stderr": "agy.stderr-receipt.json",
    "log": "agy.log-receipt.json",
    "transcript": "agy.transcript-receipt.json",
}
WORKDIR_OBSERVATION_FILENAME = "workdir-observation.json"
AGY_EVIDENCE_LIMITS = {
    "stdout_ndjson": live.MAX_LLM_RESPONSE_BYTES,
    "stderr": live.MAX_LLM_STDERR_BYTES,
    "log": live.MAX_AGY_LOG_BYTES,
    "transcript": live.MAX_AGY_TRANSCRIPT_BYTES,
}
MAX_WORKDIR_OBSERVATION_BYTES = 16 * 1024
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,191}$")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

# These values bind v2 to the exact terminal v1 request-without-result closure,
# not merely to a caller-supplied integer saying that 317 calls were used.
KNOWN_PRIOR_IDENTITY: Mapping[str, Any] = {
    "prior_global_ordinal": 317,
    "charged_call_count": 1,
    "closure_status": "request-without-result-terminal-no-replay",
    "prior_intent_digest": (
        "sha256:6a8bc6412fd93eb81663ad0657d5ca539a9d17cb045862600625024bc61f5e17"
    ),
    "prior_protocol_id": "spade-agy-1.1.23-structured-tool-denial-sentinel/v1",
    "prior_request_digest": (
        "sha256:390fbd8f10644b6fa70be7a645423a2b31314836837b5743017c1df29c4ee7f7"
    ),
    "prior_ledger_header_digest": (
        "sha256:5d5f050288ff3b4b73f110263682123aa9b81ff8fb2f8b1e7a4d0a2514dcc9d1"
    ),
    "prior_terminal_entry_digest": (
        "sha256:1c21d7093aa6c86593d9e3ac92e8872a38e951e4e1b9118e173d15bd99407ad7"
    ),
    "prior_coverage_anchor_digest": (
        "sha256:6733eb7615d048b63ad9c6c1eab2296e30bf05f2854ea1f05f13db3bf16dc213"
    ),
    "prior_runtime_agy_version": "1.1.23",
    "prior_runtime_agy_executable_digest": (
        "sha256:dea6443f3167d0ff1af9adf0bc9f96f13be85c8206a399bd33e2de87fdc39f7a"
    ),
    "receipt_leaf_count": 4,
    "receipt_manifest_digest": (
        "sha256:b9a98419e473a46fe6bc49dc397f1db39c7b540bd153ededae41a63b66f481de"
    ),
    "closure_leaf_count": 10,
    "closure_manifest_digest": (
        "sha256:59447ff7609396033d5b3d298c0a67d972fb163bca95b2ffbd7df327eb2706d1"
    ),
    "result_present": False,
    "decision_present": False,
    "anchor_digest": (
        "sha256:7441e6c0686416c4a3733727f60980e1fff965d9856546772c760c524bbb080c"
    ),
}


class SentinelError(RuntimeError):
    """A fail-closed conformance or artifact error."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class SentinelIncomplete(SentinelError):
    """The one reserved request has no durable provider disposition."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 9)


StructuredCall = Callable[..., Awaitable[live.AgyCallEvidence]]


@dataclass(frozen=True)
class RunnerDependencies:
    """The only live boundary; offline tests inject a structured fake."""

    structured_llm_call: StructuredCall
    client_or_bin: Any
    runtime_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    status: str
    intent_digest: str
    run_dir: Path
    charged_call_count: int
    provider_calls_started: int
    decision_path: Path | None = None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SentinelError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise SentinelError(f"non-finite JSON number: {value}")


def _decode_json(content: bytes, where: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SentinelError(f"invalid JSON in {where}: {exc}") from exc
    if not isinstance(value, dict):
        raise SentinelError(f"{where} must contain one JSON object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SentinelError(
            f"{where} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _required_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SentinelError(f"{where} must be a non-empty string")
    return value


def _sha256(value: object, where: str) -> str:
    text = _required_text(value, where)
    if _SHA256.fullmatch(text) is None:
        raise SentinelError(f"{where} must be a lowercase SHA-256 digest")
    return text


def _safe_id(value: object, where: str) -> str:
    text = _required_text(value, where)
    if _SAFE_ID.fullmatch(text) is None:
        raise SentinelError(f"{where} must match {_SAFE_ID.pattern}")
    return text


def _validate_timestamp(value: object, where: str) -> datetime:
    text = _required_text(value, where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SentinelError(f"{where} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise SentinelError(f"{where} must include a timezone")
    return parsed


def _utc_now() -> str:
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _reject_symlink_ancestors(target: Path) -> None:
    candidate = target
    while True:
        if candidate.is_symlink():
            raise SentinelError(f"symlinked artifact paths are forbidden: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _canonical_dir(value: Path | str, where: str, *, exists: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise SentinelError(f"{where} must be absolute")
    _reject_symlink_ancestors(path)
    resolved = path.resolve(strict=exists)
    if path != resolved:
        raise SentinelError(f"{where} must already be canonical")
    if exists and (path.is_symlink() or not path.is_dir()):
        raise SentinelError(f"{where} must be a real directory")
    if not exists and path.exists() and not path.is_dir():
        raise SentinelError(f"{where} must be a directory when it exists")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.is_symlink():
        raise SentinelError(f"symlinked artifact paths are forbidden: {cursor}")
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


def _write_immutable_bytes(target: Path, content: bytes) -> None:
    _reject_symlink_ancestors(target)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise SentinelError(f"immutable path has conflicting bytes: {target}")
        return
    _mkdir_durable(target.parent)
    _reject_symlink_ancestors(target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o644)
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError(f"zero-byte write staging {target}")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary_name, target, follow_symlinks=False)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                raise SentinelError(f"immutable path raced with different bytes: {target}")
        _fsync_directory(target.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _write_json(target: Path, value: Mapping[str, Any]) -> None:
    _write_immutable_bytes(target, _pretty_json(value))


def _read_json(target: Path) -> dict[str, Any]:
    _reject_symlink_ancestors(target)
    if target.is_symlink() or not target.is_file():
        raise SentinelError(f"required immutable JSON leaf is missing or unsafe: {target}")
    content = target.read_bytes()
    value = _decode_json(content, str(target))
    if content != _pretty_json(value):
        raise SentinelError(f"JSON leaf is not in canonical pretty form: {target}")
    return value


def _read_compact_receipt_json(target: Path) -> dict[str, Any]:
    """Read an immutable adapter receipt in its canonical compact encoding."""
    _reject_symlink_ancestors(target)
    if target.is_symlink() or not target.is_file():
        raise SentinelError(f"required immutable receipt is missing or unsafe: {target}")
    content = target.read_bytes()
    value = _decode_json(content, str(target))
    if content != live._canonical_json_bytes(value):
        raise SentinelError(f"receipt is not in canonical compact form: {target}")
    return value


@contextmanager
def _single_writer(run_dir: Path):
    _mkdir_durable(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise SentinelError(f"run directory is unsafe: {run_dir}")
    lock_path = run_dir / ".writer.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SentinelIncomplete(f"another writer owns the sentinel run: {run_dir}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "digest": _bytes_digest(content),
        "size_bytes": len(content),
    }


def _validate_self_digest(value: Mapping[str, Any], field: str, where: str) -> None:
    body = {key: item for key, item in value.items() if key != field}
    if value.get(field) != _digest(body):
        raise SentinelError(f"{where} {field} mismatch")


def _compute_prior_usage_anchor(
    prior_output_root: Path | str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate and hash the exact terminal, non-replayable v1 closure."""
    root = _canonical_dir(prior_output_root, "prior_output_root", exists=True)
    intent = _read_json(root / "intent.json")
    _validate_self_digest(intent, "intent_digest", "prior intent")
    expected_intent_fixed = {
        "schema_version": "spade-agy-conformance-sentinel-intent/v1",
        "protocol_id": "spade-agy-1.1.23-structured-tool-denial-sentinel/v1",
        "experiment_id": "spade-agy-1-1-23-tool-denial-sentinel-v1",
        "output_root": str(root),
        "provider": "agy",
        "model": MODEL,
    }
    if any(intent.get(key) != expected for key, expected in expected_intent_fixed.items()):
        raise SentinelError("prior v1 intent identity or canonical location differs")
    intent_digest = _sha256(intent.get("intent_digest"), "prior intent digest")
    experiment_id = str(expected_intent_fixed["experiment_id"])
    run_dir = root / f"{experiment_id}-{intent_digest.removeprefix('sha256:')}"
    if run_dir.is_symlink() or not run_dir.is_dir() or run_dir != run_dir.resolve():
        raise SentinelError("prior v1 run directory is missing or noncanonical")
    ledger_root = root / LEDGER_ROOT_NAME
    if intent.get("shared_ledger_root") != str(ledger_root):
        raise SentinelError("prior v1 shared ledger location differs from its seal")
    _canonical_dir(ledger_root, "prior_shared_ledger_root", exists=True)
    root_inventory = {path.name for path in root.iterdir()}
    if root_inventory != {"intent.json", LEDGER_ROOT_NAME, run_dir.name}:
        raise SentinelError("prior v1 output inventory differs from terminal stranded form")
    run_inventory = {path.name for path in run_dir.iterdir()}
    if run_inventory != {".writer.lock", "intent.json", "run-manifest.json", "calls"}:
        raise SentinelError("prior v1 run inventory differs from terminal stranded form")
    lock_path = run_dir / ".writer.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise SentinelError("prior v1 writer lock is missing or unsafe")
    if _read_json(run_dir / "intent.json") != intent:
        raise SentinelError("prior v1 run intent differs from root intent")
    run_manifest = _read_json(run_dir / "run-manifest.json")
    if (
        run_manifest.get("schema_version") != "spade-agy-conformance-sentinel-run/v1"
        or run_manifest.get("protocol_id") != intent["protocol_id"]
        or run_manifest.get("experiment_id") != experiment_id
        or run_manifest.get("intent_digest") != intent_digest
        or run_manifest.get("global_ordinal") != 317
        or run_manifest.get("new_call_cap") != 1
        or run_manifest.get("shared_ledger_root") != str(ledger_root)
    ):
        raise SentinelError("prior v1 run manifest is not bound to its reservation")

    prior_runtime = intent.get("runtime_identity")
    if not isinstance(prior_runtime, dict):
        raise SentinelError("prior v1 runtime identity is missing")
    prior_agy_version = prior_runtime.get("agy_version")
    prior_agy_digest = _sha256(
        prior_runtime.get("agy_executable_digest"),
        "prior runtime AGY executable digest",
    )
    if prior_agy_version != "1.1.23":
        raise SentinelError("prior v1 runtime was not AGY 1.1.23")

    prior_coverage_anchor = intent.get("prior_usage_anchor")
    if not isinstance(prior_coverage_anchor, dict):
        raise SentinelError("prior v1 coverage anchor is missing")
    _validate_self_digest(prior_coverage_anchor, "anchor_digest", "prior v1 coverage anchor")
    if prior_coverage_anchor.get("prior_global_ordinal") != 316:
        raise SentinelError("prior v1 coverage anchor did not close at global 316")

    header = _read_json(ledger_root / "header.json")
    _validate_self_digest(header, "header_digest", "prior ledger header")
    expected_header = {
        "schema_version": "spade-agy-conformance-sentinel-ledger/v1",
        "protocol_id": intent["protocol_id"],
        "intent_digest": intent_digest,
        "prior_usage_anchor_digest": prior_coverage_anchor["anchor_digest"],
        "prior_ledger_header_digest": prior_coverage_anchor.get("prior_ledger_header_digest"),
        "prior_terminal_entry_digest": prior_coverage_anchor.get(
            "prior_terminal_entry_digest"
        ),
        "prior_charged_calls": 316,
        "new_call_cap": 1,
        "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
        "first_new_global_ordinal": 317,
        "last_permitted_global_ordinal": 317,
    }
    if {key: header.get(key) for key in expected_header} != expected_header or set(
        header
    ) != {*expected_header, "header_digest"}:
        raise SentinelError("prior v1 ledger header budget or identity differs")

    calls_root = run_dir / "calls"
    entries_root = ledger_root / "entries"
    if calls_root.is_symlink() or not calls_root.is_dir():
        raise SentinelError("prior v1 call root is missing")
    call_dirs = list(calls_root.iterdir())
    if len(call_dirs) != 1 or call_dirs[0].is_symlink() or not call_dirs[0].is_dir():
        raise SentinelError("prior v1 must contain exactly one real call directory")
    call_dir = call_dirs[0]
    request_path = call_dir / "request.json"
    expected_call_inventory = {"request.json", *AGY_EVIDENCE_FILENAMES.values()}
    if {path.name for path in call_dir.iterdir()} != expected_call_inventory:
        raise SentinelError(
            "prior v1 call must have request and four receipts but no result"
        )
    request = _read_json(request_path)
    _validate_self_digest(request, "request_digest", "prior v1 request")
    if (
        request.get("schema_version") != "spade-agy-conformance-sentinel-call-request/v1"
        or request.get("intent_digest") != intent_digest
        or request.get("call_id") != call_dir.name
        or request.get("local_ordinal") != 1
        or request.get("global_ordinal") != 317
        or request.get("model") != MODEL
        or request.get("reservation_status") != "reserved-before-spawn"
    ):
        raise SentinelError("prior v1 request differs from the charged reservation")
    _validate_timestamp(request.get("reserved_at_utc"), "prior v1 reserved_at_utc")

    if {path.name for path in ledger_root.iterdir()} != {"header.json", "entries"}:
        raise SentinelError("prior v1 ledger inventory differs")
    if entries_root.is_symlink() or not entries_root.is_dir():
        raise SentinelError("prior v1 ledger entries are missing")
    entry_paths = list(entries_root.iterdir())
    if len(entry_paths) != 1 or entry_paths[0].name != "global-0317.json":
        raise SentinelError("prior v1 ledger must contain only global 317")
    entry_path = entry_paths[0]
    if entry_path.is_symlink() or not entry_path.is_file():
        raise SentinelError("prior v1 global 317 entry is unsafe")
    terminal_entry = _read_json(entry_path)
    _validate_self_digest(terminal_entry, "entry_digest", "prior v1 ledger entry")
    expected_entry = {
        "schema_version": "spade-agy-conformance-sentinel-ledger-entry/v1",
        "header_digest": header["header_digest"],
        "intent_digest": intent_digest,
        "prior_usage_anchor_digest": prior_coverage_anchor["anchor_digest"],
        "global_ordinal": 317,
        "local_ordinal": 1,
        "call_id": request["call_id"],
        "request_digest": request["request_digest"],
        "request_path": request_path.relative_to(root).as_posix(),
        "model": MODEL,
        "reserved_at_utc": request["reserved_at_utc"],
    }
    if {key: terminal_entry.get(key) for key in expected_entry} != expected_entry or set(
        terminal_entry
    ) != {*expected_entry, "entry_digest"}:
        raise SentinelError("prior v1 ledger entry differs from its request reservation")

    receipt_paths: list[Path] = []
    expected_receipt_schemas = {
        "stdout_ndjson": "spade-agy-sanitized-stream-receipt/v1",
        "stderr": "spade-agy-sanitized-stderr-receipt/v1",
        "log": "spade-agy-sanitized-log-receipt/v1",
        "transcript": "spade-agy-sanitized-transcript-receipt/v1",
    }
    for label, filename in AGY_EVIDENCE_FILENAMES.items():
        receipt_path = call_dir / filename
        receipt = _read_compact_receipt_json(receipt_path)
        _validate_self_digest(receipt, "receipt_digest", f"prior v1 {label} receipt")
        if receipt.get("schema_version") != expected_receipt_schemas[label]:
            raise SentinelError(f"prior v1 {label} receipt schema differs")
        receipt_paths.append(receipt_path)

    closure_paths = [
        root / "intent.json",
        run_dir / "intent.json",
        run_dir / "run-manifest.json",
        ledger_root / "header.json",
        entry_path,
        request_path,
        *receipt_paths,
    ]
    receipt_manifest = [_manifest_entry(path, root) for path in sorted(receipt_paths)]
    closure_manifest = [_manifest_entry(path, root) for path in sorted(closure_paths)]
    body = {
        "schema_version": PRIOR_ANCHOR_SCHEMA,
        "prior_global_ordinal": 317,
        "charged_call_count": 1,
        "closure_status": "request-without-result-terminal-no-replay",
        "prior_intent_digest": intent_digest,
        "prior_protocol_id": intent["protocol_id"],
        "prior_request_digest": request["request_digest"],
        "prior_ledger_header_digest": header["header_digest"],
        "prior_terminal_entry_digest": terminal_entry["entry_digest"],
        "prior_coverage_anchor_digest": prior_coverage_anchor["anchor_digest"],
        "prior_runtime_agy_version": prior_agy_version,
        "prior_runtime_agy_executable_digest": prior_agy_digest,
        "receipt_leaf_count": len(receipt_manifest),
        "receipt_manifest_digest": _digest(receipt_manifest),
        "closure_leaf_count": len(closure_manifest),
        "closure_manifest_digest": _digest(closure_manifest),
        "result_present": False,
        "decision_present": False,
    }
    anchor = {**body, "anchor_digest": _digest(body)}
    paths = {
        "output_root": str(root),
        "run_dir": str(run_dir),
        "shared_ledger_root": str(ledger_root),
        "call_dir": str(call_dir),
    }
    return paths, anchor


def _validate_known_prior(anchor: Mapping[str, Any]) -> None:
    for key, expected in KNOWN_PRIOR_IDENTITY.items():
        if anchor.get(key) != expected:
            raise SentinelError(f"prior usage anchor differs from the authorized closure: {key}")


def _runtime_identity() -> dict[str, Any]:
    executable_text = shutil.which("agy")
    if executable_text is None:
        raise SentinelError("cannot resolve the agy executable")
    executable = Path(executable_text).resolve()
    python_executable = Path(sys.executable).resolve()
    if not executable.is_file() or not python_executable.is_file():
        raise SentinelError("AGY or Python executable is not a regular file")
    agy_digest = _bytes_digest(executable.read_bytes())
    if agy_digest != EXPECTED_AGY_EXECUTABLE_DIGEST:
        raise SentinelError(
            "resolved AGY bytes are not the independently calibrated 1.1.24 executable"
        )
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
        clean = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "diff", "--quiet", "HEAD", "--"],
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SentinelError("cannot resolve the sealed local runtime identity") from exc
    if clean.returncode != 0:
        raise SentinelError("tracked source tree must be clean before sealing or execution")
    runner_files = {
        "sentinel_runner": Path(__file__).resolve(),
        "structured_agy_adapter": Path(live.__file__).resolve(),
    }
    return {
        "source_repository": str(ROOT_DIR),
        "source_revision": revision,
        "tracked_tree_clean": True,
        "runner_files": {
            name: {"path": str(path), "digest": _bytes_digest(path.read_bytes())}
            for name, path in sorted(runner_files.items())
        },
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(python_executable),
        "python_executable_digest": _bytes_digest(python_executable.read_bytes()),
        "platform": platform.platform(),
        "agy_executable": str(executable),
        "agy_executable_digest": agy_digest,
        "agy_version": EXPECTED_AGY_VERSION,
    }


def _rehash_runtime_executables(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Measure the executable bytes now; never derive this receipt from the seal."""
    result: dict[str, str] = {}
    for name in ("python", "agy"):
        key = f"{name}_executable"
        path = Path(_required_text(runtime.get(key), f"runtime_identity.{key}"))
        if not path.is_absolute():
            raise SentinelError(f"runtime_identity.{key} must be absolute")
        _reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file() or path != path.resolve():
            raise SentinelError(f"runtime_identity.{key} is unsafe")
        result[f"{key}_digest"] = _bytes_digest(path.read_bytes())
    return result


def _validate_runtime_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SentinelError("runtime_identity must be an object")
    _require_keys(
        value,
        {
            "source_repository",
            "source_revision",
            "tracked_tree_clean",
            "runner_files",
            "python_implementation",
            "python_version",
            "python_executable",
            "python_executable_digest",
            "platform",
            "agy_executable",
            "agy_executable_digest",
            "agy_version",
        },
        "runtime_identity",
    )
    for key in (
        "source_repository",
        "source_revision",
        "python_implementation",
        "python_version",
        "python_executable",
        "platform",
        "agy_executable",
        "agy_version",
    ):
        _required_text(value[key], f"runtime_identity.{key}")
    if value["tracked_tree_clean"] is not True:
        raise SentinelError("runtime_identity must seal a clean tracked tree")
    if value["agy_version"] != EXPECTED_AGY_VERSION:
        raise SentinelError(f"sentinel requires AGY {EXPECTED_AGY_VERSION} exactly")
    for key in ("python_executable_digest", "agy_executable_digest"):
        _sha256(value[key], f"runtime_identity.{key}")
    files = value["runner_files"]
    if not isinstance(files, dict) or set(files) != {
        "sentinel_runner",
        "structured_agy_adapter",
    }:
        raise SentinelError("runtime_identity.runner_files differs")
    for name, record in files.items():
        if not isinstance(record, dict):
            raise SentinelError(f"runtime runner record is invalid: {name}")
        _require_keys(record, {"path", "digest"}, f"runtime runner {name}")
        path = Path(_required_text(record["path"], f"runtime runner {name}.path"))
        if not path.is_absolute():
            raise SentinelError(f"runtime runner {name} path must be absolute")
        _reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file() or path != path.resolve():
            raise SentinelError(f"runtime runner {name} path is unsafe")
        if _bytes_digest(path.read_bytes()) != _sha256(
            record["digest"], f"runtime runner {name}.digest"
        ):
            raise SentinelError(f"runtime runner {name} bytes drifted")
    if _rehash_runtime_executables(value) != {
        "python_executable_digest": value["python_executable_digest"],
        "agy_executable_digest": value["agy_executable_digest"],
    }:
        raise SentinelError("Python or AGY executable bytes drifted")
    return value


def build_intent(
    *,
    prior_output_root: Path | str,
    output_root: Path | str,
    shared_ledger_root: Path | str,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact response-independent one-call intent."""
    prior = _canonical_dir(prior_output_root, "prior_output_root", exists=True)
    output = _canonical_dir(output_root, "output_root", exists=False)
    ledger = _canonical_dir(shared_ledger_root, "shared_ledger_root", exists=False)
    expected_output = prior.parent / OUTPUT_ROOT_NAME
    expected_ledger = expected_output / LEDGER_ROOT_NAME
    if output != expected_output or ledger != expected_ledger:
        raise SentinelError(
            "sentinel output and ledger must use the single canonical sibling paths"
        )
    prior_paths, prior_anchor = _compute_prior_usage_anchor(prior)
    _validate_known_prior(prior_anchor)
    prior_root = Path(prior_paths["output_root"])
    if _is_within(output, prior_root) or _is_within(prior_root, output):
        raise SentinelError("sentinel and prior artifact roots must be disjoint")
    runtime = dict(runtime_identity or _runtime_identity())
    _validate_runtime_identity(runtime)
    purpose_digest = _digest(PURPOSE)
    call_id = f"sentinel-{purpose_digest.removeprefix('sha256:')[:24]}"
    body = {
        "schema_version": INTENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "analysis_role": "adapter-conformance-only",
        "claim_exclusions": [
            "no-model-quality-claim",
            "no-experiment-effect-or-selector-claim",
            "no-learner-or-release-claim",
            "no-backend-identity-attestation",
        ],
        "output_root": str(output),
        "shared_ledger_root": str(ledger),
        "prior_artifacts": prior_paths,
        "prior_usage_anchor": prior_anchor,
        "provider": "agy",
        "model": MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "sentinel_call": {
            "purpose": dict(PURPOSE),
            "purpose_digest": purpose_digest,
            "call_id": call_id,
            "prompt": PROMPT,
            "prompt_digest": _digest(PROMPT),
            "system": SYSTEM,
            "system_digest": _digest(SYSTEM),
            "expected_tool": EXPECTED_TOOL,
            "expected_log_tool": EXPECTED_LOG_TOOL,
            "expected_tool_parameters_receipt": dict(EXPECTED_TOOL_PARAMETERS_RECEIPT),
            "expected_tool_state_sequence": ["ACTIVE", "ERROR"],
            "expected_response_id_count": EXPECTED_RESPONSE_ID_COUNT,
            "expected_generic_disposition": EXPECTED_GENERIC_DISPOSITION,
            "expected_generic_error": EXPECTED_GENERIC_ERROR,
        },
        "configuration": {
            "llm_timeout_seconds": TIMEOUT_SECONDS,
            "result_wall_duration_tolerance_seconds": (RESULT_WALL_DURATION_TOLERANCE_SECONDS),
            "output_format": live.AGY_OUTPUT_FORMAT,
            "log_policy": live.AGY_LOG_POLICY,
            "evidence_policy": EVIDENCE_POLICY,
            "workdir_policy": WORKDIR_POLICY,
            "privacy_policy": PRIVACY_POLICY,
            "structured_process_environment": dict(AUTO_UPDATE_ENVIRONMENT),
            "retry_policy": "no-retry-under-any-disposition",
        },
        "budget": {
            "prior_charged_calls": PRIOR_CHARGED_CALLS,
            "new_call_cap": NEW_CALL_CAP,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "first_and_last_new_global_ordinal": GLOBAL_ORDINAL,
            "planned_max_global_calls": GLOBAL_ORDINAL,
            "headroom_calls": AUTHORIZED_GLOBAL_CALL_CAP - GLOBAL_ORDINAL,
        },
        "runtime_identity": runtime,
        "release_boundary": {
            "assay_decision": "not-run-not-applicable",
            "release_authorized": False,
            "model_lock_status": "absent",
            "future_paid_experiments_require_sentinel_pass": True,
        },
    }
    intent = {**body, "intent_digest": _digest(body)}
    validate_intent(intent)
    return intent


def validate_intent(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SentinelError("sentinel intent must be an object")
    _require_keys(
        value,
        {
            "schema_version",
            "protocol_id",
            "experiment_id",
            "analysis_role",
            "claim_exclusions",
            "output_root",
            "shared_ledger_root",
            "prior_artifacts",
            "prior_usage_anchor",
            "provider",
            "model",
            "backend_identity_attested",
            "route_authority",
            "sentinel_call",
            "configuration",
            "budget",
            "runtime_identity",
            "release_boundary",
            "intent_digest",
        },
        "sentinel intent",
    )
    fixed = {
        "schema_version": INTENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "analysis_role": "adapter-conformance-only",
        "provider": "agy",
        "model": MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "claim_exclusions": [
            "no-model-quality-claim",
            "no-experiment-effect-or-selector-claim",
            "no-learner-or-release-claim",
            "no-backend-identity-attestation",
        ],
        "configuration": {
            "llm_timeout_seconds": TIMEOUT_SECONDS,
            "result_wall_duration_tolerance_seconds": (RESULT_WALL_DURATION_TOLERANCE_SECONDS),
            "output_format": live.AGY_OUTPUT_FORMAT,
            "log_policy": live.AGY_LOG_POLICY,
            "evidence_policy": EVIDENCE_POLICY,
            "workdir_policy": WORKDIR_POLICY,
            "privacy_policy": PRIVACY_POLICY,
            "structured_process_environment": dict(AUTO_UPDATE_ENVIRONMENT),
            "retry_policy": "no-retry-under-any-disposition",
        },
        "budget": {
            "prior_charged_calls": PRIOR_CHARGED_CALLS,
            "new_call_cap": NEW_CALL_CAP,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "first_and_last_new_global_ordinal": GLOBAL_ORDINAL,
            "planned_max_global_calls": GLOBAL_ORDINAL,
            "headroom_calls": AUTHORIZED_GLOBAL_CALL_CAP - GLOBAL_ORDINAL,
        },
        "release_boundary": {
            "assay_decision": "not-run-not-applicable",
            "release_authorized": False,
            "model_lock_status": "absent",
            "future_paid_experiments_require_sentinel_pass": True,
        },
    }
    if any(value.get(key) != expected for key, expected in fixed.items()):
        raise SentinelError("sentinel intent fixed protocol fields differ")
    expected_call = {
        "purpose": dict(PURPOSE),
        "purpose_digest": _digest(PURPOSE),
        "call_id": f"sentinel-{_digest(PURPOSE).removeprefix('sha256:')[:24]}",
        "prompt": PROMPT,
        "prompt_digest": _digest(PROMPT),
        "system": SYSTEM,
        "system_digest": _digest(SYSTEM),
        "expected_tool": EXPECTED_TOOL,
        "expected_log_tool": EXPECTED_LOG_TOOL,
        "expected_tool_parameters_receipt": dict(EXPECTED_TOOL_PARAMETERS_RECEIPT),
        "expected_tool_state_sequence": ["ACTIVE", "ERROR"],
        "expected_response_id_count": EXPECTED_RESPONSE_ID_COUNT,
        "expected_generic_disposition": EXPECTED_GENERIC_DISPOSITION,
        "expected_generic_error": EXPECTED_GENERIC_ERROR,
    }
    if value.get("sentinel_call") != expected_call:
        raise SentinelError("sentinel call differs from the fixed tool-denial probe")
    _validate_runtime_identity(value["runtime_identity"])
    prior_paths = value.get("prior_artifacts")
    anchor = value.get("prior_usage_anchor")
    if not isinstance(prior_paths, dict) or set(prior_paths) != {
        "output_root",
        "run_dir",
        "shared_ledger_root",
        "call_dir",
    }:
        raise SentinelError("prior artifact path seal is invalid")
    if not isinstance(anchor, dict):
        raise SentinelError("prior usage anchor is invalid")
    recomputed_paths, recomputed_anchor = _compute_prior_usage_anchor(prior_paths["output_root"])
    if prior_paths != recomputed_paths or anchor != recomputed_anchor:
        raise SentinelError("prior artifact closure changed after sentinel sealing")
    _validate_known_prior(anchor)
    prior_root = Path(prior_paths["output_root"])
    output = _canonical_dir(str(value["output_root"]), "output_root", exists=False)
    ledger = _canonical_dir(str(value["shared_ledger_root"]), "shared_ledger_root", exists=False)
    if output != prior_root.parent / OUTPUT_ROOT_NAME or ledger != output / LEDGER_ROOT_NAME:
        raise SentinelError("sealed sentinel output or ledger path is not the singleton path")
    _validate_self_digest(value, "intent_digest", "sentinel intent")
    return value


def write_intent(path: Path | str, intent: Mapping[str, Any]) -> Path:
    checked = validate_intent(dict(intent))
    target = Path(path)
    expected = Path(str(checked["output_root"])) / "intent.json"
    if not target.is_absolute() or target != expected or target.resolve(strict=False) != expected:
        raise SentinelError(f"intent output must be exactly {expected}")
    _write_json(target, checked)
    return target


def load_intent(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    intent = validate_intent(_read_json(target))
    expected = Path(str(intent["output_root"])) / "intent.json"
    if not target.is_absolute() or target != expected or target.resolve(strict=True) != expected:
        raise SentinelError(f"intent must be loaded from exactly {expected}")
    return intent


def derive_run_dir(intent: Mapping[str, Any]) -> Path:
    digest = _sha256(intent.get("intent_digest"), "intent_digest").removeprefix("sha256:")
    return Path(str(intent["output_root"])) / f"{EXPERIMENT_ID}-{digest}"


def _default_dependencies(intent: Mapping[str, Any]) -> RunnerDependencies:
    runtime = _runtime_identity()
    if runtime != intent["runtime_identity"]:
        raise SentinelIncomplete("current runtime identity differs from the sealed intent")
    executable = Path(runtime["agy_executable"])
    return RunnerDependencies(live.call_llm_with_evidence, executable, runtime)


class _Engine:
    def __init__(
        self,
        intent: dict[str, Any],
        intent_bytes: bytes,
        run_dir: Path,
        dependencies: RunnerDependencies,
    ) -> None:
        self.intent = intent
        self.intent_bytes = intent_bytes
        self.run_dir = run_dir
        self.dependencies = dependencies
        self.ledger_root = Path(intent["shared_ledger_root"])

    @property
    def call_id(self) -> str:
        return str(self.intent["sentinel_call"]["call_id"])

    @property
    def call_dir(self) -> Path:
        return self.run_dir / "calls" / self.call_id

    @property
    def request_path(self) -> Path:
        return self.call_dir / "request.json"

    @property
    def result_path(self) -> Path:
        return self.call_dir / "result.json"

    @property
    def decision_path(self) -> Path:
        return self.run_dir / "decision.json"

    def _ledger_header(self) -> dict[str, Any]:
        anchor = self.intent["prior_usage_anchor"]
        body = {
            "schema_version": LEDGER_HEADER_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "prior_usage_anchor_digest": anchor["anchor_digest"],
            "prior_ledger_header_digest": anchor["prior_ledger_header_digest"],
            "prior_terminal_entry_digest": anchor["prior_terminal_entry_digest"],
            "prior_request_digest": anchor["prior_request_digest"],
            "prior_charged_calls": PRIOR_CHARGED_CALLS,
            "new_call_cap": NEW_CALL_CAP,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "first_new_global_ordinal": GLOBAL_ORDINAL,
            "last_permitted_global_ordinal": GLOBAL_ORDINAL,
        }
        return {**body, "header_digest": _digest(body)}

    def _run_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "experiment_id": EXPERIMENT_ID,
            "intent_digest": self.intent["intent_digest"],
            "provider": "agy",
            "model": MODEL,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "prior_usage_anchor_digest": self.intent["prior_usage_anchor"]["anchor_digest"],
            "new_call_cap": NEW_CALL_CAP,
            "global_ordinal": GLOBAL_ORDINAL,
            "shared_ledger_root": str(self.ledger_root),
            "runtime_identity": self.intent["runtime_identity"],
            "structured_process_environment": dict(AUTO_UPDATE_ENVIRONMENT),
            "assay_decision": "not-run-not-applicable",
            "release_authorized": False,
            "model_lock_status": "absent",
        }

    def _validate_runtime_now(self) -> None:
        current = dict(self.dependencies.runtime_identity)
        _validate_runtime_identity(current)
        if current != self.intent["runtime_identity"]:
            raise SentinelIncomplete("injected runtime differs from the sealed intent")

    def _verify_provider_executable(self) -> None:
        client = self.dependencies.client_or_bin
        if not isinstance(client, (str, os.PathLike)):
            return
        path = Path(client).resolve()
        runtime = self.intent["runtime_identity"]
        if (
            str(path) != runtime["agy_executable"]
            or not path.is_file()
            or _bytes_digest(path.read_bytes()) != runtime["agy_executable_digest"]
        ):
            raise SentinelIncomplete("AGY executable differs from the sealed route runtime")

    def _validate_tree(self) -> None:
        output_root = Path(self.intent["output_root"])
        if output_root.exists():
            if output_root.is_symlink() or not output_root.is_dir():
                raise SentinelError(f"unsafe sentinel output root: {output_root}")
            allowed_output = {"intent.json", self.ledger_root.name, self.run_dir.name}
            extras = {path.name for path in output_root.iterdir()} - allowed_output
            if extras:
                raise SentinelError(
                    f"sentinel output root contains extra artifacts: {sorted(extras)}"
                )
        for root in (self.run_dir, self.ledger_root):
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                raise SentinelError(f"unsafe sentinel artifact root: {root}")
            for current, directories, files in os.walk(root, topdown=True, followlinks=False):
                current_path = Path(current)
                for name in (*directories, *files):
                    if (current_path / name).is_symlink():
                        raise SentinelError(f"symlink in sentinel artifacts: {current_path / name}")

    def initialize(self) -> None:
        self._validate_runtime_now()
        self._verify_provider_executable()
        current_paths, current_anchor = _compute_prior_usage_anchor(
            self.intent["prior_artifacts"]["output_root"]
        )
        if (
            current_paths != self.intent["prior_artifacts"]
            or current_anchor != self.intent["prior_usage_anchor"]
        ):
            raise SentinelIncomplete("prior charged-call closure drifted")
        _mkdir_durable(self.run_dir)
        _mkdir_durable(self.ledger_root)
        self._validate_tree()
        _write_immutable_bytes(self.run_dir / "intent.json", self.intent_bytes)
        _write_json(self.run_dir / "run-manifest.json", self._run_manifest())
        _write_json(self.ledger_root / "header.json", self._ledger_header())
        self.validate_existing()

    def _request(self, reserved_at: str) -> dict[str, Any]:
        call = self.intent["sentinel_call"]
        body = {
            "schema_version": CALL_REQUEST_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "prior_usage_anchor_digest": self.intent["prior_usage_anchor"]["anchor_digest"],
            "call_id": self.call_id,
            "local_ordinal": 1,
            "global_ordinal": GLOBAL_ORDINAL,
            "purpose": call["purpose"],
            "purpose_digest": call["purpose_digest"],
            "provider": "agy",
            "model": MODEL,
            "backend_identity_attested": False,
            "route_authority": "requested-route-only",
            "runtime_identity_digest": _digest(self.intent["runtime_identity"]),
            "agy_executable_digest": self.intent["runtime_identity"]["agy_executable_digest"],
            "structured_process_environment": dict(AUTO_UPDATE_ENVIRONMENT),
            "prompt": call["prompt"],
            "prompt_digest": call["prompt_digest"],
            "system": call["system"],
            "system_digest": call["system_digest"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "workdir_policy": WORKDIR_POLICY,
            "output_format": live.AGY_OUTPUT_FORMAT,
            "log_policy": live.AGY_LOG_POLICY,
            "evidence_policy": EVIDENCE_POLICY,
            "privacy_policy": PRIVACY_POLICY,
            "reservation_status": "reserved-before-spawn",
            "reserved_at_utc": reserved_at,
        }
        return {**body, "request_digest": _digest(body)}

    def _validate_request(self, value: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._request(str(value.get("reserved_at_utc")))
        if dict(value) != expected:
            raise SentinelError("sentinel request differs from the sealed one-call reservation")
        _validate_timestamp(value["reserved_at_utc"], "reserved_at_utc")
        return dict(value)

    def _ledger_entry(self, request: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema_version": LEDGER_ENTRY_SCHEMA,
            "header_digest": self._ledger_header()["header_digest"],
            "intent_digest": self.intent["intent_digest"],
            "prior_usage_anchor_digest": self.intent["prior_usage_anchor"]["anchor_digest"],
            "global_ordinal": GLOBAL_ORDINAL,
            "local_ordinal": 1,
            "call_id": self.call_id,
            "request_digest": request["request_digest"],
            "request_path": self.request_path.relative_to(
                Path(self.intent["output_root"])
            ).as_posix(),
            "model": MODEL,
            "reserved_at_utc": request["reserved_at_utc"],
        }
        return {**body, "entry_digest": _digest(body)}

    def _validate_workdir_audit(
        self,
        value: object,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "policy",
            "observation_file",
            "cleanup_verified_at_utc",
            "deleted_after_call",
        }:
            raise SentinelError("sentinel workdir cleanup audit fields differ")
        reference = value.get("observation_file")
        if not isinstance(reference, dict) or set(reference) != {
            "path",
            "digest",
            "size_bytes",
        }:
            raise SentinelError("sentinel workdir observation reference differs")
        expected_path = f"calls/{self.call_id}/{WORKDIR_OBSERVATION_FILENAME}"
        path = self.run_dir / expected_path
        if reference.get("path") != expected_path:
            raise SentinelError("sentinel workdir observation path is noncanonical")
        content, failure = live._read_bounded_regular_file(
            path,
            limit=MAX_WORKDIR_OBSERVATION_BYTES,
            label="sentinel_workdir_observation",
        )
        size = reference.get("size_bytes")
        if (
            failure is not None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > MAX_WORKDIR_OBSERVATION_BYTES
            or reference.get("digest") != _bytes_digest(content)
            or size != len(content)
        ):
            raise SentinelError("sentinel workdir observation bytes differ")
        observation = _decode_json(content, "sentinel workdir observation")
        if content != _pretty_json(observation):
            raise SentinelError("sentinel workdir observation is not canonical JSON")
        _require_keys(
            observation,
            {
                "schema_version",
                "intent_digest",
                "request_digest",
                "policy",
                "canary_name_digest",
                "pre_call_entries",
                "post_call_entry_count",
                "post_call_inventory_digest",
                "canary_present_after_call",
                "observed_at_utc",
                "observation_digest",
            },
            "sentinel workdir observation",
        )
        _validate_self_digest(observation, "observation_digest", "workdir observation")
        observed = _validate_timestamp(observation["observed_at_utc"], "observed_at_utc")
        cleanup = _validate_timestamp(value["cleanup_verified_at_utc"], "cleanup_verified_at_utc")
        started = _validate_timestamp(result["started_at_utc"], "started_at_utc")
        finished = _validate_timestamp(result["finished_at_utc"], "finished_at_utc")
        if not started <= observed <= cleanup <= finished:
            raise SentinelError("workdir observation and cleanup ordering is impossible")
        fixed = {
            "schema_version": WORKDIR_OBSERVATION_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "request_digest": request["request_digest"],
            "policy": WORKDIR_POLICY,
            "canary_name_digest": _bytes_digest(CANARY.encode("utf-8")),
            "pre_call_entries": [],
            "post_call_entry_count": 0,
            "post_call_inventory_digest": _digest([]),
            "canary_present_after_call": False,
        }
        observation_exact = all(
            observation.get(key) == expected for key, expected in fixed.items()
        )
        return bool(
            observation_exact
            and value.get("policy") == WORKDIR_POLICY
            and value.get("deleted_after_call") is True
        )

    def _validate_receipts(
        self, result: Mapping[str, Any], request: Mapping[str, Any]
    ) -> tuple[live.AgyCallEvidence, dict[str, bool]]:
        references = result.get("evidence_files")
        if not isinstance(references, dict) or set(references) != set(AGY_EVIDENCE_FILENAMES):
            raise SentinelError("sentinel evidence file map differs")
        raw: dict[str, bytes] = {}
        for label, filename in AGY_EVIDENCE_FILENAMES.items():
            reference = references[label]
            if not isinstance(reference, dict) or set(reference) != {
                "path",
                "digest",
                "size_bytes",
            }:
                raise SentinelError("sentinel evidence reference fields differ")
            expected_path = f"calls/{self.call_id}/{filename}"
            if reference.get("path") != expected_path:
                raise SentinelError("sentinel evidence path is noncanonical")
            size = reference.get("size_bytes")
            path = self.run_dir / expected_path
            content, failure = live._read_bounded_regular_file(
                path, limit=AGY_EVIDENCE_LIMITS[label], label=f"sentinel_{label}"
            )
            if (
                failure is not None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or size > AGY_EVIDENCE_LIMITS[label]
                or len(content) != size
                or reference.get("digest") != _bytes_digest(content)
            ):
                raise SentinelError("sentinel evidence bytes differ from their receipt")
            raw[label] = content
        summary = result.get("agy_evidence")
        if not isinstance(summary, dict):
            raise SentinelError("sentinel evidence summary is missing")
        workdir_text = summary.get("invocation_workdir")
        if not isinstance(workdir_text, str):
            raise SentinelError("sentinel evidence has no invocation workdir")
        workdir = Path(workdir_text)
        if (
            not workdir.is_absolute()
            or workdir.resolve(strict=False) != workdir
            or not workdir.name.startswith(f"spade-agy-sentinel-{self.call_id}-")
        ):
            raise SentinelError("sentinel invocation workdir is noncanonical")
        protected_roots = [
            ROOT_DIR,
            Path(self.intent["output_root"]),
            Path(self.intent["prior_artifacts"]["output_root"]),
        ]
        if any(_is_within(workdir, root) for root in protected_roots):
            raise SentinelError("sentinel workdir overlaps a protected artifact/source root")
        capture_failures = summary.get("capture_failures")
        if not isinstance(capture_failures, list) or not all(
            isinstance(item, str) and item for item in capture_failures
        ):
            raise SentinelError("sentinel capture-failure summary is invalid")
        if type(summary.get("timed_out")) is not bool:
            raise SentinelError("sentinel timeout summary is invalid")
        recomputed = live.analyze_agy_evidence(
            requested_model=MODEL,
            full_prompt=PROMPT,
            invocation_workdir=workdir,
            exit_status=result.get("exit_status"),
            timed_out=summary["timed_out"],
            capture_failures=capture_failures,
            stdout_ndjson=raw["stdout_ndjson"],
            stderr=raw["stderr"],
            log=raw["log"],
            transcript=raw["transcript"],
            sanitized_stream_receipt=True,
            sanitized_stderr_receipt=True,
            sanitized_log_receipt=True,
            sanitized_transcript_receipt=True,
        )
        if summary != recomputed.summary():
            raise SentinelError("sentinel evidence summary is not independently derivable")
        if (
            result.get("provider_disposition") != recomputed.disposition
            or result.get("response") != recomputed.response
            or result.get("error") != recomputed.error
            or result.get("exit_status") != recomputed.exit_status
        ):
            raise SentinelError("sentinel result contradicts replayed evidence")

        stream = _decode_json(raw["stdout_ndjson"], "sanitized stream receipt")
        stderr = _decode_json(raw["stderr"], "sanitized stderr receipt")
        log = _decode_json(raw["log"], "sanitized log receipt")
        transcript = _decode_json(raw["transcript"], "sanitized transcript receipt")
        events_value = stream.get("events")
        events: list[Any] = []
        if isinstance(events_value, list):
            events = events_value
        kinds = [event.get("event") if isinstance(event, dict) else None for event in events]
        step_payloads = [
            event.get("step_update")
            for event in events
            if isinstance(event, dict) and event.get("event") == "step_update"
        ]
        well_typed_steps = all(isinstance(step, dict) for step in step_payloads)
        steps = [step for step in step_payloads if isinstance(step, dict)]
        exact_ordered_steps = [
            (step.get("step_index"), step.get("step_type"), step.get("state"))
            for step in steps
        ] == [
            (0, "user_input", "DONE"),
            (1, "agent_response", "DONE"),
            (2, "tool", "ACTIVE"),
            (2, "tool", "ERROR"),
        ]
        all_step_text_deltas_blank = len(steps) == 4 and all(
            step.get("text_delta_receipt") == EXPECTED_NULL_VALUE_RECEIPT
            for step in steps
        )
        non_tool_steps = [step for step in steps if step.get("step_type") != "tool"]
        tool_steps = [step for step in steps if step.get("step_type") == "tool"]
        exact_event_sequence = kinds == [
            "init",
            "step_update",
            "step_update",
            "step_update",
            "step_update",
            "result",
        ]
        exact_non_tool_prefix = (
            well_typed_steps
            and len(non_tool_steps) == 2
            and [
                (step.get("step_index"), step.get("step_type"), step.get("state"))
                for step in non_tool_steps
            ]
            == [(0, "user_input", "DONE"), (1, "agent_response", "DONE")]
            and all(
                step.get("conversation_id") == summary.get("conversation_id")
                and step.get("tool_name") is None
                and step.get("tool_info_name") is None
                and step.get("tool_info_present") is False
                and step.get("tool_output_present") is False
                and step.get("tool_error_present") is False
                and step.get("subagent_info_present") is False
                for step in non_tool_steps
            )
        )
        exact_tool_transition = len(tool_steps) == 2 and [
            step.get("state") for step in tool_steps
        ] == ["ACTIVE", "ERROR"] and all(
            step.get("step_index") == 2
            and step.get("conversation_id") == summary.get("conversation_id")
            and step.get("tool_name") == EXPECTED_TOOL
            and step.get("tool_info_name") == EXPECTED_TOOL
            and step.get("tool_info_present") is True
            and step.get("tool_output_present") is False
            and step.get("subagent_info_present") is False
            for step in tool_steps
        )
        exact_denial_error_transition = len(tool_steps) == 2 and [
            step.get("tool_error_present") for step in tool_steps
        ] == [False, True]
        exact_tool_parameters = len(tool_steps) == 2 and all(
            step.get("tool_parameters_receipt") == EXPECTED_TOOL_PARAMETERS_RECEIPT
            for step in tool_steps
        )
        terminal = events[-1] if events and isinstance(events[-1], dict) else {}
        terminal_payload = terminal.get("result")
        direct_terminal_blank = (
            isinstance(terminal_payload, dict)
            and terminal_payload.get("status") == "SUCCESS"
            and terminal_payload.get("num_turns") == 1
            and terminal_payload.get("response") == ""
            and transcript.get("model_content_digests") == []
        )
        init_payload = events[0].get("init") if events and isinstance(events[0], dict) else None
        init_exact = (
            isinstance(init_payload, dict)
            and init_payload.get("model") == MODEL
            and init_payload.get("permission_mode") == "request-review"
            and EXPECTED_TOOL in init_payload.get("tools", [])
        )
        conversation_id = summary.get("conversation_id")
        try:
            canonical_uuid = str(uuid.UUID(str(conversation_id))) == conversation_id
        except (ValueError, AttributeError):
            canonical_uuid = False
        response_ids = log.get("response_ids")
        response_ids_exact = (
            isinstance(response_ids, list)
            and len(response_ids) == EXPECTED_RESPONSE_ID_COUNT
            and len(set(response_ids)) == EXPECTED_RESPONSE_ID_COUNT
            and all(
                isinstance(item, str)
                and item
                and len(item.encode("utf-8")) <= 256
                and all(0x21 <= ord(character) <= 0x7E for character in item)
                for item in response_ids
            )
            and tuple(response_ids) == recomputed.response_ids
        )
        log_denial = (
            log.get("soft_denied_tools") == [EXPECTED_LOG_TOOL]
            and log.get("denied_tool_confirmation_ids") == [conversation_id]
            and log.get("approved_tool_confirmation_ids") == []
        )
        denial_specific_no_execution = (
            transcript.get("tool_execution_observed") is False
            and transcript.get("tool_call_names") == [EXPECTED_TOOL]
            and log.get("approved_tool_confirmation_ids") == []
            and exact_tool_transition
            and exact_denial_error_transition
            and log_denial
            and all(step.get("tool_output_present") is False for step in steps)
            and all(step.get("subagent_info_present") is False for step in steps)
        )
        no_explicit_pre_response_failure_markers = (
            stderr.get("explicit_pre_response_failure_marker") is False
            and log.get("explicit_pre_response_failure_marker") is False
            and isinstance(terminal_payload, dict)
            and terminal_payload.get("explicit_pre_response_failure_marker") is False
        )
        canary_bytes = CANARY.encode("utf-8")
        audit = result.get("workdir_audit")
        workdir_ok = self._validate_workdir_audit(audit, request, result) and not workdir.exists()
        fresh_executable_digests = _rehash_runtime_executables(self.intent["runtime_identity"])
        sealed_executable_digests = {
            "python_executable_digest": self.intent["runtime_identity"]["python_executable_digest"],
            "agy_executable_digest": self.intent["runtime_identity"]["agy_executable_digest"],
        }
        post_call_executable_digests = result.get("post_call_executable_digests")
        runtime_unchanged = (
            isinstance(post_call_executable_digests, dict)
            and set(post_call_executable_digests) == set(sealed_executable_digests)
            and post_call_executable_digests
            == fresh_executable_digests
            == sealed_executable_digests
        )
        facts = {
            "structured_replay_exact": True,
            # The generic adapter remains fail-closed on any tool_error.  V2
            # accepts only this fully calibrated denial shape; it does not
            # weaken the generic classification.
            "generic_tool_error_classification_unchanged": (
                recomputed.disposition == EXPECTED_GENERIC_DISPOSITION
                and recomputed.response is None
                and recomputed.error == EXPECTED_GENERIC_ERROR
            ),
            "requested_and_reported_route_exact": (
                recomputed.requested_model == MODEL and recomputed.reported_model == MODEL
            ),
            "process_completed_without_capture_failure": (
                recomputed.exit_status == 0
                and recomputed.timed_out is False
                and recomputed.capture_failures == ()
            ),
            "two_response_ids_and_conversation_bound": (
                response_ids_exact
                and canonical_uuid
                and recomputed.stream_event_count == 6
                and isinstance(recomputed.terminal_event_digest, str)
                and _SHA256.fullmatch(recomputed.terminal_event_digest) is not None
            ),
            "calibrated_stream_shape_exact": (
                exact_event_sequence
                and exact_ordered_steps
                and exact_non_tool_prefix
                and init_exact
                and stream.get("shape_error") is None
            ),
            "all_step_text_deltas_blank": all_step_text_deltas_blank,
            "log_route_conversation_and_workdir_exact": (
                log.get("reported_models") == [MODEL]
                and log.get("conversation_ids") == [conversation_id]
                and log.get("workspace_dirs") == [str(workdir)]
            ),
            "transcript_prompt_and_shape_exact": (
                transcript.get("shape_error") is None
                and transcript.get("explicit_user_inputs") == 1
                and transcript.get("prompt_matches") == 1
                and transcript.get("prompt_digest")
                == _bytes_digest(PROMPT.encode("utf-8"))
            ),
            "single_runcommand_active_error_transition": (
                recomputed.tool_call_names == (EXPECTED_TOOL,)
                and exact_tool_transition
                and exact_denial_error_transition
            ),
            "runcommand_parameters_exact": exact_tool_parameters,
            "terminal_response_and_transcript_content_blank": direct_terminal_blank,
            "soft_denial_bound_to_conversation": log_denial,
            "denial_specific_no_tool_execution_evidence": denial_specific_no_execution,
            "no_explicit_pre_response_failure_markers": (
                no_explicit_pre_response_failure_markers
            ),
            "isolated_policy_config_absent": recomputed.policy_config_identity
            == {
                "relative_path": "config/config.json",
                "exists": False,
                "digest": None,
                "size_bytes": 0,
            },
            "sanitized_receipts_are_canary_free": all(
                canary_bytes not in content for content in raw.values()
            ),
            "workdir_nonexistence_observed_before_cleanup_and_deleted": workdir_ok,
            "updater_environment_sealed_at_structured_boundary": (
                request.get("structured_process_environment") == AUTO_UPDATE_ENVIRONMENT
                and result.get("structured_process_environment")
                == AUTO_UPDATE_ENVIRONMENT
                and self.intent["configuration"].get("structured_process_environment")
                == AUTO_UPDATE_ENVIRONMENT
            ),
            "python_and_agy_executables_unchanged": runtime_unchanged,
            "prior_usage_anchor_unchanged": result.get("post_call_prior_usage_anchor_digest")
            == self.intent["prior_usage_anchor"]["anchor_digest"],
        }
        return recomputed, facts

    def _validate_result(
        self, value: Mapping[str, Any], request: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        _require_keys(
            value,
            {
                "schema_version",
                "intent_digest",
                "call_id",
                "local_ordinal",
                "global_ordinal",
                "request_digest",
                "structured_process_environment",
                "provider_disposition",
                "response",
                "error",
                "exit_status",
                "agy_evidence",
                "evidence_files",
                "workdir_audit",
                "post_call_executable_digests",
                "post_call_prior_usage_anchor_digest",
                "started_at_utc",
                "finished_at_utc",
                "duration_seconds",
                "result_digest",
            },
            "sentinel result",
        )
        _validate_self_digest(value, "result_digest", "sentinel result")
        fixed = {
            "schema_version": CALL_RESULT_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "call_id": self.call_id,
            "local_ordinal": 1,
            "global_ordinal": GLOBAL_ORDINAL,
            "request_digest": request["request_digest"],
            "structured_process_environment": dict(AUTO_UPDATE_ENVIRONMENT),
        }
        if any(value.get(key) != expected for key, expected in fixed.items()):
            raise SentinelError("sentinel result is not bound to its reservation")
        reserved = _validate_timestamp(request["reserved_at_utc"], "reserved_at_utc")
        started = _validate_timestamp(value["started_at_utc"], "started_at_utc")
        finished = _validate_timestamp(value["finished_at_utc"], "finished_at_utc")
        duration = value.get("duration_seconds")
        wall_duration = (finished - started).total_seconds()
        if (
            not reserved <= started <= finished
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
            or abs(wall_duration - float(duration)) > RESULT_WALL_DURATION_TOLERANCE_SECONDS
        ):
            raise SentinelError("sentinel result timing is invalid")
        _, facts = self._validate_receipts(value, request)
        return dict(value), facts

    def _decision_body(
        self,
        result: Mapping[str, Any],
        facts: Mapping[str, bool],
        *,
        decided_at_utc: str,
    ) -> dict[str, Any]:
        all_pass = all(facts.values())
        disposition = result["provider_disposition"]
        if all_pass:
            classification = "pass"
        elif disposition == "response":
            classification = "target_not_exercised"
        elif disposition == "pre_response_provider_failure":
            classification = "inconclusive_provider_unavailable"
        else:
            classification = "failed"
        return {
            "schema_version": DECISION_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "intent_digest": self.intent["intent_digest"],
            "request_digest": result["request_digest"],
            "result_digest": result["result_digest"],
            "classification": classification,
            "pass": all_pass,
            "pass_criteria": dict(facts),
            "future_paid_google_experiments_authorized": all_pass,
            "charged_call_count": 1,
            "global_charged_calls": GLOBAL_ORDINAL,
            "authorized_global_call_cap": AUTHORIZED_GLOBAL_CALL_CAP,
            "remaining_authorized_calls": AUTHORIZED_GLOBAL_CALL_CAP - GLOBAL_ORDINAL,
            "retry_authorized": False,
            "assay_decision": "not-run-not-applicable",
            "release_authorized": False,
            "model_lock_status": "absent",
            "decided_at_utc": decided_at_utc,
        }

    def _validate_decision(
        self, value: Mapping[str, Any], result: Mapping[str, Any], facts: Mapping[str, bool]
    ) -> dict[str, Any]:
        _validate_self_digest(value, "decision_digest", "sentinel decision")
        decided = str(value.get("decided_at_utc"))
        decided_time = _validate_timestamp(decided, "decided_at_utc")
        finished_time = _validate_timestamp(result.get("finished_at_utc"), "finished_at_utc")
        if decided_time < finished_time:
            raise SentinelError("sentinel decision predates the provider result")
        expected = self._decision_body(result, facts, decided_at_utc=decided)
        if dict(value) != {**expected, "decision_digest": _digest(expected)}:
            raise SentinelError("sentinel decision differs from deterministic evidence replay")
        return dict(value)

    def _validate_static_artifacts(self) -> None:
        self._validate_tree()
        if not self.run_dir.is_dir():
            return
        if _read_json(self.run_dir / "intent.json") != self.intent:
            raise SentinelError("run-local intent differs from the root seal")
        if _read_json(self.run_dir / "run-manifest.json") != self._run_manifest():
            raise SentinelError("sentinel run manifest differs")
        if _read_json(self.ledger_root / "header.json") != self._ledger_header():
            raise SentinelError("sentinel ledger header differs")

    def validate_existing(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        self._validate_tree()
        if not self.run_dir.exists():
            if self.ledger_root.exists():
                raise SentinelError("sentinel ledger exists without its run")
            return None, None
        self._validate_static_artifacts()
        allowed_run = {".writer.lock", "intent.json", "run-manifest.json", "calls", "decision.json"}
        if {path.name for path in self.run_dir.iterdir()} - allowed_run:
            raise SentinelError("sentinel run contains extra artifacts")
        calls_root = self.run_dir / "calls"
        if calls_root.exists() and (calls_root.is_symlink() or not calls_root.is_dir()):
            raise SentinelError("sentinel calls root is unsafe")
        call_dirs = list(calls_root.iterdir()) if calls_root.is_dir() else []
        if any(path.name != self.call_id or not path.is_dir() for path in call_dirs):
            raise SentinelError("sentinel contains an unsealed call directory")
        entry_root = self.ledger_root / "entries"
        ledger_inventory = {path.name for path in self.ledger_root.iterdir()}
        if ledger_inventory - {"header.json", "entries"}:
            raise SentinelError("sentinel ledger contains extra artifacts")
        if entry_root.exists() and (entry_root.is_symlink() or not entry_root.is_dir()):
            raise SentinelError("sentinel ledger entries root is unsafe")
        entry_paths = list(entry_root.iterdir()) if entry_root.is_dir() else []
        if any(
            path.name != f"global-{GLOBAL_ORDINAL:04d}.json"
            or path.is_symlink()
            or not path.is_file()
            for path in entry_paths
        ):
            raise SentinelError("sentinel ledger contains an unsealed call ordinal")
        if self.result_path.is_file() and not self.request_path.is_file():
            raise SentinelError("sentinel result exists without its request")
        if self.decision_path.is_file() and not self.result_path.is_file():
            raise SentinelError("sentinel decision exists without a closed result")
        if not self.request_path.is_file():
            if entry_paths:
                raise SentinelError("sentinel ledger entry exists without its request")
            if calls_root.exists():
                raise SentinelError(
                    "request-less sentinel state must not contain a calls root"
                )
            return None, None
        request = self._validate_request(_read_json(self.request_path))
        ledger_path = entry_root / f"global-{GLOBAL_ORDINAL:04d}.json"
        if _read_json(ledger_path) != self._ledger_entry(request):
            raise SentinelError("sentinel ledger entry differs from its reservation")
        if not self.result_path.is_file():
            raise SentinelIncomplete(
                f"sentinel call {self.call_id} is reserved at global {GLOBAL_ORDINAL} "
                "without a durable disposition; it will not replay"
            )
        allowed_call = {
            "request.json",
            "result.json",
            WORKDIR_OBSERVATION_FILENAME,
            *AGY_EVIDENCE_FILENAMES.values(),
        }
        if {path.name for path in self.call_dir.iterdir()} != allowed_call:
            raise SentinelError("closed sentinel call inventory differs")
        result, facts = self._validate_result(_read_json(self.result_path), request)
        if not self.decision_path.is_file():
            return result, None
        decision = self._validate_decision(_read_json(self.decision_path), result, facts)
        return result, decision

    async def execute_call(self) -> dict[str, Any]:
        if self.request_path.exists():
            raise SentinelIncomplete(
                "the one sentinel reservation already exists and cannot replay"
            )
        self._validate_runtime_now()
        self._verify_provider_executable()
        reserved_at = _utc_now()
        request = self._request(reserved_at)
        _write_json(self.request_path, request)
        ledger_path = self.ledger_root / "entries" / f"global-{GLOBAL_ORDINAL:04d}.json"
        _write_json(ledger_path, self._ledger_entry(request))
        if _read_json(self.request_path) != request or _read_json(
            ledger_path
        ) != self._ledger_entry(request):
            raise SentinelIncomplete("sentinel reservation was not durably published")
        started_at = _utc_now()
        started_clock = time.monotonic()
        workdir_text: str | None = None
        post_entries: list[str] = []
        canary_present = False
        observation_reference: dict[str, Any] | None = None
        cleanup_verified_at: str | None = None
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"spade-agy-sentinel-{self.call_id}-"
            ) as workdir_name:
                workdir = Path(workdir_name).resolve(strict=True)
                workdir_text = str(workdir)
                protected = [
                    ROOT_DIR,
                    Path(self.intent["output_root"]),
                    Path(self.intent["prior_artifacts"]["output_root"]),
                ]
                if any(_is_within(workdir, root) for root in protected):
                    raise SentinelIncomplete("temporary sentinel workdir overlaps a protected root")
                pre_entries = sorted(path.name for path in workdir.iterdir())
                if pre_entries:
                    raise SentinelIncomplete(
                        "temporary sentinel workdir was not empty before spawn"
                    )
                evidence = await self.dependencies.structured_llm_call(
                    self.dependencies.client_or_bin,
                    MODEL,
                    PROMPT,
                    system=SYSTEM,
                    provider="agy",
                    workdir=workdir,
                    timeout_seconds=TIMEOUT_SECONDS,
                    evidence_log_path=workdir / "agy-cli.log",
                    process_environment=dict(AUTO_UPDATE_ENVIRONMENT),
                )
                if not isinstance(evidence, live.AgyCallEvidence):
                    raise SentinelIncomplete("structured AGY boundary returned no valid evidence")
                post_entries = sorted(path.name for path in workdir.iterdir())
                canary_present = (workdir / CANARY).exists() or (workdir / CANARY).is_symlink()
                observed_at = _utc_now()
                observation_body = {
                    "schema_version": WORKDIR_OBSERVATION_SCHEMA,
                    "intent_digest": self.intent["intent_digest"],
                    "request_digest": request["request_digest"],
                    "policy": WORKDIR_POLICY,
                    "canary_name_digest": _bytes_digest(CANARY.encode("utf-8")),
                    "pre_call_entries": [],
                    "post_call_entry_count": len(post_entries),
                    "post_call_inventory_digest": _digest(post_entries),
                    "canary_present_after_call": canary_present,
                    "observed_at_utc": observed_at,
                }
                observation = {
                    **observation_body,
                    "observation_digest": _digest(observation_body),
                }
                observation_path = self.call_dir / WORKDIR_OBSERVATION_FILENAME
                _write_json(observation_path, observation)
                if _read_json(observation_path) != observation:
                    raise SentinelIncomplete(
                        "workdir observation was not durable before temporary cleanup"
                    )
                observation_bytes = observation_path.read_bytes()
                observation_reference = {
                    "path": observation_path.relative_to(self.run_dir).as_posix(),
                    "digest": _bytes_digest(observation_bytes),
                    "size_bytes": len(observation_bytes),
                }
                post_persistence_entries = sorted(path.name for path in workdir.iterdir())
                post_persistence_canary = (workdir / CANARY).exists() or (
                    workdir / CANARY
                ).is_symlink()
                if (
                    post_persistence_entries != post_entries
                    or post_persistence_canary != canary_present
                ):
                    raise SentinelIncomplete(
                        "temporary workdir changed after its durable observation"
                    )
            workdir_deleted = workdir_text is not None and not Path(workdir_text).exists()
            cleanup_verified_at = _utc_now()
        except SentinelIncomplete:
            raise
        except Exception as exc:
            raise SentinelIncomplete(
                f"sentinel request is reserved but its disposition is not durable: {exc}"
            ) from exc
        if (
            workdir_text is None
            or not workdir_deleted
            or observation_reference is None
            or cleanup_verified_at is None
        ):
            raise SentinelIncomplete("sentinel workdir cleanup could not be proven")

        evidence_payloads = {
            "stdout_ndjson": evidence.stdout_ndjson,
            "stderr": evidence.stderr,
            "log": evidence.log,
            "transcript": evidence.transcript,
        }
        if any(
            not isinstance(content, bytes) or len(content) > AGY_EVIDENCE_LIMITS[label]
            for label, content in evidence_payloads.items()
        ):
            raise SentinelIncomplete("structured AGY boundary returned invalid receipt bytes")
        references: dict[str, dict[str, Any]] = {}
        for label, filename in AGY_EVIDENCE_FILENAMES.items():
            content = evidence_payloads[label]
            path = self.call_dir / filename
            _write_immutable_bytes(path, content)
            references[label] = {
                "path": path.relative_to(self.run_dir).as_posix(),
                "digest": _bytes_digest(content),
                "size_bytes": len(content),
            }
        try:
            self._validate_runtime_now()
        except SentinelError as exc:
            raise SentinelIncomplete(
                f"sentinel request is reserved and post-call runtime validation failed: {exc}"
            ) from exc
        post_call_executable_digests = _rehash_runtime_executables(self.intent["runtime_identity"])
        current_paths, current_anchor = _compute_prior_usage_anchor(
            self.intent["prior_artifacts"]["output_root"]
        )
        if (
            current_paths != self.intent["prior_artifacts"]
            or current_anchor != self.intent["prior_usage_anchor"]
        ):
            raise SentinelIncomplete("prior usage closure drifted during the sentinel")
        result_body = {
            "schema_version": CALL_RESULT_SCHEMA,
            "intent_digest": self.intent["intent_digest"],
            "call_id": self.call_id,
            "local_ordinal": 1,
            "global_ordinal": GLOBAL_ORDINAL,
            "request_digest": request["request_digest"],
            "structured_process_environment": dict(AUTO_UPDATE_ENVIRONMENT),
            "provider_disposition": evidence.disposition,
            "response": evidence.response,
            "error": evidence.error,
            "exit_status": evidence.exit_status,
            "agy_evidence": evidence.summary(),
            "evidence_files": references,
            "workdir_audit": {
                "policy": WORKDIR_POLICY,
                "observation_file": observation_reference,
                "cleanup_verified_at_utc": cleanup_verified_at,
                "deleted_after_call": workdir_deleted,
            },
            "post_call_executable_digests": post_call_executable_digests,
            "post_call_prior_usage_anchor_digest": current_anchor["anchor_digest"],
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "duration_seconds": time.monotonic() - started_clock,
        }
        result = {**result_body, "result_digest": _digest(result_body)}
        self._validate_result(result, request)
        _write_json(self.result_path, result)
        return result

    def ensure_decision(self, result: Mapping[str, Any]) -> dict[str, Any]:
        request = self._validate_request(_read_json(self.request_path))
        checked_result, facts = self._validate_result(result, request)
        if self.decision_path.is_file():
            return self._validate_decision(_read_json(self.decision_path), checked_result, facts)
        body = self._decision_body(checked_result, facts, decided_at_utc=_utc_now())
        decision = {**body, "decision_digest": _digest(body)}
        self._validate_decision(decision, checked_result, facts)
        _write_json(self.decision_path, decision)
        return decision


def _dry_validate_existing(intent: dict[str, Any], dependencies: RunnerDependencies) -> RunResult:
    run_dir = derive_run_dir(intent)
    engine = _Engine(intent, _pretty_json(intent), run_dir, dependencies)
    engine._validate_runtime_now()
    engine._verify_provider_executable()
    result, decision = engine.validate_existing()
    if decision is not None:
        status = str(decision["classification"])
    elif result is not None:
        status = "closed-decision-pending"
    else:
        status = "validated"
    return RunResult(
        status=status,
        intent_digest=intent["intent_digest"],
        run_dir=run_dir,
        charged_call_count=1 if result is not None else 0,
        provider_calls_started=0,
        decision_path=engine.decision_path if decision is not None else None,
    )


async def run_sentinel(
    intent_path: Path | str,
    *,
    execute: bool = False,
    acknowledged_new_call_cap: int | None = None,
    dependencies: RunnerDependencies | None = None,
) -> RunResult:
    """Dry-validate by default, or execute/resume the one sealed request."""
    intent = load_intent(intent_path)
    resolved = dependencies or _default_dependencies(intent)
    if not execute:
        return _dry_validate_existing(intent, resolved)
    if acknowledged_new_call_cap != NEW_CALL_CAP:
        raise SentinelError("--acknowledge-new-call-cap must exactly equal 1")
    run_dir = derive_run_dir(intent)
    engine = _Engine(intent, _pretty_json(intent), run_dir, resolved)
    provider_calls_started = 0
    with _single_writer(run_dir):
        engine.initialize()
        result, decision = engine.validate_existing()
        if result is None:
            provider_calls_started = 1
            result = await engine.execute_call()
        if decision is None:
            decision = engine.ensure_decision(result)
        final_result, final_decision = engine.validate_existing()
        if final_result is None or final_decision != decision:
            raise SentinelError("sentinel terminal closure did not revalidate")
    return RunResult(
        status=str(decision["classification"]),
        intent_digest=intent["intent_digest"],
        run_dir=run_dir,
        charged_call_count=1,
        provider_calls_started=provider_calls_started,
        decision_path=engine.decision_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("sentinel-plan", help="seal the exact one-call intent")
    plan.add_argument("--prior-output-root", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--shared-ledger-root", required=True)
    plan.add_argument("--output", required=True)
    run = commands.add_parser("run", help="dry-validate or explicitly execute/resume")
    run.add_argument("--intent", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--acknowledge-new-call-cap", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "sentinel-plan":
            intent = build_intent(
                prior_output_root=Path(args.prior_output_root),
                output_root=Path(args.output_root),
                shared_ledger_root=Path(args.shared_ledger_root),
            )
            write_intent(Path(args.output), intent)
            print(intent["intent_digest"])
            return 0
        result = asyncio.run(
            run_sentinel(
                args.intent,
                execute=args.execute,
                acknowledged_new_call_cap=args.acknowledge_new_call_cap,
            )
        )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "intent_digest": result.intent_digest,
                    "run_dir": str(result.run_dir),
                    "charged_call_count": result.charged_call_count,
                    "provider_calls_started": result.provider_calls_started,
                    "decision_path": str(result.decision_path) if result.decision_path else None,
                },
                sort_keys=True,
            )
        )
        if args.execute and result.status != "pass":
            return 3
        return 0
    except SentinelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
