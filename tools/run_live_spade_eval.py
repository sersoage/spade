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
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


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

DESIGNER_PROMPT = """You are an expert Environment Designer. Create a clean,
self-contained Python environment for a cognitive reasoning puzzle in the skill
area: {skill}.

Security and interface requirements:
1. Import only modules explicitly allowed by ProofPack: math, random, re,
   heapq, collections, itertools, typing, dataclasses, json, functools, string,
   and copy. Never access files, the network, processes, environment variables,
   frames, globals, or Python object internals.
2. The class name must end in Env.
3. Implement __init__(self, max_turns=10, **kwargs).
4. Implement reset(self, seed=None) -> (observation: str, info: dict).
5. Implement solution() returning one answer or a list of turn-by-turn actions.
6. Implement step(action) -> (observation, reward, terminated, truncated, info).
7. Every seed must be deterministic. A correct completed episode returns 1.0;
   incorrect actions return 0.0 and must not terminate immediately.
8. Tell the player to respond with \\boxed{{action}}.

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


def _agy_environment() -> dict[str, str]:
    """Pass only startup/config variables to agy, not unrelated API credentials."""
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
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
    prefixes = ("AGY_", "ANTIGRAVITY_")
    return {
        key: value
        for key, value in os.environ.items()
        if key in exact or key.startswith(prefixes)
    }


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
        cmd = [
            str(client_or_bin),
            "-p",
            full_prompt,
            "--disable-slash-commands",
            "--sandbox",
            "--print-timeout",
            f"{max(1, int(timeout_seconds))}s",
        ]
        if model != "agy-subscription":
            cmd.extend(["--model", model])
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workdir) if workdir else None,
                env=_agy_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            raise LiveEvalError(f"agy could not start: {exc}", 4) from exc
        try:
            stdout, stderr = await _communicate_agy_bounded(
                proc,
                timeout_seconds=timeout_seconds + 5.0,
            )
        except asyncio.TimeoutError as exc:
            raise LiveEvalError(f"agy call timed out after {timeout_seconds:.0f}s", 4) from exc
        if proc.returncode != 0:
            error_text = stderr.decode(errors="replace").strip()
            raise LiveEvalError(
                f"agy failed with exit {proc.returncode}: {error_text[-1000:]}",
                4,
            )
        response = stdout.decode(errors="replace").strip()
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
            raise LiveEvalError(f"{provider} call timed out after {timeout_seconds:.0f}s", 4) from exc
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
                f"\n\nAction at turn {turn}: {action}\n"
                f"Environment response: {next_observation}"
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
            f"   Hint accepted ({len(hint)} characters; "
            "explicit-answer leakage heuristic passed)"
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
