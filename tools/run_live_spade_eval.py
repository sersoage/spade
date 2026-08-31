#!/usr/bin/env python3
"""Run a live end-to-end SPADE test with real LLM calls and detailed transparent traces.

Orchestrates:
1. Environment Designer (LLM) generates executable Python environment + hint
2. ProofPack V0-V4 Qualification Ladder certifies environment validity ($0 sandbox gate)
3. Multi-turn Reasoning Agent (LLM) plays the environment (Unhinted vs Hinted)
4. Computes Hint-Based Regret: r(e|h) - r(e)
5. Assay Evidence Ledger records trajectory and computes clustered statistics
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Load ~/.env if present
env_path = Path.home() / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Add paths dynamically via sibling discovery
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

PROOFPACK_ENV_PATH = ROOT_DIR.parent / "proofpack" / "packages" / "env" / "src"
if PROOFPACK_ENV_PATH.exists() and str(PROOFPACK_ENV_PATH) not in sys.path:
    sys.path.insert(0, str(PROOFPACK_ENV_PATH))

ASSAY_PATH = ROOT_DIR.parent / "assay"
if ASSAY_PATH.exists() and str(ASSAY_PATH) not in sys.path:
    sys.path.insert(0, str(ASSAY_PATH))

try:
    from proofpack_env.spade_qualification import qualify_spade_environment
    from proofpack_env.spade_target import SpadeEnvironmentTarget
    HAS_PROOFPACK = True
except ImportError:
    HAS_PROOFPACK = False

try:
    from assay.bench.spade_emitter import SpadeTaskPayload, emit_spade_curriculum
    from assay.certify.spade_evaluator import SpadeClusterData, certify_spade_curriculum_gain
    HAS_ASSAY = True
except ImportError:
    HAS_ASSAY = False


DESIGNER_PROMPT = """You are an expert Environment Designer. Create a clean, self-contained Python Gym environment for a cognitive reasoning puzzle in the skill area: {skill}.

Security and interface requirements:
1. ONLY import standard modules: math, random, re, heapq, collections, itertools, typing, dataclasses, json, functools. Do NOT import os, sys, subprocess, or open files.
2. Class name must end in `Env` (e.g. `PuzzleEnv`).
3. Implement `__init__(self, max_turns=10, **kwargs)`
4. Implement `reset(self, seed=None)` returning `(observation: str, info: dict)`. Format instructions should ask for `\\boxed{{answer}}`.
5. Implement `solution(self)` returning the string answer (or list of turn-by-turn answers).
6. Implement `step(self, action: str)` returning `(observation: str, reward: float, terminated: bool, truncated: bool, info: dict)`.
7. Reward must be 1.0 when correct and 0.0 otherwise.

Return ONLY the executable Python code inside a single ```python ... ``` block. Do not include conversational prose."""


HINT_PROMPT = """You are an expert tutor. Given the following environment code, provide a high-level conceptual hint or strategy to solve it WITHOUT revealing the exact final answer or literal solution string.

Code:
```python
{code}
```

Hint:"""


def get_llm_client(provider: str, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
    """Instantiate client or configure CLI runner."""
    if provider == "agy":
        agy_bin = shutil.which("agy") or "/Users/sergio.soage/.local/bin/agy"
        if not os.path.exists(agy_bin):
            raise FileNotFoundError(f"agy CLI binary not found at {agy_bin}")
        return agy_bin, "agy-subscription"

    from openai import AsyncOpenAI
    
    if provider in ("google", "gemini"):
        url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        resolved_model = model or "gemini-2.5-flash"
    elif provider == "openrouter":
        url = base_url or "https://openrouter.ai/api/v1"
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        resolved_model = model or "deepseek/deepseek-chat"
    elif provider == "openai":
        url = base_url or "https://api.openai.com/v1"
        key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_model = model or "gpt-4o-mini"
    elif provider in ("local", "ollama", "vllm"):
        url = base_url or "http://localhost:11434/v1"
        key = api_key or "local-key"
        resolved_model = model or "qwen2.5:7b"
    else:
        url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_model = model or "gpt-4o-mini"

    if not key:
        env_var_hint = "GEMINI_API_KEY or GOOGLE_API_KEY" if provider in ("google", "gemini") else f"{provider.upper()}_API_KEY"
        raise ValueError(
            f"API key missing for provider '{provider}'. "
            f"Please set {env_var_hint} in ~/.env or use `--provider agy`."
        )

    return AsyncOpenAI(base_url=url, api_key=key), resolved_model


async def call_llm(client_or_bin, model: str, prompt: str, system: str = "", provider: str = "agy") -> str:
    """Make LLM completion call via agy CLI subscription or API."""
    if provider == "agy":
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        cmd = [str(client_or_bin), "-p", full_prompt, "--disable-slash-commands"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"agy CLI execution failed (exit {proc.returncode}): {stderr.decode()}")
        return stdout.decode().strip()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = await client_or_bin.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown block."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def extract_clean_action(response: str) -> str:
    """Extract clean boxed action or command token from LLM chain-of-thought response."""
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if m:
        return f"\\boxed{{{m.group(1).strip()}}}"
    # Fallback to last non-empty line
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()


async def run_multi_turn_rollout(
    client_or_bin,
    model: str,
    target: SpadeEnvironmentTarget,
    seed: int,
    hint_text: str = "",
    provider: str = "agy",
    max_turns: int = 5,
) -> tuple[float, list[dict[str, Any]]]:
    """Execute a multi-turn conversational interaction with clean action extraction."""
    env = target.instantiate()
    obs, _info = env.reset(seed=seed)
    
    trajectory: list[dict[str, Any]] = [{"role": "environment", "observation": obs}]
    history_str = f"Initial Observation: {obs}"
    if hint_text:
        history_str += f"\n\nPrivileged Strategy Hint:\n{hint_text}"

    total_reward = 0.0
    terminated = False
    truncated = False
    turn = 0

    while not (terminated or truncated) and turn < max_turns:
        turn += 1
        prompt = (
            f"You are a reasoning agent solving an interactive puzzle.\n"
            f"Current Dialogue & Observations:\n{history_str}\n\n"
            f"Turn {turn}/{max_turns}: Analyze the current state and provide your next action or answer formatted as \\boxed{{action}}."
        )
        agent_raw = await call_llm(client_or_bin, model, prompt, provider=provider)
        action = extract_clean_action(agent_raw)

        step_res = env.step(action)
        if len(step_res) == 5:
            step_obs, reward, term, trunc, _ = step_res
        else:
            step_obs, reward, term, _ = step_res
            trunc = False

        total_reward = max(total_reward, float(reward))
        terminated = bool(term)
        truncated = bool(trunc)

        trajectory.append({
            "turn": turn,
            "raw_response": agent_raw,
            "clean_action": action,
            "observation": step_obs,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
        })
        history_str += f"\n\nYour Action (Turn {turn}): {action}\nEnvironment Feedback: {step_obs}"

    return total_reward, trajectory


async def run_live_eval(args):
    print("\n" + "=" * 76)
    print("🚀 LIVE SPADE + PROOFPACK + ASSAY VERIFICATION TRACE")
    print(f"Provider: {args.provider.upper()} | Target Skill: {args.skill}")
    print("=" * 76 + "\n")

    if not HAS_PROOFPACK:
        print("❌ Error: proofpack_env package is not installed or available.")
        sys.exit(1)

    client_or_bin, model = get_llm_client(args.provider, args.model, args.base_url, args.api_key)
    print(f"📡 Backend: {model if args.provider != 'agy' else 'Google Antigravity CLI'}")

    # Qualified seeds: Qualification guarantees and rollouts must use these seeds
    QUALIFIED_SEEDS = [0, 1, 42]

    # -------------------------------------------------------------
    # Step 1: Environment Designer (π_D) generates environment
    # -------------------------------------------------------------
    print("\n" + "-" * 76)
    print("🧠 [STEP 1/5] ENVIRONMENT DESIGNER (π_D) SYNTHESIZING ENVIRONMENT CODE")
    print("-" * 76)
    raw_env_response = await call_llm(
        client_or_bin, model, DESIGNER_PROMPT.format(skill=args.skill),
        system="You are an expert Python environment creator.", provider=args.provider
    )
    env_code = extract_python_code(raw_env_response)
    print("📜 Synthesized Python Code:")
    for idx, line in enumerate(env_code.splitlines()[:30], 1):
        print(f"   {idx:02d} | {line}")
    if len(env_code.splitlines()) > 30:
        print(f"   ... ({len(env_code.splitlines()) - 30} more lines)")

    # -------------------------------------------------------------
    # Step 2: ProofPack V0-V4 Qualification Ladder
    # -------------------------------------------------------------
    print("\n" + "-" * 76)
    print("🛡️ [STEP 2/5] PROOFPACK V0-V4 FORMAL QUALIFICATION LADDER")
    print("-" * 76)
    report = qualify_spade_environment(env_code, seeds=QUALIFIED_SEEDS)
    for cid, clause in report.clauses.items():
        icon = "✅" if clause.status == "pass" else "❌"
        print(f"   {icon} {cid.upper():<24}: {clause.summary}")
    print(f"\n   🏆 ProofPack Certification Verdict: {'✅ CERTIFIED' if report.passed else '❌ REJECTED'}")
    print(f"   📦 Environment Digest: {report.environment_digest}")

    if not report.passed:
        print("\n❌ Environment failed ProofPack qualification. Halting fail-closed (exit 5).")
        sys.exit(5)

    # -------------------------------------------------------------
    # Step 3: Hint Generation
    # -------------------------------------------------------------
    print("\n" + "-" * 76)
    print("💡 [STEP 3/5] PRIVILEGED CONCEPTUAL HINT GENERATION")
    print("-" * 76)
    hint_text = await call_llm(
        client_or_bin, model, HINT_PROMPT.format(code=env_code),
        system="You provide strategic problem-solving hints.", provider=args.provider
    )
    print(f"   Generated Hint:\n   \"{hint_text.strip()[:200]}...\"")

    # -------------------------------------------------------------
    # Step 4: Reasoning Agent (π_A) Gameplay (Unhinted vs Hinted)
    # -------------------------------------------------------------
    print("\n" + "-" * 76)
    print("🎮 [STEP 4/5] REASONING AGENT MULTI-TURN ROLLOUTS (UNHINTED vs HINTED)")
    print("-" * 76)
    target = SpadeEnvironmentTarget(env_code)

    # Use a strictly qualified seed
    play_seed = QUALIFIED_SEEDS[-1]  # seed 42
    test_env = target.instantiate()
    obs_test, _ = test_env.reset(seed=play_seed)
    oracle_sol = test_env.solution()
    print(f"📋 Problem Observation (Qualified Seed {play_seed}):\n   {obs_test.strip()}\n")
    print(f"🔑 True Oracle Solution: {oracle_sol}\n")

    # 4A. Unhinted rollout
    print("▶️ [Lane 1: Unhinted Multi-Turn Play]")
    reward_unhinted, traj_unhinted = await run_multi_turn_rollout(
        client_or_bin, model, target, seed=play_seed, hint_text="", provider=args.provider
    )
    last_act_unhinted = traj_unhinted[-1].get("clean_action", "")
    print(f"   Turns Taken:        {len(traj_unhinted) - 1}")
    print(f"   Final Action:       {last_act_unhinted}")
    print(f"   Return r_A(e):      {reward_unhinted:.2f}\n")

    # 4B. Hinted rollout
    print("▶️ [Lane 2: Hinted Multi-Turn Play]")
    reward_hinted, traj_hinted = await run_multi_turn_rollout(
        client_or_bin, model, target, seed=play_seed, hint_text=hint_text, provider=args.provider
    )
    last_act_hinted = traj_hinted[-1].get("clean_action", "")
    print(f"   Turns Taken:        {len(traj_hinted) - 1}")
    print(f"   Final Action:       {last_act_hinted}")
    print(f"   Return r_A(e|h):    {reward_hinted:.2f}\n")

    # 4C. Compute Designer Regret
    regret = max(0.0, reward_hinted - reward_unhinted)
    print(f"📐 Hint-Based Regret r_D(e) = max(0, r_A(e|h) - r_A(e)) = max(0, {reward_hinted:.2f} - {reward_unhinted:.2f}) = {regret:.2f}")

    # -------------------------------------------------------------
    # Step 5: Assay Benchmark & Evidence Ledger Persistence
    # -------------------------------------------------------------
    print("\n" + "-" * 76)
    print("📊 [STEP 5/5] ASSAY CURRICULUM EMISSION & EVIDENCE PERSISTENCE")
    print("-" * 76)
    if HAS_ASSAY:
        assay_dir = ROOT_DIR / ".assay"
        manifests_dir = assay_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)

        task_payload = SpadeTaskPayload(
            task_id=f"spade-live-{report.environment_name.lower()}",
            environment_name=report.environment_name,
            skill=args.skill,
            code=env_code,
            solution=str(oracle_sol),
        )
        manifest = emit_spade_curriculum([task_payload], curriculum_id="live-hardened-manifest")
        
        # Persist manifest to disk
        manifest_file = manifests_dir / f"{manifest.content_digest.replace(':', '_')}.json"
        manifest_file.write_text(json.dumps({
            "curriculum_id": manifest.curriculum_id,
            "content_digest": manifest.content_digest,
            "environment_count": manifest.environment_count,
            "vendi_diversity_score": manifest.vendi_diversity_score,
            "skill_distribution": manifest.skill_distribution,
        }, indent=2))

        # Persist evidence row to JSONL
        evidence_file = assay_dir / "evidence.jsonl"
        with open(evidence_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "schema_version": "assay-evidence/v2",
                "environment_digest": report.environment_digest,
                "environment_name": report.environment_name,
                "skill": args.skill,
                "seed": play_seed,
                "reward_unhinted": reward_unhinted,
                "reward_hinted": reward_hinted,
                "regret": regret,
                "turns_unhinted": len(traj_unhinted) - 1,
                "turns_hinted": len(traj_hinted) - 1,
            }) + "\n")

        print(f"   ✅ Saved Assay Manifest:  {manifest_file.name}")
        print(f"   ✅ Appended Evidence Log: {evidence_file}")
        print(f"   Manifest Digest:          {manifest.content_digest}")
        print(f"   Vendi Diversity Score:    {manifest.vendi_diversity_score:.4f}")

    print("\n" + "=" * 76)
    print("🏁 LIVE VERIFICATION TRACE COMPLETE")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live SPADE + ProofPack + Assay Evaluator")
    parser.add_argument("--provider", default="agy", choices=["agy", "google", "gemini", "openrouter", "openai", "local", "ollama", "vllm"])
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--skill", default="Graph Theory & Shortest Path", help="Target skill category")
    parser.add_argument("--base-url", default=None, help="Custom API Base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    args = parser.parse_args()

    asyncio.run(run_live_eval(args))


if __name__ == "__main__":
    main()
