"""Sealed Slime launch plans and post-run validation for the learner assay.

This module is deliberately offline: it describes remote Slime/SGLang jobs and
validates their receipts, but never starts a job or contacts a model endpoint.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from spade.core.learner_branch_pools import (
    SEALED_LEARNER_POOL_MANIFEST_DIGEST,
    PoolBundle,
    load_learner_pool_manifest,
)
from spade.slime.static_pool import SCHEDULE_SCHEMA, select_static_games


SLIME_PLAN_SCHEMA = "spade-slime-learner-branch-assay-plan/v1"
SLIME_LAUNCH_SCHEMA = "spade-slime-learner-branch-launch/v1"
SLIME_RESULTS_SCHEMA = "spade-slime-learner-branch-results/v1"
SLIME_VALIDATION_SCHEMA = "spade-slime-learner-branch-validation/v1"
SLIME_BACKEND = "slime-megatron-sglang"
SLIME_EXECUTION_STATE = "plan-only-hold"
SLIME_EVIDENCE_STATE = "artifact-backed-collector-not-implemented"
SLIME_HOLD_DECISION = "hold-no-trusted-artifact-collector"
SLIME_CLAIM_SCOPE = "slime-paper-backend-causal-learner-assay-plan-only"
SLIME_SUBMODULE_REVISION = "bf14dc21f9500746447f2572d0692e981c4d2a7e"
DEFAULT_PAIRED_SEEDS = (0, 1, 2, 3, 4, 5)
DEFAULT_HELDOUT_EPISODE_SEEDS = tuple(range(16))

_ARMS = ("coverage_forced", "redundant_historical")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_PATH_RE = re.compile(r"/[A-Za-z0-9._/+@=-]+(?:/[A-Za-z0-9._+@=-]+)*")
_SOURCE_FILES = (
    "cmd/games/_train_fixed_gpt55.sh",
    "cmd/models/qwen3-8B.sh",
    "train_spade_slime.py",
    "spade/slime/arguments.py",
    "spade/slime/branch_assay.py",
    "spade/slime/spade_rollout.py",
    "spade/slime/static_pool.py",
    "spade/slime/token_accounting.py",
)
_TOKEN_POPULATIONS = ("real", "padded", "total")
_TOKEN_ROLES = ("all", "actor", "environment", "unknown")
_TOKEN_MEASURES = (
    "samples",
    "prompt_tokens",
    "response_tokens",
    "loss_mask_tokens",
    "sequence_tokens",
)
_COMPUTE_MEASURES = ("sequence_tokens", "loss_mask_tokens")
_SOURCE_TOPOLOGY_KEYS = (
    "rollout/topology/actor_instances_requested",
    "rollout/topology/actor_instances_succeeded",
    "rollout/topology/actor_instances_failed",
    "rollout/topology/actor_trajectories_filtered",
    "rollout/topology/environment_trajectories_filtered",
)


class SlimeAssayError(ValueError):
    """A Slime assay plan or result violates its sealed contract."""


@dataclass(frozen=True)
class SlimeAssayPlan:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    launch_specs: tuple[Mapping[str, Any], ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SlimeAssayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SlimeAssayError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _read_json(path: Path, where: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SlimeAssayError(f"{where} must be a regular non-symlinked file: {path}")
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlimeAssayError(f"cannot read {where}: {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise SlimeAssayError(f"{where} must contain a JSON object")
    return result


def _sealed(value: Mapping[str, Any], field: str, where: str) -> str:
    observed = value.get(field)
    if not isinstance(observed, str) or _DIGEST_RE.fullmatch(observed) is None:
        raise SlimeAssayError(f"{where}.{field} is not a lowercase sha256 digest")
    body = {key: item for key, item in value.items() if key != field}
    if observed != _digest(body):
        raise SlimeAssayError(f"{where}.{field} does not bind its artifact")
    return observed


def _sha256(value: object, where: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SlimeAssayError(f"{where} must be a lowercase sha256 digest")
    return value


def _nonempty(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise SlimeAssayError(f"{where} must be a non-empty string without NUL")
    return value


def _absolute_path(value: object, where: str) -> str:
    text = _nonempty(value, where)
    if _SAFE_PATH_RE.fullmatch(text) is None or ".." in Path(text).parts:
        raise SlimeAssayError(f"{where} must be a normalized absolute POSIX path")
    return text.rstrip("/") or "/"


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SlimeAssayError(f"{where} must be a positive integer")
    return value


def _exact_keys(value: object, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SlimeAssayError(f"{where} keys differ: {observed}")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write((_canonical_json(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _source_inventory(source_root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for relative in _SOURCE_FILES:
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise SlimeAssayError(f"required Slime assay source is unavailable: {path}")
        raw = path.read_bytes()
        inventory.append({"path": relative, "size_bytes": len(raw), "sha256": _bytes_digest(raw)})
    return inventory


def _optimizer_contract() -> dict[str, object]:
    return {
        "initial_state": "fresh-zero-state-from-identical-model-weights-and-run-seed",
        "optimizer": "adam",
        "learning_rate": 1e-6,
        "learning_rate_schedule": "constant",
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.98,
        "advantage_estimator": "grpo",
        "grpo_std_normalization": False,
        "rewards_normalization": False,
        "kl_loss": True,
        "kl_loss_coefficient": 0.005,
        "kl_loss_type": "low_var_kl",
        "entropy_coefficient": 0.0,
        "epsilon_clip": 0.2,
        "epsilon_clip_high": 0.28,
        "tis": True,
    }


def _rollout_topology_contract() -> dict[str, object]:
    return {
        "real_episode_groups": 192,
        "padded_episode_groups": 0,
        "actor_instances_requested": 192,
        "actor_instances_succeeded": 192,
        "actor_instances_failed": 0,
        "actor_trajectories_filtered": 0,
        "environment_trajectories_filtered": 0,
        "failure_upsampling_permitted": False,
    }


def _effective_arguments(
    *,
    hf_checkpoint: str,
    reference_checkpoint: str,
    output_dir: str,
    pool_dir: str,
    seed: int,
    schedule_id: str,
    num_rollouts: int,
) -> list[str]:
    """Return the exact argv tokens passed to ``train_spade_slime``."""

    return [
        "python3",
        "-m",
        "train_spade_slime",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "8",
        "--colocate",
        # cmd/models/qwen3-8B.sh
        "--swiglu",
        "--num-layers",
        "36",
        "--hidden-size",
        "4096",
        "--ffn-hidden-size",
        "12288",
        "--num-attention-heads",
        "32",
        "--group-query-attention",
        "--num-query-groups",
        "8",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--rotary-base",
        "1000000",
        "--vocab-size",
        "151936",
        "--kv-channels",
        "128",
        "--qk-layernorm",
        "--untie-embeddings-and-output-weights",
        # Checkpoint arguments.
        "--hf-checkpoint",
        hf_checkpoint,
        "--ref-load",
        reference_checkpoint,
        "--save",
        f"{output_dir}/Qwen3-8B_fixed_gpt55/",
        "--save-interval",
        str(num_rollouts),
        # Rollout arguments.
        "--data-source-path",
        "spade.slime.data_source.SpadeDataSource",
        "--rollout-function-path",
        "spade.slime.spade_rollout.spade_generate_rollout",
        "--num-rollout",
        str(num_rollouts),
        "--rollout-batch-size",
        "12",
        "--n-samples-per-prompt",
        "1",
        "--rollout-max-response-len",
        "8192",
        "--rollout-temperature",
        "1.0",
        "--seed",
        str(seed),
        "--global-batch-size",
        "192",
        "--use-dynamic-global-batch-size",
        "--balance-data",
        "--apply-chat-template-kwargs",
        '{"enable_thinking":false}',
        # SPADE arguments.
        "--spade-gamma1",
        "0.98",
        "--spade-gamma2",
        "0.85",
        "--spade-actor-temperature",
        "0.6",
        "--spade-actor-max-tokens",
        "8192",
        "--spade-max-context-length",
        "32768",
        "--spade-max-turns",
        "25",
        "--spade-gamma",
        "0.99",
        "--spade-game-regeneration-interval",
        "0",
        "--spade-skills",
        "Mathematical_Reasoning",
        "Logical_Deduction",
        "Spatial_Reasoning",
        "Pattern_Recognition",
        "Optimization",
        "Causal_Inference",
        "--spade-skills-per-regen",
        "0",
        "--spade-num-games-per-rollout",
        "12",
        "--spade-games-dir",
        pool_dir,
        "--spade-game-difficulty",
        "medium",
        "--spade-trajectories-per-game",
        "16",
        "--spade-cache-dir",
        f"{output_dir}/spade_games_cache",
        "--spade-env-generation-template",
        "qwen3_multiturn_game_generation",
        "--spade-actor-template",
        "qwen3_game",
        "--spade-reward-normalization",
        "grpo",
        "--spade-no-train-on-env-trajectories",
        "--spade-proposer-training-delay",
        "0",
        "--spade-static-game-pool",
        "--spade-no-replacement",
        "--spade-fixed-pool-seed",
        str(seed),
        "--spade-static-pool-schedule-id",
        schedule_id,
        "--spade-require-complete-rollout",
        "--spade-actor-enable-thinking",
        # Eval is configured but cannot fire inside this short assay.
        "--eval-interval",
        "100000",
        "--eval-config",
        f"{output_dir}/eval_aime.yaml",
        "--skip-eval-before-train",
        "--apply-chat-template",
        "--rm-type",
        "math",
        # Optimizer and GRPO.
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--advantage-estimator",
        "grpo",
        "--disable-grpo-std-normalization",
        "--disable-rewards-normalization",
        "--use-kl-loss",
        "--kl-loss-coef",
        "0.005",
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        "0.00",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--use-tis",
        # Qwen3-8B performance, SGLang, and numerical arguments.
        "--tensor-model-parallel-size",
        "2",
        "--sequence-parallel",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "2",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "8192",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-mem-fraction-static",
        "0.7",
        "--sglang-tool-call-parser",
        "qwen",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
    ]


def _rollout_schedule(pool: Mapping[str, Any], seed: int, schedule_id: str, count: int) -> list:
    names = [Path(str(entry["basename"])) for entry in pool["entries"]]
    return [
        {
            "rollout_id": rollout_id,
            "slot_basenames": [
                path.name
                for path in select_static_games(
                    names,
                    12,
                    seed=seed,
                    schedule_id=schedule_id,
                    rollout_id=rollout_id,
                )
            ],
        }
        for rollout_id in range(count)
    ]


def _heldout_sampling_seed(schedule_id: str, basename: str, replicate_id: int) -> int:
    material = (
        f"spade-slime-heldout-sampling-v1\0{schedule_id}\0{basename}\0{replicate_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _heldout_games(bundle: PoolBundle) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for entry in bundle.manifest["pools"]["heldout_v4"]["entries"]:
        basename = str(entry["basename"])
        replicates = []
        for replicate_id, environment_seed in enumerate(DEFAULT_HELDOUT_EPISODE_SEEDS):
            replicates.append(
                {
                    "replicate_id": replicate_id,
                    "environment_seed": environment_seed,
                    "sampling_seed": _heldout_sampling_seed(
                        bundle.schedule_id, basename, replicate_id
                    ),
                }
            )
        games.append({"basename": basename, "replicates": replicates})
    return games


def _launch_spec(
    *,
    bundle: PoolBundle,
    arm: str,
    pair_index: int,
    order_position: int,
    seed: int,
    num_rollouts: int,
    remote_pool_root: str,
    remote_output_root: str,
    hf_checkpoint: str,
    hf_checkpoint_digest: str,
    reference_checkpoint: str,
    reference_checkpoint_digest: str,
    runtime: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = f"pair-{pair_index:02d}-seed-{seed:010d}-{arm.replace('_', '-')}"
    output_dir = f"{remote_output_root}/{run_id}"
    pool_dir = f"{remote_pool_root}/{arm}"
    schedule_id = bundle.schedule_id
    environment = {
        "FIXED_MODEL_SIZE": "8b",
        "STATIC_POOL_DIR": pool_dir,
        "MIN_POOL_GAMES": "12",
        "NUM_ROLLOUT": str(num_rollouts),
        "ROLLOUT_BATCH_SIZE": "12",
        "GLOBAL_BATCH_SIZE": "192",
        "MAX_CONTEXT_LENGTH": "32768",
        "MAX_TURNS": "25",
        "SPADE_NUM_GAMES_PER_ROLLOUT": "12",
        "SPADE_TRAJECTORIES_PER_GAME": "16",
        "SPADE_FIXED_POOL_SEED": str(seed),
        "SPADE_STATIC_POOL_SCHEDULE_ID": schedule_id,
        "SPADE_REQUIRE_COMPLETE_ROLLOUT": "1",
        "TRAIN_SEED": str(seed),
        "KL_COEF": "0.005",
        "ACTOR_MAX_TOKENS": "8192",
        "MODEL_ARGS_ROTARY_BASE": "1000000",
        "EVAL_INTERVAL": "100000",
        "MASTER_ADDR": "127.0.0.1",
        "WORKSPACE_DIR": "/mnt/spade-workspace",
        "SPADE_PLAY_CANARY": "1",
        "HF_CHECKPOINT": hf_checkpoint,
        "REF_CHECKPOINT": reference_checkpoint,
        "OUTPUT_DIR": output_dir,
        "SAVE_INTERVAL": str(num_rollouts),
        "SKIP_RUNTIME_INSTALL": "1",
        "DISABLE_WANDB": "1",
    }
    body: dict[str, Any] = {
        "schema_version": SLIME_LAUNCH_SCHEMA,
        "backend": SLIME_BACKEND,
        "run_id": run_id,
        "pair_id": f"pair-{pair_index:02d}",
        "pair_index": pair_index,
        "order_position": order_position,
        "arm": arm,
        "seed": seed,
        "pool_manifest_digest": bundle.manifest["manifest_digest"],
        "pool_name": arm,
        "initial_state": {
            "model": {"uri": hf_checkpoint, "artifact_digest": hf_checkpoint_digest},
            "reference_model": {
                "uri": reference_checkpoint,
                "artifact_digest": reference_checkpoint_digest,
            },
            "load_checkpoint": None,
            "optimizer": _optimizer_contract(),
        },
        "source": source,
        "runtime": runtime,
        "launcher": {
            "working_directory": "/workspace/spade",
            "argv": ["cmd/games/_train_fixed_gpt55.sh"],
            "environment": environment,
            "required_secret_environment": [],
            "forbidden_environment": [
                "LOAD_DIR",
                "PREPARE_ONLY",
                "WANDB_API_KEY",
                "WANDB_ENTITY",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "AGY_API_KEY",
            ],
            "effective_training_arguments": _effective_arguments(
                hf_checkpoint=hf_checkpoint,
                reference_checkpoint=reference_checkpoint,
                output_dir=output_dir,
                pool_dir=pool_dir,
                seed=seed,
                schedule_id=schedule_id,
                num_rollouts=num_rollouts,
            ),
        },
        "training": {
            "num_rollouts": num_rollouts,
            "games_per_rollout": 12,
            "trajectories_per_game": 16,
            "global_batch_size": 192,
            "expected_trajectories": num_rollouts * 192,
            "schedule_schema": SCHEDULE_SCHEMA,
            "schedule_id": schedule_id,
            "rollout_schedule": _rollout_schedule(
                bundle.manifest["pools"][arm], seed, schedule_id, num_rollouts
            ),
            "rollout_actor_source": "current-slime-checkpoint-via-in-job-sglang",
            "external_actor_substitution": "forbidden",
            "topology_gate": _rollout_topology_contract(),
        },
        "heldout_evaluation": {
            "pool_name": "heldout_v4",
            "pool_manifest_digest": bundle.manifest["manifest_digest"],
            "games": _heldout_games(bundle),
            "plays_per_game": len(DEFAULT_HELDOUT_EPISODE_SEEDS),
            "request_parameters": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_response_tokens": 8192,
                "enable_thinking": True,
            },
            "trajectory_parameters": {
                "max_turns": 25,
                "conversation_history": True,
                "action_extraction": "first-boxed-payload-else-raw-response",
                "action_submission": "single-boxed-action",
                "success_semantics": "terminated-and-terminal-reward-strictly-positive",
                "truncation_semantics": "failure",
                "error_policy": "invalidate-entire-assay",
            },
            "actor_source": "final-slime-checkpoint-via-dedicated-sglang",
            "served_checkpoint_digest_must_equal_final_checkpoint": True,
            "external_actor_substitution": "forbidden",
        },
    }
    return {**body, "launch_spec_digest": _digest(body)}


def materialize_slime_assay_plan(
    pool_manifest_or_root: Path | str,
    output_dir: Path | str,
    *,
    remote_pool_root: str,
    remote_output_root: str,
    hf_checkpoint: str,
    hf_checkpoint_digest: str,
    reference_checkpoint: str,
    reference_checkpoint_digest: str,
    runtime_image: str,
    runtime_image_digest: str,
    spade_source_revision: str,
    source_root: Path | str,
    num_rollouts: int = 16,
    paired_seeds: Sequence[int] = DEFAULT_PAIRED_SEEDS,
    token_tolerance: float = 0.05,
) -> SlimeAssayPlan:
    """Write one immutable plan and 12 exact, counterbalanced Slime launch specs."""

    bundle = load_learner_pool_manifest(pool_manifest_or_root, verify_files=True)
    if bundle.heldout_v4_dir is None:
        raise SlimeAssayError("Slime assay plan requires the sealed heldout_v4 pool")
    if bundle.manifest["manifest_digest"] != SEALED_LEARNER_POOL_MANIFEST_DIGEST:
        raise SlimeAssayError("Slime assay plan requires the canonical sealed learner pool")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise SlimeAssayError(f"output directory already exists; refusing overwrite: {output}")
    num_rollouts = _positive_int(num_rollouts, "num_rollouts")
    seeds = tuple(paired_seeds)
    if (
        len(seeds) != 6
        or len(set(seeds)) != 6
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise SlimeAssayError("paired_seeds must contain exactly six unique non-negative integers")
    if (
        isinstance(token_tolerance, bool)
        or not isinstance(token_tolerance, (int, float))
        or not math.isfinite(float(token_tolerance))
        or not 0 <= float(token_tolerance) <= 0.05
    ):
        raise SlimeAssayError("token_tolerance must be finite and between 0 and 0.05")

    remote_pool_root = _absolute_path(remote_pool_root, "remote_pool_root")
    remote_output_root = _absolute_path(remote_output_root, "remote_output_root")
    hf_checkpoint = _absolute_path(hf_checkpoint, "hf_checkpoint")
    reference_checkpoint = _absolute_path(reference_checkpoint, "reference_checkpoint")
    hf_checkpoint_digest = _sha256(hf_checkpoint_digest, "hf_checkpoint_digest")
    reference_checkpoint_digest = _sha256(
        reference_checkpoint_digest, "reference_checkpoint_digest"
    )
    runtime_image = _nonempty(runtime_image, "runtime_image")
    runtime_image_digest = _sha256(runtime_image_digest, "runtime_image_digest")
    if not runtime_image.endswith("@" + runtime_image_digest):
        raise SlimeAssayError("runtime_image must be pinned to runtime_image_digest")
    if (
        not isinstance(spade_source_revision, str)
        or _REVISION_RE.fullmatch(spade_source_revision) is None
    ):
        raise SlimeAssayError("spade_source_revision must be a lowercase 40-hex revision")
    source_inventory = _source_inventory(Path(source_root))
    source: dict[str, Any] = {
        "spade_revision": spade_source_revision,
        "clean_checkout_required": True,
        "slime_submodule_revision": SLIME_SUBMODULE_REVISION,
        "critical_file_inventory": source_inventory,
    }
    runtime = {
        "container_image": runtime_image,
        "container_image_digest": runtime_image_digest,
        "platform": "linux-amd64-cuda",
        "gpu_count": "8",
        "job_system": "ray",
        "inference_engine": "sglang",
        "runtime_dependency_install": "forbidden",
    }

    specs: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    for pair_index, seed in enumerate(seeds):
        ordered_arms = _ARMS if pair_index % 2 == 0 else tuple(reversed(_ARMS))
        for order_position, arm in enumerate(ordered_arms):
            spec = _launch_spec(
                bundle=bundle,
                arm=arm,
                pair_index=pair_index,
                order_position=order_position,
                seed=seed,
                num_rollouts=num_rollouts,
                remote_pool_root=remote_pool_root,
                remote_output_root=remote_output_root,
                hf_checkpoint=hf_checkpoint,
                hf_checkpoint_digest=hf_checkpoint_digest,
                reference_checkpoint=reference_checkpoint,
                reference_checkpoint_digest=reference_checkpoint_digest,
                runtime=runtime,
                source=source,
            )
            specs.append(spec)
            schedule.append(
                {
                    "ordinal": len(schedule),
                    "pair_id": spec["pair_id"],
                    "seed": seed,
                    "order_position": order_position,
                    "arm": arm,
                    "run_id": spec["run_id"],
                    "launch_spec_path": f"launch-specs/{spec['run_id']}.json",
                    "launch_spec_digest": spec["launch_spec_digest"],
                }
            )

    body: dict[str, Any] = {
        "schema_version": SLIME_PLAN_SCHEMA,
        "backend": SLIME_BACKEND,
        "execution_state": SLIME_EXECUTION_STATE,
        "evidence_state": SLIME_EVIDENCE_STATE,
        "claim_scope": SLIME_CLAIM_SCOPE,
        "pool_manifest_digest": bundle.manifest["manifest_digest"],
        "pool_schedule_id": bundle.schedule_id,
        "paired_seeds": list(seeds),
        "num_pairs": len(seeds),
        "num_runs": len(specs),
        "num_rollouts_per_run": num_rollouts,
        "counterbalance": "coverage-first-even-pair;control-first-odd-pair",
        "launch_schedule": schedule,
        "source": source,
        "runtime": runtime,
        "initial_state": {
            "model": {"uri": hf_checkpoint, "artifact_digest": hf_checkpoint_digest},
            "reference_model": {
                "uri": reference_checkpoint,
                "artifact_digest": reference_checkpoint_digest,
            },
            "optimizer": _optimizer_contract(),
        },
        "compute_gate": {
            "population": "real",
            "role": "all",
            "measures": list(_COMPUTE_MEASURES),
            "maximum_pairwise_relative_difference": float(token_tolerance),
            "padded_tokens_excluded": True,
        },
        "rollout_topology_gate": _rollout_topology_contract(),
        "heldout_protocol": specs[0]["heldout_evaluation"],
        "checkpoint_provenance": {
            "training_rollout": "each rollout must use the previous Slime optimizer output via in-job SGLang",
            "heldout": "dedicated SGLang must serve the run's final Slime optimizer output",
            "agy_google": "sealed environment/hint/design provenance only",
            "external_actor_substitution": "forbidden",
        },
        "primary_analysis": {
            "endpoint": "heldout_success_rate",
            "unit": "paired_training_seed",
            "direction": "coverage_forced_minus_redundant_historical",
            "test": "exact-one-sided-paired-sign-flip",
            "test_assumption": "paired differences are exchangeable under the sharp null",
            "assignment": "deterministic counterbalance; job slots are not randomized",
            "alpha": 0.05,
            "decision": "disabled until an artifact-backed trusted collector is implemented",
        },
    }
    manifest = {**body, "plan_digest": _digest(body)}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755)
    try:
        launch_dir = output / "launch-specs"
        launch_dir.mkdir(mode=0o755)
        for spec in specs:
            _write_new(launch_dir / f"{spec['run_id']}.json", spec)
        _write_new(output / "slime-assay-plan.json", manifest)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return load_slime_assay_plan(output, verify_files=True)


def load_slime_assay_plan(
    manifest_or_root: Path | str, *, verify_files: bool = True
) -> SlimeAssayPlan:
    """Load and strictly revalidate an immutable Slime assay plan."""

    supplied = Path(manifest_or_root)
    manifest_path = (
        supplied if supplied.name == "slime-assay-plan.json" else supplied / "slime-assay-plan.json"
    )
    root = manifest_path.parent
    manifest = _read_json(manifest_path, "Slime assay plan")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "backend",
            "execution_state",
            "evidence_state",
            "claim_scope",
            "pool_manifest_digest",
            "pool_schedule_id",
            "paired_seeds",
            "num_pairs",
            "num_runs",
            "num_rollouts_per_run",
            "counterbalance",
            "launch_schedule",
            "source",
            "runtime",
            "initial_state",
            "compute_gate",
            "rollout_topology_gate",
            "heldout_protocol",
            "checkpoint_provenance",
            "primary_analysis",
            "plan_digest",
        },
        "Slime assay plan",
    )
    _sealed(manifest, "plan_digest", "Slime assay plan")
    if (
        manifest.get("schema_version") != SLIME_PLAN_SCHEMA
        or manifest.get("backend") != SLIME_BACKEND
    ):
        raise SlimeAssayError("Slime assay plan schema or backend differs")
    if (
        manifest.get("execution_state") != SLIME_EXECUTION_STATE
        or manifest.get("evidence_state") != SLIME_EVIDENCE_STATE
        or manifest.get("claim_scope") != SLIME_CLAIM_SCOPE
    ):
        raise SlimeAssayError("Slime assay plan-only claim scope differs")
    if (
        _sha256(manifest.get("pool_manifest_digest"), "pool manifest digest")
        != SEALED_LEARNER_POOL_MANIFEST_DIGEST
    ):
        raise SlimeAssayError("Slime assay pool is not the canonical sealed learner pool")
    _nonempty(manifest.get("pool_schedule_id"), "pool schedule id")
    seeds = manifest.get("paired_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 6
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != 6
    ):
        raise SlimeAssayError("Slime assay must contain six paired seeds")
    if manifest.get("num_pairs") != 6 or manifest.get("num_runs") != 12:
        raise SlimeAssayError("Slime assay pair/run counts differ")
    if (
        manifest.get("counterbalance") != "coverage-first-even-pair;control-first-odd-pair"
        or manifest.get("rollout_topology_gate") != _rollout_topology_contract()
    ):
        raise SlimeAssayError("Slime assay counterbalance or rollout topology differs")
    num_rollouts = _positive_int(manifest.get("num_rollouts_per_run"), "num_rollouts_per_run")
    compute_gate = manifest.get("compute_gate")
    if (
        not isinstance(compute_gate, dict)
        or set(compute_gate)
        != {
            "population",
            "role",
            "measures",
            "maximum_pairwise_relative_difference",
            "padded_tokens_excluded",
        }
        or compute_gate.get("population") != "real"
        or compute_gate.get("role") != "all"
        or compute_gate.get("measures") != list(_COMPUTE_MEASURES)
        or compute_gate.get("padded_tokens_excluded") is not True
        or not isinstance(compute_gate.get("maximum_pairwise_relative_difference"), (int, float))
        or not 0 <= float(compute_gate["maximum_pairwise_relative_difference"]) <= 0.05
    ):
        raise SlimeAssayError("Slime assay compute gate differs")
    provenance = manifest.get("checkpoint_provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {"training_rollout", "heldout", "agy_google", "external_actor_substitution"}
        or provenance.get("external_actor_substitution") != "forbidden"
        or provenance.get("agy_google") != "sealed environment/hint/design provenance only"
    ):
        raise SlimeAssayError("Slime assay checkpoint provenance differs")
    source = manifest.get("source")
    source_inventory = source.get("critical_file_inventory") if isinstance(source, dict) else None
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "spade_revision",
            "clean_checkout_required",
            "slime_submodule_revision",
            "critical_file_inventory",
        }
        or source.get("clean_checkout_required") is not True
        or source.get("slime_submodule_revision") != SLIME_SUBMODULE_REVISION
        or not isinstance(source.get("spade_revision"), str)
        or _REVISION_RE.fullmatch(source["spade_revision"]) is None
        or not isinstance(source_inventory, list)
        or [item.get("path") for item in source_inventory if isinstance(item, dict)]
        != list(_SOURCE_FILES)
    ):
        raise SlimeAssayError("Slime assay source identity differs")
    for item in source_inventory:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size_bytes", "sha256"}
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
        ):
            raise SlimeAssayError("Slime assay critical source inventory differs")
        _sha256(item.get("sha256"), f"critical source {item.get('path')} digest")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "container_image",
            "container_image_digest",
            "platform",
            "gpu_count",
            "job_system",
            "inference_engine",
            "runtime_dependency_install",
        }
        or runtime.get("platform") != "linux-amd64-cuda"
        or runtime.get("gpu_count") != "8"
        or runtime.get("job_system") != "ray"
        or runtime.get("inference_engine") != "sglang"
        or runtime.get("runtime_dependency_install") != "forbidden"
        or not isinstance(runtime.get("container_image"), str)
        or not runtime["container_image"]
    ):
        raise SlimeAssayError("Slime assay runtime identity differs")
    runtime_digest = _sha256(
        runtime.get("container_image_digest"), "runtime container image digest"
    )
    if not runtime["container_image"].endswith("@" + runtime_digest):
        raise SlimeAssayError("Slime assay runtime image is not digest pinned")
    initial_state = manifest.get("initial_state")
    if (
        not isinstance(initial_state, dict)
        or set(initial_state) != {"model", "reference_model", "optimizer"}
        or initial_state.get("optimizer") != _optimizer_contract()
        or not isinstance(initial_state.get("model"), dict)
        or set(initial_state["model"]) != {"uri", "artifact_digest"}
        or not isinstance(initial_state.get("reference_model"), dict)
        or set(initial_state["reference_model"]) != {"uri", "artifact_digest"}
    ):
        raise SlimeAssayError("Slime assay initial model or optimizer differs")
    _sha256(initial_state["model"].get("artifact_digest"), "initial model digest")
    _absolute_path(initial_state["model"].get("uri"), "initial model uri")
    _sha256(
        initial_state["reference_model"].get("artifact_digest"),
        "initial reference model digest",
    )
    _absolute_path(initial_state["reference_model"].get("uri"), "reference model uri")
    heldout_protocol = manifest.get("heldout_protocol")
    if (
        not isinstance(heldout_protocol, dict)
        or set(heldout_protocol)
        != {
            "pool_name",
            "pool_manifest_digest",
            "games",
            "plays_per_game",
            "request_parameters",
            "trajectory_parameters",
            "actor_source",
            "served_checkpoint_digest_must_equal_final_checkpoint",
            "external_actor_substitution",
        }
        or heldout_protocol.get("pool_name") != "heldout_v4"
        or heldout_protocol.get("pool_manifest_digest") != manifest.get("pool_manifest_digest")
        or not isinstance(heldout_protocol.get("games"), list)
        or len(heldout_protocol["games"]) != 12
        or heldout_protocol.get("plays_per_game") != len(DEFAULT_HELDOUT_EPISODE_SEEDS)
        or heldout_protocol.get("request_parameters")
        != {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_response_tokens": 8192,
            "enable_thinking": True,
        }
        or heldout_protocol.get("trajectory_parameters")
        != {
            "max_turns": 25,
            "conversation_history": True,
            "action_extraction": "first-boxed-payload-else-raw-response",
            "action_submission": "single-boxed-action",
            "success_semantics": "terminated-and-terminal-reward-strictly-positive",
            "truncation_semantics": "failure",
            "error_policy": "invalidate-entire-assay",
        }
        or heldout_protocol.get("actor_source") != "final-slime-checkpoint-via-dedicated-sglang"
        or heldout_protocol.get("served_checkpoint_digest_must_equal_final_checkpoint") is not True
        or heldout_protocol.get("external_actor_substitution") != "forbidden"
    ):
        raise SlimeAssayError("Slime assay heldout protocol differs")
    heldout_names: list[str] = []
    for game in heldout_protocol["games"]:
        if not isinstance(game, dict) or set(game) != {"basename", "replicates"}:
            raise SlimeAssayError("Slime heldout game protocol keys differ")
        basename = game.get("basename")
        replicates = game.get("replicates")
        if (
            not isinstance(basename, str)
            or not basename.startswith("game_")
            or not basename.endswith(".py")
            or not isinstance(replicates, list)
            or len(replicates) != len(DEFAULT_HELDOUT_EPISODE_SEEDS)
        ):
            raise SlimeAssayError("Slime heldout game protocol differs")
        heldout_names.append(basename)
        for replicate_id, replicate in enumerate(replicates):
            if (
                not isinstance(replicate, dict)
                or set(replicate) != {"replicate_id", "environment_seed", "sampling_seed"}
                or replicate.get("replicate_id") != replicate_id
                or replicate.get("environment_seed") != DEFAULT_HELDOUT_EPISODE_SEEDS[replicate_id]
                or replicate.get("sampling_seed")
                != _heldout_sampling_seed(manifest["pool_schedule_id"], basename, replicate_id)
            ):
                raise SlimeAssayError("Slime heldout paired seed protocol differs")
    if len(set(heldout_names)) != 12:
        raise SlimeAssayError("Slime heldout game basenames differ")
    primary = manifest.get("primary_analysis")
    if (
        not isinstance(primary, dict)
        or set(primary)
        != {
            "endpoint",
            "unit",
            "direction",
            "test",
            "test_assumption",
            "assignment",
            "alpha",
            "decision",
        }
        or primary.get("endpoint") != "heldout_success_rate"
        or primary.get("unit") != "paired_training_seed"
        or primary.get("direction") != "coverage_forced_minus_redundant_historical"
        or primary.get("test") != "exact-one-sided-paired-sign-flip"
        or primary.get("test_assumption")
        != "paired differences are exchangeable under the sharp null"
        or primary.get("assignment") != "deterministic counterbalance; job slots are not randomized"
        or primary.get("alpha") != 0.05
        or primary.get("decision")
        != "disabled until an artifact-backed trusted collector is implemented"
    ):
        raise SlimeAssayError("Slime assay primary analysis differs")
    schedule = manifest.get("launch_schedule")
    if not isinstance(schedule, list) or len(schedule) != 12:
        raise SlimeAssayError("Slime launch schedule must contain 12 runs")

    specs: list[Mapping[str, Any]] = []
    seen_runs: set[str] = set()
    for ordinal, item in enumerate(schedule):
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "ordinal",
                "pair_id",
                "seed",
                "order_position",
                "arm",
                "run_id",
                "launch_spec_path",
                "launch_spec_digest",
            }
            or item.get("ordinal") != ordinal
        ):
            raise SlimeAssayError("Slime launch schedule ordinal differs")
        pair_index = ordinal // 2
        expected_arms = _ARMS if pair_index % 2 == 0 else tuple(reversed(_ARMS))
        if (
            item.get("pair_id") != f"pair-{pair_index:02d}"
            or item.get("seed") != seeds[pair_index]
            or item.get("order_position") != ordinal % 2
            or item.get("arm") != expected_arms[ordinal % 2]
        ):
            raise SlimeAssayError("Slime launch counterbalance differs")
        run_id = _nonempty(item.get("run_id"), "launch run_id")
        if run_id in seen_runs:
            raise SlimeAssayError(f"duplicate Slime launch run_id: {run_id}")
        seen_runs.add(run_id)
        expected_path = f"launch-specs/{run_id}.json"
        if item.get("launch_spec_path") != expected_path:
            raise SlimeAssayError(f"Slime launch spec path differs: {run_id}")
        spec = _read_json(root / expected_path, f"Slime launch spec {run_id}")
        _exact_keys(
            spec,
            {
                "schema_version",
                "backend",
                "run_id",
                "pair_id",
                "pair_index",
                "order_position",
                "arm",
                "seed",
                "pool_manifest_digest",
                "pool_name",
                "initial_state",
                "source",
                "runtime",
                "launcher",
                "training",
                "heldout_evaluation",
                "launch_spec_digest",
            },
            f"Slime launch spec {run_id}",
        )
        digest = _sealed(spec, "launch_spec_digest", f"Slime launch spec {run_id}")
        if digest != item.get("launch_spec_digest"):
            raise SlimeAssayError(f"Slime launch spec binding differs: {run_id}")
        if (
            spec.get("schema_version") != SLIME_LAUNCH_SCHEMA
            or spec.get("backend") != SLIME_BACKEND
            or spec.get("run_id") != run_id
            or spec.get("pair_id") != item.get("pair_id")
            or spec.get("pair_index") != pair_index
            or spec.get("order_position") != item.get("order_position")
            or spec.get("arm") != item.get("arm")
            or spec.get("seed") != item.get("seed")
            or spec.get("pool_manifest_digest") != manifest.get("pool_manifest_digest")
            or spec.get("pool_name") != spec.get("arm")
            or spec.get("source") != source
            or spec.get("runtime") != runtime
            or not isinstance(spec.get("initial_state"), dict)
            or set(spec["initial_state"])
            != {"model", "reference_model", "load_checkpoint", "optimizer"}
            or spec.get("initial_state", {}).get("model") != initial_state["model"]
            or spec.get("initial_state", {}).get("reference_model")
            != initial_state["reference_model"]
            or spec.get("initial_state", {}).get("optimizer") != initial_state["optimizer"]
            or spec.get("initial_state", {}).get("load_checkpoint") is not None
            or not isinstance(spec.get("training"), dict)
            or spec.get("training", {}).get("num_rollouts") != num_rollouts
            or spec.get("training", {}).get("games_per_rollout") != 12
            or spec.get("training", {}).get("trajectories_per_game") != 16
            or spec.get("training", {}).get("global_batch_size") != 192
            or spec.get("training", {}).get("expected_trajectories") != num_rollouts * 192
            or spec.get("training", {}).get("schedule_schema") != SCHEDULE_SCHEMA
            or spec.get("training", {}).get("schedule_id") != manifest.get("pool_schedule_id")
            or spec.get("training", {}).get("rollout_actor_source")
            != "current-slime-checkpoint-via-in-job-sglang"
            or spec.get("training", {}).get("external_actor_substitution") != "forbidden"
            or spec.get("training", {}).get("topology_gate") != _rollout_topology_contract()
            or spec.get("heldout_evaluation") != manifest.get("heldout_protocol")
        ):
            raise SlimeAssayError(f"Slime launch spec contract differs: {run_id}")
        rollout_schedule = spec.get("training", {}).get("rollout_schedule")
        if (
            set(spec["training"])
            != {
                "num_rollouts",
                "games_per_rollout",
                "trajectories_per_game",
                "global_batch_size",
                "expected_trajectories",
                "schedule_schema",
                "schedule_id",
                "rollout_schedule",
                "rollout_actor_source",
                "external_actor_substitution",
                "topology_gate",
            }
            or not isinstance(rollout_schedule, list)
            or any(not isinstance(entry, dict) for entry in rollout_schedule)
            or [entry.get("rollout_id") for entry in rollout_schedule] != list(range(num_rollouts))
            or any(
                set(entry) != {"rollout_id", "slot_basenames"}
                or not isinstance(entry.get("slot_basenames"), list)
                or len(entry["slot_basenames"]) != 12
                or len(set(entry["slot_basenames"])) != 12
                or any(
                    not isinstance(name, str)
                    or not name.startswith("game_")
                    or not name.endswith(".py")
                    for name in entry["slot_basenames"]
                )
                for entry in rollout_schedule
            )
        ):
            raise SlimeAssayError(f"Slime rollout schedule differs: {run_id}")
        launcher = spec.get("launcher")
        environment = launcher.get("environment") if isinstance(launcher, dict) else None
        if (
            not isinstance(launcher, dict)
            or set(launcher)
            != {
                "working_directory",
                "argv",
                "environment",
                "required_secret_environment",
                "forbidden_environment",
                "effective_training_arguments",
            }
            or launcher.get("working_directory") != "/workspace/spade"
            or launcher.get("argv") != ["cmd/games/_train_fixed_gpt55.sh"]
            or launcher.get("required_secret_environment") != []
            or launcher.get("forbidden_environment")
            != [
                "LOAD_DIR",
                "PREPARE_ONLY",
                "WANDB_API_KEY",
                "WANDB_ENTITY",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "AGY_API_KEY",
            ]
            or not isinstance(environment, dict)
            or set(environment)
            != {
                "FIXED_MODEL_SIZE",
                "STATIC_POOL_DIR",
                "MIN_POOL_GAMES",
                "NUM_ROLLOUT",
                "ROLLOUT_BATCH_SIZE",
                "GLOBAL_BATCH_SIZE",
                "MAX_CONTEXT_LENGTH",
                "MAX_TURNS",
                "SPADE_NUM_GAMES_PER_ROLLOUT",
                "SPADE_TRAJECTORIES_PER_GAME",
                "SPADE_FIXED_POOL_SEED",
                "SPADE_STATIC_POOL_SCHEDULE_ID",
                "SPADE_REQUIRE_COMPLETE_ROLLOUT",
                "TRAIN_SEED",
                "KL_COEF",
                "ACTOR_MAX_TOKENS",
                "MODEL_ARGS_ROTARY_BASE",
                "EVAL_INTERVAL",
                "MASTER_ADDR",
                "WORKSPACE_DIR",
                "SPADE_PLAY_CANARY",
                "HF_CHECKPOINT",
                "REF_CHECKPOINT",
                "OUTPUT_DIR",
                "SAVE_INTERVAL",
                "SKIP_RUNTIME_INSTALL",
                "DISABLE_WANDB",
            }
            or environment.get("FIXED_MODEL_SIZE") != "8b"
            or environment.get("MIN_POOL_GAMES") != "12"
            or environment.get("NUM_ROLLOUT") != str(num_rollouts)
            or environment.get("ROLLOUT_BATCH_SIZE") != "12"
            or environment.get("GLOBAL_BATCH_SIZE") != "192"
            or environment.get("MAX_CONTEXT_LENGTH") != "32768"
            or environment.get("MAX_TURNS") != "25"
            or environment.get("SPADE_NUM_GAMES_PER_ROLLOUT") != "12"
            or environment.get("SPADE_TRAJECTORIES_PER_GAME") != "16"
            or environment.get("SPADE_FIXED_POOL_SEED") != str(spec["seed"])
            or environment.get("TRAIN_SEED") != str(spec["seed"])
            or environment.get("KL_COEF") != "0.005"
            or environment.get("ACTOR_MAX_TOKENS") != "8192"
            or environment.get("MODEL_ARGS_ROTARY_BASE") != "1000000"
            or environment.get("EVAL_INTERVAL") != "100000"
            or environment.get("MASTER_ADDR") != "127.0.0.1"
            or environment.get("WORKSPACE_DIR") != "/mnt/spade-workspace"
            or environment.get("SPADE_PLAY_CANARY") != "1"
            or environment.get("SPADE_STATIC_POOL_SCHEDULE_ID") != manifest.get("pool_schedule_id")
            or environment.get("SPADE_REQUIRE_COMPLETE_ROLLOUT") != "1"
            or environment.get("SAVE_INTERVAL") != str(num_rollouts)
            or environment.get("SKIP_RUNTIME_INSTALL") != "1"
            or environment.get("DISABLE_WANDB") != "1"
            or "LOAD_DIR" in environment
        ):
            raise SlimeAssayError(f"Slime launcher contract differs: {run_id}")
        hf_checkpoint = _absolute_path(environment.get("HF_CHECKPOINT"), "HF_CHECKPOINT")
        reference_checkpoint = _absolute_path(environment.get("REF_CHECKPOINT"), "REF_CHECKPOINT")
        pool_dir = _absolute_path(environment.get("STATIC_POOL_DIR"), "STATIC_POOL_DIR")
        output_dir = _absolute_path(environment.get("OUTPUT_DIR"), "OUTPUT_DIR")
        if (
            hf_checkpoint != initial_state["model"]["uri"]
            or reference_checkpoint != initial_state["reference_model"]["uri"]
            or launcher.get("effective_training_arguments")
            != _effective_arguments(
                hf_checkpoint=hf_checkpoint,
                reference_checkpoint=reference_checkpoint,
                output_dir=output_dir,
                pool_dir=pool_dir,
                seed=spec["seed"],
                schedule_id=manifest["pool_schedule_id"],
                num_rollouts=num_rollouts,
            )
        ):
            raise SlimeAssayError(f"Slime effective training arguments differ: {run_id}")
        specs.append(spec)

    for pair_index in range(6):
        left, right = specs[pair_index * 2 : pair_index * 2 + 2]
        left_slots = [item["slot_basenames"] for item in left["training"]["rollout_schedule"]]
        right_slots = [item["slot_basenames"] for item in right["training"]["rollout_schedule"]]
        if left_slots != right_slots:
            raise SlimeAssayError(f"pair-{pair_index:02d} rollout basename slots do not align")

    if verify_files:
        if {path.name for path in root.iterdir()} != {"slime-assay-plan.json", "launch-specs"}:
            raise SlimeAssayError("Slime assay plan root inventory differs")
        launch_dir = root / "launch-specs"
        if launch_dir.is_symlink() or not launch_dir.is_dir():
            raise SlimeAssayError("Slime launch-spec directory is unavailable")
        if {path.name for path in launch_dir.iterdir()} != {
            f"{item['run_id']}.json" for item in schedule
        }:
            raise SlimeAssayError("Slime launch-spec inventory differs")
        for path in (manifest_path, *(launch_dir.iterdir())):
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise SlimeAssayError(f"Slime assay plan artifact is writable: {path}")
    return SlimeAssayPlan(root, manifest_path, manifest, tuple(specs))


def _token_metric_keys() -> set[str]:
    return {
        f"rollout/tokens/{population}/{role}/{measure}"
        for population in _TOKEN_POPULATIONS
        for role in _TOKEN_ROLES
        for measure in _TOKEN_MEASURES
    }


def _validate_token_metrics(value: object, where: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _token_metric_keys():
        raise SlimeAssayError(f"{where} token metric inventory differs")
    metrics: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise SlimeAssayError(f"{where}.{key} must be a non-negative integer")
        metrics[key] = item
    for role in _TOKEN_ROLES:
        for measure in _TOKEN_MEASURES:
            total = metrics[f"rollout/tokens/total/{role}/{measure}"]
            recomputed = sum(
                metrics[f"rollout/tokens/{population}/{role}/{measure}"]
                for population in ("real", "padded")
            )
            if total != recomputed:
                raise SlimeAssayError(f"{where} total != real + padded: {role}/{measure}")
    for population in _TOKEN_POPULATIONS:
        for measure in _TOKEN_MEASURES:
            total = metrics[f"rollout/tokens/{population}/all/{measure}"]
            recomputed = sum(
                metrics[f"rollout/tokens/{population}/{role}/{measure}"]
                for role in ("actor", "environment", "unknown")
            )
            if total != recomputed:
                raise SlimeAssayError(f"{where} all roles do not rederive: {population}/{measure}")
        for role in _TOKEN_ROLES:
            prompt = metrics[f"rollout/tokens/{population}/{role}/prompt_tokens"]
            response = metrics[f"rollout/tokens/{population}/{role}/response_tokens"]
            sequence = metrics[f"rollout/tokens/{population}/{role}/sequence_tokens"]
            loss_mask = metrics[f"rollout/tokens/{population}/{role}/loss_mask_tokens"]
            if sequence != prompt + response or loss_mask > response:
                raise SlimeAssayError(f"{where} token shapes differ: {population}/{role}")
    if (
        metrics["rollout/tokens/real/all/sequence_tokens"] <= 0
        or metrics["rollout/tokens/real/all/response_tokens"] <= 0
        or metrics["rollout/tokens/real/all/loss_mask_tokens"] <= 0
    ):
        raise SlimeAssayError(f"{where} contains no real action/loss-mask training tokens")
    return metrics


def _topology_metric_keys() -> set[str]:
    return {
        f"rollout/groups/{population}/{role}/episode_groups"
        for population in _TOKEN_POPULATIONS
        for role in _TOKEN_ROLES
    } | set(_SOURCE_TOPOLOGY_KEYS)


def _validate_topology_metrics(value: object, where: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _topology_metric_keys():
        raise SlimeAssayError(f"{where} topology metric inventory differs")
    metrics: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise SlimeAssayError(f"{where}.{key} must be a non-negative integer")
        metrics[key] = item
    for role in _TOKEN_ROLES:
        if metrics[f"rollout/groups/total/{role}/episode_groups"] != sum(
            metrics[f"rollout/groups/{population}/{role}/episode_groups"]
            for population in ("real", "padded")
        ):
            raise SlimeAssayError(f"{where} total episode groups do not rederive: {role}")
    for population in _TOKEN_POPULATIONS:
        if metrics[f"rollout/groups/{population}/all/episode_groups"] != sum(
            metrics[f"rollout/groups/{population}/{role}/episode_groups"]
            for role in ("actor", "environment", "unknown")
        ):
            raise SlimeAssayError(f"{where} episode-group roles do not rederive: {population}")
    expected = {
        "rollout/groups/real/all/episode_groups": 192,
        "rollout/groups/real/actor/episode_groups": 192,
        "rollout/groups/real/environment/episode_groups": 0,
        "rollout/groups/real/unknown/episode_groups": 0,
        "rollout/groups/padded/all/episode_groups": 0,
        "rollout/groups/padded/actor/episode_groups": 0,
        "rollout/groups/padded/environment/episode_groups": 0,
        "rollout/groups/padded/unknown/episode_groups": 0,
        "rollout/groups/total/all/episode_groups": 192,
        "rollout/groups/total/actor/episode_groups": 192,
        "rollout/groups/total/environment/episode_groups": 0,
        "rollout/groups/total/unknown/episode_groups": 0,
        "rollout/topology/actor_instances_requested": 192,
        "rollout/topology/actor_instances_succeeded": 192,
        "rollout/topology/actor_instances_failed": 0,
        "rollout/topology/actor_trajectories_filtered": 0,
        "rollout/topology/environment_trajectories_filtered": 0,
    }
    if metrics != expected:
        raise SlimeAssayError(f"{where} is not the sealed 192-real/zero-pad/zero-failure topology")
    return metrics


def _cross_validate_rollout_metrics(
    token_metrics: Mapping[str, int],
    topology_metrics: Mapping[str, int],
    where: str,
) -> None:
    """Close token/sample counters against the sealed episode-group topology."""

    for population in _TOKEN_POPULATIONS:
        for role in _TOKEN_ROLES:
            groups = topology_metrics[f"rollout/groups/{population}/{role}/episode_groups"]
            samples = token_metrics[f"rollout/tokens/{population}/{role}/samples"]
            token_values = [
                token_metrics[f"rollout/tokens/{population}/{role}/{measure}"]
                for measure in _TOKEN_MEASURES
                if measure != "samples"
            ]
            if groups == 0 and (samples != 0 or any(token_values)):
                raise SlimeAssayError(
                    f"{where} has tokens or samples for a zero-group population: "
                    f"{population}/{role}"
                )
            if groups > 0 and samples < groups:
                raise SlimeAssayError(
                    f"{where} has fewer samples than episode groups: {population}/{role}"
                )


def _validate_heldout(
    value: object, spec: Mapping[str, Any], final_checkpoint_digest: str, where: str
) -> float:
    protocol = spec["heldout_evaluation"]
    value = _exact_keys(
        value,
        {
            "actor_source",
            "backend",
            "served_checkpoint_digest",
            "pool_manifest_digest",
            "pool_name",
            "request_parameters",
            "trajectory_parameters",
            "games",
            "aggregate",
        },
        f"{where} heldout result",
    )
    if (
        value.get("actor_source") != "final-slime-checkpoint-via-dedicated-sglang"
        or value.get("backend") != SLIME_BACKEND
        or value.get("served_checkpoint_digest") != final_checkpoint_digest
        or value.get("pool_manifest_digest") != protocol["pool_manifest_digest"]
        or value.get("pool_name") != protocol["pool_name"]
        or value.get("request_parameters") != protocol["request_parameters"]
        or value.get("trajectory_parameters") != protocol["trajectory_parameters"]
    ):
        raise SlimeAssayError(f"{where} heldout checkpoint, pool, or execution contract differs")
    games = value.get("games")
    expected_games = protocol["games"]
    expected_names = [item["basename"] for item in expected_games]
    if (
        not isinstance(games, list)
        or any(not isinstance(item, dict) for item in games)
        or [item.get("basename") for item in games] != expected_names
    ):
        raise SlimeAssayError(f"{where} heldout game inventory differs")
    total_wins = 0
    total_plays = 0
    for game, expected_game in zip(games, expected_games):
        _exact_keys(game, {"basename", "episodes"}, f"{where} heldout game result")
        episodes = game.get("episodes")
        expected_replicates = expected_game["replicates"]
        if not isinstance(episodes, list) or len(episodes) != len(expected_replicates):
            raise SlimeAssayError(f"{where} heldout episode inventory differs")
        for episode, expected in zip(episodes, expected_replicates):
            _exact_keys(
                episode,
                {
                    "replicate_id",
                    "environment_seed",
                    "sampling_seed",
                    "termination",
                    "terminal_reward",
                    "total_reward",
                    "turns",
                    "error",
                },
                f"{where} heldout episode",
            )
            if (
                episode.get("replicate_id") != expected["replicate_id"]
                or episode.get("environment_seed") != expected["environment_seed"]
                or episode.get("sampling_seed") != expected["sampling_seed"]
                or episode.get("termination") not in {"terminated", "truncated"}
                or isinstance(episode.get("terminal_reward"), bool)
                or not isinstance(episode.get("terminal_reward"), (int, float))
                or not math.isfinite(float(episode["terminal_reward"]))
                or isinstance(episode.get("total_reward"), bool)
                or not isinstance(episode.get("total_reward"), (int, float))
                or not math.isfinite(float(episode["total_reward"]))
                or isinstance(episode.get("turns"), bool)
                or not isinstance(episode.get("turns"), int)
                or not 1 <= episode["turns"] <= protocol["trajectory_parameters"]["max_turns"]
                or episode.get("error") is not None
            ):
                raise SlimeAssayError(f"{where} heldout episode is invalid or failed")
            total_wins += int(
                episode["termination"] == "terminated" and episode["terminal_reward"] > 0
            )
            total_plays += 1
    expected_plays = len(expected_names) * protocol["plays_per_game"]
    if total_plays != expected_plays:
        raise SlimeAssayError(f"{where} heldout play count differs")
    success_rate = total_wins / total_plays
    aggregate = value.get("aggregate")
    if (
        not isinstance(aggregate, dict)
        or set(aggregate) != {"total_games", "total_plays", "total_successes", "success_rate"}
        or aggregate.get("total_games") != len(expected_names)
        or aggregate.get("total_plays") != total_plays
        or aggregate.get("total_successes") != total_wins
        or not math.isclose(
            float(aggregate.get("success_rate", -1)), success_rate, rel_tol=0, abs_tol=1e-15
        )
    ):
        raise SlimeAssayError(f"{where} heldout aggregate does not rederive")
    return success_rate


def _sign_flip_p_value(deltas: Sequence[float]) -> float:
    observed = sum(deltas) / len(deltas)
    permutations = [
        sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas)
        for signs in itertools.product((-1, 1), repeat=len(deltas))
    ]
    return sum(value >= observed - 1e-15 for value in permutations) / len(permutations)


def validate_slime_assay_results(
    plan_or_root: Path | str,
    results_path: Path | str,
) -> Mapping[str, Any]:
    """Check self-supplied receipt consistency and return an unconditional HOLD.

    No artifact-backed collector exists yet, so even internally consistent input
    is not treated as trusted execution evidence and cannot produce a positive
    learner or causal decision.
    """

    plan = load_slime_assay_plan(plan_or_root, verify_files=True)
    results = _read_json(Path(results_path), "Slime assay results")
    _exact_keys(
        results,
        {"schema_version", "backend", "plan_digest", "runs", "results_digest"},
        "Slime assay results",
    )
    results_digest = _sealed(results, "results_digest", "Slime assay results")
    if (
        results.get("schema_version") != SLIME_RESULTS_SCHEMA
        or results.get("backend") != SLIME_BACKEND
        or results.get("plan_digest") != plan.manifest["plan_digest"]
    ):
        raise SlimeAssayError("Slime assay result schema, backend, or plan binding differs")
    runs = results.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise SlimeAssayError("Slime assay results must contain exactly 12 runs")
    by_run: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        _exact_keys(
            run,
            {
                "run_id",
                "status",
                "backend",
                "launch_spec_digest",
                "arm",
                "seed",
                "external_actor_substitution",
                "rollouts",
                "final_checkpoint_digest",
                "heldout",
            },
            "Slime assay run result",
        )
        run_id = _nonempty(run.get("run_id"), "result run_id")
        if run_id in by_run:
            raise SlimeAssayError(f"duplicate Slime assay run result: {run_id}")
        by_run[run_id] = run

    run_summaries: dict[str, dict[str, Any]] = {}
    for spec in plan.launch_specs:
        run_id = str(spec["run_id"])
        run = by_run.get(run_id)
        if run is None:
            raise SlimeAssayError(f"missing Slime assay run result: {run_id}")
        where = f"run {run_id}"
        if (
            run.get("status") != "complete"
            or run.get("backend") != SLIME_BACKEND
            or run.get("launch_spec_digest") != spec["launch_spec_digest"]
            or run.get("arm") != spec["arm"]
            or run.get("seed") != spec["seed"]
            or run.get("external_actor_substitution") is not False
        ):
            raise SlimeAssayError(f"{where} identity or completion differs")
        initial_digest = spec["initial_state"]["model"]["artifact_digest"]
        rollout_results = run.get("rollouts")
        schedule = spec["training"]["rollout_schedule"]
        if not isinstance(rollout_results, list) or len(rollout_results) != len(schedule):
            raise SlimeAssayError(f"{where} rollout count differs")
        previous_checkpoint = initial_digest
        token_totals = {measure: 0 for measure in _COMPUTE_MEASURES}
        for expected, rollout in zip(schedule, rollout_results):
            _exact_keys(
                rollout,
                {
                    "rollout_id",
                    "schedule_id",
                    "slot_basenames",
                    "actor_source",
                    "inference_checkpoint_digest",
                    "optimizer_input_checkpoint_digest",
                    "optimizer_output_checkpoint_digest",
                    "token_metrics",
                    "topology_metrics",
                },
                f"{where} rollout receipt",
            )
            rollout_id = expected["rollout_id"]
            if (
                rollout.get("rollout_id") != rollout_id
                or rollout.get("slot_basenames") != expected["slot_basenames"]
                or rollout.get("schedule_id") != spec["training"]["schedule_id"]
                or rollout.get("actor_source") != "current-slime-checkpoint-via-in-job-sglang"
                or rollout.get("inference_checkpoint_digest") != previous_checkpoint
                or rollout.get("optimizer_input_checkpoint_digest") != previous_checkpoint
            ):
                raise SlimeAssayError(f"{where} rollout {rollout_id} provenance differs")
            previous_checkpoint = _sha256(
                rollout.get("optimizer_output_checkpoint_digest"),
                f"{where} rollout {rollout_id} optimizer output",
            )
            metrics = _validate_token_metrics(
                rollout.get("token_metrics"), f"{where} rollout {rollout_id}"
            )
            topology = _validate_topology_metrics(
                rollout.get("topology_metrics"), f"{where} rollout {rollout_id}"
            )
            _cross_validate_rollout_metrics(
                metrics,
                topology,
                f"{where} rollout {rollout_id}",
            )
            for measure in _COMPUTE_MEASURES:
                token_totals[measure] += metrics[f"rollout/tokens/real/all/{measure}"]
        final_checkpoint = _sha256(run.get("final_checkpoint_digest"), f"{where} final checkpoint")
        if final_checkpoint != previous_checkpoint:
            raise SlimeAssayError(f"{where} final checkpoint does not close optimizer chain")
        if final_checkpoint == initial_digest:
            raise SlimeAssayError(f"{where} final checkpoint did not change from initial state")
        success_rate = _validate_heldout(run.get("heldout"), spec, final_checkpoint, where)
        run_summaries[run_id] = {
            "pair_id": spec["pair_id"],
            "arm": spec["arm"],
            "seed": spec["seed"],
            "real_token_totals": token_totals,
            "heldout_success_rate": success_rate,
            "final_checkpoint_digest": final_checkpoint,
        }

    pair_summaries: list[dict[str, Any]] = []
    deltas: list[float] = []
    tolerance = float(plan.manifest["compute_gate"]["maximum_pairwise_relative_difference"])
    for pair_index, seed in enumerate(plan.manifest["paired_seeds"]):
        pair_id = f"pair-{pair_index:02d}"
        arm_runs = {
            summary["arm"]: summary
            for summary in run_summaries.values()
            if summary["pair_id"] == pair_id
        }
        if set(arm_runs) != set(_ARMS):
            raise SlimeAssayError(f"{pair_id} does not contain both arms")
        compute: dict[str, Any] = {}
        for measure in _COMPUTE_MEASURES:
            treatment = arm_runs["coverage_forced"]["real_token_totals"][measure]
            control = arm_runs["redundant_historical"]["real_token_totals"][measure]
            denominator = max(treatment, control)
            relative_difference = abs(treatment - control) / denominator if denominator else 0.0
            if relative_difference > tolerance + 1e-15:
                raise SlimeAssayError(
                    f"{pair_id} {measure} exceeds paired compute tolerance: "
                    f"{relative_difference:.6f} > {tolerance:.6f}"
                )
            compute[measure] = {
                "coverage_forced": treatment,
                "redundant_historical": control,
                "relative_difference": relative_difference,
            }
        delta = (
            arm_runs["coverage_forced"]["heldout_success_rate"]
            - arm_runs["redundant_historical"]["heldout_success_rate"]
        )
        deltas.append(delta)
        pair_summaries.append(
            {
                "pair_id": pair_id,
                "seed": seed,
                "compute": compute,
                "coverage_forced_success_rate": arm_runs["coverage_forced"]["heldout_success_rate"],
                "redundant_historical_success_rate": arm_runs["redundant_historical"][
                    "heldout_success_rate"
                ],
                "delta": delta,
            }
        )

    mean_delta = sum(deltas) / len(deltas)
    p_value = _sign_flip_p_value(deltas)
    alpha = float(plan.manifest["primary_analysis"]["alpha"])
    body: dict[str, Any] = {
        "schema_version": SLIME_VALIDATION_SCHEMA,
        "status": SLIME_EXECUTION_STATE,
        "backend": SLIME_BACKEND,
        "plan_digest": plan.manifest["plan_digest"],
        "results_digest": results_digest,
        "evidence_state": SLIME_EVIDENCE_STATE,
        "receipt_consistency_gate": "pass",
        "rollout_topology_gate": "pass",
        "paired_compute_gate": "pass",
        "pair_summaries": pair_summaries,
        "primary_analysis": {
            "endpoint": "heldout_success_rate",
            "paired_deltas": deltas,
            "mean_delta": mean_delta,
            "one_sided_exact_sign_flip_p_value": p_value,
            "test_assumption": "paired differences are exchangeable under the sharp null",
            "assignment": "deterministic counterbalance; job slots are not randomized",
            "alpha": alpha,
            "interpretation": "exploratory-only; inputs are not trusted execution evidence",
        },
        "decision": SLIME_HOLD_DECISION,
        "causal_evidence": "not-established",
        "claim_scope": (
            "plan and self-supplied receipt consistency only; no learner-improvement, "
            "causal, execution, or cross-backend evidence"
        ),
    }
    return {**body, "validation_digest": _digest(body)}


__all__ = [
    "DEFAULT_HELDOUT_EPISODE_SEEDS",
    "DEFAULT_PAIRED_SEEDS",
    "SLIME_BACKEND",
    "SLIME_CLAIM_SCOPE",
    "SLIME_EVIDENCE_STATE",
    "SLIME_EXECUTION_STATE",
    "SLIME_HOLD_DECISION",
    "SLIME_LAUNCH_SCHEMA",
    "SLIME_PLAN_SCHEMA",
    "SLIME_RESULTS_SCHEMA",
    "SLIME_VALIDATION_SCHEMA",
    "SlimeAssayError",
    "SlimeAssayPlan",
    "load_slime_assay_plan",
    "materialize_slime_assay_plan",
    "validate_slime_assay_results",
]
