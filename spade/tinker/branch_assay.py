"""Pure protocol primitives for the paired Tinker learner assay.

This module deliberately imports neither Tinker nor ProofPack. The evidence
protocol and sensitivity analysis are therefore testable on an offline host.
The retained service boundary is disabled by the v1 runner's permanent HOLD.

The plan compares training on two fixed curricula from one asserted common
optimizer-bearing state. It is not an archive-selection proxy, does not train
the environment proposer, and cannot emit a causal learner claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class BranchAssayError(ValueError):
    """The learner-assay contract is incomplete, ambiguous, or inconsistent."""


PROTOCOL_ID = "spade-tinker-coverage-forced-learner-assay/v1"
INTENT_SCHEMA = "spade-tinker-learner-assay-intent/v1"
BASE_REQUEST_SCHEMA = "spade-tinker-common-base-request/v1"
BASE_RESULT_SCHEMA = "spade-tinker-common-base-result/v1"
BASE_RECEIPT_SCHEMA = "spade-tinker-common-base-receipt/v1"
BRANCH_REQUEST_SCHEMA = "spade-tinker-learner-branch-request/v1"
BRANCH_EXECUTION_SCHEMA = "spade-tinker-learner-branch-execution/v1"
SCORE_SCHEMA = "spade-tinker-heldout-score/v1"
PAIR_COMPLETE_SCHEMA = "spade-tinker-learner-pair-complete/v1"
PAIR_TERMINAL_ERROR_SCHEMA = "spade-tinker-learner-pair-terminal-error/v1"
AGGREGATE_SCHEMA = "spade-tinker-learner-assay-aggregate/v1"

ARMS = ("coverage_forced", "redundant_historical")
TRAIN_STRATA = ("c001", "c003", "c004", "c005", "c006", "c007")
HELDOUT_STRATA = (
    "c002",
    "c008",
    "c009",
    "c010",
    "c011",
    "c012",
    "c013",
    "c014",
    "c015",
    "c016",
    "c017",
    "c018",
)

PAIR_COUNT = 6
UPDATES_PER_BRANCH = 16
TRAIN_GAMES_PER_ARM = 12
TRAJECTORIES_PER_GAME = 8
EPISODES_PER_UPDATE = TRAIN_GAMES_PER_ARM * TRAJECTORIES_PER_GAME
MAX_TRAIN_TURNS = 2
# Every sealed v4 held-out environment has native horizon 10.  The live
# adapter loads that native value and rejects any source whose horizon exceeds
# this prospective service-cost ceiling.
MAX_HELDOUT_NATIVE_TURNS = 10
TRAIN_POSITIONS_PER_EPISODE = 4096
HELDOUT_CONTEXT_POSITIONS = 32768
TRAIN_POSITIONS_PER_UPDATE = EPISODES_PER_UPDATE * TRAIN_POSITIONS_PER_EPISODE
TRAIN_POSITIONS_PER_BRANCH = UPDATES_PER_BRANCH * TRAIN_POSITIONS_PER_UPDATE
HELDOUT_SEEDS_PER_GAME = 8
HELDOUT_EPISODES_PER_CHECKPOINT = len(HELDOUT_STRATA) * HELDOUT_SEEDS_PER_GAME
EXACT_SIGN_FLIP_COUNT = 2**PAIR_COUNT
MAX_SAMPLE_CALLS_PER_BRANCH = (
    UPDATES_PER_BRANCH * EPISODES_PER_UPDATE * MAX_TRAIN_TURNS
    + HELDOUT_EPISODES_PER_CHECKPOINT * MAX_HELDOUT_NATIVE_TURNS
    + 1  # restored-state canary
)
MAX_SAMPLE_CALLS_TOTAL = PAIR_COUNT * 2 * MAX_SAMPLE_CALLS_PER_BRANCH + 1

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RENDERER = "qwen3_disable_thinking_preserve_history"
DEFAULT_TINKER_ENDPOINT = "https://tinker.thinkingmachines.dev/services/tinker-prod"
DEFAULT_LORA_RANK = 32
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_ACTOR_TEMPERATURE = 0.6
DEFAULT_ACTOR_TOP_P = 0.95
DEFAULT_ACTOR_TOP_K = 20
DEFAULT_ACTOR_MAX_TOKENS = 1024
DEFAULT_MAX_CONCURRENT_SAMPLES = 16
DEFAULT_MINIMUM_EFFECT = 0.05
AUTHORIZED_POOL_MANIFEST_DIGEST = (
    "sha256:c10b3be0acd42df8b4bf3b695c02eab0cdc83678e699f8e860890e70b2783b1d"
)

_DIGEST_KEYS = {
    INTENT_SCHEMA: "intent_digest",
    BASE_REQUEST_SCHEMA: "request_digest",
    BASE_RESULT_SCHEMA: "result_digest",
    BASE_RECEIPT_SCHEMA: "receipt_digest",
    BRANCH_REQUEST_SCHEMA: "request_digest",
    BRANCH_EXECUTION_SCHEMA: "execution_digest",
    SCORE_SCHEMA: "score_digest",
    PAIR_COMPLETE_SCHEMA: "pair_digest",
    PAIR_TERMINAL_ERROR_SCHEMA: "error_digest",
    AGGREGATE_SCHEMA: "aggregate_digest",
}


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical JSON encoding used for scientific seals."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BranchAssayError(f"value is not canonical JSON: {exc}") from exc


def object_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    tail = value.removeprefix("sha256:")
    return len(tail) == 64 and all(char in "0123456789abcdef" for char in tail)


def seal_document(body: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    if digest_key in body:
        raise BranchAssayError(f"unsealed body unexpectedly contains {digest_key}")
    sealed = dict(body)
    sealed[digest_key] = object_digest(body)
    return sealed


def validate_seal(
    document: Mapping[str, Any],
    *,
    schema: str,
    exact_keys: set[str] | None = None,
) -> dict[str, Any]:
    digest_key = _DIGEST_KEYS[schema]
    if document.get("schema_version") != schema:
        raise BranchAssayError(f"document does not use {schema}")
    if exact_keys is not None and set(document) != exact_keys:
        raise BranchAssayError(f"{schema} keys differ from the sealed schema")
    digest = document.get(digest_key)
    if not _is_digest(digest):
        raise BranchAssayError(f"{schema} lacks a canonical {digest_key}")
    body = {key: value for key, value in document.items() if key != digest_key}
    if object_digest(body) != digest:
        raise BranchAssayError(f"{schema} self-digest mismatch")
    return dict(document)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise BranchAssayError(f"refusing to overwrite evidence artifact: {path}")
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BranchAssayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path, *, maximum_bytes: int = 8_000_000) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise BranchAssayError(f"JSON evidence path is missing or unsafe: {path}")
    if path.stat().st_size > maximum_bytes:
        raise BranchAssayError(f"JSON evidence exceeds size limit: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, BranchAssayError) as exc:
        raise BranchAssayError(f"cannot parse JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BranchAssayError(f"JSON evidence is not an object: {path}")
    return value


def file_digest(path: Path, *, maximum_bytes: int = 64_000_000) -> str:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise BranchAssayError(f"file is missing or unsafe: {path}")
    if path.stat().st_size > maximum_bytes:
        raise BranchAssayError(f"file exceeds size limit: {path}")
    return bytes_digest(path.read_bytes())


def derive_seed(*parts: object) -> int:
    """Derive a stable, nonnegative 31-bit seed from public schedule fields."""

    payload = "\0".join(str(part) for part in (PROTOCOL_ID, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def build_pair_schedule(assignment_seed: int, sampling_seed: int) -> list[dict[str, Any]]:
    """Build six independent within-pair treatment assignments.

    The seeded assignment is materialized in the intent. Re-running this
    function is validation, not a source of post-outcome randomness. The six
    independently generated bits support a ``2**6`` sign-flip sensitivity
    reference set under exchangeability, but this protocol does not
    authenticate the seed as external random entropy.
    """

    for value, field in (
        (assignment_seed, "assignment_seed"),
        (sampling_seed, "sampling_seed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise BranchAssayError(f"{field} must be an integer")
    rng = random.Random(assignment_seed)
    schedule: list[dict[str, Any]] = []
    for index in range(PAIR_COUNT):
        pair_id = f"pair-{index + 1:02d}"
        treatment_in_slot_a = bool(rng.getrandbits(1))
        arm_a, arm_b = ARMS if treatment_in_slot_a else ("redundant_historical", "coverage_forced")
        # Execution order is independently counterbalanced and prospectively
        # fixed. A production launcher may run both slots concurrently.
        execution_order = ["slot_a", "slot_b"] if index % 2 == 0 else ["slot_b", "slot_a"]
        schedule.append(
            {
                "pair_id": pair_id,
                # Stochastic learner/evaluation seeds are independent of the
                # treatment-assignment seed. Changing only assignment labels
                # therefore leaves every potential-outcome seed unchanged.
                "pair_seed": derive_seed("pair", sampling_seed, pair_id),
                "slot_assignments": {"slot_a": arm_a, "slot_b": arm_b},
                "execution_order": execution_order,
            }
        )
    return schedule


def _require_absolute_file(path_text: object, digest: object, field: str) -> Path:
    if not isinstance(path_text, str):
        raise BranchAssayError(f"{field}_path must be text")
    path = Path(path_text)
    if not _is_digest(digest):
        raise BranchAssayError(f"{field}_digest is invalid")
    if file_digest(path) != digest:
        raise BranchAssayError(f"{field} bytes differ from the intent")
    return path


def build_intent(
    *,
    source_actor_plan_path: Path,
    source_actor_plan_digest: str,
    pool_manifest_path: Path,
    output_root: Path,
    assignment_seed: int,
    sampling_seed: int,
    runtime_identity: Mapping[str, Any],
    model_name: str = DEFAULT_MODEL,
    renderer_name: str = DEFAULT_RENDERER,
) -> dict[str, Any]:
    """Create, but do not write, the fixed prospective assay intent."""

    for path, label in (
        (source_actor_plan_path, "source actor plan"),
        (pool_manifest_path, "pool manifest"),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise BranchAssayError(f"{label} must be an absolute regular file")
    if not _is_digest(source_actor_plan_digest):
        raise BranchAssayError("source_actor_plan_digest is invalid")
    source_plan = read_json(source_actor_plan_path)
    if source_plan.get("actor_plan_digest") != source_actor_plan_digest:
        raise BranchAssayError("source actor-plan digest is not the expected sealed digest")
    source_body = {key: value for key, value in source_plan.items() if key != "actor_plan_digest"}
    if object_digest(source_body) != source_actor_plan_digest:
        raise BranchAssayError("source actor-plan self-digest mismatch")
    if not output_root.is_absolute() or output_root.is_symlink():
        raise BranchAssayError("output_root must be an absolute, non-symlink path")
    if not isinstance(model_name, str) or not model_name:
        raise BranchAssayError("model_name must be non-empty")
    if not isinstance(renderer_name, str) or not renderer_name:
        raise BranchAssayError("renderer_name must be non-empty")
    if model_name != DEFAULT_MODEL or renderer_name != DEFAULT_RENDERER:
        raise BranchAssayError(
            "this protocol version is fixed to the supported Qwen3-8B Tinker learner"
        )
    if not runtime_identity:
        raise BranchAssayError("runtime_identity cannot be empty")

    pool_digest = file_digest(pool_manifest_path)
    pool_manifest = read_json(pool_manifest_path)
    pool_manifest_digest = pool_manifest.get("manifest_digest")
    if pool_manifest_digest != AUTHORIZED_POOL_MANIFEST_DIGEST:
        raise BranchAssayError("pool manifest is not the authorized 12/12/12 bundle")
    pool_body = {key: value for key, value in pool_manifest.items() if key != "manifest_digest"}
    if object_digest(pool_body) != pool_manifest_digest:
        raise BranchAssayError("pool manifest self-digest mismatch")
    body: dict[str, Any] = {
        "schema_version": INTENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source_actor_plan_path": str(source_actor_plan_path),
        "source_actor_plan_digest": source_actor_plan_digest,
        "pool_manifest_path": str(pool_manifest_path),
        "pool_manifest_file_digest": pool_digest,
        "pool_manifest_digest": pool_manifest_digest,
        "output_root": str(output_root),
        "runtime_identity": dict(runtime_identity),
        "service_endpoint": DEFAULT_TINKER_ENDPOINT,
        "assignment_seed": assignment_seed,
        "assignment_seed_record": (
            "user-entered-local-record;external-entropy-not-authenticated;"
            "assumption-based-analysis-only"
        ),
        "sampling_seed": sampling_seed,
        "pair_schedule": build_pair_schedule(assignment_seed, sampling_seed),
        "learner": {
            "backend": "tinker",
            "checkpoint_sampling_backend": "tinker",
            "model_name": model_name,
            "renderer_name": renderer_name,
            "lora_rank": DEFAULT_LORA_RANK,
            "initialization_seed": derive_seed("initial-state", sampling_seed),
            "learning_rate": DEFAULT_LEARNING_RATE,
            "loss_function": "importance_sampling",
            "optimizer": "adam",
            "optimizer_beta1": 0.9,
            "optimizer_beta2": 0.95,
            "optimizer_epsilon": 1e-12,
            "reward_normalization": "within-game-grpo-zscore",
            "gamma": 0.99,
        },
        "training": {
            "arm_names": list(ARMS),
            "strata": list(TRAIN_STRATA),
            "games_per_arm": TRAIN_GAMES_PER_ARM,
            "trajectories_per_game": TRAJECTORIES_PER_GAME,
            "episodes_per_update": EPISODES_PER_UPDATE,
            "updates": UPDATES_PER_BRANCH,
            "maximum_turns": MAX_TRAIN_TURNS,
            "actor_temperature": DEFAULT_ACTOR_TEMPERATURE,
            "actor_top_p": DEFAULT_ACTOR_TOP_P,
            "actor_top_k": DEFAULT_ACTOR_TOP_K,
            "actor_max_tokens": DEFAULT_ACTOR_MAX_TOKENS,
            "maximum_concurrent_samples": DEFAULT_MAX_CONCURRENT_SAMPLES,
            "submitted_positions_per_episode": TRAIN_POSITIONS_PER_EPISODE,
            "submitted_positions_per_update": TRAIN_POSITIONS_PER_UPDATE,
            "submitted_positions_per_branch": TRAIN_POSITIONS_PER_BRANCH,
            "padding_advantage": 0.0,
            "failure_policy": "fail-pair-no-upsampling-no-outcome-deletion",
        },
        "evaluation": {
            "heldout_source": "sealed-v4-qualified-selections",
            "heldout_strata": list(HELDOUT_STRATA),
            "seeds_per_game": HELDOUT_SEEDS_PER_GAME,
            "episodes_per_checkpoint": HELDOUT_EPISODES_PER_CHECKPOINT,
            "hints": False,
            "endpoint": "mean-terminal-reward",
            "respect_native_horizon": True,
            "maximum_native_turns": MAX_HELDOUT_NATIVE_TURNS,
            "maximum_context_positions": HELDOUT_CONTEXT_POSITIONS,
            "checkpoint_step": UPDATES_PER_BRANCH,
        },
        "resource_ceilings": {
            "branch_count": PAIR_COUNT * 2,
            "forward_backward_calls": PAIR_COUNT * 2 * UPDATES_PER_BRANCH,
            "optimizer_step_calls": PAIR_COUNT * 2 * UPDATES_PER_BRANCH,
            "submitted_training_positions": PAIR_COUNT * 2 * TRAIN_POSITIONS_PER_BRANCH,
            "maximum_sample_calls_including_canaries": MAX_SAMPLE_CALLS_TOTAL,
            "maximum_completion_tokens_per_sample_call": DEFAULT_ACTOR_MAX_TOKENS,
            "policy": "hard-fail-at-any-fixed-loop-or-token-ceiling",
        },
        "evidence_authentication": {
            "assignment_entropy": "unattested-user-entered-local-record",
            "optimizer_and_checkpoint_receipts": (
                "tinker-sdk-path-only;local-process-self-attestation"
            ),
            "causal_claim_authorized": False,
            "live_execution_authorized": False,
        },
        "analysis": {
            "experimental_unit": "matched-training-run-pair",
            "pair_count": PAIR_COUNT,
            "assignment_mechanism": "six-seeded-within-pair-arm-to-slot-assignments",
            "assignment_provenance": "unattested-user-entered-seed",
            "exact_assignments": EXACT_SIGN_FLIP_COUNT,
            "test": "two-sided-sign-flip-sensitivity-under-within-pair-exchangeability",
            "alpha": 0.05,
            "minimum_effect": DEFAULT_MINIMUM_EFFECT,
            "sensitivity_gate": "p<=0.05-and-mean-effect>=0.05",
            "causal_decision": "permanent-hold-for-this-protocol-version",
        },
        "claim_boundary": (
            "Exploratory paired association for the fixed coverage-forced versus "
            "redundant-historical curricula under a local-process trust and within-pair "
            "exchangeability assumption; no causal learner, promotion, release, active-QD, "
            "proposer-learning, Slime, Google-model, or general-capability claim."
        ),
    }
    return seal_document(body, "intent_digest")


_INTENT_KEYS = {
    "schema_version",
    "protocol_id",
    "source_actor_plan_path",
    "source_actor_plan_digest",
    "pool_manifest_path",
    "pool_manifest_file_digest",
    "pool_manifest_digest",
    "output_root",
    "runtime_identity",
    "service_endpoint",
    "assignment_seed",
    "assignment_seed_record",
    "sampling_seed",
    "pair_schedule",
    "learner",
    "training",
    "evaluation",
    "resource_ceilings",
    "evidence_authentication",
    "analysis",
    "claim_boundary",
    "intent_digest",
}


def validate_intent(
    intent: Mapping[str, Any], *, verify_bound_files: bool = True
) -> dict[str, Any]:
    """Recompute every fixed design field and both source-file bindings."""

    value = validate_seal(intent, schema=INTENT_SCHEMA, exact_keys=_INTENT_KEYS)
    if value.get("protocol_id") != PROTOCOL_ID:
        raise BranchAssayError("intent protocol differs")
    assignment_seed = value.get("assignment_seed")
    sampling_seed = value.get("sampling_seed")
    if value.get("pair_schedule") != build_pair_schedule(assignment_seed, sampling_seed):
        raise BranchAssayError("pair schedule differs from its assignment/sampling seeds")
    if value.get("service_endpoint") != DEFAULT_TINKER_ENDPOINT:
        raise BranchAssayError("Tinker service endpoint differs from the fixed production endpoint")
    if value.get("assignment_seed_record") != (
        "user-entered-local-record;external-entropy-not-authenticated;"
        "assumption-based-analysis-only"
    ):
        raise BranchAssayError("assignment-seed provenance record differs")
    learner = value.get("learner")
    training = value.get("training")
    evaluation = value.get("evaluation")
    analysis = value.get("analysis")
    if not all(isinstance(item, dict) for item in (learner, training, evaluation, analysis)):
        raise BranchAssayError("intent scientific sections must be objects")
    expected_learner = {
        "backend": "tinker",
        "checkpoint_sampling_backend": "tinker",
        "model_name": DEFAULT_MODEL,
        "renderer_name": DEFAULT_RENDERER,
        "lora_rank": DEFAULT_LORA_RANK,
        "initialization_seed": derive_seed("initial-state", sampling_seed),
        "learning_rate": DEFAULT_LEARNING_RATE,
        "loss_function": "importance_sampling",
        "optimizer": "adam",
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.95,
        "optimizer_epsilon": 1e-12,
        "reward_normalization": "within-game-grpo-zscore",
        "gamma": 0.99,
    }
    if learner != expected_learner:
        raise BranchAssayError(
            "learner identity/hyperparameters differ or substitute another sampling backend"
        )
    expected_training = {
        "arm_names": list(ARMS),
        "strata": list(TRAIN_STRATA),
        "games_per_arm": TRAIN_GAMES_PER_ARM,
        "trajectories_per_game": TRAJECTORIES_PER_GAME,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "updates": UPDATES_PER_BRANCH,
        "maximum_turns": MAX_TRAIN_TURNS,
        "actor_temperature": DEFAULT_ACTOR_TEMPERATURE,
        "actor_top_p": DEFAULT_ACTOR_TOP_P,
        "actor_top_k": DEFAULT_ACTOR_TOP_K,
        "actor_max_tokens": DEFAULT_ACTOR_MAX_TOKENS,
        "maximum_concurrent_samples": DEFAULT_MAX_CONCURRENT_SAMPLES,
        "submitted_positions_per_episode": TRAIN_POSITIONS_PER_EPISODE,
        "submitted_positions_per_update": TRAIN_POSITIONS_PER_UPDATE,
        "submitted_positions_per_branch": TRAIN_POSITIONS_PER_BRANCH,
        "padding_advantage": 0.0,
        "failure_policy": "fail-pair-no-upsampling-no-outcome-deletion",
    }
    if training != expected_training:
        raise BranchAssayError("training design differs from the fixed protocol")
    expected_evaluation = {
        "heldout_source": "sealed-v4-qualified-selections",
        "heldout_strata": list(HELDOUT_STRATA),
        "seeds_per_game": HELDOUT_SEEDS_PER_GAME,
        "episodes_per_checkpoint": HELDOUT_EPISODES_PER_CHECKPOINT,
        "hints": False,
        "endpoint": "mean-terminal-reward",
        "respect_native_horizon": True,
        "maximum_native_turns": MAX_HELDOUT_NATIVE_TURNS,
        "maximum_context_positions": HELDOUT_CONTEXT_POSITIONS,
        "checkpoint_step": UPDATES_PER_BRANCH,
    }
    if evaluation != expected_evaluation:
        raise BranchAssayError("held-out endpoint differs")
    expected_resource_ceilings = {
        "branch_count": PAIR_COUNT * 2,
        "forward_backward_calls": PAIR_COUNT * 2 * UPDATES_PER_BRANCH,
        "optimizer_step_calls": PAIR_COUNT * 2 * UPDATES_PER_BRANCH,
        "submitted_training_positions": PAIR_COUNT * 2 * TRAIN_POSITIONS_PER_BRANCH,
        "maximum_sample_calls_including_canaries": MAX_SAMPLE_CALLS_TOTAL,
        "maximum_completion_tokens_per_sample_call": DEFAULT_ACTOR_MAX_TOKENS,
        "policy": "hard-fail-at-any-fixed-loop-or-token-ceiling",
    }
    if value.get("resource_ceilings") != expected_resource_ceilings:
        raise BranchAssayError("resource ceilings differ from the fixed protocol")
    expected_evidence_authentication = {
        "assignment_entropy": "unattested-user-entered-local-record",
        "optimizer_and_checkpoint_receipts": (
            "tinker-sdk-path-only;local-process-self-attestation"
        ),
        "causal_claim_authorized": False,
        "live_execution_authorized": False,
    }
    if value.get("evidence_authentication") != expected_evidence_authentication:
        raise BranchAssayError("evidence-authentication HOLD boundary differs")
    expected_analysis = {
        "experimental_unit": "matched-training-run-pair",
        "pair_count": PAIR_COUNT,
        "assignment_mechanism": "six-seeded-within-pair-arm-to-slot-assignments",
        "assignment_provenance": "unattested-user-entered-seed",
        "exact_assignments": EXACT_SIGN_FLIP_COUNT,
        "test": "two-sided-sign-flip-sensitivity-under-within-pair-exchangeability",
        "alpha": 0.05,
        "minimum_effect": DEFAULT_MINIMUM_EFFECT,
        "sensitivity_gate": "p<=0.05-and-mean-effect>=0.05",
        "causal_decision": "permanent-hold-for-this-protocol-version",
    }
    if analysis != expected_analysis:
        raise BranchAssayError("analysis topology or decision gate differs")
    if not isinstance(value.get("runtime_identity"), dict) or not value["runtime_identity"]:
        raise BranchAssayError("runtime identity is absent")
    output_root = Path(value.get("output_root", ""))
    if (
        not output_root.is_absolute()
        or output_root.is_symlink()
        or output_root.resolve() != output_root
    ):
        raise BranchAssayError("output_root is unsafe")
    if verify_bound_files:
        _require_absolute_file(
            value.get("pool_manifest_path"),
            value.get("pool_manifest_file_digest"),
            "pool_manifest_file",
        )
        pool_manifest = read_json(Path(value["pool_manifest_path"]))
        pool_digest = value.get("pool_manifest_digest")
        if (
            pool_digest != AUTHORIZED_POOL_MANIFEST_DIGEST
            or pool_manifest.get("manifest_digest") != pool_digest
            or object_digest(
                {key: item for key, item in pool_manifest.items() if key != "manifest_digest"}
            )
            != pool_digest
        ):
            raise BranchAssayError("bound pool manifest semantic digest differs")
        actor_path = Path(str(value.get("source_actor_plan_path", "")))
        actor = read_json(actor_path)
        actor_digest = value.get("source_actor_plan_digest")
        if actor.get("actor_plan_digest") != actor_digest:
            raise BranchAssayError("bound actor-plan digest differs")
        body = {key: item for key, item in actor.items() if key != "actor_plan_digest"}
        if object_digest(body) != actor_digest:
            raise BranchAssayError("bound actor-plan self-digest mismatch")
    return value


class CommonStateBoundary(Protocol):
    """Injected live boundary used only by :func:`prepare_common_base_state`."""

    async def create_and_save_common_state(
        self,
        *,
        service_endpoint: str,
        model_name: str,
        lora_rank: int,
        initialization_seed: int,
        state_name: str,
        metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """Create one LoRA state and save weights plus optimizer state."""


class TrainingBoundary(Protocol):
    """Injected same-checkpoint Tinker boundary for one learner branch."""

    @property
    def pad_token_id(self) -> int:
        """Tokenizer pad token used only in zero-advantage compute padding."""

    async def attest_runtime(self) -> Mapping[str, Any]:
        """Return the current SDK/model/renderer capability receipt."""

    async def restore_with_optimizer(
        self, *, state_uri: str, metadata: Mapping[str, str]
    ) -> object:
        """Create an independent client from the common optimizer-bearing state."""

    async def canary_token_digest(self, client: object, *, seed: int) -> str:
        """Return the deterministic public canary token digest for this state."""

    async def collect_training_rollouts(
        self,
        client: object,
        *,
        pool_dir: Path,
        pool_entries: Sequence[Mapping[str, Any]],
        pair_id: str,
        pair_seed: int,
        update: int,
    ) -> Sequence[EpisodeRollout | Mapping[str, Any]]:
        """Collect the exact 12x8 on-policy trajectories for one update."""

    async def train_update(
        self,
        client: object,
        *,
        datums: Sequence[PaddedEpisodeDatum],
        update: int,
        learning_rate: float,
    ) -> Mapping[str, Any]:
        """Submit one forward/backward and exactly one optimizer step."""

    async def save_checkpoint(self, client: object, *, name: str) -> Mapping[str, Any]:
        """Save optimizer-bearing state plus sampling weights after an update."""

    async def evaluate_checkpoint(
        self,
        *,
        sampler_uri: str,
        heldout_dir: Path,
        heldout_entries: Sequence[Mapping[str, Any]],
        pair_id: str,
        pair_seed: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Evaluate the final learned checkpoint on the sealed held-out panel."""


_RUNTIME_ATTESTATION_KEYS = {
    "backend",
    "service_endpoint",
    "model_name",
    "renderer_name",
    "sdk_version",
    "server_capabilities_digest",
}


def validate_runtime_attestation(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject dependency, route, or renderer drift before branch restoration."""

    if not isinstance(value, Mapping) or set(value) != _RUNTIME_ATTESTATION_KEYS:
        raise BranchAssayError("Tinker runtime attestation differs from its exact schema")
    result = dict(value)
    learner = intent["learner"]
    runtime = intent.get("runtime_identity")
    try:
        sealed_sdk_version = runtime["distributions"]["tinker"]["version"]
    except (KeyError, TypeError) as exc:
        raise BranchAssayError("intent lacks a sealed Tinker SDK distribution version") from exc
    if not isinstance(sealed_sdk_version, str) or not sealed_sdk_version:
        raise BranchAssayError("intent Tinker SDK distribution version is invalid")
    expected = {
        "backend": "tinker",
        "service_endpoint": intent["service_endpoint"],
        "model_name": learner["model_name"],
        "renderer_name": learner["renderer_name"],
        "sdk_version": sealed_sdk_version,
        "server_capabilities_digest": base_receipt["server_capabilities_digest"],
    }
    if result != expected:
        raise BranchAssayError("Tinker runtime differs from the common-state receipt")
    return result


_BASE_REQUEST_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "backend",
    "service_endpoint",
    "model_name",
    "lora_rank",
    "initialization_seed",
    "sdk_version",
    "state_name",
    "operation",
    "retry_policy",
    "request_digest",
}
_BASE_RESULT_KEYS = {
    "schema_version",
    "protocol_id",
    "request_digest",
    "status",
    "backend",
    "service_endpoint",
    "model_name",
    "lora_rank",
    "initialization_seed",
    "state_uri",
    "state_uri_digest",
    "optimizer_state_included",
    "sdk_version",
    "server_capabilities_digest",
    "canary_token_digest",
    "public_response_metadata",
    "error_type",
    "result_digest",
}
_BASE_RECEIPT_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "request_digest",
    "result_digest",
    "backend",
    "service_endpoint",
    "model_name",
    "lora_rank",
    "initialization_seed",
    "state_uri",
    "state_uri_digest",
    "optimizer_state_included",
    "sdk_version",
    "server_capabilities_digest",
    "canary_token_digest",
    "receipt_digest",
}


def _validate_public_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise BranchAssayError("public_response_metadata must be an object")
    denied = {"api_key", "authorization", "password", "secret", "credential", "cookie"}
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key.lower() in denied:
            raise BranchAssayError("base-state result contains a forbidden metadata key")
        if not isinstance(item, str) or len(item.encode("utf-8")) > 2048:
            raise BranchAssayError("base-state public metadata values must be bounded text")
        output[key] = item
    return output


def _validate_tinker_state_uri(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("tinker://")
        or len(value) > 2048
        or any(marker in value for marker in ("?", "#", "@", "\n", "\r"))
    ):
        raise BranchAssayError("common state URI is missing or unsafe")
    return value


def _base_request(intent: Mapping[str, Any]) -> dict[str, Any]:
    learner = intent["learner"]
    try:
        sdk_version = intent["runtime_identity"]["distributions"]["tinker"]["version"]
    except (KeyError, TypeError) as exc:
        raise BranchAssayError("intent lacks a sealed Tinker SDK distribution version") from exc
    if not isinstance(sdk_version, str) or not sdk_version:
        raise BranchAssayError("intent Tinker SDK distribution version is invalid")
    body = {
        "schema_version": BASE_REQUEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": intent["intent_digest"],
        "backend": "tinker",
        "service_endpoint": intent["service_endpoint"],
        "model_name": learner["model_name"],
        "lora_rank": learner["lora_rank"],
        "initialization_seed": learner["initialization_seed"],
        "sdk_version": sdk_version,
        "state_name": f"spade-learner-common-{intent['intent_digest'][7:19]}",
        "operation": "create_lora_then_save_optimizer_bearing_state",
        "retry_policy": "never-retry-after-request-persistence",
    }
    return seal_document(body, "request_digest")


def validate_base_result(result: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_seal(result, schema=BASE_RESULT_SCHEMA, exact_keys=_BASE_RESULT_KEYS)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "request_digest": request["request_digest"],
        "backend": request["backend"],
        "service_endpoint": request["service_endpoint"],
        "model_name": request["model_name"],
        "lora_rank": request["lora_rank"],
        "initialization_seed": request["initialization_seed"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise BranchAssayError("base-state result does not bind its request")
    status = value.get("status")
    if status not in {"success", "ambiguous"}:
        raise BranchAssayError("base-state result status is invalid")
    _validate_public_metadata(value.get("public_response_metadata"))
    if status == "ambiguous":
        if any(
            value.get(key) is not None
            for key in (
                "state_uri",
                "state_uri_digest",
                "optimizer_state_included",
                "sdk_version",
                "server_capabilities_digest",
                "canary_token_digest",
            )
        ):
            raise BranchAssayError("ambiguous base-state result claims usable state")
        if not isinstance(value.get("error_type"), str) or not value["error_type"]:
            raise BranchAssayError("ambiguous base-state result lacks error type")
        return value
    uri = _validate_tinker_state_uri(value.get("state_uri"))
    if value.get("state_uri_digest") != bytes_digest(uri.encode("utf-8")):
        raise BranchAssayError("common state URI digest mismatch")
    if value.get("optimizer_state_included") is not True:
        raise BranchAssayError("common state is not optimizer-bearing")
    if not isinstance(value.get("sdk_version"), str) or not value["sdk_version"]:
        raise BranchAssayError("base-state result lacks SDK version")
    if value["sdk_version"] != request["sdk_version"]:
        raise BranchAssayError("base-state result SDK version differs from its request")
    for field in ("server_capabilities_digest", "canary_token_digest"):
        if not _is_digest(value.get(field)):
            raise BranchAssayError(f"base-state result lacks {field}")
    if value.get("error_type") is not None:
        raise BranchAssayError("successful base-state result contains an error")
    return value


def _base_receipt(
    intent: Mapping[str, Any], request: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "schema_version": BASE_RECEIPT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": intent["intent_digest"],
        "request_digest": request["request_digest"],
        "result_digest": result["result_digest"],
        "backend": result["backend"],
        "service_endpoint": result["service_endpoint"],
        "model_name": result["model_name"],
        "lora_rank": result["lora_rank"],
        "initialization_seed": result["initialization_seed"],
        "state_uri": result["state_uri"],
        "state_uri_digest": result["state_uri_digest"],
        "optimizer_state_included": result["optimizer_state_included"],
        "sdk_version": result["sdk_version"],
        "server_capabilities_digest": result["server_capabilities_digest"],
        "canary_token_digest": result["canary_token_digest"],
    }
    return seal_document(body, "receipt_digest")


def validate_base_receipt(
    receipt: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    request: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = validate_seal(receipt, schema=BASE_RECEIPT_SCHEMA, exact_keys=_BASE_RECEIPT_KEYS)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("intent_digest") != intent.get(
        "intent_digest"
    ):
        raise BranchAssayError("base-state receipt does not bind the intent")
    uri = _validate_tinker_state_uri(value.get("state_uri"))
    if value.get("state_uri_digest") != bytes_digest(uri.encode("utf-8")):
        raise BranchAssayError("base-state receipt URI digest mismatch")
    learner = intent["learner"]
    for field in ("model_name", "lora_rank", "initialization_seed"):
        if value.get(field) != learner.get(field):
            raise BranchAssayError(f"base-state receipt {field} differs from intent")
    if (
        value.get("backend") != learner.get("backend")
        or value.get("service_endpoint") != intent.get("service_endpoint")
        or value.get("optimizer_state_included") is not True
    ):
        raise BranchAssayError("base-state receipt backend/state semantics differ")
    try:
        sealed_sdk_version = intent["runtime_identity"]["distributions"]["tinker"]["version"]
    except (KeyError, TypeError) as exc:
        raise BranchAssayError("intent lacks a sealed Tinker SDK distribution version") from exc
    if value.get("sdk_version") != sealed_sdk_version:
        raise BranchAssayError("base-state receipt SDK version differs from intent")
    if request is not None and value.get("request_digest") != request.get("request_digest"):
        raise BranchAssayError("base-state receipt request link differs")
    if result is not None:
        if result.get("status") != "success" or value.get("result_digest") != result.get(
            "result_digest"
        ):
            raise BranchAssayError("base-state receipt result link differs")
        for field in (
            "backend",
            "service_endpoint",
            "model_name",
            "lora_rank",
            "initialization_seed",
            "state_uri",
            "state_uri_digest",
            "optimizer_state_included",
            "sdk_version",
            "server_capabilities_digest",
            "canary_token_digest",
        ):
            if value.get(field) != result.get(field):
                raise BranchAssayError(f"base-state receipt/result {field} differs")
    return value


def _validated_run_dir(intent: Mapping[str, Any], run_dir: Path) -> Path:
    if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.resolve() != run_dir:
        raise BranchAssayError("run_dir must be an absolute non-symlink path")
    output_root = Path(intent["output_root"])
    root = output_root.resolve()
    candidate = run_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BranchAssayError("run_dir escapes the sealed output root") from exc
    current = candidate
    while current != root:
        if current.exists() and current.is_symlink():
            raise BranchAssayError("run_dir contains a symlink component")
        current = current.parent
    return candidate


async def prepare_common_base_state(
    *,
    intent: Mapping[str, Any],
    run_dir: Path,
    boundary: CommonStateBoundary,
) -> dict[str, Any]:
    """Make the sole common initial state with fail-closed call chronology."""

    valid_intent = validate_intent(intent)
    root = _validated_run_dir(valid_intent, run_dir)
    evidence = root / "common-base-state"
    request_path = evidence / "request.json"
    result_path = evidence / "result.json"
    receipt_path = evidence / "receipt.json"
    presence = tuple(path.exists() for path in (request_path, result_path, receipt_path))
    if presence == (True, True, True):
        request = validate_seal(
            read_json(request_path), schema=BASE_REQUEST_SCHEMA, exact_keys=_BASE_REQUEST_KEYS
        )
        result = validate_base_result(read_json(result_path), request)
        if result["status"] != "success":
            raise BranchAssayError("base-state call ended ambiguous; use a fresh sealed run")
        return validate_base_receipt(
            read_json(receipt_path), intent=valid_intent, request=request, result=result
        )
    if any(presence):
        raise BranchAssayError("base-state evidence is partial or ambiguous; do not retry")

    request = _base_request(valid_intent)
    _atomic_write_json(request_path, request)
    learner = valid_intent["learner"]
    try:
        raw = await boundary.create_and_save_common_state(
            service_endpoint=valid_intent["service_endpoint"],
            model_name=learner["model_name"],
            lora_rank=learner["lora_rank"],
            initialization_seed=learner["initialization_seed"],
            state_name=request["state_name"],
            metadata={
                "protocol_id": PROTOCOL_ID,
                "intent_digest": valid_intent["intent_digest"],
                "purpose": "common-optimizer-bearing-initial-state",
            },
        )
        if raw.get("service_endpoint") != valid_intent["service_endpoint"]:
            raise BranchAssayError("common-state boundary used a different service endpoint")
        public_metadata = _validate_public_metadata(raw.get("public_response_metadata", {}))
        uri = _validate_tinker_state_uri(raw.get("state_uri"))
        result_body = {
            "schema_version": BASE_RESULT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "request_digest": request["request_digest"],
            "status": "success",
            "backend": "tinker",
            "service_endpoint": valid_intent["service_endpoint"],
            "model_name": learner["model_name"],
            "lora_rank": learner["lora_rank"],
            "initialization_seed": learner["initialization_seed"],
            "state_uri": uri,
            "state_uri_digest": bytes_digest(uri.encode("utf-8")),
            "optimizer_state_included": raw.get("optimizer_state_included"),
            "sdk_version": raw.get("sdk_version"),
            "server_capabilities_digest": raw.get("server_capabilities_digest"),
            "canary_token_digest": raw.get("canary_token_digest"),
            "public_response_metadata": public_metadata,
            "error_type": None,
        }
    except Exception as exc:
        result_body = {
            "schema_version": BASE_RESULT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "request_digest": request["request_digest"],
            "status": "ambiguous",
            "backend": "tinker",
            "service_endpoint": valid_intent["service_endpoint"],
            "model_name": learner["model_name"],
            "lora_rank": learner["lora_rank"],
            "initialization_seed": learner["initialization_seed"],
            "state_uri": None,
            "state_uri_digest": None,
            "optimizer_state_included": None,
            "sdk_version": None,
            "server_capabilities_digest": None,
            "canary_token_digest": None,
            "public_response_metadata": {},
            "error_type": type(exc).__name__,
        }
        result = seal_document(result_body, "result_digest")
        _atomic_write_json(result_path, result)
        raise BranchAssayError(
            "base-state boundary failed after reservation; state is ambiguous and cannot be retried"
        ) from exc

    result = seal_document(result_body, "result_digest")
    validate_base_result(result, request)
    _atomic_write_json(result_path, result)
    receipt = _base_receipt(valid_intent, request, result)
    validate_base_receipt(receipt, intent=valid_intent, request=request, result=result)
    _atomic_write_json(receipt_path, receipt)
    return receipt


def validate_common_base_tree(*, intent: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    root = _validated_run_dir(valid_intent, run_dir) / "common-base-state"
    expected = {"request.json", "result.json", "receipt.json"}
    if not root.is_dir() or root.is_symlink():
        raise BranchAssayError("common base-state evidence directory is missing or unsafe")
    observed = {path.name for path in root.iterdir()}
    if observed != expected or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise BranchAssayError("common base-state evidence inventory differs")
    request = validate_seal(
        read_json(root / "request.json"),
        schema=BASE_REQUEST_SCHEMA,
        exact_keys=_BASE_REQUEST_KEYS,
    )
    if request != _base_request(valid_intent):
        raise BranchAssayError("common base-state request differs from intent")
    result = validate_base_result(read_json(root / "result.json"), request)
    if result["status"] != "success":
        raise BranchAssayError("common base-state result is not successful")
    return validate_base_receipt(
        read_json(root / "receipt.json"),
        intent=valid_intent,
        request=request,
        result=result,
    )


def branch_requests(
    intent: Mapping[str, Any], base_receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Materialize all twelve branches from the one sealed common state."""

    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    requests: list[dict[str, Any]] = []
    for pair in valid_intent["pair_schedule"]:
        for slot_id in pair["execution_order"]:
            arm = pair["slot_assignments"][slot_id]
            body = {
                "schema_version": BRANCH_REQUEST_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "intent_digest": valid_intent["intent_digest"],
                "base_receipt_digest": receipt["receipt_digest"],
                "common_state_uri": receipt["state_uri"],
                "common_state_uri_digest": receipt["state_uri_digest"],
                "restore_method": "create_training_client_from_state_with_optimizer_async",
                "pair_id": pair["pair_id"],
                "slot_id": slot_id,
                "arm": arm,
                "pair_seed": pair["pair_seed"],
                "pool_manifest_file_digest": valid_intent["pool_manifest_file_digest"],
                "updates": UPDATES_PER_BRANCH,
                "episodes_per_update": EPISODES_PER_UPDATE,
                "submitted_positions_per_update": TRAIN_POSITIONS_PER_UPDATE,
                "submitted_positions_total": TRAIN_POSITIONS_PER_BRANCH,
                "seed_derivation": (
                    "sha256-domain-separated-pair-update-game-replicate-turn;"
                    "arm-and-slot-excluded-for-pairing/v1"
                ),
                "failure_policy": "fail-whole-pair-no-retry-after-remote-training-request",
            }
            requests.append(seal_document(body, "request_digest"))
    if len(requests) != PAIR_COUNT * 2:
        raise AssertionError("fixed branch topology was not materialized")
    return tuple(requests)


_BRANCH_REQUEST_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "base_receipt_digest",
    "common_state_uri",
    "common_state_uri_digest",
    "restore_method",
    "pair_id",
    "slot_id",
    "arm",
    "pair_seed",
    "pool_manifest_file_digest",
    "updates",
    "episodes_per_update",
    "submitted_positions_per_update",
    "submitted_positions_total",
    "seed_derivation",
    "failure_policy",
    "request_digest",
}


def validate_branch_request(
    request: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    value = validate_seal(request, schema=BRANCH_REQUEST_SCHEMA, exact_keys=_BRANCH_REQUEST_KEYS)
    candidates = {
        (item["pair_id"], item["slot_id"]): item for item in branch_requests(valid_intent, receipt)
    }
    key = (value.get("pair_id"), value.get("slot_id"))
    if key not in candidates or value != candidates[key]:
        raise BranchAssayError("branch request differs from the prospective schedule")
    return value


@dataclass(frozen=True)
class PaddedEpisodeDatum:
    """Backend-neutral importance-sampling datum with fixed submitted length."""

    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    advantages: tuple[float, ...]
    actual_positions: int
    action_positions: int
    padded_positions: int

    @property
    def submitted_positions(self) -> int:
        return len(self.input_tokens)


@dataclass(frozen=True)
class EpisodeRollout:
    """One successful service/environment rollout before GRPO normalization."""

    game_basename: str
    replicate: int
    environment_seed: int
    sampling_seeds: tuple[int, ...]
    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    action_logprobs: tuple[float, ...]
    action_turn_indices: tuple[int, ...]
    raw_reward: float
    turn_count: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_basename": self.game_basename,
            "replicate": self.replicate,
            "environment_seed": self.environment_seed,
            "sampling_seeds": list(self.sampling_seeds),
            "tokens": list(self.tokens),
            "loss_mask": list(self.loss_mask),
            "action_logprobs": list(self.action_logprobs),
            "action_turn_indices": list(self.action_turn_indices),
            "raw_reward": self.raw_reward,
            "turn_count": self.turn_count,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeRollout":
        expected = {
            "game_basename",
            "replicate",
            "environment_seed",
            "sampling_seeds",
            "tokens",
            "loss_mask",
            "action_logprobs",
            "action_turn_indices",
            "raw_reward",
            "turn_count",
            "status",
        }
        if set(value) != expected:
            raise BranchAssayError("episode rollout differs from its exact schema")
        try:
            result = cls(
                game_basename=value["game_basename"],
                replicate=value["replicate"],
                environment_seed=value["environment_seed"],
                sampling_seeds=tuple(value["sampling_seeds"]),
                tokens=tuple(value["tokens"]),
                loss_mask=tuple(value["loss_mask"]),
                action_logprobs=tuple(value["action_logprobs"]),
                action_turn_indices=tuple(value["action_turn_indices"]),
                raw_reward=value["raw_reward"],
                turn_count=value["turn_count"],
                status=value["status"],
            )
        except (KeyError, TypeError) as exc:
            raise BranchAssayError(f"episode rollout is malformed: {exc}") from exc
        return result


def training_seed(
    *,
    pair_id: str,
    pair_seed: int,
    update: int,
    game_basename: str,
    replicate: int,
    turn: int | None = None,
) -> int:
    if update not in range(UPDATES_PER_BRANCH):
        raise BranchAssayError("training update is outside the sealed schedule")
    if replicate not in range(TRAJECTORIES_PER_GAME):
        raise BranchAssayError("training replicate is outside the sealed schedule")
    if turn is not None and turn not in range(MAX_TRAIN_TURNS):
        raise BranchAssayError("training turn is outside the sealed schedule")
    purpose = "train-env" if turn is None else f"train-sampling-turn-{turn}"
    return derive_seed(purpose, pair_id, pair_seed, update, game_basename, replicate)


def pad_episode_datum(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[int],
    action_logprobs: Sequence[float],
    action_advantages: Sequence[float],
    pad_token_id: int,
    submitted_positions: int = TRAIN_POSITIONS_PER_EPISODE,
) -> PaddedEpisodeDatum:
    """Right-pad one full episode to an exact next-token training budget.

    ``loss_mask`` is aligned to ``tokens`` and marks assistant/action tokens.
    The first token cannot be an action because next-token loss has no preceding
    model input. Padding has zero log probability and zero advantage, so it
    contributes compute positions but no gradient.
    """

    if isinstance(submitted_positions, bool) or not isinstance(submitted_positions, int):
        raise BranchAssayError("submitted_positions must be an integer")
    if submitted_positions <= 0:
        raise BranchAssayError("submitted_positions must be positive")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise BranchAssayError("pad_token_id must be a nonnegative integer")
    token_list = list(tokens)
    mask_list = list(loss_mask)
    if len(token_list) < 2 or len(mask_list) != len(token_list):
        raise BranchAssayError("episode tokens and aligned loss mask are incomplete")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_list
    ):
        raise BranchAssayError("episode tokens must be nonnegative integers")
    if any(mask not in (0, 1) or isinstance(mask, bool) for mask in mask_list):
        raise BranchAssayError("loss_mask must contain only integer zero/one values")
    if mask_list[0] != 0:
        raise BranchAssayError("the first episode token cannot be an action target")
    action_count = sum(mask_list)
    if action_count == 0:
        raise BranchAssayError("episode has no trainable action tokens")
    if len(action_logprobs) != action_count or len(action_advantages) != action_count:
        raise BranchAssayError("action logprobs/advantages do not match the action mask")
    for label, values in (
        ("action_logprobs", action_logprobs),
        ("action_advantages", action_advantages),
    ):
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
            raise BranchAssayError(f"{label} must be numeric")
        if any(not math.isfinite(float(item)) for item in values):
            raise BranchAssayError(f"{label} must be finite")
    actual_positions = len(token_list) - 1
    if actual_positions > submitted_positions:
        raise BranchAssayError(
            f"episode requires {actual_positions} positions, exceeding sealed budget "
            f"{submitted_positions}"
        )

    pad_count = submitted_positions - actual_positions
    full_tokens = token_list + [pad_token_id] * pad_count
    full_mask = mask_list + [0] * pad_count
    # The vectors index target tokens, hence the one-token shift.
    target_mask = full_mask[1:]
    logprob_iter = iter(float(item) for item in action_logprobs)
    advantage_iter = iter(float(item) for item in action_advantages)
    full_logprobs: list[float] = []
    full_advantages: list[float] = []
    for is_action in target_mask:
        if is_action:
            full_logprobs.append(next(logprob_iter))
            full_advantages.append(next(advantage_iter))
        else:
            full_logprobs.append(0.0)
            full_advantages.append(0.0)
    try:
        next(logprob_iter)
        raise AssertionError("unconsumed action log probability")
    except StopIteration:
        pass
    try:
        next(advantage_iter)
        raise AssertionError("unconsumed action advantage")
    except StopIteration:
        pass

    datum = PaddedEpisodeDatum(
        input_tokens=tuple(full_tokens[:-1]),
        target_tokens=tuple(full_tokens[1:]),
        logprobs=tuple(full_logprobs),
        advantages=tuple(full_advantages),
        actual_positions=actual_positions,
        action_positions=action_count,
        padded_positions=pad_count,
    )
    if not all(
        len(value) == submitted_positions
        for value in (
            datum.input_tokens,
            datum.target_tokens,
            datum.logprobs,
            datum.advantages,
        )
    ):
        raise AssertionError("padded datum length invariant failed")
    return datum


def normalize_and_pad_training_rollouts(
    *,
    rollouts: Sequence[EpisodeRollout | Mapping[str, Any]],
    expected_basenames: Sequence[str],
    pair_id: str,
    pair_seed: int,
    update: int,
    pad_token_id: int,
    gamma: float = 0.99,
) -> tuple[dict[str, tuple[PaddedEpisodeDatum, ...]], dict[str, Any]]:
    """Validate a complete rollout batch, normalize per game, and pad exactly."""

    basenames = tuple(expected_basenames)
    if (
        len(basenames) != TRAIN_GAMES_PER_ARM
        or len(set(basenames)) != TRAIN_GAMES_PER_ARM
        or any(
            Path(name).name != name or not name.startswith("game_") or not name.endswith(".py")
            for name in basenames
        )
    ):
        raise BranchAssayError("expected training basenames must be twelve safe unique slots")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0.0 <= gamma <= 1.0:
        raise BranchAssayError("gamma must be in [0,1]")
    converted = [
        item if isinstance(item, EpisodeRollout) else EpisodeRollout.from_mapping(item)
        for item in rollouts
    ]
    if len(converted) != EPISODES_PER_UPDATE:
        raise BranchAssayError("training rollout count differs; upsampling is forbidden")
    by_key: dict[tuple[str, int], EpisodeRollout] = {}
    for item in converted:
        key = (item.game_basename, item.replicate)
        if key in by_key:
            raise BranchAssayError("training rollout key is duplicated")
        if item.game_basename not in basenames or item.replicate not in range(
            TRAJECTORIES_PER_GAME
        ):
            raise BranchAssayError("training rollout key is outside the sealed batch")
        expected_env_seed = training_seed(
            pair_id=pair_id,
            pair_seed=pair_seed,
            update=update,
            game_basename=item.game_basename,
            replicate=item.replicate,
        )
        expected_sampling = tuple(
            training_seed(
                pair_id=pair_id,
                pair_seed=pair_seed,
                update=update,
                game_basename=item.game_basename,
                replicate=item.replicate,
                turn=turn,
            )
            for turn in range(item.turn_count)
        )
        if item.environment_seed != expected_env_seed or item.sampling_seeds != expected_sampling:
            raise BranchAssayError("training rollout seeds differ from the paired schedule")
        if (
            isinstance(item.turn_count, bool)
            or not isinstance(item.turn_count, int)
            or item.turn_count not in range(1, MAX_TRAIN_TURNS + 1)
        ):
            raise BranchAssayError("training rollout turn count differs")
        if item.status not in {"completed", "truncated", "timeout"}:
            raise BranchAssayError("failed/provider-error training rollouts cannot be substituted")
        if (
            isinstance(item.raw_reward, bool)
            or not isinstance(item.raw_reward, (int, float))
            or not math.isfinite(float(item.raw_reward))
            or not 0.0 <= float(item.raw_reward) <= 1.0
        ):
            raise BranchAssayError("training terminal reward must be finite in [0,1]")
        if len(item.loss_mask) != len(item.tokens) or item.loss_mask[0] != 0:
            raise BranchAssayError("training rollout token mask is not fully aligned")
        action_count = sum(item.loss_mask)
        if (
            len(item.action_logprobs) != action_count
            or len(item.action_turn_indices) != action_count
        ):
            raise BranchAssayError("training rollout action metadata differs")
        if any(turn not in range(item.turn_count) for turn in item.action_turn_indices):
            raise BranchAssayError("training action turn index differs")
        by_key[key] = item
    expected_keys = {
        (basename, replicate)
        for basename in basenames
        for replicate in range(TRAJECTORIES_PER_GAME)
    }
    if set(by_key) != expected_keys:
        raise BranchAssayError("training rollout topology differs")

    padded: dict[str, tuple[PaddedEpisodeDatum, ...]] = {}
    raw_group_means: dict[str, float] = {}
    mixed_game_count = 0
    for basename in basenames:
        group = [by_key[(basename, replicate)] for replicate in range(TRAJECTORIES_PER_GAME)]
        rewards = [float(item.raw_reward) for item in group]
        mean = math.fsum(rewards) / len(rewards)
        variance = math.fsum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        standard_deviation = math.sqrt(variance)
        if standard_deviation > 0.0:
            mixed_game_count += 1
        raw_group_means[basename] = mean
        game_datums: list[PaddedEpisodeDatum] = []
        for item in group:
            normalized_reward = (float(item.raw_reward) - mean) / (standard_deviation + 1e-8)
            action_advantages = tuple(
                normalized_reward * (float(gamma) ** (item.turn_count - turn - 1))
                for turn in item.action_turn_indices
            )
            game_datums.append(
                pad_episode_datum(
                    tokens=item.tokens,
                    loss_mask=item.loss_mask,
                    action_logprobs=item.action_logprobs,
                    action_advantages=action_advantages,
                    pad_token_id=pad_token_id,
                )
            )
        padded[basename] = tuple(game_datums)
    audit = audit_training_batch(padded)
    audit.update(
        {
            "mixed_reward_games": mixed_game_count,
            "raw_group_means": raw_group_means,
        }
    )
    return padded, audit


@dataclass(frozen=True)
class BranchExecution:
    """Validated in-memory result of one complete 16-update learner branch."""

    score: Mapping[str, Any]
    update_audits: tuple[Mapping[str, Any], ...]
    checkpoint_receipts: tuple[Mapping[str, Any], ...]
    canary_token_digest: str


def _pool_section(
    pool_manifest: Mapping[str, Any], arm: str
) -> tuple[str, list[Mapping[str, Any]]]:
    pools = pool_manifest.get("pools")
    if not isinstance(pools, Mapping) or arm not in pools:
        raise BranchAssayError(f"pool manifest omits {arm}")
    pool = pools[arm]
    if not isinstance(pool, Mapping):
        raise BranchAssayError(f"pool manifest {arm} section is invalid")
    relative_dir = pool.get("relative_dir")
    entries = pool.get("entries")
    if relative_dir != arm or not isinstance(entries, list) or len(entries) != TRAIN_GAMES_PER_ARM:
        raise BranchAssayError(f"pool manifest {arm} topology differs")
    if any(not isinstance(item, Mapping) for item in entries):
        raise BranchAssayError(f"pool manifest {arm} entry is invalid")
    basenames = [item.get("basename") for item in entries]
    if len(set(basenames)) != TRAIN_GAMES_PER_ARM:
        raise BranchAssayError(f"pool manifest {arm} basenames differ")
    return relative_dir, list(entries)


def _checkpoint_uris(value: Mapping[str, Any], *, update: int) -> tuple[str, str]:
    exact = {"state_uri", "sampler_uri", "completed_update", "public_metrics"}
    if set(value) != exact or value.get("completed_update") != update + 1:
        raise BranchAssayError("checkpoint receipt differs from the completed update")
    state_uri = _validate_tinker_state_uri(value.get("state_uri"))
    sampler_uri = _validate_tinker_state_uri(value.get("sampler_uri"))
    _validate_public_metadata(value.get("public_metrics"))
    return state_uri, sampler_uri


async def execute_branch(
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    branch_request: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    pool_root: Path,
    boundary: TrainingBoundary,
) -> BranchExecution:
    """Run one fixed branch through its final direct held-out evaluation.

    This function contains no SDK import. Production evidence persistence and
    exactly-once reservations are supplied by the CLI boundary; unit tests can
    inject a fully offline boundary.
    """

    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    request = validate_branch_request(branch_request, intent=valid_intent, base_receipt=receipt)
    if not pool_root.is_absolute() or pool_root.is_symlink() or not pool_root.is_dir():
        raise BranchAssayError("pool root is missing or unsafe")
    relative_dir, entries = _pool_section(pool_manifest, request["arm"])
    pool_dir = pool_root / relative_dir
    if pool_dir.is_symlink() or not pool_dir.is_dir():
        raise BranchAssayError("branch pool directory is missing or unsafe")
    basenames = [str(item["basename"]) for item in entries]
    observed = {path.name for path in pool_dir.iterdir()}
    if observed != set(basenames) or any(
        path.is_symlink() or not path.is_file() for path in pool_dir.iterdir()
    ):
        raise BranchAssayError("branch pool inventory differs")

    runtime_attestation = await boundary.attest_runtime()
    validate_runtime_attestation(
        runtime_attestation,
        intent=valid_intent,
        base_receipt=receipt,
    )
    client = await boundary.restore_with_optimizer(
        state_uri=receipt["state_uri"],
        metadata={
            "protocol_id": PROTOCOL_ID,
            "intent_digest": valid_intent["intent_digest"],
            "branch_request_digest": request["request_digest"],
            "pair_id": request["pair_id"],
            "slot_id": request["slot_id"],
            "arm": request["arm"],
        },
    )
    canary = await boundary.canary_token_digest(
        client, seed=valid_intent["learner"]["initialization_seed"]
    )
    if canary != receipt["canary_token_digest"]:
        raise BranchAssayError("restored branch canary differs from the common initial state")

    update_audits: list[Mapping[str, Any]] = []
    checkpoints: list[Mapping[str, Any]] = []
    submitted_total = 0
    final_state_uri: str | None = None
    final_sampler_uri: str | None = None
    for update in range(UPDATES_PER_BRANCH):
        raw_rollouts = await boundary.collect_training_rollouts(
            client,
            pool_dir=pool_dir,
            pool_entries=entries,
            pair_id=request["pair_id"],
            pair_seed=request["pair_seed"],
            update=update,
        )
        padded_by_game, audit = normalize_and_pad_training_rollouts(
            rollouts=raw_rollouts,
            expected_basenames=basenames,
            pair_id=request["pair_id"],
            pair_seed=request["pair_seed"],
            update=update,
            pad_token_id=boundary.pad_token_id,
            gamma=valid_intent["learner"]["gamma"],
        )
        datums = [datum for basename in basenames for datum in padded_by_game[basename]]
        train_result = await boundary.train_update(
            client,
            datums=datums,
            update=update,
            learning_rate=valid_intent["learner"]["learning_rate"],
        )
        if set(train_result) != {
            "completed_update",
            "forward_backward_calls",
            "optimizer_step_calls",
            "submitted_positions",
            "public_metrics",
        }:
            raise BranchAssayError("training result differs from its exact public schema")
        if (
            train_result.get("completed_update") != update + 1
            or train_result.get("forward_backward_calls") != 1
            or train_result.get("optimizer_step_calls") != 1
            or train_result.get("submitted_positions") != TRAIN_POSITIONS_PER_UPDATE
        ):
            raise BranchAssayError("training update call/compute accounting differs")
        _validate_public_metadata(train_result.get("public_metrics"))
        submitted_total += TRAIN_POSITIONS_PER_UPDATE
        checkpoint = await boundary.save_checkpoint(
            client,
            name=(f"{request['pair_id']}-{request['slot_id']}-update-{update + 1:02d}"),
        )
        final_state_uri, final_sampler_uri = _checkpoint_uris(checkpoint, update=update)
        update_audits.append(dict(audit))
        checkpoints.append(dict(checkpoint))

    if submitted_total != TRAIN_POSITIONS_PER_BRANCH:
        raise BranchAssayError("branch total submitted training positions differ")
    if final_state_uri is None or final_sampler_uri is None:
        raise AssertionError("fixed training loop did not produce a final checkpoint")
    heldout = pool_manifest.get("pools", {}).get("heldout_v4")
    if not isinstance(heldout, Mapping) or heldout.get("relative_dir") != "heldout_v4":
        raise BranchAssayError("pool manifest omits the held-out panel")
    heldout_entries = heldout.get("entries")
    if not isinstance(heldout_entries, list) or len(heldout_entries) != len(HELDOUT_STRATA):
        raise BranchAssayError("held-out pool topology differs")
    heldout_dir = pool_root / "heldout_v4"
    outcomes = await boundary.evaluate_checkpoint(
        sampler_uri=final_sampler_uri,
        heldout_dir=heldout_dir,
        heldout_entries=heldout_entries,
        pair_id=request["pair_id"],
        pair_seed=request["pair_seed"],
    )
    score = build_heldout_score(
        intent=valid_intent,
        branch_request=request,
        base_receipt=receipt,
        final_state_uri=final_state_uri,
        final_sampler_uri=final_sampler_uri,
        completed_updates=UPDATES_PER_BRANCH,
        submitted_training_positions=submitted_total,
        outcomes=outcomes,
    )
    return BranchExecution(
        score=score,
        update_audits=tuple(update_audits),
        checkpoint_receipts=tuple(checkpoints),
        canary_token_digest=canary,
    )


_UPDATE_AUDIT_KEYS = {
    "games",
    "episodes",
    "submitted_positions",
    "actual_positions",
    "action_positions",
    "padded_positions",
    "mixed_reward_games",
    "raw_group_means",
}
_BRANCH_EXECUTION_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "base_receipt_digest",
    "branch_request_digest",
    "pair_id",
    "slot_id",
    "arm",
    "canary_token_digest",
    "update_audits",
    "checkpoint_receipts",
    "completed_updates",
    "submitted_training_positions",
    "execution_digest",
}


def _validate_update_audit(
    value: object,
    *,
    expected_basenames: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _UPDATE_AUDIT_KEYS:
        raise BranchAssayError("update audit differs from its exact schema")
    result = dict(value)
    integer_fields = {
        "games": TRAIN_GAMES_PER_ARM,
        "episodes": EPISODES_PER_UPDATE,
        "submitted_positions": TRAIN_POSITIONS_PER_UPDATE,
    }
    for field, expected in integer_fields.items():
        if isinstance(result.get(field), bool) or result.get(field) != expected:
            raise BranchAssayError(f"update audit {field} differs")
    for field in ("actual_positions", "action_positions", "padded_positions"):
        item = result.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BranchAssayError(f"update audit {field} is invalid")
    if (
        result["actual_positions"] + result["padded_positions"] != TRAIN_POSITIONS_PER_UPDATE
        or not EPISODES_PER_UPDATE <= result["action_positions"] <= result["actual_positions"]
    ):
        raise BranchAssayError("update audit token accounting is inconsistent")
    mixed = result.get("mixed_reward_games")
    if isinstance(mixed, bool) or not isinstance(mixed, int) or mixed not in range(13):
        raise BranchAssayError("update audit mixed-game count is invalid")
    means = result.get("raw_group_means")
    if not isinstance(means, Mapping) or set(means) != expected_basenames:
        raise BranchAssayError("update audit game means differ from the sealed pool")
    for mean in means.values():
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
            or not 0.0 <= float(mean) <= 1.0
        ):
            raise BranchAssayError("update audit game mean is invalid")
    return result


def build_branch_execution_receipt(
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    branch_request: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    execution: BranchExecution,
) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    request = validate_branch_request(
        branch_request,
        intent=valid_intent,
        base_receipt=receipt,
    )
    _, entries = _pool_section(pool_manifest, request["arm"])
    basenames = {str(item["basename"]) for item in entries}
    audits = [
        _validate_update_audit(item, expected_basenames=basenames)
        for item in execution.update_audits
    ]
    if len(audits) != UPDATES_PER_BRANCH:
        raise BranchAssayError("branch execution update-audit count differs")
    checkpoints = [dict(item) for item in execution.checkpoint_receipts]
    if len(checkpoints) != UPDATES_PER_BRANCH:
        raise BranchAssayError("branch execution checkpoint count differs")
    for update, checkpoint in enumerate(checkpoints):
        _checkpoint_uris(checkpoint, update=update)
    if execution.canary_token_digest != receipt["canary_token_digest"]:
        raise BranchAssayError("branch execution canary differs from the common state")
    body = {
        "schema_version": BRANCH_EXECUTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": valid_intent["intent_digest"],
        "base_receipt_digest": receipt["receipt_digest"],
        "branch_request_digest": request["request_digest"],
        "pair_id": request["pair_id"],
        "slot_id": request["slot_id"],
        "arm": request["arm"],
        "canary_token_digest": execution.canary_token_digest,
        "update_audits": audits,
        "checkpoint_receipts": checkpoints,
        "completed_updates": UPDATES_PER_BRANCH,
        "submitted_training_positions": TRAIN_POSITIONS_PER_BRANCH,
    }
    return seal_document(body, "execution_digest")


def validate_branch_execution_receipt(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    branch_request: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    score: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sealed = validate_seal(
        value,
        schema=BRANCH_EXECUTION_SCHEMA,
        exact_keys=_BRANCH_EXECUTION_KEYS,
    )
    synthetic = BranchExecution(
        score={},
        update_audits=tuple(sealed.get("update_audits", ())),
        checkpoint_receipts=tuple(sealed.get("checkpoint_receipts", ())),
        canary_token_digest=sealed.get("canary_token_digest"),
    )
    rebuilt = build_branch_execution_receipt(
        intent=intent,
        base_receipt=base_receipt,
        branch_request=branch_request,
        pool_manifest=pool_manifest,
        execution=synthetic,
    )
    if rebuilt != sealed:
        raise BranchAssayError("branch execution differs from deterministic recomputation")
    if score is not None:
        valid_score = validate_heldout_score(
            score,
            intent=intent,
            branch_request=branch_request,
            base_receipt=base_receipt,
        )
        final_checkpoint = sealed["checkpoint_receipts"][-1]
        if (
            valid_score["final_state_uri"] != final_checkpoint["state_uri"]
            or valid_score["final_sampler_uri"] != final_checkpoint["sampler_uri"]
        ):
            raise BranchAssayError("held-out score is not bound to the final checkpoint")
    return sealed


def audit_training_batch(
    datums_by_game: Mapping[str, Sequence[PaddedEpisodeDatum]],
) -> dict[str, int]:
    """Fail closed unless a batch has 12x8 unique, equally padded episodes."""

    if len(datums_by_game) != TRAIN_GAMES_PER_ARM:
        raise BranchAssayError("training batch does not contain exactly 12 games")
    datums: list[PaddedEpisodeDatum] = []
    for game_id, game_datums in datums_by_game.items():
        if not isinstance(game_id, str) or not game_id:
            raise BranchAssayError("training game identifier is invalid")
        if len(game_datums) != TRAJECTORIES_PER_GAME:
            raise BranchAssayError("training batch is not balanced at eight episodes per game")
        datums.extend(game_datums)
    if len(datums) != EPISODES_PER_UPDATE:
        raise BranchAssayError("training batch episode count differs")
    if any(item.submitted_positions != TRAIN_POSITIONS_PER_EPISODE for item in datums):
        raise BranchAssayError("training batch contains a noncanonical token budget")
    submitted = sum(item.submitted_positions for item in datums)
    if submitted != TRAIN_POSITIONS_PER_UPDATE:
        raise BranchAssayError("training batch submitted-position total differs")
    return {
        "games": len(datums_by_game),
        "episodes": len(datums),
        "submitted_positions": submitted,
        "actual_positions": sum(item.actual_positions for item in datums),
        "action_positions": sum(item.action_positions for item in datums),
        "padded_positions": sum(item.padded_positions for item in datums),
    }


def evaluation_seed_pair(
    *, pair_id: str, pair_seed: int, stratum_id: str, replicate: int
) -> tuple[int, int]:
    if stratum_id not in HELDOUT_STRATA:
        raise BranchAssayError("evaluation stratum is outside the sealed panel")
    if replicate not in range(HELDOUT_SEEDS_PER_GAME):
        raise BranchAssayError("evaluation replicate is outside the sealed panel")
    # Arm is intentionally absent: both checkpoints in a pair see identical
    # environment and sampling randomness.
    return (
        derive_seed("heldout-env", pair_id, pair_seed, stratum_id, replicate),
        derive_seed("heldout-sampling", pair_id, pair_seed, stratum_id, replicate),
    )


_OUTCOME_KEYS = {
    "stratum_id",
    "replicate",
    "environment_seed",
    "sampling_seed",
    "reward",
    "terminated",
    "truncated",
    "turn_count",
    "actions",
    "response_token_digests",
    "parse_failure_turn",
    "trajectory_digest",
    "provider_error",
}

_OUTCOME_TRAJECTORY_KEYS = _OUTCOME_KEYS - {"trajectory_digest"}
MAX_ACTION_CHARACTERS = 32_768
MAX_TRAJECTORY_ACTION_CHARACTERS = 131_072


def _validated_outcomes(
    outcomes: Sequence[Mapping[str, Any]], *, pair_id: str, pair_seed: int
) -> list[dict[str, Any]]:
    if len(outcomes) != HELDOUT_EPISODES_PER_CHECKPOINT:
        raise BranchAssayError("held-out score must contain exactly 96 outcomes")
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in outcomes:
        if not isinstance(raw, Mapping) or set(raw) != _OUTCOME_KEYS:
            raise BranchAssayError("held-out outcome differs from its exact schema")
        item = dict(raw)
        key = (item.get("stratum_id"), item.get("replicate"))
        if key in observed:
            raise BranchAssayError("held-out outcome key is duplicated")
        if key[0] not in HELDOUT_STRATA or key[1] not in range(HELDOUT_SEEDS_PER_GAME):
            raise BranchAssayError("held-out outcome key is outside the sealed panel")
        env_seed, sampling_seed = evaluation_seed_pair(
            pair_id=pair_id,
            pair_seed=pair_seed,
            stratum_id=key[0],
            replicate=key[1],
        )
        if item.get("environment_seed") != env_seed or item.get("sampling_seed") != sampling_seed:
            raise BranchAssayError("held-out outcome seeds differ from the paired schedule")
        reward = item.get("reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or not 0.0 <= float(reward) <= 1.0
        ):
            raise BranchAssayError("held-out terminal reward must be finite in [0,1]")
        for field in ("terminated", "truncated"):
            if not isinstance(item.get(field), bool):
                raise BranchAssayError(f"held-out {field} must be boolean")
        if (
            isinstance(item.get("turn_count"), bool)
            or not isinstance(item.get("turn_count"), int)
            or item["turn_count"] not in range(1, MAX_HELDOUT_NATIVE_TURNS + 1)
        ):
            raise BranchAssayError("held-out turn_count exceeds the sealed native-horizon ceiling")
        actions = item.get("actions")
        response_digests = item.get("response_token_digests")
        parse_failure_turn = item.get("parse_failure_turn")
        if not isinstance(actions, list) or any(
            not isinstance(action, str) or len(action) > MAX_ACTION_CHARACTERS for action in actions
        ):
            raise BranchAssayError("held-out actions are not bounded public strings")
        if sum(len(action) for action in actions) > MAX_TRAJECTORY_ACTION_CHARACTERS:
            raise BranchAssayError("held-out action trajectory exceeds its sealed size ceiling")
        if (
            not isinstance(response_digests, list)
            or len(response_digests) != item["turn_count"]
            or any(not _is_digest(digest) for digest in response_digests)
        ):
            raise BranchAssayError("held-out response-token digest topology differs")
        if parse_failure_turn is None:
            if len(actions) != item["turn_count"]:
                raise BranchAssayError("held-out action count differs from sampled turns")
        elif (
            isinstance(parse_failure_turn, bool)
            or not isinstance(parse_failure_turn, int)
            or parse_failure_turn != item["turn_count"] - 1
            or len(actions) != parse_failure_turn
            or item["terminated"]
            or not item["truncated"]
        ):
            raise BranchAssayError("held-out parser-failure trajectory is inconsistent")
        if item.get("trajectory_digest") != object_digest(
            {key_name: item[key_name] for key_name in _OUTCOME_TRAJECTORY_KEYS}
        ):
            raise BranchAssayError("held-out trajectory digest differs")
        if item.get("provider_error") is not None:
            raise BranchAssayError("provider errors invalidate a held-out score")
        observed[key] = item
    expected = {
        (stratum, replicate)
        for stratum in HELDOUT_STRATA
        for replicate in range(HELDOUT_SEEDS_PER_GAME)
    }
    if set(observed) != expected:
        raise BranchAssayError("held-out outcome topology differs")
    return [observed[key] for key in sorted(observed)]


def build_heldout_score(
    *,
    intent: Mapping[str, Any],
    branch_request: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    final_state_uri: str,
    final_sampler_uri: str,
    completed_updates: int,
    submitted_training_positions: int,
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    request = validate_branch_request(branch_request, intent=valid_intent, base_receipt=receipt)
    if completed_updates != UPDATES_PER_BRANCH:
        raise BranchAssayError("held-out score is not from the fixed final update")
    if submitted_training_positions != TRAIN_POSITIONS_PER_BRANCH:
        raise BranchAssayError("branch training compute differs from the sealed budget")
    state_uri = _validate_tinker_state_uri(final_state_uri)
    sampler_uri = _validate_tinker_state_uri(final_sampler_uri)
    valid_outcomes = _validated_outcomes(
        outcomes, pair_id=request["pair_id"], pair_seed=request["pair_seed"]
    )
    mean_reward = math.fsum(float(item["reward"]) for item in valid_outcomes) / len(valid_outcomes)
    body = {
        "schema_version": SCORE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": valid_intent["intent_digest"],
        "branch_request_digest": request["request_digest"],
        "base_receipt_digest": receipt["receipt_digest"],
        "pair_id": request["pair_id"],
        "slot_id": request["slot_id"],
        "arm": request["arm"],
        "learner_backend": "tinker",
        "rollout_backend": "tinker",
        "evaluation_backend": "tinker",
        "model_name": valid_intent["learner"]["model_name"],
        "completed_updates": completed_updates,
        "submitted_training_positions": submitted_training_positions,
        "final_state_uri": state_uri,
        "final_state_uri_digest": bytes_digest(state_uri.encode("utf-8")),
        "final_sampler_uri": sampler_uri,
        "final_sampler_uri_digest": bytes_digest(sampler_uri.encode("utf-8")),
        "outcomes": valid_outcomes,
        "mean_terminal_reward": mean_reward,
    }
    return seal_document(body, "score_digest")


_SCORE_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "branch_request_digest",
    "base_receipt_digest",
    "pair_id",
    "slot_id",
    "arm",
    "learner_backend",
    "rollout_backend",
    "evaluation_backend",
    "model_name",
    "completed_updates",
    "submitted_training_positions",
    "final_state_uri",
    "final_state_uri_digest",
    "final_sampler_uri",
    "final_sampler_uri_digest",
    "outcomes",
    "mean_terminal_reward",
    "score_digest",
}


def validate_heldout_score(
    score: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    branch_request: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    request = validate_branch_request(branch_request, intent=valid_intent, base_receipt=receipt)
    value = validate_seal(score, schema=SCORE_SCHEMA, exact_keys=_SCORE_KEYS)
    rebuilt = build_heldout_score(
        intent=valid_intent,
        branch_request=request,
        base_receipt=receipt,
        final_state_uri=value.get("final_state_uri"),
        final_sampler_uri=value.get("final_sampler_uri"),
        completed_updates=value.get("completed_updates"),
        submitted_training_positions=value.get("submitted_training_positions"),
        outcomes=value.get("outcomes", []),
    )
    if rebuilt != value:
        raise BranchAssayError("held-out score differs from deterministic recomputation")
    if not (
        value.get("learner_backend")
        == value.get("rollout_backend")
        == value.get("evaluation_backend")
        == valid_intent["learner"]["backend"]
    ):
        raise BranchAssayError("score substitutes a model/backend for the learned checkpoint")
    return value


_PAIR_COMPLETE_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "base_receipt_digest",
    "pair_id",
    "branch_request_digests",
    "branch_execution_digests",
    "score_digests",
    "status",
    "pair_digest",
}
_PAIR_TERMINAL_ERROR_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "base_receipt_digest",
    "pair_id",
    "branch_request_digests",
    "status",
    "error_type",
    "retry_authorized",
    "error_digest",
}


def build_pair_complete(
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    pair_id: str,
    requests: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    pool_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    expected_requests = {
        item["slot_id"]: item
        for item in branch_requests(valid_intent, receipt)
        if item["pair_id"] == pair_id
    }
    if set(expected_requests) != {"slot_a", "slot_b"} or len(requests) != 2:
        raise BranchAssayError("pair requests differ from the sealed schedule")
    supplied_requests: dict[str, dict[str, Any]] = {}
    for raw in requests:
        request = validate_branch_request(
            raw,
            intent=valid_intent,
            base_receipt=receipt,
        )
        if request["pair_id"] != pair_id or request["slot_id"] in supplied_requests:
            raise BranchAssayError("pair request topology differs")
        supplied_requests[request["slot_id"]] = request
    if supplied_requests != expected_requests:
        raise BranchAssayError("pair request bytes differ from the sealed schedule")

    execution_by_request = {
        item.get("branch_request_digest"): item for item in executions if isinstance(item, Mapping)
    }
    score_by_request = {
        item.get("branch_request_digest"): item for item in scores if isinstance(item, Mapping)
    }
    expected_digests = {item["request_digest"] for item in expected_requests.values()}
    if (
        len(executions) != 2
        or len(scores) != 2
        or len(execution_by_request) != 2
        or len(score_by_request) != 2
        or set(execution_by_request) != expected_digests
        or set(score_by_request) != expected_digests
    ):
        raise BranchAssayError("pair execution/score topology differs")
    valid_executions: list[dict[str, Any]] = []
    valid_scores: list[dict[str, Any]] = []
    for request in expected_requests.values():
        score = validate_heldout_score(
            score_by_request[request["request_digest"]],
            intent=valid_intent,
            branch_request=request,
            base_receipt=receipt,
        )
        execution = validate_branch_execution_receipt(
            execution_by_request[request["request_digest"]],
            intent=valid_intent,
            base_receipt=receipt,
            branch_request=request,
            pool_manifest=pool_manifest,
            score=score,
        )
        valid_executions.append(execution)
        valid_scores.append(score)
    body = {
        "schema_version": PAIR_COMPLETE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": valid_intent["intent_digest"],
        "base_receipt_digest": receipt["receipt_digest"],
        "pair_id": pair_id,
        "branch_request_digests": sorted(expected_digests),
        "branch_execution_digests": sorted(item["execution_digest"] for item in valid_executions),
        "score_digests": sorted(item["score_digest"] for item in valid_scores),
        "status": "complete",
    }
    return seal_document(body, "pair_digest")


def validate_pair_complete(
    value: Mapping[str, Any],
    **rebuild_arguments: Any,
) -> dict[str, Any]:
    sealed = validate_seal(
        value,
        schema=PAIR_COMPLETE_SCHEMA,
        exact_keys=_PAIR_COMPLETE_KEYS,
    )
    rebuilt = build_pair_complete(**rebuild_arguments)
    if sealed != rebuilt:
        raise BranchAssayError("pair completion differs from deterministic recomputation")
    return sealed


def build_pair_terminal_error(
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    pair_id: str,
    requests: Sequence[Mapping[str, Any]],
    error_type: str,
) -> dict[str, Any]:
    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    if not isinstance(error_type, str) or not error_type or len(error_type) > 256:
        raise BranchAssayError("terminal pair error type is invalid")
    valid_requests = [
        validate_branch_request(item, intent=valid_intent, base_receipt=receipt)
        for item in requests
    ]
    if (
        len(valid_requests) != 2
        or {item["pair_id"] for item in valid_requests} != {pair_id}
        or {item["slot_id"] for item in valid_requests} != {"slot_a", "slot_b"}
    ):
        raise BranchAssayError("terminal pair request topology differs")
    body = {
        "schema_version": PAIR_TERMINAL_ERROR_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": valid_intent["intent_digest"],
        "base_receipt_digest": receipt["receipt_digest"],
        "pair_id": pair_id,
        "branch_request_digests": sorted(item["request_digest"] for item in valid_requests),
        "status": "terminal_ambiguous",
        "error_type": error_type,
        "retry_authorized": False,
    }
    return seal_document(body, "error_digest")


def validate_pair_terminal_error(
    value: Mapping[str, Any],
    **rebuild_arguments: Any,
) -> dict[str, Any]:
    sealed = validate_seal(
        value,
        schema=PAIR_TERMINAL_ERROR_SCHEMA,
        exact_keys=_PAIR_TERMINAL_ERROR_KEYS,
    )
    rebuilt = build_pair_terminal_error(**rebuild_arguments)
    if sealed != rebuilt:
        raise BranchAssayError("terminal pair error differs from deterministic recomputation")
    return sealed


def exact_two_sided_sign_flip(differences: Sequence[float]) -> tuple[float, int, int]:
    if len(differences) != PAIR_COUNT:
        raise BranchAssayError("exact test requires six paired training-run differences")
    values = []
    for item in differences:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BranchAssayError("paired differences must be numeric")
        value = float(item)
        if not math.isfinite(value):
            raise BranchAssayError("paired differences must be finite")
        values.append(value)
    observed = abs(math.fsum(values) / len(values))
    extreme = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=PAIR_COUNT):
        statistic = abs(math.fsum(sign * value for sign, value in zip(signs, values)) / len(values))
        if statistic + tolerance >= observed:
            extreme += 1
    return extreme / EXACT_SIGN_FLIP_COUNT, extreme, EXACT_SIGN_FLIP_COUNT


def analyze_scores(
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    scores: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    pair_completions: Sequence[Mapping[str, Any]],
    pool_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the pair estimand and assumption-based sign-flip sensitivity."""

    valid_intent = validate_intent(intent)
    receipt = validate_base_receipt(base_receipt, intent=valid_intent)
    requests = branch_requests(valid_intent, receipt)
    request_by_digest = {item["request_digest"]: item for item in requests}
    if len(scores) != PAIR_COUNT * 2 or len(executions) != PAIR_COUNT * 2:
        raise BranchAssayError("analysis requires exactly twelve branch executions and scores")
    execution_by_digest = {
        item.get("branch_request_digest"): item for item in executions if isinstance(item, Mapping)
    }
    if len(execution_by_digest) != PAIR_COUNT * 2 or set(execution_by_digest) != set(
        request_by_digest
    ):
        raise BranchAssayError("analysis branch-execution topology differs")
    validated: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in scores:
        digest = raw.get("branch_request_digest") if isinstance(raw, Mapping) else None
        request = request_by_digest.get(digest)
        if request is None:
            raise BranchAssayError("score does not correspond to a sealed branch request")
        score = validate_heldout_score(
            raw, intent=valid_intent, branch_request=request, base_receipt=receipt
        )
        key = (score["pair_id"], score["arm"])
        if key in validated:
            raise BranchAssayError("duplicate score for pair/arm")
        validated[key] = score
    expected = {(item["pair_id"], arm) for item in valid_intent["pair_schedule"] for arm in ARMS}
    if set(validated) != expected:
        raise BranchAssayError("score pair/arm topology differs")

    completions_by_pair = {
        item.get("pair_id"): item for item in pair_completions if isinstance(item, Mapping)
    }
    expected_pair_ids = {item["pair_id"] for item in valid_intent["pair_schedule"]}
    if (
        len(pair_completions) != PAIR_COUNT
        or len(completions_by_pair) != PAIR_COUNT
        or set(completions_by_pair) != expected_pair_ids
    ):
        raise BranchAssayError("analysis pair-completion topology differs")
    valid_completions: list[dict[str, Any]] = []
    valid_executions: list[dict[str, Any]] = []
    for pair_id in sorted(expected_pair_ids):
        pair_requests = [item for item in requests if item["pair_id"] == pair_id]
        pair_executions = [execution_by_digest[item["request_digest"]] for item in pair_requests]
        pair_scores = [validated[(pair_id, item["arm"])] for item in pair_requests]
        completion = validate_pair_complete(
            completions_by_pair[pair_id],
            intent=valid_intent,
            base_receipt=receipt,
            pair_id=pair_id,
            requests=pair_requests,
            executions=pair_executions,
            scores=pair_scores,
            pool_manifest=pool_manifest,
        )
        valid_completions.append(completion)
        valid_executions.extend(pair_executions)

    pair_effects: list[dict[str, Any]] = []
    differences: list[float] = []
    for pair in valid_intent["pair_schedule"]:
        pair_id = pair["pair_id"]
        treatment = validated[(pair_id, "coverage_forced")]["mean_terminal_reward"]
        control = validated[(pair_id, "redundant_historical")]["mean_terminal_reward"]
        difference = float(treatment) - float(control)
        differences.append(difference)
        pair_effects.append(
            {
                "pair_id": pair_id,
                "coverage_forced_mean": treatment,
                "redundant_historical_mean": control,
                "difference": difference,
            }
        )
    effect = math.fsum(differences) / PAIR_COUNT
    p_value, extreme, total = exact_two_sided_sign_flip(differences)
    sensitivity_gate_passed = p_value <= 0.05 and effect >= DEFAULT_MINIMUM_EFFECT
    body = {
        "schema_version": AGGREGATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "intent_digest": valid_intent["intent_digest"],
        "base_receipt_digest": receipt["receipt_digest"],
        "score_digests": sorted(item["score_digest"] for item in validated.values()),
        "branch_execution_digests": sorted(item["execution_digest"] for item in valid_executions),
        "pair_completion_digests": sorted(item["pair_digest"] for item in valid_completions),
        "pair_effects": pair_effects,
        "mean_effect": effect,
        "minimum_effect": DEFAULT_MINIMUM_EFFECT,
        "exact_test": "two-sided-sign-flip-sensitivity-under-within-pair-exchangeability",
        "exact_extreme_assignments": extreme,
        "exact_assignments": total,
        "p_value": p_value,
        "sensitivity_gate_passed": sensitivity_gate_passed,
        # This protocol cannot authenticate either the externally sourced
        # assignment entropy or the optimizer/checkpoint objects returned as
        # bare Tinker URIs.  A favorable numerical sensitivity result must
        # therefore never be converted into a causal learner decision.
        "scientific_gate_passed": False,
        "causal_claim_authorized": False,
        "supports_narrow_claim": False,
        "decision": "hold_plan_only_unattested_assignment_and_provider_state",
        "claim_boundary": valid_intent["claim_boundary"],
        "promotion_authorized": False,
        "release_authorized": False,
        "model_lock_written": False,
    }
    return seal_document(body, "aggregate_digest")


_AGGREGATE_KEYS = {
    "schema_version",
    "protocol_id",
    "intent_digest",
    "base_receipt_digest",
    "score_digests",
    "branch_execution_digests",
    "pair_completion_digests",
    "pair_effects",
    "mean_effect",
    "minimum_effect",
    "exact_test",
    "exact_extreme_assignments",
    "exact_assignments",
    "p_value",
    "sensitivity_gate_passed",
    "scientific_gate_passed",
    "causal_claim_authorized",
    "supports_narrow_claim",
    "decision",
    "claim_boundary",
    "promotion_authorized",
    "release_authorized",
    "model_lock_written",
    "aggregate_digest",
}


def validate_aggregate(
    aggregate: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    scores: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    pair_completions: Sequence[Mapping[str, Any]],
    pool_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(aggregate, schema=AGGREGATE_SCHEMA, exact_keys=_AGGREGATE_KEYS)
    rebuilt = analyze_scores(
        intent=intent,
        base_receipt=base_receipt,
        scores=scores,
        executions=executions,
        pair_completions=pair_completions,
        pool_manifest=pool_manifest,
    )
    if value != rebuilt:
        raise BranchAssayError("aggregate differs from deterministic recomputation")
    return value
