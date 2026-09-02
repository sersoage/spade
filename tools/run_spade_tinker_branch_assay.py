#!/usr/bin/env python3
"""Seal and validate the plan-only Tinker learner-assay evidence protocol.

Offline commands import no Tinker SDK and make no network calls. This protocol
version's live guard unconditionally rejects service access because its entropy
and remote-state evidence cannot be authenticated.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spade.core.learner_branch_pools import (  # noqa: E402
    SEALED_ACTOR_PLAN_DIGEST,
    load_learner_pool_manifest,
)
from spade.tinker.branch_assay import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MAX_CONCURRENT_SAMPLES,
    DEFAULT_RENDERER,
    DEFAULT_TINKER_ENDPOINT,
    HELDOUT_CONTEXT_POSITIONS,
    HELDOUT_SEEDS_PER_GAME,
    MAX_HELDOUT_NATIVE_TURNS,
    MAX_TRAIN_TURNS,
    TRAJECTORIES_PER_GAME,
    BranchAssayError,
    EpisodeRollout,
    analyze_scores,
    branch_requests,
    build_branch_execution_receipt,
    build_intent,
    build_pair_complete,
    build_pair_terminal_error,
    bytes_digest,
    canonical_json_bytes,
    derive_seed,
    execute_branch,
    evaluation_seed_pair,
    file_digest,
    object_digest,
    prepare_common_base_state,
    read_json,
    validate_aggregate,
    validate_branch_execution_receipt,
    validate_branch_request,
    validate_common_base_tree,
    validate_heldout_score,
    validate_intent,
    validate_pair_complete,
    validate_pair_terminal_error,
    validate_runtime_attestation,
    training_seed,
)


_ENVIRONMENT_RNG_LOCK = threading.Lock()
_LINEAGE_SCHEMA = "spade-tinker-learner-assay-lineage-lock/v1"
_LINEAGE_FILENAME = "lineage-lock.json"


def _canonical_output_root(*, pool_manifest_digest: str, source_actor_plan_digest: str) -> Path:
    """Return the sole repository-local root authorized for this source lineage."""

    return (
        ROOT_DIR
        / ".assay"
        / (
            "spade-tinker-learner-"
            f"{source_actor_plan_digest.removeprefix('sha256:')[:12]}-"
            f"{pool_manifest_digest.removeprefix('sha256:')[:12]}-v1"
        )
    ).resolve()


def _lineage_lock(intent: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": _LINEAGE_SCHEMA,
        "protocol_id": intent["protocol_id"],
        "source_actor_plan_digest": intent["source_actor_plan_digest"],
        "pool_manifest_digest": intent["pool_manifest_digest"],
        "intent_digest": intent["intent_digest"],
        "canonical_intent_path": str(Path(intent["output_root"]) / "intent.json"),
        "authorization_scope": "one-prospective-assay-lineage-no-reseal-or-rerun",
    }
    return {**body, "lineage_digest": object_digest(body)}


def _validate_lineage_lock(intent: Mapping[str, Any], *, intent_path: Path) -> None:
    expected_root = _canonical_output_root(
        pool_manifest_digest=str(intent["pool_manifest_digest"]),
        source_actor_plan_digest=str(intent["source_actor_plan_digest"]),
    )
    if Path(intent["output_root"]) != expected_root:
        raise BranchAssayError("output root is not the canonical source-lineage root")
    canonical_intent = expected_root / "intent.json"
    if intent_path != canonical_intent:
        raise BranchAssayError("intent is not at its canonical one-lineage path")
    lock_path = expected_root / _LINEAGE_FILENAME
    lock = read_json(lock_path)
    if set(lock) != {
        "schema_version",
        "protocol_id",
        "source_actor_plan_digest",
        "pool_manifest_digest",
        "intent_digest",
        "canonical_intent_path",
        "authorization_scope",
        "lineage_digest",
    } or lock != _lineage_lock(intent):
        raise BranchAssayError("one-lineage lock is missing, altered, or bound to another intent")
    canonical_run_name = "spade-tinker-learner-" + str(intent["intent_digest"]).removeprefix(
        "sha256:"
    )
    entries = list(expected_root.iterdir())
    names = {item.name for item in entries}
    expected_names = {"intent.json", _LINEAGE_FILENAME}
    if canonical_run_name in names:
        expected_names.add(canonical_run_name)
    if names != expected_names or any(item.is_symlink() for item in entries):
        raise BranchAssayError("canonical lineage-root inventory differs")
    by_name = {item.name: item for item in entries}
    if not by_name["intent.json"].is_file() or not by_name[_LINEAGE_FILENAME].is_file():
        raise BranchAssayError("canonical lineage-root files are unsafe")
    if canonical_run_name in expected_names and not by_name[canonical_run_name].is_dir():
        raise BranchAssayError("canonical run artifact is not a directory")


def _read_sealed_environment(
    game_path: Path,
    entry: Mapping[str, Any],
) -> str:
    """Read one manifest-bound game into an immutable in-memory snapshot."""

    if (
        not game_path.is_absolute()
        or game_path.is_symlink()
        or not game_path.is_file()
        or game_path.resolve(strict=True) != game_path
        or game_path.name != entry.get("basename")
    ):
        raise BranchAssayError(f"game path is missing or unsafe: {game_path}")
    raw = game_path.read_bytes()
    if len(raw) != entry.get("size_bytes") or bytes_digest(raw) != entry.get("environment_digest"):
        raise BranchAssayError(f"game bytes differ from the sealed pool: {game_path}")
    try:
        code = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BranchAssayError(f"game source is not UTF-8: {game_path}") from exc
    if object_digest(code) != entry.get("code_digest"):
        raise BranchAssayError(f"game code digest differs from the sealed pool: {game_path}")
    return code


def _call_with_isolated_global_rng(
    operation: object,
    *,
    seed: int | None = None,
    state: tuple[object, object] | None = None,
) -> tuple[object, tuple[object, object]]:
    """Run one generated-environment operation with episode-local global RNG state.

    Several sealed games use module-global ``random``/``numpy.random``.  Their
    state must not bleed between concurrently sampled episodes or between the
    two paired learner branches.  The lock also makes save/restore atomic
    with respect to this runner's environment workers.
    """

    import numpy as np

    if not callable(operation) or (seed is None) == (state is None):
        raise BranchAssayError("environment RNG operation needs exactly one seed/state")
    with _ENVIRONMENT_RNG_LOCK:
        ambient_python = random.getstate()
        ambient_numpy = np.random.get_state()
        try:
            if state is None:
                random.seed(seed)
                np.random.seed(int(seed) % (2**32))
            else:
                random.setstate(state[0])
                np.random.set_state(state[1])
            result = operation()
            episode_state = (random.getstate(), np.random.get_state())
        finally:
            random.setstate(ambient_python)
            np.random.set_state(ambient_numpy)
    return result, episode_state


def _pretty(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise BranchAssayError(f"refusing to overwrite artifact: {path}")
    with path.open("xb") as handle:
        handle.write(_pretty(value))
        handle.flush()
        os.fsync(handle.fileno())


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    head = result.stdout.strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise BranchAssayError("cannot resolve a canonical SPADE git head")
    return head


def _git_object(root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise BranchAssayError(f"cannot resolve canonical git object {revision}")
    return value


def _distribution_identity(
    distribution_name: str,
    module_name: str,
    *,
    bind_package_files: bool = False,
) -> dict[str, Any]:
    """Bind an installed wheel's inventory plus its executable package entrypoint."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise BranchAssayError(
            f"required runtime distribution is unavailable: {distribution_name}"
        ) from exc
    files = distribution.files
    if files is None:
        raise BranchAssayError(f"{distribution_name} distribution has no installed-file record")

    def one_file(suffix: str) -> Path:
        matches = [item for item in files if str(item).endswith(suffix)]
        if len(matches) != 1:
            raise BranchAssayError(f"{distribution_name} distribution lacks one canonical {suffix}")
        return Path(distribution.locate_file(matches[0])).resolve(strict=True)

    def exact_file(relative: str) -> Path:
        matches = [item for item in files if str(item) == relative]
        if len(matches) != 1:
            raise BranchAssayError(
                f"{distribution_name} distribution lacks one canonical {relative}"
            )
        return Path(distribution.locate_file(matches[0])).resolve(strict=True)

    record = one_file(".dist-info/RECORD")
    metadata_file = one_file(".dist-info/METADATA")
    package_entry = exact_file(f"{module_name}/__init__.py")
    identity: dict[str, Any] = {
        "version": distribution.version,
        "record_path": str(record),
        "record_digest": file_digest(record),
        "metadata_path": str(metadata_file),
        "metadata_digest": file_digest(metadata_file),
        "package_entry_path": str(package_entry),
        "package_entry_digest": file_digest(package_entry),
    }
    if bind_package_files:
        package_files = sorted(
            (
                item
                for item in files
                if str(item).startswith(f"{module_name}/")
                and "__pycache__" not in str(item)
                and not str(item).endswith((".pyc", ".pyo"))
            ),
            key=str,
        )
        if not package_files:
            raise BranchAssayError(f"{distribution_name} package inventory is empty")
        installed_digests: dict[str, str] = {}
        for item in package_files:
            path = Path(distribution.locate_file(item)).resolve(strict=True)
            installed_digests[str(item)] = file_digest(path)
        identity["installed_package_file_count"] = len(installed_digests)
        identity["installed_package_files_digest"] = object_digest(installed_digests)
    return identity


def _verify_imported_distribution_module(
    module: object,
    *,
    distribution_name: str,
    module_name: str,
    sealed_identity: Mapping[str, Any],
    bind_package_files: bool = False,
) -> None:
    """Reject local shadow imports and post-seal installed-package drift."""

    if not isinstance(sealed_identity, Mapping):
        raise BranchAssayError(f"sealed {distribution_name} distribution identity is absent")
    current_identity = _distribution_identity(
        distribution_name,
        module_name,
        bind_package_files=bind_package_files,
    )
    if current_identity != dict(sealed_identity):
        raise BranchAssayError(f"imported {distribution_name} distribution differs from the seal")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise BranchAssayError(f"imported {module_name} module has no canonical origin")
    origin = Path(module_file)
    expected = Path(str(current_identity["package_entry_path"]))
    if (
        not origin.is_absolute()
        or origin.is_symlink()
        or not origin.is_file()
        or origin.resolve(strict=True) != expected
        or file_digest(origin.resolve(strict=True)) != current_identity["package_entry_digest"]
    ):
        raise BranchAssayError(f"imported {module_name} module shadows its sealed distribution")


def _runtime_identity() -> dict[str, Any]:
    files = (
        ROOT_DIR / "spade/tinker/branch_assay.py",
        ROOT_DIR / "tools/run_spade_tinker_branch_assay.py",
        ROOT_DIR / "spade/core/learner_branch_pools.py",
        ROOT_DIR / "spade/core/envs/synthetic_game_env.py",
        ROOT_DIR / "spade/core/envs/tool_use_base_env.py",
        ROOT_DIR / "spade/core/utils/parsing.py",
        ROOT_DIR / "spade/core/utils/rewards.py",
        ROOT_DIR / "tinker-cookbook/tinker_cookbook/renderers/base.py",
        ROOT_DIR / "tinker-cookbook/tinker_cookbook/renderers/qwen3.py",
        ROOT_DIR / "tinker-cookbook/tinker_cookbook/tokenizer_utils.py",
    )
    python_executable = Path(sys.executable).resolve(strict=True)
    return {
        "spade_git_head": _git_head(ROOT_DIR),
        "tinker_cookbook_gitlink": _git_object(ROOT_DIR, "HEAD:tinker-cookbook"),
        "files": {str(path.relative_to(ROOT_DIR)): file_digest(path) for path in files},
        "python": {
            "version": sys.version,
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "executable_path": str(python_executable),
            "executable_digest": file_digest(python_executable),
        },
        "distributions": {
            "tinker": _distribution_identity("tinker", "tinker", bind_package_files=True),
            "torch": _distribution_identity("torch", "torch"),
            "numpy": _distribution_identity("numpy", "numpy"),
        },
        "proofpack": "source-qualification-only;not-in-live-learner-process",
    }


def _require_pinned_cookbook(package: object) -> None:
    package_file = getattr(package, "__file__", None)
    if not isinstance(package_file, str):
        raise BranchAssayError("cannot resolve the imported Tinker cookbook")
    expected_root = (ROOT_DIR / "tinker-cookbook").resolve()
    try:
        Path(package_file).resolve(strict=True).relative_to(expected_root)
    except (OSError, ValueError) as exc:
        raise BranchAssayError(
            "imported Tinker cookbook is not the repository-pinned checkout"
        ) from exc
    if _git_head(expected_root) != _git_object(ROOT_DIR, "HEAD:tinker-cookbook"):
        raise BranchAssayError("imported Tinker cookbook checkout differs from the gitlink")


def _prime_environment_runtime() -> None:
    """Load every repeatedly used SPADE environment module before rehashing it."""

    from spade.core.envs import synthetic_game_env, tool_use_base_env
    from spade.core.utils import parsing, rewards

    _ = synthetic_game_env, tool_use_base_env, parsing, rewards


def _load_intent(path: Path) -> dict[str, Any]:
    _prime_environment_runtime()
    intent = validate_intent(read_json(path))
    _validate_lineage_lock(intent, intent_path=path)
    if intent["runtime_identity"] != _runtime_identity():
        raise BranchAssayError("runtime identity differs from the sealed intent")
    return intent


def _run_dir(intent: Mapping[str, Any]) -> Path:
    return Path(intent["output_root"]) / (
        "spade-tinker-learner-" + str(intent["intent_digest"]).removeprefix("sha256:")
    )


def _capability_receipt(capabilities: object, *, model_name: str) -> dict[str, Any]:
    """Canonicalize only the stable public model names from server capabilities."""

    supported = getattr(capabilities, "supported_models", None)
    if not isinstance(supported, (list, tuple)):
        raise BranchAssayError("Tinker capabilities omit the supported-model inventory")
    model_names: list[str] = []
    for item in supported:
        value = item if isinstance(item, str) else getattr(item, "model_name", None)
        if not isinstance(value, str) or not value:
            raise BranchAssayError("Tinker capabilities contain a malformed model entry")
        model_names.append(value)
    canonical_names = sorted(set(model_names))
    if len(canonical_names) != len(model_names):
        raise BranchAssayError("Tinker capabilities contain duplicate model entries")
    if model_name not in canonical_names:
        raise BranchAssayError(
            f"Tinker server capabilities do not advertise the sealed model {model_name}"
        )
    return {"supported_model_names": canonical_names}


async def _runtime_attestation(
    service: object, *, service_endpoint: str, model_name: str, renderer_name: str
) -> dict[str, Any]:
    capabilities = await service.get_server_capabilities_async()
    public_capabilities = _capability_receipt(capabilities, model_name=model_name)
    return {
        "backend": "tinker",
        "service_endpoint": service_endpoint,
        "model_name": model_name,
        "renderer_name": renderer_name,
        "sdk_version": importlib.metadata.version("tinker"),
        "server_capabilities_digest": bytes_digest(canonical_json_bytes(public_capabilities)),
    }


class _TinkerCommonStateBoundary:
    """Small live adapter; importing this class itself does not import Tinker."""

    def __init__(self, *, intent: Mapping[str, Any]):
        self.renderer_name = str(intent["learner"]["renderer_name"])
        service_endpoint = str(intent["service_endpoint"])
        if service_endpoint != DEFAULT_TINKER_ENDPOINT:
            raise BranchAssayError("live adapter refuses a noncanonical Tinker endpoint")
        self.service_endpoint = service_endpoint
        distributions = intent.get("runtime_identity", {}).get("distributions", {})
        if not isinstance(distributions, Mapping):
            raise BranchAssayError("sealed runtime distribution identity is absent")
        self.runtime_distributions = distributions

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
        try:
            import tinker
            import tinker_cookbook
            from tinker_cookbook.renderers.qwen3 import Qwen3DisableThinkingRenderer
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as exc:
            raise BranchAssayError(
                "Tinker SDK/cookbook is unavailable; install the pinned runtime first"
            ) from exc
        _require_pinned_cookbook(tinker_cookbook)
        _verify_imported_distribution_module(
            tinker,
            distribution_name="tinker",
            module_name="tinker",
            sealed_identity=self.runtime_distributions.get("tinker", {}),
            bind_package_files=True,
        )

        if service_endpoint != self.service_endpoint:
            raise BranchAssayError("common-state request endpoint differs from the sealed endpoint")
        service = tinker.ServiceClient(base_url=self.service_endpoint)
        runtime = await _runtime_attestation(
            service,
            service_endpoint=self.service_endpoint,
            model_name=model_name,
            renderer_name=self.renderer_name,
        )

        training_client = await service.create_lora_training_client_async(
            base_model=model_name,
            rank=lora_rank,
            seed=initialization_seed,
            user_metadata=dict(metadata),
        )
        save_future = await training_client.save_state_async(state_name)
        saved = await save_future.result_async()
        state_uri = getattr(saved, "path", None)
        if not isinstance(state_uri, str):
            raise BranchAssayError("Tinker save_state response lacks a state URI")

        # This is both a renderer/model capability gate and a public fingerprint
        # of the just-saved zero-step state. No game or outcome enters the canary.
        tokenizer = get_tokenizer(model_name)
        if self.renderer_name != "qwen3_disable_thinking_preserve_history":
            raise BranchAssayError("sealed renderer is unsupported by the live adapter")
        renderer = Qwen3DisableThinkingRenderer(tokenizer, strip_thinking_from_history=False)
        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": "Reply with exactly: SPADE_STATE_CANARY"}]
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        params = tinker.SamplingParams(
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            seed=initialization_seed,
            stop=renderer.get_stop_sequences(),
        )
        sample = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        )
        if len(sample.sequences) != 1 or not sample.sequences[0].tokens:
            raise BranchAssayError("zero-step state canary returned no tokens")
        canary_digest = bytes_digest(canonical_json_bytes(list(sample.sequences[0].tokens)))
        return {
            "service_endpoint": self.service_endpoint,
            "state_uri": state_uri,
            "optimizer_state_included": True,
            "sdk_version": runtime["sdk_version"],
            "server_capabilities_digest": runtime["server_capabilities_digest"],
            "canary_token_digest": canary_digest,
            "public_response_metadata": {},
        }


class _TinkerTrainingBoundary:
    """Live same-checkpoint rollout/train/eval adapter for the narrow assay."""

    def __init__(self, *, intent: Mapping[str, Any], max_concurrent: int):
        if max_concurrent <= 0:
            raise BranchAssayError("max_concurrent must be positive")
        try:
            import tinker
            import tinker_cookbook
            import numpy as np
            import torch
            from spade.core.envs.synthetic_game_env import SyntheticGameEnv
            from spade.core.envs.tool_use_base_env import ToolUseBaseEnv
            from spade.core.utils.parsing import parse_action
            from spade.core.utils.rewards import episode_reward
            from tinker_cookbook.renderers.base import get_text_content
            from tinker_cookbook.renderers.qwen3 import Qwen3DisableThinkingRenderer
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as exc:
            raise BranchAssayError(
                "Tinker SDK/cookbook/torch is unavailable; install the pinned runtime first"
            ) from exc
        _require_pinned_cookbook(tinker_cookbook)
        distributions = intent.get("runtime_identity", {}).get("distributions", {})
        if not isinstance(distributions, Mapping):
            raise BranchAssayError("sealed runtime distribution identity is absent")
        for module, distribution_name, bind_package_files in (
            (tinker, "tinker", True),
            (torch, "torch", False),
            (np, "numpy", False),
        ):
            _verify_imported_distribution_module(
                module,
                distribution_name=distribution_name,
                module_name=distribution_name,
                sealed_identity=distributions.get(distribution_name, {}),
                bind_package_files=bind_package_files,
            )
        self.tinker = tinker
        self.torch = torch
        self.synthetic_game_env_class = SyntheticGameEnv
        self.tool_use_base_env_class = ToolUseBaseEnv
        self.parse_action = parse_action
        self.episode_reward = episode_reward
        self.get_text_content = get_text_content
        self.service_endpoint = str(intent["service_endpoint"])
        if self.service_endpoint != DEFAULT_TINKER_ENDPOINT:
            raise BranchAssayError("live adapter refuses a noncanonical Tinker endpoint")
        self.service = tinker.ServiceClient(base_url=self.service_endpoint)
        self.model_name = str(intent["learner"]["model_name"])
        self.renderer_name = str(intent["learner"]["renderer_name"])
        if self.renderer_name != "qwen3_disable_thinking_preserve_history":
            raise BranchAssayError("sealed renderer is unsupported by the live adapter")
        self.tokenizer = get_tokenizer(self.model_name)
        self.renderer = Qwen3DisableThinkingRenderer(
            self.tokenizer, strip_thinking_from_history=False
        )
        if not self.renderer.has_extension_property:
            raise BranchAssayError("renderer lacks the multi-turn sequence-extension property")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise BranchAssayError("learner tokenizer has no usable pad/eos token")
        self._pad_token_id = pad
        self.max_concurrent = max_concurrent
        # One boundary is shared by both concurrently executed branches in a
        # matched pair. This semaphore is therefore the sealed pair-wide
        # logical-sampling ceiling, not a per-branch limit that could double.
        self.sample_semaphore = asyncio.Semaphore(max_concurrent)
        self.training = intent["training"]
        self.learner = intent["learner"]
        self._runtime_attestation: dict[str, Any] | None = None

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    async def attest_runtime(self) -> Mapping[str, Any]:
        if self._runtime_attestation is None:
            self._runtime_attestation = await _runtime_attestation(
                self.service,
                service_endpoint=self.service_endpoint,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
            )
        return dict(self._runtime_attestation)

    async def restore_with_optimizer(
        self, *, state_uri: str, metadata: Mapping[str, str]
    ) -> object:
        return await self.service.create_training_client_from_state_with_optimizer_async(
            state_uri, user_metadata=dict(metadata)
        )

    async def _sampling_client(self, training_client: object):
        return await training_client.save_weights_and_get_sampling_client_async()

    async def _sample_tokens(
        self,
        sampling_client: object,
        *,
        prompt_tokens: Sequence[int],
        seed: int,
        maximum_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[list[int], list[float], str]:
        params = self.tinker.SamplingParams(
            max_tokens=maximum_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=self.renderer.get_stop_sequences(),
        )
        response = await sampling_client.sample_async(
            prompt=self.tinker.ModelInput.from_ints(list(prompt_tokens)),
            num_samples=1,
            sampling_params=params,
        )
        if len(response.sequences) != 1:
            raise BranchAssayError("Tinker sampling returned a non-singleton response")
        sequence = response.sequences[0]
        tokens = list(sequence.tokens)
        if not tokens:
            raise BranchAssayError("Tinker sampling returned no tokens")
        if sequence.logprobs is None or len(sequence.logprobs) != len(tokens):
            raise BranchAssayError("Tinker sampling omitted aligned action log probabilities")
        logprobs = [float(item) for item in sequence.logprobs]
        if any(not math.isfinite(item) for item in logprobs):
            raise BranchAssayError("Tinker sampling returned non-finite log probabilities")
        stop_reason = str(sequence.stop_reason)
        return tokens, logprobs, stop_reason

    async def canary_token_digest(self, client: object, *, seed: int) -> str:
        sampling_client = await self._sampling_client(client)
        prompt = self.renderer.build_generation_prompt(
            [{"role": "user", "content": "Reply with exactly: SPADE_STATE_CANARY"}]
        ).to_ints()
        tokens, _, _ = await self._sample_tokens(
            sampling_client,
            prompt_tokens=prompt,
            seed=seed,
            maximum_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        return bytes_digest(canonical_json_bytes(tokens))

    async def _play_episode(
        self,
        sampling_client: object,
        *,
        game_path: Path,
        game_entry: Mapping[str, Any],
        environment_seed: int,
        sampling_seeds: Sequence[int],
        maximum_turns: int,
        training: bool,
    ) -> dict[str, Any]:
        game_code = _read_sealed_environment(game_path, game_entry)

        def make_and_reset() -> tuple[object, str, int]:
            loader_turn_budget = maximum_turns if training else 1_000_000
            environment = self.synthetic_game_env_class(
                game_code=game_code,
                max_turns=loader_turn_budget,
                respect_game_max_turns=True,
            )
            effective_turns = maximum_turns
            if not training:
                native_turns = getattr(environment.game, "max_turns", None)
                if (
                    isinstance(native_turns, bool)
                    or not isinstance(native_turns, int)
                    or native_turns <= 0
                    or native_turns > MAX_HELDOUT_NATIVE_TURNS
                ):
                    raise BranchAssayError("held-out game has no supported finite native horizon")
                effective_turns = native_turns
            observation, _ = environment.reset(seed=environment_seed)
            return environment, observation, effective_turns

        initialized, episode_rng_state = await asyncio.wait_for(
            asyncio.to_thread(
                _call_with_isolated_global_rng,
                make_and_reset,
                seed=environment_seed,
            ),
            timeout=30.0,
        )
        env, initial_observation, effective_turns = initialized
        tokens: list[int] = []
        loss_mask: list[int] = []
        action_logprobs: list[float] = []
        action_turn_indices: list[int] = []
        rewards: list[float] = []
        terminated = False
        truncated = False
        turn_count = 0
        actions: list[str] = []
        response_token_digests: list[str] = []
        parse_failure_turn: int | None = None
        try:
            messages: list[dict[str, Any]] = [{"role": "user", "content": initial_observation}]
            for turn in range(effective_turns):
                prompt = self.renderer.build_generation_prompt(messages).to_ints()
                if not tokens:
                    tokens.extend(prompt)
                    loss_mask.extend([0] * len(prompt))
                else:
                    if prompt[: len(tokens)] != tokens:
                        raise BranchAssayError(
                            "renderer broke the sealed multi-turn sequence-extension contract"
                        )
                    delta = prompt[len(tokens) :]
                    tokens.extend(delta)
                    loss_mask.extend([0] * len(delta))
                remaining = 4097 - len(tokens)
                if not training:
                    remaining = HELDOUT_CONTEXT_POSITIONS + 1 - len(tokens)
                if remaining <= 0:
                    raise BranchAssayError("episode prompt exceeds the sealed token budget")
                maximum_tokens = min(int(self.training["actor_max_tokens"]), remaining)
                response_tokens, response_logprobs, _ = await self._sample_tokens(
                    sampling_client,
                    prompt_tokens=prompt,
                    seed=sampling_seeds[turn],
                    maximum_tokens=maximum_tokens,
                    temperature=float(self.training["actor_temperature"]),
                    top_p=float(self.training["actor_top_p"]),
                    top_k=int(self.training["actor_top_k"]),
                )
                tokens.extend(response_tokens)
                response_token_digests.append(bytes_digest(canonical_json_bytes(response_tokens)))
                loss_mask.extend([1] * len(response_tokens))
                action_logprobs.extend(response_logprobs)
                action_turn_indices.extend([turn] * len(response_tokens))
                turn_count += 1
                parsed_message, parse_success = self.renderer.parse_response(response_tokens)
                if not parse_success:
                    parse_failure_turn = turn
                    truncated = True
                    break
                messages.append(parsed_message)
                action = self.parse_action(self.get_text_content(parsed_message), "boxed")
                if not isinstance(action, str):
                    raise BranchAssayError("parsed held-out action is not public text")
                actions.append(action)
                stepped, episode_rng_state = await asyncio.wait_for(
                    asyncio.to_thread(
                        _call_with_isolated_global_rng,
                        lambda: env.step(action),
                        state=episode_rng_state,
                    ),
                    timeout=10.0,
                )
                obs, reward, terminated, truncated, _ = stepped
                rewards.append(float(reward))
                if terminated or truncated:
                    break
                messages.append({"role": "user", "content": obs})
        finally:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        _call_with_isolated_global_rng,
                        env.close,
                        state=episode_rng_state,
                    ),
                    timeout=5.0,
                )
            except Exception:
                pass
        # The operation executed the verified in-memory snapshot.  This second
        # read additionally makes any mid-episode path drift terminal before an
        # outcome can enter evidence.
        _read_sealed_environment(game_path, game_entry)
        if turn_count == 0:
            raise BranchAssayError("episode produced no actor turn")
        status = "completed" if terminated else "truncated" if truncated else "timeout"
        raw_reward = float(self.episode_reward(rewards, terminated))
        if training:
            return {
                "tokens": tokens,
                "loss_mask": loss_mask,
                "action_logprobs": action_logprobs,
                "action_turn_indices": action_turn_indices,
                "raw_reward": raw_reward,
                "turn_count": turn_count,
                "status": status,
            }
        return {
            "reward": raw_reward,
            "terminated": terminated,
            "truncated": truncated or not terminated,
            "turn_count": turn_count,
            "actions": actions,
            "response_token_digests": response_token_digests,
            "parse_failure_turn": parse_failure_turn,
            "provider_error": None,
        }

    async def collect_training_rollouts(
        self,
        client: object,
        *,
        pool_dir: Path,
        pool_entries: Sequence[Mapping[str, Any]],
        pair_id: str,
        pair_seed: int,
        update: int,
    ) -> Sequence[EpisodeRollout]:
        sampling_client = await self._sampling_client(client)

        async def one(entry: Mapping[str, Any], replicate: int) -> EpisodeRollout:
            basename = str(entry["basename"])
            env_seed = training_seed(
                pair_id=pair_id,
                pair_seed=pair_seed,
                update=update,
                game_basename=basename,
                replicate=replicate,
            )
            sampling_seeds = tuple(
                training_seed(
                    pair_id=pair_id,
                    pair_seed=pair_seed,
                    update=update,
                    game_basename=basename,
                    replicate=replicate,
                    turn=turn,
                )
                for turn in range(MAX_TRAIN_TURNS)
            )
            async with self.sample_semaphore:
                result = await self._play_episode(
                    sampling_client,
                    game_path=pool_dir / basename,
                    game_entry=entry,
                    environment_seed=env_seed,
                    sampling_seeds=sampling_seeds,
                    maximum_turns=MAX_TRAIN_TURNS,
                    training=True,
                )
            used_sampling_seeds = sampling_seeds[: result["turn_count"]]
            return EpisodeRollout(
                game_basename=basename,
                replicate=replicate,
                environment_seed=env_seed,
                sampling_seeds=used_sampling_seeds,
                tokens=tuple(result["tokens"]),
                loss_mask=tuple(result["loss_mask"]),
                action_logprobs=tuple(result["action_logprobs"]),
                action_turn_indices=tuple(result["action_turn_indices"]),
                raw_reward=result["raw_reward"],
                turn_count=result["turn_count"],
                status=result["status"],
            )

        tasks = [
            one(entry, replicate)
            for entry in pool_entries
            for replicate in range(TRAJECTORIES_PER_GAME)
        ]
        return await asyncio.gather(*tasks)

    async def train_update(
        self,
        client: object,
        *,
        datums: Sequence[object],
        update: int,
        learning_rate: float,
    ) -> Mapping[str, Any]:
        tinker_datums = []
        for datum in datums:
            tinker_datums.append(
                self.tinker.Datum(
                    model_input=self.tinker.ModelInput.from_ints(list(datum.input_tokens)),
                    loss_fn_inputs={
                        "target_tokens": self.tinker.TensorData.from_torch(
                            self.torch.tensor(datum.target_tokens, dtype=self.torch.long)
                        ),
                        "logprobs": self.tinker.TensorData.from_torch(
                            self.torch.tensor(datum.logprobs, dtype=self.torch.float32)
                        ),
                        "advantages": self.tinker.TensorData.from_torch(
                            self.torch.tensor(datum.advantages, dtype=self.torch.float32)
                        ),
                    },
                )
            )
        forward = await client.forward_backward_async(
            tinker_datums, loss_fn=self.learner["loss_function"]
        )
        forward_result = await forward.result_async()
        adam = self.tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=float(self.learner["optimizer_beta1"]),
            beta2=float(self.learner["optimizer_beta2"]),
            eps=float(self.learner["optimizer_epsilon"]),
        )
        optimizer = await client.optim_step_async(adam)
        optimizer_result = await optimizer.result_async()
        metric_source = {
            **(getattr(forward_result, "metrics", None) or {}),
            **(getattr(optimizer_result, "metrics", None) or {}),
        }
        public_metrics = {
            str(key): str(value)[:2048] for key, value in sorted(metric_source.items())
        }
        return {
            "completed_update": update + 1,
            "forward_backward_calls": 1,
            "optimizer_step_calls": 1,
            "submitted_positions": sum(datum.submitted_positions for datum in datums),
            "public_metrics": public_metrics,
        }

    async def save_checkpoint(self, client: object, *, name: str) -> Mapping[str, Any]:
        state_future = await client.save_state_async(name)
        sampler_future = await client.save_weights_for_sampler_async(name)
        state = await state_future.result_async()
        sampler = await sampler_future.result_async()
        completed_update = int(name.rsplit("-", 1)[-1])
        return {
            "state_uri": state.path,
            "sampler_uri": sampler.path,
            "completed_update": completed_update,
            "public_metrics": {},
        }

    async def evaluate_checkpoint(
        self,
        *,
        sampler_uri: str,
        heldout_dir: Path,
        heldout_entries: Sequence[Mapping[str, Any]],
        pair_id: str,
        pair_seed: int,
    ) -> Sequence[Mapping[str, Any]]:
        sampling_client = self.service.create_sampling_client(model_path=sampler_uri)

        async def one(entry: Mapping[str, Any], replicate: int) -> Mapping[str, Any]:
            full_stratum = str(entry["stratum_id"])
            stratum = full_stratum.split("-", 1)[0]
            env_seed, sampling_root = evaluation_seed_pair(
                pair_id=pair_id,
                pair_seed=pair_seed,
                stratum_id=stratum,
                replicate=replicate,
            )
            sampling_seeds = tuple(
                derive_seed("heldout-sampling-turn", sampling_root, turn)
                for turn in range(MAX_HELDOUT_NATIVE_TURNS)
            )
            async with self.sample_semaphore:
                outcome = await self._play_episode(
                    sampling_client,
                    game_path=heldout_dir / str(entry["basename"]),
                    game_entry=entry,
                    environment_seed=env_seed,
                    sampling_seeds=sampling_seeds,
                    maximum_turns=MAX_HELDOUT_NATIVE_TURNS,
                    training=False,
                )
            outcome_body = {
                "stratum_id": stratum,
                "replicate": replicate,
                "environment_seed": env_seed,
                "sampling_seed": sampling_root,
                **outcome,
            }
            return {
                **outcome_body,
                "trajectory_digest": object_digest(outcome_body),
            }

        tasks = [
            one(entry, replicate)
            for entry in heldout_entries
            for replicate in range(HELDOUT_SEEDS_PER_GAME)
        ]
        return await asyncio.gather(*tasks)


def _require_live_authorization(arguments: argparse.Namespace) -> None:
    del arguments
    raise BranchAssayError(
        "this protocol version is a preparatory plan-only HOLD; live Tinker access is disabled "
        "because assignment entropy and provider checkpoint/state receipts are not authenticated"
    )


def _seal_intent(arguments: argparse.Namespace) -> dict[str, Any]:
    bundle = load_learner_pool_manifest(arguments.pool_manifest, verify_files=True)
    if bundle.heldout_v4_dir is None:
        raise BranchAssayError("pool bundle has no sealed held-out v4 panel")
    canonical_root = _canonical_output_root(
        pool_manifest_digest=str(bundle.manifest["manifest_digest"]),
        source_actor_plan_digest=SEALED_ACTOR_PLAN_DIGEST,
    )
    canonical_intent = canonical_root / "intent.json"
    if arguments.output_root.resolve() != canonical_root:
        raise BranchAssayError(f"output root must be the canonical lineage root: {canonical_root}")
    if arguments.intent.resolve() != canonical_intent:
        raise BranchAssayError(f"intent path must be canonical: {canonical_intent}")
    if canonical_root.exists() or canonical_root.is_symlink():
        raise BranchAssayError(
            "this source lineage is already sealed or ambiguous; no duplicate assay is authorized"
        )
    intent = build_intent(
        source_actor_plan_path=arguments.actor_plan.resolve(),
        source_actor_plan_digest=SEALED_ACTOR_PLAN_DIGEST,
        pool_manifest_path=bundle.manifest_path.resolve(),
        output_root=canonical_root,
        assignment_seed=arguments.assignment_seed,
        sampling_seed=arguments.sampling_seed,
        runtime_identity=_runtime_identity(),
        model_name=arguments.model,
        renderer_name=arguments.renderer,
    )
    # Intent first and lock second is deliberately fail-closed: a crash leaves
    # a nonempty canonical root which can neither run nor be silently resealed.
    _write_new(canonical_intent, intent)
    _write_new(canonical_root / _LINEAGE_FILENAME, _lineage_lock(intent))
    _validate_lineage_lock(intent, intent_path=canonical_intent)
    return intent


async def _prepare_base(arguments: argparse.Namespace) -> dict[str, Any]:
    _require_live_authorization(arguments)
    intent = _load_intent(arguments.intent.resolve())
    return await prepare_common_base_state(
        intent=intent,
        run_dir=_run_dir(intent),
        boundary=_TinkerCommonStateBoundary(
            intent=intent,
        ),
    )


def _exact_directory(path: Path, expected: set[str], *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise BranchAssayError(f"{label} is missing or unsafe")
    entries = list(path.iterdir())
    if {item.name for item in entries} != expected or any(item.is_symlink() for item in entries):
        raise BranchAssayError(f"{label} inventory differs")


def _replay_heldout_score(
    *,
    score: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    pool_root: Path,
) -> None:
    """Deterministically replay every recorded public action against sealed bytes."""

    from spade.core.envs.synthetic_game_env import SyntheticGameEnv
    from spade.core.utils.rewards import episode_reward

    heldout = pool_manifest.get("pools", {}).get("heldout_v4")
    if not isinstance(heldout, Mapping) or heldout.get("relative_dir") != "heldout_v4":
        raise BranchAssayError("pool manifest omits the held-out replay panel")
    entries = heldout.get("entries")
    if not isinstance(entries, list):
        raise BranchAssayError("held-out replay panel is malformed")
    by_stratum: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BranchAssayError("held-out replay entry is malformed")
        stratum = str(entry.get("stratum_id", "")).split("-", 1)[0]
        if stratum in by_stratum:
            raise BranchAssayError("held-out replay stratum is duplicated")
        by_stratum[stratum] = entry
    code_by_stratum = {
        stratum: _read_sealed_environment(pool_root / "heldout_v4" / str(entry["basename"]), entry)
        for stratum, entry in by_stratum.items()
    }
    for outcome in score.get("outcomes", []):
        if not isinstance(outcome, Mapping):
            raise BranchAssayError("held-out replay outcome is malformed")
        stratum = outcome.get("stratum_id")
        if stratum not in code_by_stratum:
            raise BranchAssayError("held-out replay outcome has no sealed environment")

        def make_and_reset() -> tuple[object, int]:
            environment = SyntheticGameEnv(
                game_code=code_by_stratum[stratum],
                max_turns=1_000_000,
                respect_game_max_turns=True,
            )
            native_turns = getattr(environment.game, "max_turns", None)
            if (
                isinstance(native_turns, bool)
                or not isinstance(native_turns, int)
                or native_turns <= 0
                or native_turns > MAX_HELDOUT_NATIVE_TURNS
            ):
                raise BranchAssayError("held-out replay has an unsupported native horizon")
            environment.reset(seed=outcome["environment_seed"])
            return environment, native_turns

        initialized, episode_rng_state = _call_with_isolated_global_rng(
            make_and_reset,
            seed=outcome["environment_seed"],
        )
        environment, native_turns = initialized
        actions = outcome["actions"]
        rewards: list[float] = []
        terminated = False
        truncated = False
        try:
            for action_index, action in enumerate(actions):
                if action_index >= native_turns or terminated or truncated:
                    raise BranchAssayError("held-out action trajectory continues past termination")
                stepped, episode_rng_state = _call_with_isolated_global_rng(
                    lambda action=action: environment.step(action),
                    state=episode_rng_state,
                )
                _, reward, terminated, truncated, _ = stepped
                rewards.append(float(reward))
            parse_failure_turn = outcome["parse_failure_turn"]
            if parse_failure_turn is not None:
                if terminated or truncated:
                    raise BranchAssayError("parser failure follows an already-terminal action")
                truncated = True
            elif not terminated and not truncated and len(actions) != native_turns:
                raise BranchAssayError("held-out action trajectory stops before its native horizon")
        finally:
            try:
                _call_with_isolated_global_rng(
                    environment.close,
                    state=episode_rng_state,
                )
            except Exception:
                pass
        replayed = {
            "reward": float(episode_reward(rewards, terminated)),
            "terminated": bool(terminated),
            "truncated": bool(truncated or not terminated),
            "turn_count": len(actions) + (1 if outcome["parse_failure_turn"] is not None else 0),
        }
        if any(outcome[field] != value for field, value in replayed.items()):
            raise BranchAssayError("held-out reward/flags differ from deterministic action replay")

    # No replay may consume bytes different from the manifest, even if a path
    # is mutated and restored between the initial bundle load and this audit.
    for stratum, entry in by_stratum.items():
        _read_sealed_environment(pool_root / "heldout_v4" / str(entry["basename"]), entry)


def _load_complete_pair(
    *,
    pair_root: Path,
    pair_id: str,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    pool_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _exact_directory(
        pair_root,
        {"slot_a", "slot_b", "pair-complete.json"},
        label=f"completed {pair_id}",
    )
    expected_requests = {
        item["slot_id"]: item
        for item in branch_requests(intent, receipt)
        if item["pair_id"] == pair_id
    }
    requests: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for slot_id in ("slot_a", "slot_b"):
        slot_root = pair_root / slot_id
        _exact_directory(
            slot_root,
            {"request.json", "execution.json", "heldout-score.json"},
            label=f"{pair_id}/{slot_id}",
        )
        request = validate_branch_request(
            read_json(slot_root / "request.json"),
            intent=intent,
            base_receipt=receipt,
        )
        if request != expected_requests[slot_id]:
            raise BranchAssayError(f"{pair_id}/{slot_id} request differs")
        score = validate_heldout_score(
            read_json(slot_root / "heldout-score.json", maximum_bytes=32_000_000),
            intent=intent,
            branch_request=request,
            base_receipt=receipt,
        )
        _replay_heldout_score(
            score=score,
            pool_manifest=pool_manifest,
            pool_root=pool_root,
        )
        execution = validate_branch_execution_receipt(
            read_json(slot_root / "execution.json", maximum_bytes=32_000_000),
            intent=intent,
            base_receipt=receipt,
            branch_request=request,
            pool_manifest=pool_manifest,
            score=score,
        )
        requests.append(request)
        executions.append(execution)
        scores.append(score)
    complete = validate_pair_complete(
        read_json(pair_root / "pair-complete.json"),
        intent=intent,
        base_receipt=receipt,
        pair_id=pair_id,
        requests=requests,
        executions=executions,
        scores=scores,
        pool_manifest=pool_manifest,
    )
    return executions, scores, complete


def _load_terminal_pair(
    *,
    pair_root: Path,
    pair_id: str,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_directory(
        pair_root,
        {"slot_a", "slot_b", "pair-terminal-error.json"},
        label=f"terminal {pair_id}",
    )
    requests: list[dict[str, Any]] = []
    for slot_id in ("slot_a", "slot_b"):
        slot_root = pair_root / slot_id
        _exact_directory(slot_root, {"request.json"}, label=f"{pair_id}/{slot_id}")
        request = validate_branch_request(
            read_json(slot_root / "request.json"),
            intent=intent,
            base_receipt=receipt,
        )
        if request["pair_id"] != pair_id or request["slot_id"] != slot_id:
            raise BranchAssayError("terminal pair request topology differs")
        requests.append(request)
    error = read_json(pair_root / "pair-terminal-error.json")
    return validate_pair_terminal_error(
        error,
        intent=intent,
        base_receipt=receipt,
        pair_id=pair_id,
        requests=requests,
        error_type=error.get("error_type"),
    )


def _validate_run_evidence(
    *,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    pool_root: Path,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate the exact run/pair frontier and return all rooted leaves."""

    run_dir = _run_dir(intent)
    allowed_root = {"common-base-state"}
    branches_root = run_dir / "branches"
    aggregate_path = run_dir / "aggregate.json"
    if branches_root.exists() or branches_root.is_symlink():
        allowed_root.add("branches")
    if aggregate_path.exists() or aggregate_path.is_symlink():
        allowed_root.add("aggregate.json")
    _exact_directory(run_dir, allowed_root, label="learner assay run root")

    schedule_ids = [item["pair_id"] for item in intent["pair_schedule"]]
    all_executions: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    all_completions: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    observed_ids: list[str] = []
    if "branches" in allowed_root:
        if branches_root.is_symlink() or not branches_root.is_dir():
            raise BranchAssayError("branches root is unsafe")
        entries = list(branches_root.iterdir())
        if any(item.is_symlink() or not item.is_dir() for item in entries):
            raise BranchAssayError("branches root contains a noncanonical pair entry")
        observed_names = {item.name for item in entries}
        if not observed_names.issubset(set(schedule_ids)):
            raise BranchAssayError("branches root contains an unknown pair")
        observed_ids = [pair_id for pair_id in schedule_ids if pair_id in observed_names]
        if observed_ids != schedule_ids[: len(observed_ids)]:
            raise BranchAssayError("pair evidence is not a sealed-order prefix")
        for index, pair_id in enumerate(observed_ids):
            pair_root = branches_root / pair_id
            names = {item.name for item in pair_root.iterdir()} if pair_root.is_dir() else set()
            if "pair-complete.json" in names:
                executions, scores, complete = _load_complete_pair(
                    pair_root=pair_root,
                    pair_id=pair_id,
                    intent=intent,
                    receipt=receipt,
                    pool_manifest=pool_manifest,
                    pool_root=pool_root,
                )
                all_executions.extend(executions)
                all_scores.extend(scores)
                all_completions.append(complete)
            elif "pair-terminal-error.json" in names:
                if index != len(observed_ids) - 1:
                    raise BranchAssayError("terminal pair is followed by later pair evidence")
                terminal = _load_terminal_pair(
                    pair_root=pair_root,
                    pair_id=pair_id,
                    intent=intent,
                    receipt=receipt,
                )
            else:
                raise BranchAssayError(
                    f"{pair_id} is a partial/ambiguous pair; no resume is authorized"
                )

    if aggregate_path.exists():
        if terminal is not None or len(all_completions) != len(schedule_ids):
            raise BranchAssayError("aggregate exists without all six completed pairs")
        aggregate = validate_aggregate(
            read_json(aggregate_path, maximum_bytes=64_000_000),
            intent=intent,
            base_receipt=receipt,
            scores=all_scores,
            executions=all_executions,
            pair_completions=all_completions,
            pool_manifest=pool_manifest,
        )
    else:
        aggregate = None
    if require_complete and (
        terminal is not None or len(all_completions) != len(schedule_ids) or aggregate is not None
    ):
        raise BranchAssayError("analysis requires six complete pairs and no pre-existing aggregate")
    return {
        "executions": all_executions,
        "scores": all_scores,
        "pair_completions": all_completions,
        "terminal": terminal,
        "aggregate": aggregate,
        "observed_pair_ids": observed_ids,
    }


def _validate(arguments: argparse.Namespace) -> dict[str, Any]:
    intent = _load_intent(arguments.intent.resolve())
    bundle = load_learner_pool_manifest(Path(intent["pool_manifest_path"]), verify_files=True)
    if bundle.manifest_path.resolve() != Path(intent["pool_manifest_path"]).resolve():
        raise BranchAssayError("pool manifest canonical path differs")
    result: dict[str, Any] = {
        "intent_digest": intent["intent_digest"],
        "pool_manifest_digest": bundle.manifest["manifest_digest"],
        "run_dir": str(_run_dir(intent)),
        "provider_calls": 0,
        "learner_updates": 0,
    }
    base_dir = _run_dir(intent) / "common-base-state"
    if base_dir.exists():
        receipt = validate_common_base_tree(intent=intent, run_dir=_run_dir(intent))
        result["base_receipt_digest"] = receipt["receipt_digest"]
        result["branch_request_count"] = len(branch_requests(intent, receipt))
        evidence = _validate_run_evidence(
            intent=intent,
            receipt=receipt,
            pool_manifest=bundle.manifest,
            pool_root=bundle.root.resolve(),
            require_complete=False,
        )
        result["completed_pair_count"] = len(evidence["pair_completions"])
        result["terminal_pair"] = (
            evidence["terminal"]["pair_id"] if evidence["terminal"] is not None else None
        )
        result["aggregate_digest"] = (
            evidence["aggregate"]["aggregate_digest"] if evidence["aggregate"] is not None else None
        )
    else:
        if _run_dir(intent).exists() or _run_dir(intent).is_symlink():
            raise BranchAssayError("run root exists without complete common-state evidence")
        result["base_receipt_digest"] = None
        result["branch_request_count"] = 0
        result["completed_pair_count"] = 0
        result["terminal_pair"] = None
        result["aggregate_digest"] = None
    return result


def _analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    intent = _load_intent(arguments.intent.resolve())
    run_dir = _run_dir(intent)
    receipt = validate_common_base_tree(intent=intent, run_dir=run_dir)
    bundle = load_learner_pool_manifest(Path(intent["pool_manifest_path"]), verify_files=True)
    evidence = _validate_run_evidence(
        intent=intent,
        receipt=receipt,
        pool_manifest=bundle.manifest,
        pool_root=bundle.root.resolve(),
        require_complete=True,
    )
    aggregate = analyze_scores(
        intent=intent,
        base_receipt=receipt,
        scores=evidence["scores"],
        executions=evidence["executions"],
        pair_completions=evidence["pair_completions"],
        pool_manifest=bundle.manifest,
    )
    validate_aggregate(
        aggregate,
        intent=intent,
        base_receipt=receipt,
        scores=evidence["scores"],
        executions=evidence["executions"],
        pair_completions=evidence["pair_completions"],
        pool_manifest=bundle.manifest,
    )
    destination = run_dir / "aggregate.json"
    _write_new(destination, aggregate)
    frozen = _validate_run_evidence(
        intent=intent,
        receipt=receipt,
        pool_manifest=bundle.manifest,
        pool_root=bundle.root.resolve(),
        require_complete=False,
    )["aggregate"]
    if frozen != aggregate:
        raise BranchAssayError("persisted aggregate differs from its verified bytes")
    return aggregate


async def _run_pair(arguments: argparse.Namespace) -> dict[str, Any]:
    _require_live_authorization(arguments)
    intent = _load_intent(arguments.intent.resolve())
    run_dir = _run_dir(intent)
    receipt = validate_common_base_tree(intent=intent, run_dir=run_dir)
    bundle = load_learner_pool_manifest(Path(intent["pool_manifest_path"]), verify_files=True)
    schedule = intent["pair_schedule"]
    pair_ids = [item["pair_id"] for item in schedule]
    if arguments.pair_id not in pair_ids:
        raise BranchAssayError("pair_id is outside the sealed schedule")
    pair_index = pair_ids.index(arguments.pair_id)
    evidence = _validate_run_evidence(
        intent=intent,
        receipt=receipt,
        pool_manifest=bundle.manifest,
        pool_root=bundle.root.resolve(),
        require_complete=False,
    )
    if evidence["aggregate"] is not None or evidence["terminal"] is not None:
        raise BranchAssayError("terminal or aggregate evidence forbids further pair launches")
    if evidence["observed_pair_ids"] != pair_ids[:pair_index]:
        raise BranchAssayError("pairs must be launched in their sealed order")
    branches_root = run_dir / "branches"
    pair_root = branches_root / arguments.pair_id
    if pair_root.exists() or pair_root.is_symlink():
        raise BranchAssayError(
            "pair evidence already exists; completed and ambiguous pairs are never overwritten"
        )

    requests = [
        item for item in branch_requests(intent, receipt) if item["pair_id"] == arguments.pair_id
    ]
    if len(requests) != 2:
        raise AssertionError("sealed pair does not contain two branches")

    # Local imports/renderer construction and a read-only server capability
    # check happen before the irreversible pair reservation.  They cannot
    # expose an outcome or update a checkpoint.
    boundary = _TinkerTrainingBoundary(
        intent=intent,
        max_concurrent=DEFAULT_MAX_CONCURRENT_SAMPLES,
    )
    validate_runtime_attestation(
        await boundary.attest_runtime(),
        intent=intent,
        base_receipt=receipt,
    )
    # Boundary construction performs every lazy runtime import used during a
    # branch. Revalidate those source bytes and the complete immutable game
    # bundle after imports but before reserving either expensive branch.
    if _load_intent(arguments.intent.resolve()) != intent:
        raise BranchAssayError("intent/runtime bytes drifted during branch preflight")
    preflight_bundle = load_learner_pool_manifest(
        Path(intent["pool_manifest_path"]), verify_files=True
    )
    if (
        preflight_bundle.root.resolve() != bundle.root.resolve()
        or preflight_bundle.manifest != bundle.manifest
    ):
        raise BranchAssayError("pool bundle drifted during branch preflight")

    pair_root.mkdir(parents=True)
    for request in requests:
        slot_root = pair_root / request["slot_id"]
        slot_root.mkdir()
        _write_new(slot_root / "request.json", request)

    async def execute_verified_branch(request: Mapping[str, Any]):
        before = load_learner_pool_manifest(Path(intent["pool_manifest_path"]), verify_files=True)
        if before.root.resolve() != bundle.root.resolve() or before.manifest != bundle.manifest:
            raise BranchAssayError("pool bundle drifted before branch execution")
        execution = await execute_branch(
            intent=intent,
            base_receipt=receipt,
            branch_request=request,
            pool_manifest=bundle.manifest,
            pool_root=bundle.root.resolve(),
            boundary=boundary,
        )
        after = load_learner_pool_manifest(Path(intent["pool_manifest_path"]), verify_files=True)
        if after.root.resolve() != bundle.root.resolve() or after.manifest != bundle.manifest:
            raise BranchAssayError("pool bundle drifted during branch execution")
        if _load_intent(arguments.intent.resolve()) != intent:
            raise BranchAssayError("intent/runtime bytes drifted during branch execution")
        return execution

    try:
        executions = await asyncio.gather(
            *(execute_verified_branch(request) for request in requests)
        )
    except BaseException as exc:
        # The persisted branch reservations make the pair permanently unusable;
        # this avoids selecting a rerun based on partially observed outcomes.
        error = build_pair_terminal_error(
            intent=intent,
            base_receipt=receipt,
            pair_id=arguments.pair_id,
            requests=requests,
            error_type=type(exc).__name__,
        )
        _write_new(pair_root / "pair-terminal-error.json", error)
        raise BranchAssayError(
            "pair failed after reservation; this assay is terminal and authorizes no retry"
        ) from exc

    execution_receipts: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for request, execution in zip(requests, executions):
        slot_root = pair_root / request["slot_id"]
        execution_value = build_branch_execution_receipt(
            intent=intent,
            base_receipt=receipt,
            branch_request=request,
            pool_manifest=bundle.manifest,
            execution=execution,
        )
        validate_branch_execution_receipt(
            execution_value,
            intent=intent,
            base_receipt=receipt,
            branch_request=request,
            pool_manifest=bundle.manifest,
            score=execution.score,
        )
        _write_new(slot_root / "execution.json", execution_value)
        _write_new(slot_root / "heldout-score.json", execution.score)
        execution_receipts.append(execution_value)
        scores.append(dict(execution.score))
    pair_complete = build_pair_complete(
        intent=intent,
        base_receipt=receipt,
        pair_id=arguments.pair_id,
        requests=requests,
        executions=execution_receipts,
        scores=scores,
        pool_manifest=bundle.manifest,
    )
    _write_new(pair_root / "pair-complete.json", pair_complete)
    persisted = _load_complete_pair(
        pair_root=pair_root,
        pair_id=arguments.pair_id,
        intent=intent,
        receipt=receipt,
        pool_manifest=bundle.manifest,
        pool_root=bundle.root.resolve(),
    )[2]
    if persisted != pair_complete:
        raise BranchAssayError("persisted pair completion differs from its verified bytes")
    return pair_complete


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal-intent", help="write a prospective zero-call intent")
    seal.add_argument("--actor-plan", required=True, type=Path)
    seal.add_argument("--pool-manifest", required=True, type=Path)
    seal.add_argument("--output-root", required=True, type=Path)
    seal.add_argument("--intent", required=True, type=Path)
    seal.add_argument("--assignment-seed", required=True, type=int)
    seal.add_argument("--sampling-seed", required=True, type=int)
    seal.add_argument("--model", default=DEFAULT_MODEL)
    seal.add_argument("--renderer", default=DEFAULT_RENDERER)

    validate = subparsers.add_parser("validate", help="offline intent/evidence validation")
    validate.add_argument("--intent", required=True, type=Path)

    prepare = subparsers.add_parser(
        "prepare-base", help="create the sole optimizer-bearing common state"
    )
    prepare.add_argument("--intent", required=True, type=Path)
    prepare.add_argument("--allow-live", action="store_true")

    analyze = subparsers.add_parser("analyze", help="offline exact paired analysis")
    analyze.add_argument("--intent", required=True, type=Path)

    run_pair = subparsers.add_parser(
        "run-pair", help="run the next sealed matched pair through training and evaluation"
    )
    run_pair.add_argument("--intent", required=True, type=Path)
    run_pair.add_argument("--pair-id", required=True)
    run_pair.add_argument("--allow-live", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "seal-intent":
            value = _seal_intent(arguments)
        elif arguments.command == "validate":
            value = _validate(arguments)
        elif arguments.command == "prepare-base":
            value = asyncio.run(_prepare_base(arguments))
        elif arguments.command == "analyze":
            value = _analyze(arguments)
        elif arguments.command == "run-pair":
            value = asyncio.run(_run_pair(arguments))
        else:  # pragma: no cover - argparse owns the command set
            raise AssertionError(arguments.command)
    except (BranchAssayError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_pretty(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
