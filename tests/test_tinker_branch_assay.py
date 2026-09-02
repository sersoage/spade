from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import spade.tinker.branch_assay as branch_assay

from spade.tinker.branch_assay import (
    ARMS,
    DEFAULT_TINKER_ENDPOINT,
    EXACT_SIGN_FLIP_COUNT,
    HELDOUT_SEEDS_PER_GAME,
    HELDOUT_STRATA,
    PAIR_COUNT,
    TRAIN_GAMES_PER_ARM,
    TRAIN_POSITIONS_PER_BRANCH,
    TRAIN_POSITIONS_PER_EPISODE,
    TRAJECTORIES_PER_GAME,
    BranchExecution,
    BranchAssayError,
    EpisodeRollout,
    analyze_scores,
    branch_requests,
    build_branch_execution_receipt,
    build_heldout_score,
    build_intent,
    build_pair_complete,
    build_pair_schedule,
    bytes_digest,
    execute_branch,
    normalize_and_pad_training_rollouts,
    object_digest,
    pad_episode_datum,
    prepare_common_base_state,
    training_seed,
    validate_aggregate,
    validate_base_receipt,
    validate_common_base_tree,
    validate_heldout_score,
    validate_intent,
)


_TEST_POOL_BODY = {"schema_version": "test-pools/v1"}
_TEST_POOL_DIGEST = object_digest(_TEST_POOL_BODY)
_TEST_RUNTIME_IDENTITY = {
    "spade_git_head": "a" * 40,
    "distributions": {"tinker": {"version": "test-sdk"}},
}


@pytest.fixture(autouse=True)
def _authorize_test_pool(monkeypatch):
    """Keep production pinned to c10b while unit tests use a tiny local fixture."""

    monkeypatch.setattr(
        branch_assay,
        "AUTHORIZED_POOL_MANIFEST_DIGEST",
        _TEST_POOL_DIGEST,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sealed_actor_plan(path: Path) -> tuple[dict, str]:
    body = {"schema_version": "test-actor-plan/v1", "candidate_evidence": []}
    digest = object_digest(body)
    plan = {**body, "actor_plan_digest": digest}
    _write_json(path, plan)
    return plan, digest


def _intent(tmp_path: Path) -> dict:
    actor_path = (tmp_path / "actor-plan.json").resolve()
    _, actor_digest = _sealed_actor_plan(actor_path)
    manifest_path = (tmp_path / "learner-pools-manifest.json").resolve()
    _write_json(
        manifest_path,
        {**_TEST_POOL_BODY, "manifest_digest": _TEST_POOL_DIGEST},
    )
    output_root = (tmp_path / "runs").resolve()
    return build_intent(
        source_actor_plan_path=actor_path,
        source_actor_plan_digest=actor_digest,
        pool_manifest_path=manifest_path,
        output_root=output_root,
        assignment_seed=20260902,
        sampling_seed=910247,
        runtime_identity=_TEST_RUNTIME_IDENTITY,
    )


class _FakeCommonState:
    def __init__(self, *, fail: bool = False):
        self.calls = 0
        self.fail = fail

    async def create_and_save_common_state(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("sanitized failure")
        assert kwargs["model_name"] == "Qwen/Qwen3-8B"
        assert kwargs["service_endpoint"] == DEFAULT_TINKER_ENDPOINT
        return {
            "service_endpoint": DEFAULT_TINKER_ENDPOINT,
            "state_uri": "tinker://run-test/state/common",
            "optimizer_state_included": True,
            "sdk_version": "test-sdk",
            "server_capabilities_digest": bytes_digest(b"capabilities"),
            "canary_token_digest": bytes_digest(b"canary"),
            "public_response_metadata": {"request_id": "public-test-id"},
        }


def _base(tmp_path: Path, intent: dict) -> tuple[Path, dict]:
    run_dir = Path(intent["output_root"]) / "run"
    boundary = _FakeCommonState()
    receipt = asyncio.run(
        prepare_common_base_state(intent=intent, run_dir=run_dir, boundary=boundary)
    )
    assert boundary.calls == 1
    return run_dir, receipt


def test_schedule_uses_six_independent_coin_flips_and_counterbalanced_execution() -> None:
    schedule = build_pair_schedule(20260902, 910247)
    assert len(schedule) == PAIR_COUNT
    assert all(set(item["slot_assignments"].values()) == set(ARMS) for item in schedule)
    assert [item["slot_assignments"]["slot_a"] for item in schedule] == [
        "coverage_forced",
        "redundant_historical",
        "coverage_forced",
        "coverage_forced",
        "coverage_forced",
        "redundant_historical",
    ]
    assert {tuple(item["execution_order"]) for item in schedule} == {
        ("slot_a", "slot_b"),
        ("slot_b", "slot_a"),
    }
    assert schedule == build_pair_schedule(20260902, 910247)
    reassigned = build_pair_schedule(20260903, 910247)
    assert schedule != reassigned
    assert [item["pair_seed"] for item in schedule] == [item["pair_seed"] for item in reassigned]
    resampled = build_pair_schedule(20260902, 910248)
    assert [item["slot_assignments"] for item in schedule] == [
        item["slot_assignments"] for item in resampled
    ]
    assert [item["pair_seed"] for item in schedule] != [item["pair_seed"] for item in resampled]


def test_intent_binds_files_and_recomputes_schedule(tmp_path: Path) -> None:
    intent = _intent(tmp_path)
    assert validate_intent(intent) == intent
    changed = json.loads(json.dumps(intent))
    changed["pair_schedule"][0]["pair_seed"] += 1
    body = {key: value for key, value in changed.items() if key != "intent_digest"}
    changed["intent_digest"] = object_digest(body)
    with pytest.raises(BranchAssayError, match="pair schedule"):
        validate_intent(changed)

    changed_endpoint = json.loads(json.dumps(intent))
    changed_endpoint["service_endpoint"] = "https://example.invalid/tinker"
    endpoint_body = {
        key: value for key, value in changed_endpoint.items() if key != "intent_digest"
    }
    changed_endpoint["intent_digest"] = object_digest(endpoint_body)
    with pytest.raises(BranchAssayError, match="service endpoint"):
        validate_intent(changed_endpoint)

    changed_assignment = json.loads(json.dumps(intent))
    assignments = changed_assignment["pair_schedule"][0]["slot_assignments"]
    assignments["slot_a"], assignments["slot_b"] = (
        assignments["slot_b"],
        assignments["slot_a"],
    )
    assignment_body = {
        key: value for key, value in changed_assignment.items() if key != "intent_digest"
    }
    changed_assignment["intent_digest"] = object_digest(assignment_body)
    with pytest.raises(BranchAssayError, match="pair schedule"):
        validate_intent(changed_assignment)

    substituted = json.loads(json.dumps(intent))
    substituted["learner"]["checkpoint_sampling_backend"] = "agy-google"
    substituted_body = {key: value for key, value in substituted.items() if key != "intent_digest"}
    substituted["intent_digest"] = object_digest(substituted_body)
    with pytest.raises(BranchAssayError, match="substitute another sampling backend"):
        validate_intent(substituted)

    claim_enabled = json.loads(json.dumps(intent))
    claim_enabled["evidence_authentication"]["causal_claim_authorized"] = True
    claim_body = {key: value for key, value in claim_enabled.items() if key != "intent_digest"}
    claim_enabled["intent_digest"] = object_digest(claim_body)
    with pytest.raises(BranchAssayError, match="authentication HOLD"):
        validate_intent(claim_enabled)

    Path(intent["pool_manifest_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(BranchAssayError, match="bytes differ"):
        validate_intent(intent)


def test_common_state_is_one_call_optimizer_bearing_and_resume_safe(tmp_path: Path) -> None:
    intent = _intent(tmp_path)
    run_dir = Path(intent["output_root"]) / "run"
    boundary = _FakeCommonState()
    first = asyncio.run(
        prepare_common_base_state(intent=intent, run_dir=run_dir, boundary=boundary)
    )
    second = asyncio.run(
        prepare_common_base_state(intent=intent, run_dir=run_dir, boundary=boundary)
    )
    assert first == second
    assert boundary.calls == 1
    assert first["optimizer_state_included"] is True
    assert validate_common_base_tree(intent=intent, run_dir=run_dir) == first

    requests = branch_requests(intent, first)
    assert len(requests) == 12
    assert {item["common_state_uri_digest"] for item in requests} == {first["state_uri_digest"]}
    assert {item["base_receipt_digest"] for item in requests} == {first["receipt_digest"]}
    for pair_id in {item["pair_id"] for item in requests}:
        assert {item["arm"] for item in requests if item["pair_id"] == pair_id} == set(ARMS)


def test_ambiguous_common_state_never_retries(tmp_path: Path) -> None:
    intent = _intent(tmp_path)
    run_dir = Path(intent["output_root"]) / "run"
    boundary = _FakeCommonState(fail=True)
    with pytest.raises(BranchAssayError, match="cannot be retried"):
        asyncio.run(prepare_common_base_state(intent=intent, run_dir=run_dir, boundary=boundary))
    assert boundary.calls == 1
    with pytest.raises(BranchAssayError, match="partial or ambiguous"):
        asyncio.run(prepare_common_base_state(intent=intent, run_dir=run_dir, boundary=boundary))
    assert boundary.calls == 1


def test_common_state_rejects_endpoint_substitution(tmp_path: Path) -> None:
    class _WrongEndpoint(_FakeCommonState):
        async def create_and_save_common_state(self, **kwargs):
            result = dict(await super().create_and_save_common_state(**kwargs))
            result["service_endpoint"] = "https://example.invalid/tinker"
            return result

    intent = _intent(tmp_path)
    run_dir = Path(intent["output_root"]) / "run"
    with pytest.raises(BranchAssayError, match="cannot be retried"):
        asyncio.run(
            prepare_common_base_state(
                intent=intent,
                run_dir=run_dir,
                boundary=_WrongEndpoint(),
            )
        )


def test_base_receipt_tamper_and_path_escape_fail(tmp_path: Path) -> None:
    intent = _intent(tmp_path)
    run_dir, receipt = _base(tmp_path, intent)
    tampered = {**receipt, "state_uri": "tinker://other/state"}
    with pytest.raises(BranchAssayError, match="self-digest"):
        validate_base_receipt(tampered, intent=intent)
    with pytest.raises(BranchAssayError, match="escapes"):
        asyncio.run(
            prepare_common_base_state(
                intent=intent,
                run_dir=(tmp_path / "outside").resolve(),
                boundary=_FakeCommonState(),
            )
        )
    assert run_dir.exists()


def test_padding_is_exact_and_zero_advantage(tmp_path: Path) -> None:
    _ = tmp_path
    datum = pad_episode_datum(
        tokens=[10, 11, 12, 13],
        loss_mask=[0, 0, 1, 1],
        action_logprobs=[-0.2, -0.3],
        action_advantages=[1.0, 1.0],
        pad_token_id=0,
        submitted_positions=8,
    )
    assert len(datum.input_tokens) == len(datum.target_tokens) == 8
    assert len(datum.logprobs) == len(datum.advantages) == 8
    assert datum.actual_positions == 3
    assert datum.padded_positions == 5
    assert datum.action_positions == 2
    assert datum.advantages[-5:] == (0.0,) * 5

    with pytest.raises(BranchAssayError, match="exceeding"):
        pad_episode_datum(
            tokens=list(range(11)),
            loss_mask=[0] + [1] * 10,
            action_logprobs=[-1.0] * 10,
            action_advantages=[1.0] * 10,
            pad_token_id=0,
            submitted_positions=8,
        )


def _rollouts(pair_id: str, pair_seed: int, update: int) -> list[EpisodeRollout]:
    results: list[EpisodeRollout] = []
    for game_index in range(TRAIN_GAMES_PER_ARM):
        basename = f"game_{game_index:03d}.py"
        for replicate in range(TRAJECTORIES_PER_GAME):
            reward = float(replicate % 2)
            results.append(
                EpisodeRollout(
                    game_basename=basename,
                    replicate=replicate,
                    environment_seed=training_seed(
                        pair_id=pair_id,
                        pair_seed=pair_seed,
                        update=update,
                        game_basename=basename,
                        replicate=replicate,
                    ),
                    sampling_seeds=(
                        training_seed(
                            pair_id=pair_id,
                            pair_seed=pair_seed,
                            update=update,
                            game_basename=basename,
                            replicate=replicate,
                            turn=0,
                        ),
                    ),
                    tokens=(10, 11, 12, 13),
                    loss_mask=(0, 0, 1, 1),
                    action_logprobs=(-0.2, -0.3),
                    action_turn_indices=(0, 0),
                    raw_reward=reward,
                    turn_count=1,
                    status="completed",
                )
            )
    return results


def test_complete_batch_is_balanced_seeded_normalized_and_compute_fixed() -> None:
    basenames = [f"game_{index:03d}.py" for index in range(TRAIN_GAMES_PER_ARM)]
    padded, audit = normalize_and_pad_training_rollouts(
        rollouts=_rollouts("pair-01", 1234, 0),
        expected_basenames=basenames,
        pair_id="pair-01",
        pair_seed=1234,
        update=0,
        pad_token_id=0,
    )
    assert len(padded) == 12
    assert all(len(group) == 8 for group in padded.values())
    assert audit["episodes"] == 96
    assert audit["mixed_reward_games"] == 12
    assert audit["submitted_positions"] == 96 * TRAIN_POSITIONS_PER_EPISODE

    missing = _rollouts("pair-01", 1234, 0)[:-1]
    with pytest.raises(BranchAssayError, match="upsampling is forbidden"):
        normalize_and_pad_training_rollouts(
            rollouts=missing,
            expected_basenames=basenames,
            pair_id="pair-01",
            pair_seed=1234,
            update=0,
            pad_token_id=0,
        )


class _FakeTrainingBoundary:
    pad_token_id = 0

    def __init__(self, canary: str):
        self.canary = canary
        self.train_calls = 0
        self.save_calls = 0

    async def attest_runtime(self):
        return {
            "backend": "tinker",
            "service_endpoint": DEFAULT_TINKER_ENDPOINT,
            "model_name": "Qwen/Qwen3-8B",
            "renderer_name": "qwen3_disable_thinking_preserve_history",
            "sdk_version": "test-sdk",
            "server_capabilities_digest": bytes_digest(b"capabilities"),
        }

    async def restore_with_optimizer(self, *, state_uri, metadata):
        assert state_uri == "tinker://run-test/state/common"
        assert metadata["arm"] in ARMS
        return object()

    async def canary_token_digest(self, client, *, seed):
        _ = client, seed
        return self.canary

    async def collect_training_rollouts(
        self, client, *, pool_dir, pool_entries, pair_id, pair_seed, update
    ):
        _ = client, pool_dir
        values = _rollouts(pair_id, pair_seed, update)
        expected_names = [entry["basename"] for entry in pool_entries]
        assert [f"game_{index:03d}.py" for index in range(12)] == expected_names
        return values

    async def train_update(self, client, *, datums, update, learning_rate):
        _ = client
        assert len(datums) == 96
        assert all(item.submitted_positions == TRAIN_POSITIONS_PER_EPISODE for item in datums)
        assert learning_rate == 1e-6
        self.train_calls += 1
        return {
            "completed_update": update + 1,
            "forward_backward_calls": 1,
            "optimizer_step_calls": 1,
            "submitted_positions": 96 * TRAIN_POSITIONS_PER_EPISODE,
            "public_metrics": {},
        }

    async def save_checkpoint(self, client, *, name):
        _ = client
        self.save_calls += 1
        update = int(name.rsplit("-", 1)[-1])
        return {
            "state_uri": f"tinker://fake/{name}/state",
            "sampler_uri": f"tinker://fake/{name}/sampler",
            "completed_update": update,
            "public_metrics": {},
        }

    async def evaluate_checkpoint(
        self, *, sampler_uri, heldout_dir, heldout_entries, pair_id, pair_seed
    ):
        _ = sampler_uri, heldout_dir, heldout_entries
        pair = {"pair_id": pair_id, "pair_seed": pair_seed}
        return _outcomes(pair, 0.5)


def _fake_pool_bundle(tmp_path: Path) -> tuple[Path, dict]:
    root = (tmp_path / "pools").resolve()
    pools = {}
    game_code = (
        "class ReplayEnv:\n"
        "    max_turns = 1\n"
        "    def reset(self, seed=None):\n"
        "        return 'test observation', {}\n"
        "    def step(self, action):\n"
        "        reward = 0.5 if action == 'answer' else 0.0\n"
        "        return 'done', reward, True, False, {}\n"
    )
    encoded_game = game_code.encode("utf-8")
    for name in (*ARMS, "heldout_v4"):
        directory = root / name
        directory.mkdir(parents=True)
        count = 12
        entries = []
        for index in range(count):
            suffix = "heldout" if name == "heldout_v4" else "train"
            basename = f"game_{index:03d}.py"
            (directory / basename).write_bytes(encoded_game)
            entries.append(
                {
                    "basename": basename,
                    "stratum_id": (
                        f"{HELDOUT_STRATA[index]}-{suffix}"
                        if name == "heldout_v4"
                        else f"c{index:03d}-{suffix}"
                    ),
                    "size_bytes": len(encoded_game),
                    "environment_digest": bytes_digest(encoded_game),
                    "code_digest": object_digest(game_code),
                }
            )
        pools[name] = {"relative_dir": name, "entries": entries}
    return root, {"pools": pools}


def test_full_sixteen_update_branch_runs_on_injected_boundary(tmp_path: Path) -> None:
    intent = _intent(tmp_path)
    _, receipt = _base(tmp_path, intent)
    request = branch_requests(intent, receipt)[0]
    pool_root, pool_manifest = _fake_pool_bundle(tmp_path)
    boundary = _FakeTrainingBoundary(receipt["canary_token_digest"])
    execution = asyncio.run(
        execute_branch(
            intent=intent,
            base_receipt=receipt,
            branch_request=request,
            pool_manifest=pool_manifest,
            pool_root=pool_root,
            boundary=boundary,
        )
    )
    assert boundary.train_calls == boundary.save_calls == 16
    assert len(execution.update_audits) == len(execution.checkpoint_receipts) == 16
    assert execution.score["completed_updates"] == 16
    assert execution.score["submitted_training_positions"] == TRAIN_POSITIONS_PER_BRANCH

    bad_boundary = _FakeTrainingBoundary(bytes_digest(b"wrong state"))
    with pytest.raises(BranchAssayError, match="canary differs"):
        asyncio.run(
            execute_branch(
                intent=intent,
                base_receipt=receipt,
                branch_request=request,
                pool_manifest=pool_manifest,
                pool_root=pool_root,
                boundary=bad_boundary,
            )
        )


def _outcomes(pair: dict, reward: float) -> list[dict]:
    from spade.tinker.branch_assay import evaluation_seed_pair

    values = []
    for stratum in HELDOUT_STRATA:
        for replicate in range(HELDOUT_SEEDS_PER_GAME):
            env_seed, sample_seed = evaluation_seed_pair(
                pair_id=pair["pair_id"],
                pair_seed=pair["pair_seed"],
                stratum_id=stratum,
                replicate=replicate,
            )
            body = {
                "stratum_id": stratum,
                "replicate": replicate,
                "environment_seed": env_seed,
                "sampling_seed": sample_seed,
                "reward": reward,
                "terminated": True,
                "truncated": False,
                "turn_count": 1,
                "actions": ["answer"],
                "response_token_digests": [bytes_digest(b"test response tokens")],
                "parse_failure_turn": None,
                "provider_error": None,
            }
            values.append({**body, "trajectory_digest": object_digest(body)})
    return values


def _execution_receipt(
    *,
    intent: dict,
    receipt: dict,
    request: dict,
    pool_manifest: dict,
    final_state_uri: str,
    final_sampler_uri: str,
) -> dict:
    basenames = [item["basename"] for item in pool_manifest["pools"][request["arm"]]["entries"]]
    audits = tuple(
        {
            "games": 12,
            "episodes": 96,
            "submitted_positions": 96 * TRAIN_POSITIONS_PER_EPISODE,
            "actual_positions": 288,
            "action_positions": 192,
            "padded_positions": 96 * TRAIN_POSITIONS_PER_EPISODE - 288,
            "mixed_reward_games": 12,
            "raw_group_means": {basename: 0.5 for basename in basenames},
        }
        for _ in range(16)
    )
    checkpoints = tuple(
        {
            "state_uri": (
                final_state_uri
                if update == 15
                else f"tinker://{request['pair_id']}/{request['slot_id']}/{update + 1}/state"
            ),
            "sampler_uri": (
                final_sampler_uri
                if update == 15
                else f"tinker://{request['pair_id']}/{request['slot_id']}/{update + 1}/sampler"
            ),
            "completed_update": update + 1,
            "public_metrics": {},
        }
        for update in range(16)
    )
    execution = BranchExecution(
        score={},
        update_audits=audits,
        checkpoint_receipts=checkpoints,
        canary_token_digest=receipt["canary_token_digest"],
    )
    return build_branch_execution_receipt(
        intent=intent,
        base_receipt=receipt,
        branch_request=request,
        pool_manifest=pool_manifest,
        execution=execution,
    )


def test_six_pair_exact_analysis_and_backend_substitution_rejection(tmp_path: Path) -> None:
    intent = _intent(tmp_path)
    _, receipt = _base(tmp_path, intent)
    requests = branch_requests(intent, receipt)
    pair_by_id = {item["pair_id"]: item for item in intent["pair_schedule"]}
    pool_root, pool_manifest = _fake_pool_bundle(tmp_path)
    scores = []
    executions = []
    for request in requests:
        reward = 0.75 if request["arm"] == "coverage_forced" else 0.50
        final_state_uri = f"tinker://{request['pair_id']}/{request['slot_id']}/state"
        final_sampler_uri = f"tinker://{request['pair_id']}/{request['slot_id']}/sampler"
        score = build_heldout_score(
            intent=intent,
            branch_request=request,
            base_receipt=receipt,
            final_state_uri=final_state_uri,
            final_sampler_uri=final_sampler_uri,
            completed_updates=16,
            submitted_training_positions=TRAIN_POSITIONS_PER_BRANCH,
            outcomes=_outcomes(pair_by_id[request["pair_id"]], reward),
        )
        assert (
            validate_heldout_score(
                score,
                intent=intent,
                branch_request=request,
                base_receipt=receipt,
            )
            == score
        )
        scores.append(score)
        executions.append(
            _execution_receipt(
                intent=intent,
                receipt=receipt,
                request=request,
                pool_manifest=pool_manifest,
                final_state_uri=final_state_uri,
                final_sampler_uri=final_sampler_uri,
            )
        )

    pair_completions = []
    for pair in intent["pair_schedule"]:
        pair_requests = [item for item in requests if item["pair_id"] == pair["pair_id"]]
        request_digests = {item["request_digest"] for item in pair_requests}
        pair_executions = [
            item for item in executions if item["branch_request_digest"] in request_digests
        ]
        pair_scores = [item for item in scores if item["branch_request_digest"] in request_digests]
        pair_completions.append(
            build_pair_complete(
                intent=intent,
                base_receipt=receipt,
                pair_id=pair["pair_id"],
                requests=pair_requests,
                executions=pair_executions,
                scores=pair_scores,
                pool_manifest=pool_manifest,
            )
        )

    aggregate = analyze_scores(
        intent=intent,
        base_receipt=receipt,
        scores=scores,
        executions=executions,
        pair_completions=pair_completions,
        pool_manifest=pool_manifest,
    )
    assert aggregate["sensitivity_gate_passed"] is True
    assert aggregate["scientific_gate_passed"] is False
    assert aggregate["causal_claim_authorized"] is False
    assert aggregate["supports_narrow_claim"] is False
    assert aggregate["decision"] == "hold_plan_only_unattested_assignment_and_provider_state"
    assert aggregate["mean_effect"] == 0.25
    assert aggregate["p_value"] == 2 / EXACT_SIGN_FLIP_COUNT
    assert aggregate["promotion_authorized"] is False
    assert aggregate["release_authorized"] is False
    assert (
        validate_aggregate(
            aggregate,
            intent=intent,
            base_receipt=receipt,
            scores=scores,
            executions=executions,
            pair_completions=pair_completions,
            pool_manifest=pool_manifest,
        )
        == aggregate
    )

    substituted = dict(scores[0])
    substituted["evaluation_backend"] = "agy-google"
    body = {key: value for key, value in substituted.items() if key != "score_digest"}
    substituted["score_digest"] = object_digest(body)
    with pytest.raises(BranchAssayError, match="recomputation|substitutes"):
        validate_heldout_score(
            substituted,
            intent=intent,
            branch_request=requests[0],
            base_receipt=receipt,
        )


def test_runner_rejects_unknown_and_partial_pair_evidence(tmp_path: Path) -> None:
    from tools.run_spade_tinker_branch_assay import _run_dir, _validate_run_evidence

    intent = _intent(tmp_path)
    run_dir = _run_dir(intent)
    receipt = asyncio.run(
        prepare_common_base_state(
            intent=intent,
            run_dir=run_dir,
            boundary=_FakeCommonState(),
        )
    )
    pool_root, pool_manifest = _fake_pool_bundle(tmp_path)
    pair = intent["pair_schedule"][0]
    requests = [
        item for item in branch_requests(intent, receipt) if item["pair_id"] == pair["pair_id"]
    ]
    executions = []
    scores = []
    pair_root = run_dir / "branches" / pair["pair_id"]
    for request in requests:
        slot_root = pair_root / request["slot_id"]
        final_state_uri = f"tinker://{request['pair_id']}/{request['slot_id']}/state"
        final_sampler_uri = f"tinker://{request['pair_id']}/{request['slot_id']}/sampler"
        score = build_heldout_score(
            intent=intent,
            branch_request=request,
            base_receipt=receipt,
            final_state_uri=final_state_uri,
            final_sampler_uri=final_sampler_uri,
            completed_updates=16,
            submitted_training_positions=TRAIN_POSITIONS_PER_BRANCH,
            outcomes=_outcomes(pair, 0.5),
        )
        execution = _execution_receipt(
            intent=intent,
            receipt=receipt,
            request=request,
            pool_manifest=pool_manifest,
            final_state_uri=final_state_uri,
            final_sampler_uri=final_sampler_uri,
        )
        _write_json(slot_root / "request.json", request)
        _write_json(slot_root / "execution.json", execution)
        _write_json(slot_root / "heldout-score.json", score)
        executions.append(execution)
        scores.append(score)
    complete = build_pair_complete(
        intent=intent,
        base_receipt=receipt,
        pair_id=pair["pair_id"],
        requests=requests,
        executions=executions,
        scores=scores,
        pool_manifest=pool_manifest,
    )
    _write_json(pair_root / "pair-complete.json", complete)

    evidence = _validate_run_evidence(
        intent=intent,
        receipt=receipt,
        pool_manifest=pool_manifest,
        pool_root=pool_root,
        require_complete=False,
    )
    assert evidence["observed_pair_ids"] == ["pair-01"]
    assert len(evidence["executions"]) == len(evidence["scores"]) == 2

    unknown = pair_root / "unknown.json"
    unknown.write_text("{}\n", encoding="utf-8")
    with pytest.raises(BranchAssayError, match="inventory differs"):
        _validate_run_evidence(
            intent=intent,
            receipt=receipt,
            pool_manifest=pool_manifest,
            pool_root=pool_root,
            require_complete=False,
        )
    unknown.unlink()

    partial_root = run_dir / "branches" / "pair-02" / "slot_a"
    partial_root.mkdir(parents=True)
    with pytest.raises(BranchAssayError, match="partial/ambiguous"):
        _validate_run_evidence(
            intent=intent,
            receipt=receipt,
            pool_manifest=pool_manifest,
            pool_root=pool_root,
            require_complete=False,
        )


def test_heldout_reward_is_replayed_and_game_drift_fails_closed(tmp_path: Path) -> None:
    from tools.run_spade_tinker_branch_assay import (
        _read_sealed_environment,
        _replay_heldout_score,
    )

    intent = _intent(tmp_path)
    _, receipt = _base(tmp_path, intent)
    request = branch_requests(intent, receipt)[0]
    pair = next(item for item in intent["pair_schedule"] if item["pair_id"] == request["pair_id"])
    pool_root, pool_manifest = _fake_pool_bundle(tmp_path)
    final_state_uri = f"tinker://{request['pair_id']}/{request['slot_id']}/state"
    final_sampler_uri = f"tinker://{request['pair_id']}/{request['slot_id']}/sampler"
    score = build_heldout_score(
        intent=intent,
        branch_request=request,
        base_receipt=receipt,
        final_state_uri=final_state_uri,
        final_sampler_uri=final_sampler_uri,
        completed_updates=16,
        submitted_training_positions=TRAIN_POSITIONS_PER_BRANCH,
        outcomes=_outcomes(pair, 0.5),
    )
    _replay_heldout_score(
        score=score,
        pool_manifest=pool_manifest,
        pool_root=pool_root,
    )

    fabricated = _outcomes(pair, 0.5)
    fabricated[0]["reward"] = 0.75
    trajectory_body = {
        key: value for key, value in fabricated[0].items() if key != "trajectory_digest"
    }
    fabricated[0]["trajectory_digest"] = object_digest(trajectory_body)
    fabricated_score = build_heldout_score(
        intent=intent,
        branch_request=request,
        base_receipt=receipt,
        final_state_uri=final_state_uri,
        final_sampler_uri=final_sampler_uri,
        completed_updates=16,
        submitted_training_positions=TRAIN_POSITIONS_PER_BRANCH,
        outcomes=fabricated,
    )
    assert (
        validate_heldout_score(
            fabricated_score,
            intent=intent,
            branch_request=request,
            base_receipt=receipt,
        )
        == fabricated_score
    )
    with pytest.raises(BranchAssayError, match="deterministic action replay"):
        _replay_heldout_score(
            score=fabricated_score,
            pool_manifest=pool_manifest,
            pool_root=pool_root,
        )

    heldout_entry = pool_manifest["pools"]["heldout_v4"]["entries"][0]
    heldout_path = pool_root / "heldout_v4" / heldout_entry["basename"]
    _read_sealed_environment(heldout_path, heldout_entry)
    heldout_path.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(BranchAssayError, match="sealed pool"):
        _read_sealed_environment(heldout_path, heldout_entry)


def test_seal_intent_uses_one_canonical_lineage_and_refuses_reseal(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.run_spade_tinker_branch_assay as runner

    actor_path = (tmp_path / "actor-plan.json").resolve()
    _, actor_digest = _sealed_actor_plan(actor_path)
    manifest_path = (tmp_path / "pools" / "learner-pools-manifest.json").resolve()
    _write_json(
        manifest_path,
        {**_TEST_POOL_BODY, "manifest_digest": _TEST_POOL_DIGEST},
    )
    bundle = SimpleNamespace(
        root=manifest_path.parent,
        manifest_path=manifest_path,
        manifest={**_TEST_POOL_BODY, "manifest_digest": _TEST_POOL_DIGEST},
        heldout_v4_dir=manifest_path.parent / "heldout_v4",
    )
    monkeypatch.setattr(runner, "ROOT_DIR", tmp_path.resolve())
    monkeypatch.setattr(runner, "SEALED_ACTOR_PLAN_DIGEST", actor_digest)
    monkeypatch.setattr(runner, "load_learner_pool_manifest", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        runner,
        "_runtime_identity",
        lambda: _TEST_RUNTIME_IDENTITY,
    )
    output_root = runner._canonical_output_root(
        pool_manifest_digest=_TEST_POOL_DIGEST,
        source_actor_plan_digest=actor_digest,
    )
    arguments = Namespace(
        pool_manifest=manifest_path,
        actor_plan=actor_path,
        output_root=output_root,
        intent=output_root / "intent.json",
        assignment_seed=20260902,
        sampling_seed=910247,
        model="Qwen/Qwen3-8B",
        renderer="qwen3_disable_thinking_preserve_history",
    )
    intent = runner._seal_intent(arguments)
    runner._validate_lineage_lock(intent, intent_path=arguments.intent)
    (output_root / "alternate-intent.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(BranchAssayError, match="lineage-root inventory"):
        runner._validate_lineage_lock(intent, intent_path=arguments.intent)
    (output_root / "alternate-intent.json").unlink()
    with pytest.raises(BranchAssayError, match="no duplicate assay"):
        runner._seal_intent(arguments)


def test_live_guard_and_capability_gate_are_offline_and_fail_closed(monkeypatch) -> None:
    from tools.run_spade_tinker_branch_assay import (
        _capability_receipt,
        _require_live_authorization,
    )

    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(BranchAssayError, match="plan-only HOLD"):
        _require_live_authorization(Namespace(allow_live=False))
    with pytest.raises(BranchAssayError, match="plan-only HOLD"):
        _require_live_authorization(Namespace(allow_live=True))

    capabilities = SimpleNamespace(
        supported_models=[
            SimpleNamespace(model_name="Qwen/Qwen3-8B"),
            SimpleNamespace(model_name="Qwen/Qwen3.5-4B"),
        ]
    )
    assert _capability_receipt(capabilities, model_name="Qwen/Qwen3-8B") == {
        "supported_model_names": ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-4B"]
    }
    with pytest.raises(BranchAssayError, match="do not advertise"):
        _capability_receipt(capabilities, model_name="Qwen/retired-model")


def test_pair_sampling_limit_is_shared_across_both_concurrent_branches() -> None:
    from tools.run_spade_tinker_branch_assay import _TinkerTrainingBoundary

    boundary = object.__new__(_TinkerTrainingBoundary)
    boundary.sample_semaphore = asyncio.Semaphore(2)
    active = 0
    peak = 0

    async def sampling_client(client):
        return client

    async def play_episode(*args, **kwargs):
        nonlocal active, peak
        _ = args, kwargs
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {
            "tokens": [1, 2],
            "loss_mask": [0, 1],
            "action_logprobs": [-0.1],
            "action_turn_indices": [0],
            "raw_reward": 1.0,
            "turn_count": 1,
            "status": "completed",
        }

    boundary._sampling_client = sampling_client
    boundary._play_episode = play_episode
    entries = [{"basename": f"game_{index:03d}.py"} for index in range(12)]

    async def run_both():
        return await asyncio.gather(
            boundary.collect_training_rollouts(
                object(),
                pool_dir=Path("/tmp/unused"),
                pool_entries=entries,
                pair_id="pair-01",
                pair_seed=101,
                update=0,
            ),
            boundary.collect_training_rollouts(
                object(),
                pool_dir=Path("/tmp/unused"),
                pool_entries=entries,
                pair_id="pair-01",
                pair_seed=101,
                update=0,
            ),
        )

    first, second = asyncio.run(run_both())
    assert len(first) == len(second) == 96
    assert peak == 2


def test_runtime_distribution_identity_fails_closed_when_missing(monkeypatch) -> None:
    import tools.run_spade_tinker_branch_assay as runner

    def missing(_name):
        raise runner.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(runner.importlib.metadata, "distribution", missing)
    with pytest.raises(BranchAssayError, match="distribution is unavailable"):
        runner._distribution_identity("tinker", "tinker")


def test_imported_module_origin_must_match_sealed_distribution(tmp_path: Path, monkeypatch) -> None:
    import tools.run_spade_tinker_branch_assay as runner

    package_entry = (tmp_path / "site-packages" / "tinker" / "__init__.py").resolve()
    package_entry.parent.mkdir(parents=True)
    package_entry.write_text("VERSION = 'sealed'\n", encoding="utf-8")
    identity = {
        "version": "test-sdk",
        "record_path": str((tmp_path / "RECORD").resolve()),
        "record_digest": bytes_digest(b"record"),
        "metadata_path": str((tmp_path / "METADATA").resolve()),
        "metadata_digest": bytes_digest(b"metadata"),
        "package_entry_path": str(package_entry),
        "package_entry_digest": runner.file_digest(package_entry),
    }
    monkeypatch.setattr(runner, "_distribution_identity", lambda *args, **kwargs: identity)
    runner._verify_imported_distribution_module(
        SimpleNamespace(__file__=str(package_entry)),
        distribution_name="tinker",
        module_name="tinker",
        sealed_identity=identity,
    )

    shadow = (tmp_path / "tinker.py").resolve()
    shadow.write_text("VERSION = 'shadow'\n", encoding="utf-8")
    with pytest.raises(BranchAssayError, match="shadows its sealed distribution"):
        runner._verify_imported_distribution_module(
            SimpleNamespace(__file__=str(shadow)),
            distribution_name="tinker",
            module_name="tinker",
            sealed_identity=identity,
        )


def test_generated_environment_global_rng_is_episode_isolated(monkeypatch) -> None:
    import random
    import sys

    from tools.run_spade_tinker_branch_assay import _call_with_isolated_global_rng

    class _FakeNumpyRandom:
        value = 0

        @classmethod
        def seed(cls, value):
            cls.value = int(value)

        @classmethod
        def get_state(cls):
            return cls.value

        @classmethod
        def set_state(cls, value):
            cls.value = value

        @classmethod
        def random(cls):
            cls.value = (cls.value * 1_103_515_245 + 12_345) % (2**31)
            return cls.value / (2**31)

    fake_numpy = SimpleNamespace(random=_FakeNumpyRandom)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    original_python = random.getstate()
    original_numpy = fake_numpy.random.get_state()
    try:
        random.seed(991)
        fake_numpy.random.seed(991)
        ambient_python = random.getstate()
        ambient_numpy = fake_numpy.random.get_state()

        first, state = _call_with_isolated_global_rng(
            lambda: (random.random(), float(fake_numpy.random.random())),
            seed=42,
        )
        second, _ = _call_with_isolated_global_rng(
            lambda: (random.random(), float(fake_numpy.random.random())),
            state=state,
        )
        repeated_first, repeated_state = _call_with_isolated_global_rng(
            lambda: (random.random(), float(fake_numpy.random.random())),
            seed=42,
        )
        repeated_second, _ = _call_with_isolated_global_rng(
            lambda: (random.random(), float(fake_numpy.random.random())),
            state=repeated_state,
        )
        assert (first, second) == (repeated_first, repeated_second)
        assert random.getstate() == ambient_python
        assert fake_numpy.random.get_state() == ambient_numpy
    finally:
        random.setstate(original_python)
        fake_numpy.random.set_state(original_numpy)
