import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from spade.core.learner_branch_pools import LearnerPoolError, materialize_learner_pools
from spade.slime.branch_assay import (
    SLIME_BACKEND,
    SLIME_HOLD_DECISION,
    SLIME_RESULTS_SCHEMA,
    SlimeAssayError,
    load_slime_assay_plan,
    materialize_slime_assay_plan,
    validate_slime_assay_results,
)


_REPO_ROOT = Path(__file__).parents[1]
_DEFAULT_ACTOR_PLAN = Path(
    "/Users/sergio.soage/code/spade-baseline-stack-20260901/spade/.assay/"
    "spade-coverage-forced-proxy-v2/"
    "spade-google-coverage-forced-matched-swap-v2-"
    "df1a06c7fb854d5267ec4d1e41cd44c1ffd229018bf0f5a0dcde512fa47c4c09/"
    "actor-plan.json"
)
_ACTOR_PLAN = Path(os.environ.get("SPADE_SEALED_ACTOR_PLAN", _DEFAULT_ACTOR_PLAN))
_D1 = "sha256:" + "1" * 64
_D2 = "sha256:" + "2" * 64
_D3 = "sha256:" + "3" * 64


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _seal(value: dict, field: str) -> dict:
    body = {key: item for key, item in value.items() if key != field}
    return {
        **body,
        field: "sha256:" + hashlib.sha256(_canonical(body).encode()).hexdigest(),
    }


@pytest.fixture(scope="module")
def assay(tmp_path_factory: pytest.TempPathFactory):
    if not _ACTOR_PLAN.is_file():
        pytest.skip("sealed df1a/fc798 actor-plan fixture is not available")
    root = tmp_path_factory.mktemp("slime-assay")
    bundle = materialize_learner_pools(_ACTOR_PLAN, root / "pools")
    plan = materialize_slime_assay_plan(
        bundle.root,
        root / "plan",
        remote_pool_root="/scratch/spade-learner-pools",
        remote_output_root="/scratch/spade-learner-runs",
        hf_checkpoint="/scratch/models/Qwen3-8B",
        hf_checkpoint_digest=_D1,
        reference_checkpoint="/scratch/models/Qwen3-8B_torch_dist",
        reference_checkpoint_digest=_D2,
        runtime_image="registry.example/spade@sha256:" + "3" * 64,
        runtime_image_digest=_D3,
        spade_source_revision="a" * 40,
        source_root=_REPO_ROOT,
    )
    return bundle, plan, root


def _token_metrics(sequence_tokens: int = 2880, loss_mask_tokens: int = 900) -> dict:
    response_tokens = 960
    prompt_tokens = sequence_tokens - response_tokens
    values = {
        "samples": 192,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "loss_mask_tokens": loss_mask_tokens,
        "sequence_tokens": sequence_tokens,
    }
    metrics: dict[str, int] = {}
    for population in ("real", "padded", "total"):
        for role in ("all", "actor", "environment", "unknown"):
            for measure in (
                "samples",
                "prompt_tokens",
                "response_tokens",
                "loss_mask_tokens",
                "sequence_tokens",
            ):
                populated = population in ("real", "total") and role in ("all", "actor")
                metrics[f"rollout/tokens/{population}/{role}/{measure}"] = (
                    values[measure] if populated else 0
                )
    return metrics


def _topology_metrics() -> dict[str, int]:
    metrics = {
        f"rollout/groups/{population}/{role}/episode_groups": (
            192 if population in ("real", "total") and role in ("all", "actor") else 0
        )
        for population in ("real", "padded", "total")
        for role in ("all", "actor", "environment", "unknown")
    }
    metrics.update(
        {
            "rollout/topology/actor_instances_requested": 192,
            "rollout/topology/actor_instances_succeeded": 192,
            "rollout/topology/actor_instances_failed": 0,
            "rollout/topology/actor_trajectories_filtered": 0,
            "rollout/topology/environment_trajectories_filtered": 0,
        }
    )
    return metrics


def _checkpoint(run_id: str, rollout_id: int) -> str:
    material = f"{run_id}:{rollout_id}".encode()
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _result_payload(plan, *, imbalanced_run: str | None = None) -> dict:
    runs = []
    for spec in plan.launch_specs:
        previous = spec["initial_state"]["model"]["artifact_digest"]
        rollouts = []
        for scheduled in spec["training"]["rollout_schedule"]:
            rollout_id = scheduled["rollout_id"]
            output = _checkpoint(spec["run_id"], rollout_id)
            metrics = _token_metrics(
                sequence_tokens=(3100 if spec["run_id"] == imbalanced_run else 2880)
            )
            rollouts.append(
                {
                    "rollout_id": rollout_id,
                    "schedule_id": spec["training"]["schedule_id"],
                    "slot_basenames": scheduled["slot_basenames"],
                    "actor_source": "current-slime-checkpoint-via-in-job-sglang",
                    "inference_checkpoint_digest": previous,
                    "optimizer_input_checkpoint_digest": previous,
                    "optimizer_output_checkpoint_digest": output,
                    "token_metrics": metrics,
                    "topology_metrics": _topology_metrics(),
                }
            )
            previous = output
        won = spec["arm"] == "coverage_forced"
        games = [
            {
                "basename": game["basename"],
                "episodes": [
                    {
                        **replicate,
                        "termination": "terminated",
                        "terminal_reward": float(won),
                        "total_reward": float(won),
                        "turns": 1,
                        "error": None,
                    }
                    for replicate in game["replicates"]
                ],
            }
            for game in spec["heldout_evaluation"]["games"]
        ]
        total_plays = sum(len(game["episodes"]) for game in games)
        total_wins = total_plays if won else 0
        runs.append(
            {
                "run_id": spec["run_id"],
                "status": "complete",
                "backend": SLIME_BACKEND,
                "launch_spec_digest": spec["launch_spec_digest"],
                "arm": spec["arm"],
                "seed": spec["seed"],
                "external_actor_substitution": False,
                "rollouts": rollouts,
                "final_checkpoint_digest": previous,
                "heldout": {
                    "actor_source": "final-slime-checkpoint-via-dedicated-sglang",
                    "backend": SLIME_BACKEND,
                    "served_checkpoint_digest": previous,
                    "pool_manifest_digest": spec["pool_manifest_digest"],
                    "pool_name": "heldout_v4",
                    "request_parameters": spec["heldout_evaluation"]["request_parameters"],
                    "trajectory_parameters": spec["heldout_evaluation"]["trajectory_parameters"],
                    "games": games,
                    "aggregate": {
                        "total_games": len(games),
                        "total_plays": total_plays,
                        "total_successes": total_wins,
                        "success_rate": total_wins / total_plays,
                    },
                },
            }
        )
    return _seal(
        {
            "schema_version": SLIME_RESULTS_SCHEMA,
            "backend": SLIME_BACKEND,
            "plan_digest": plan.manifest["plan_digest"],
            "runs": runs,
        },
        "results_digest",
    )


def _write_results(path: Path, value: dict) -> None:
    path.write_text(_canonical(value) + "\n")


def test_plan_seals_six_paired_counterbalanced_executable_launches(assay) -> None:
    bundle, plan, _ = assay
    assert plan.manifest["execution_state"] == "plan-only-hold"
    assert plan.manifest["evidence_state"] == "artifact-backed-collector-not-implemented"
    assert plan.manifest["num_pairs"] == 6
    assert plan.manifest["num_runs"] == 12
    assert plan.manifest["num_rollouts_per_run"] == 16
    assert plan.manifest["compute_gate"]["maximum_pairwise_relative_difference"] == 0.05

    for pair_index in range(6):
        left, right = plan.launch_specs[pair_index * 2 : pair_index * 2 + 2]
        assert {left["arm"], right["arm"]} == {
            "coverage_forced",
            "redundant_historical",
        }
        assert left["seed"] == right["seed"]
        assert left["arm"] == ("coverage_forced" if pair_index % 2 == 0 else "redundant_historical")
        assert [item["slot_basenames"] for item in left["training"]["rollout_schedule"]] == [
            item["slot_basenames"] for item in right["training"]["rollout_schedule"]
        ]
        for spec in (left, right):
            env = spec["launcher"]["environment"]
            assert env["SPADE_STATIC_POOL_SCHEDULE_ID"] == bundle.schedule_id
            assert env["SPADE_FIXED_POOL_SEED"] == str(spec["seed"])
            assert env["TRAIN_SEED"] == str(spec["seed"])
            assert env["SPADE_REQUIRE_COMPLETE_ROLLOUT"] == "1"
            assert env["SKIP_RUNTIME_INSTALL"] == "1"
            assert env["DISABLE_WANDB"] == "1"
            assert "OPENAI_API_KEY" in spec["launcher"]["forbidden_environment"]
            assert spec["training"]["rollout_actor_source"].startswith("current-slime")
            argv = spec["launcher"]["effective_training_arguments"]
            assert argv.count("--tensor-model-parallel-size") == 1
            assert argv.count("--spade-require-complete-rollout") == 1
            assert "--use-wandb" not in argv

    assert plan.manifest["primary_analysis"]["test_assumption"] == (
        "paired differences are exchangeable under the sharp null"
    )
    assert plan.manifest["primary_analysis"]["assignment"].endswith("job slots are not randomized")
    assert load_slime_assay_plan(plan.manifest_path).manifest == plan.manifest


def test_fixed_launcher_prepare_path_accepts_sealed_twelve_game_pool(assay) -> None:
    bundle, plan, _ = assay
    first = plan.launch_specs[0]
    environment = {
        **os.environ,
        **first["launcher"]["environment"],
        "STATIC_POOL_DIR": str(bundle.coverage_forced_dir),
        "PREPARE_ONLY": "1",
        "WANDB_ENTITY": "offline-test",
    }
    completed = subprocess.run(
        ["bash", str(_REPO_ROOT / "cmd/games/_train_fixed_gpt55.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert bundle.schedule_id in completed.stdout


def test_result_validator_checks_receipts_but_can_only_return_hold(assay) -> None:
    _, plan, root = assay
    path = root / "passing-results.json"
    _write_results(path, _result_payload(plan))
    validation = validate_slime_assay_results(plan.root, path)
    assert validation["receipt_consistency_gate"] == "pass"
    assert validation["paired_compute_gate"] == "pass"
    assert validation["status"] == "plan-only-hold"
    assert validation["causal_evidence"] == "not-established"
    assert validation["decision"] == SLIME_HOLD_DECISION
    assert validation["primary_analysis"]["one_sided_exact_sign_flip_p_value"] == 1 / 64


def test_result_validator_fails_closed_on_external_actor_or_compute_imbalance(assay) -> None:
    _, plan, root = assay
    first_run = plan.launch_specs[0]["run_id"]
    external = _result_payload(plan)
    external["runs"][0]["heldout"]["actor_source"] = "agy-google"
    external = _seal(external, "results_digest")
    external_path = root / "external-results.json"
    _write_results(external_path, external)
    with pytest.raises(SlimeAssayError, match="heldout checkpoint, pool, or execution contract"):
        validate_slime_assay_results(plan.root, external_path)

    imbalanced = _result_payload(plan, imbalanced_run=first_run)
    imbalance_path = root / "imbalanced-results.json"
    _write_results(imbalance_path, imbalanced)
    with pytest.raises(SlimeAssayError, match="exceeds paired compute tolerance"):
        validate_slime_assay_results(plan.root, imbalance_path)


def test_result_validator_rejects_padding_seed_drift_and_unknown_claims(assay) -> None:
    _, plan, root = assay
    padded = _result_payload(plan)
    topology = padded["runs"][0]["rollouts"][0]["topology_metrics"]
    topology["rollout/groups/real/all/episode_groups"] = 191
    topology["rollout/groups/real/actor/episode_groups"] = 191
    topology["rollout/groups/padded/all/episode_groups"] = 1
    topology["rollout/groups/padded/actor/episode_groups"] = 1
    padded = _seal(padded, "results_digest")
    padded_path = root / "padded-results.json"
    _write_results(padded_path, padded)
    with pytest.raises(SlimeAssayError, match="zero-pad/zero-failure topology"):
        validate_slime_assay_results(plan.root, padded_path)

    failed = _result_payload(plan)
    failed_topology = failed["runs"][0]["rollouts"][0]["topology_metrics"]
    failed_topology["rollout/topology/actor_instances_succeeded"] = 191
    failed_topology["rollout/topology/actor_instances_failed"] = 1
    failed = _seal(failed, "results_digest")
    failed_path = root / "failure-upsample-results.json"
    _write_results(failed_path, failed)
    with pytest.raises(SlimeAssayError, match="zero-pad/zero-failure topology"):
        validate_slime_assay_results(plan.root, failed_path)

    seed_drift = _result_payload(plan)
    seed_drift["runs"][0]["heldout"]["games"][0]["episodes"][0]["sampling_seed"] += 1
    seed_drift = _seal(seed_drift, "results_digest")
    seed_path = root / "seed-drift-results.json"
    _write_results(seed_path, seed_drift)
    with pytest.raises(SlimeAssayError, match="invalid or failed"):
        validate_slime_assay_results(plan.root, seed_path)

    arbitrary_won = _result_payload(plan)
    arbitrary_won["runs"][0]["heldout"]["games"][0]["episodes"][0]["won"] = True
    arbitrary_won = _seal(arbitrary_won, "results_digest")
    won_path = root / "arbitrary-won-results.json"
    _write_results(won_path, arbitrary_won)
    with pytest.raises(SlimeAssayError, match="heldout episode keys differ"):
        validate_slime_assay_results(plan.root, won_path)

    unknown = _result_payload(plan)
    unknown["runs"][0]["release_authorized"] = True
    unknown = _seal(unknown, "results_digest")
    unknown_path = root / "unknown-claim-results.json"
    _write_results(unknown_path, unknown)
    with pytest.raises(SlimeAssayError, match="keys differ"):
        validate_slime_assay_results(plan.root, unknown_path)


def test_result_validator_cross_closes_tokens_groups_and_checkpoint_change(assay) -> None:
    _, plan, root = assay
    padded_tokens = _result_payload(plan)
    token_metrics = padded_tokens["runs"][0]["rollouts"][0]["token_metrics"]
    additions = {
        "samples": 1,
        "prompt_tokens": 3,
        "response_tokens": 2,
        "loss_mask_tokens": 1,
        "sequence_tokens": 5,
    }
    for role in ("all", "actor"):
        for measure, amount in additions.items():
            token_metrics[f"rollout/tokens/padded/{role}/{measure}"] = amount
            token_metrics[f"rollout/tokens/total/{role}/{measure}"] += amount
    padded_tokens = _seal(padded_tokens, "results_digest")
    padded_tokens_path = root / "zero-group-nonzero-token-results.json"
    _write_results(padded_tokens_path, padded_tokens)
    with pytest.raises(SlimeAssayError, match="zero-group population"):
        validate_slime_assay_results(plan.root, padded_tokens_path)

    no_action_tokens = _result_payload(plan)
    token_metrics = no_action_tokens["runs"][0]["rollouts"][0]["token_metrics"]
    token_metrics["rollout/tokens/real/all/loss_mask_tokens"] = 0
    token_metrics["rollout/tokens/real/actor/loss_mask_tokens"] = 0
    token_metrics["rollout/tokens/total/all/loss_mask_tokens"] = 0
    token_metrics["rollout/tokens/total/actor/loss_mask_tokens"] = 0
    no_action_tokens = _seal(no_action_tokens, "results_digest")
    no_action_path = root / "zero-action-token-results.json"
    _write_results(no_action_path, no_action_tokens)
    with pytest.raises(SlimeAssayError, match="no real action/loss-mask"):
        validate_slime_assay_results(plan.root, no_action_path)

    unchanged = _result_payload(plan)
    spec = plan.launch_specs[0]
    run = unchanged["runs"][0]
    initial = spec["initial_state"]["model"]["artifact_digest"]
    for rollout in run["rollouts"]:
        rollout["inference_checkpoint_digest"] = initial
        rollout["optimizer_input_checkpoint_digest"] = initial
        rollout["optimizer_output_checkpoint_digest"] = initial
    run["final_checkpoint_digest"] = initial
    run["heldout"]["served_checkpoint_digest"] = initial
    unchanged = _seal(unchanged, "results_digest")
    unchanged_path = root / "unchanged-checkpoint-results.json"
    _write_results(unchanged_path, unchanged)
    with pytest.raises(SlimeAssayError, match="did not change from initial"):
        validate_slime_assay_results(plan.root, unchanged_path)


def test_planner_rejects_resealed_identical_training_pools(assay) -> None:
    bundle, _, root = assay
    counterfeit_root = root / "counterfeit-identical-pools"
    shutil.copytree(bundle.root, counterfeit_root)
    manifest_path = counterfeit_root / "learner-pools-manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["pools"]["redundant_historical"]["entries"] = json.loads(
        json.dumps(manifest["pools"]["coverage_forced"]["entries"])
    )
    manifest = _seal(manifest, "manifest_digest")
    manifest_path.write_text(_canonical(manifest) + "\n")
    manifest_path.chmod(0o444)

    with pytest.raises(LearnerPoolError, match="authorized df1a/fc798/v4"):
        materialize_slime_assay_plan(
            counterfeit_root,
            root / "counterfeit-plan",
            remote_pool_root="/scratch/pools",
            remote_output_root="/scratch/runs",
            hf_checkpoint="/scratch/models/Qwen3-8B",
            hf_checkpoint_digest=_D1,
            reference_checkpoint="/scratch/models/Qwen3-8B_torch_dist",
            reference_checkpoint_digest=_D2,
            runtime_image="registry.example/spade@" + _D3,
            runtime_image_digest=_D3,
            spade_source_revision="a" * 40,
            source_root=_REPO_ROOT,
        )


def test_cli_bootstraps_repository_imports_from_an_unrelated_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools/run_spade_learner_branch_assay.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "plan-slime" in completed.stdout


def test_plan_and_launch_spec_schemas_reject_unknown_claims(assay) -> None:
    _, plan, root = assay
    bad_plan_root = root / "bad-plan-schema"
    shutil.copytree(plan.root, bad_plan_root)
    bad_plan_path = bad_plan_root / "slime-assay-plan.json"
    bad_plan_path.chmod(0o644)
    bad_plan = json.loads(bad_plan_path.read_text())
    bad_plan["release_authorized"] = True
    bad_plan_path.write_text(_canonical(_seal(bad_plan, "plan_digest")) + "\n")
    bad_plan_path.chmod(0o444)
    with pytest.raises(SlimeAssayError, match="plan keys differ"):
        load_slime_assay_plan(bad_plan_root)

    bad_spec_root = root / "bad-spec-schema"
    shutil.copytree(plan.root, bad_spec_root)
    bad_spec_plan_path = bad_spec_root / "slime-assay-plan.json"
    bad_spec_plan_path.chmod(0o644)
    bad_spec_plan = json.loads(bad_spec_plan_path.read_text())
    first = bad_spec_plan["launch_schedule"][0]
    bad_spec_path = bad_spec_root / first["launch_spec_path"]
    bad_spec_path.chmod(0o644)
    bad_spec = json.loads(bad_spec_path.read_text())
    bad_spec["release_authorized"] = True
    bad_spec = _seal(bad_spec, "launch_spec_digest")
    bad_spec_path.write_text(_canonical(bad_spec) + "\n")
    bad_spec_path.chmod(0o444)
    first["launch_spec_digest"] = bad_spec["launch_spec_digest"]
    bad_spec_plan = _seal(bad_spec_plan, "plan_digest")
    bad_spec_plan_path.write_text(_canonical(bad_spec_plan) + "\n")
    bad_spec_plan_path.chmod(0o444)
    with pytest.raises(SlimeAssayError, match="launch spec .* keys differ"):
        load_slime_assay_plan(bad_spec_root)
