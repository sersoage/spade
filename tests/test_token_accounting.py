from types import SimpleNamespace

import pytest

from spade.slime.token_accounting import (
    TokenAccountingError,
    rollout_source_topology_metrics,
    rollout_token_accounting_metrics,
)


def _sample(
    *,
    total: int,
    response: int,
    loss_mask: list[int] | None,
    role: str | None,
    pad: bool = False,
    remove: bool = False,
) -> SimpleNamespace:
    metadata = {"spade_is_pad": pad}
    if role is not None:
        metadata["role"] = role
    return SimpleNamespace(
        tokens=list(range(total)),
        response_length=response,
        loss_mask=loss_mask,
        metadata=metadata,
        remove_sample=remove,
    )


def test_pads_are_excluded_from_real_metrics_and_totals_rederive() -> None:
    groups = [
        [_sample(total=10, response=4, loss_mask=[1, 1, 0, 1], role="actor")],
        [_sample(total=8, response=3, loss_mask=None, role="environment")],
        [_sample(total=10, response=4, loss_mask=[0, 0, 0, 0], role="actor", pad=True)],
    ]
    metrics = rollout_token_accounting_metrics(groups)

    assert metrics["rollout/tokens/real/all/samples"] == 2
    assert metrics["rollout/tokens/padded/all/samples"] == 1
    assert metrics["rollout/tokens/real/all/prompt_tokens"] == 11
    assert metrics["rollout/tokens/real/all/response_tokens"] == 7
    assert metrics["rollout/tokens/real/all/loss_mask_tokens"] == 6
    assert metrics["rollout/tokens/padded/all/sequence_tokens"] == 10
    assert metrics["rollout/groups/real/all/episode_groups"] == 2
    assert metrics["rollout/groups/padded/all/episode_groups"] == 1
    assert metrics["rollout/groups/total/all/episode_groups"] == 3

    for measure in (
        "samples",
        "prompt_tokens",
        "response_tokens",
        "loss_mask_tokens",
        "sequence_tokens",
    ):
        assert metrics[f"rollout/tokens/total/all/{measure}"] == (
            metrics[f"rollout/tokens/real/all/{measure}"]
            + metrics[f"rollout/tokens/padded/all/{measure}"]
        )

    for population in ("real", "padded", "total"):
        for measure in (
            "samples",
            "prompt_tokens",
            "response_tokens",
            "loss_mask_tokens",
            "sequence_tokens",
        ):
            assert metrics[f"rollout/tokens/{population}/all/{measure}"] == sum(
                metrics[f"rollout/tokens/{population}/{role}/{measure}"]
                for role in ("actor", "environment", "unknown")
            )
        assert metrics[f"rollout/tokens/{population}/all/sequence_tokens"] == (
            metrics[f"rollout/tokens/{population}/all/prompt_tokens"]
            + metrics[f"rollout/tokens/{population}/all/response_tokens"]
        )


def test_unknown_roles_and_removed_samples_remain_rederivable() -> None:
    metrics = rollout_token_accounting_metrics(
        [[_sample(total=5, response=2, loss_mask=[1, 1], role=None, remove=True)]]
    )
    assert metrics["rollout/tokens/real/unknown/samples"] == 1
    assert metrics["rollout/tokens/real/unknown/loss_mask_tokens"] == 0
    assert metrics["rollout/tokens/total/all/sequence_tokens"] == 5


def test_invalid_token_shapes_fail_closed() -> None:
    with pytest.raises(TokenAccountingError, match="loss_mask length"):
        rollout_token_accounting_metrics(
            [[_sample(total=5, response=2, loss_mask=[1], role="actor")]]
        )


def test_empty_or_mixed_episode_groups_fail_closed() -> None:
    with pytest.raises(TokenAccountingError, match="must not be empty"):
        rollout_token_accounting_metrics([[]])
    with pytest.raises(TokenAccountingError, match="share role and padding status"):
        rollout_token_accounting_metrics(
            [
                [
                    _sample(total=5, response=2, loss_mask=[1, 1], role="actor"),
                    _sample(total=5, response=2, loss_mask=[1, 1], role="environment"),
                ]
            ]
        )


def test_source_topology_metrics_enforce_complete_no_upsample_contract() -> None:
    collect_info = {
        "num_instances": 192,
        "num_succeeded": 192,
        "num_failed": 0,
        "rollout/num_filtered_actor_trajectories": 0,
        "rollout/num_filtered_env_trajectories": 0,
    }
    metrics = rollout_source_topology_metrics(
        collect_info,
        192,
        require_complete=True,
    )

    assert metrics == {
        "rollout/topology/actor_instances_requested": 192,
        "rollout/topology/actor_instances_succeeded": 192,
        "rollout/topology/actor_instances_failed": 0,
        "rollout/topology/actor_trajectories_filtered": 0,
        "rollout/topology/environment_trajectories_filtered": 0,
    }


def test_source_topology_metrics_fail_closed_on_partial_or_invalid_counts() -> None:
    partial = {
        "num_instances": 192,
        "num_succeeded": 191,
        "num_failed": 1,
        "rollout/num_filtered_actor_trajectories": 0,
        "rollout/num_filtered_env_trajectories": 0,
    }
    with pytest.raises(TokenAccountingError, match="complete-rollout contract failed"):
        rollout_source_topology_metrics(partial, 192, require_complete=True)

    invalid = dict(partial, num_failed=0.5)
    with pytest.raises(TokenAccountingError, match="num_failed"):
        rollout_source_topology_metrics(invalid, 192)
