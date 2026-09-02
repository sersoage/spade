"""Exact token accounting for the final SPADE-to-Slime rollout payload."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real
from typing import Any


class TokenAccountingError(ValueError):
    """A final sample violates the token-shape contract used by Slime."""


def rollout_source_topology_metrics(
    collect_info: Any,
    expected_actor_groups: int,
    *,
    require_complete: bool = False,
) -> dict[str, int]:
    """Normalize source counters and optionally enforce a no-upsample topology."""

    if not isinstance(collect_info, dict):
        raise TokenAccountingError("collect_info must be a mapping")
    if (
        isinstance(expected_actor_groups, bool)
        or not isinstance(expected_actor_groups, int)
        or expected_actor_groups <= 0
    ):
        raise TokenAccountingError("expected_actor_groups must be a positive integer")
    source_fields = {
        "rollout/topology/actor_instances_requested": "num_instances",
        "rollout/topology/actor_instances_succeeded": "num_succeeded",
        "rollout/topology/actor_instances_failed": "num_failed",
        "rollout/topology/actor_trajectories_filtered": ("rollout/num_filtered_actor_trajectories"),
        "rollout/topology/environment_trajectories_filtered": (
            "rollout/num_filtered_env_trajectories"
        ),
    }
    metrics: dict[str, int] = {}
    for metric, source in source_fields.items():
        value = collect_info.get(source, 0)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise TokenAccountingError(f"collect_info.{source} must be a non-negative integer")
        metrics[metric] = int(value)
    expected = {
        "rollout/topology/actor_instances_requested": expected_actor_groups,
        "rollout/topology/actor_instances_succeeded": expected_actor_groups,
        "rollout/topology/actor_instances_failed": 0,
        "rollout/topology/actor_trajectories_filtered": 0,
        "rollout/topology/environment_trajectories_filtered": 0,
    }
    if require_complete and metrics != expected:
        raise TokenAccountingError(
            "complete-rollout contract failed before optimizer step: "
            f"observed={metrics}, expected={expected}"
        )
    return metrics


_POPULATIONS = ("real", "padded", "total")
_ROLES = ("all", "actor", "environment", "unknown")
_MEASURES = ("samples", "prompt", "response", "loss_mask", "sequence")


def _empty_counts() -> dict[str, dict[str, dict[str, int]]]:
    return {
        population: {role: {measure: 0 for measure in _MEASURES} for role in _ROLES}
        for population in _POPULATIONS
    }


def _sample_counts(sample: Any) -> tuple[str, bool, dict[str, int]]:
    tokens = getattr(sample, "tokens", None)
    response_length = getattr(sample, "response_length", None)
    if not isinstance(tokens, (list, tuple)):
        raise TokenAccountingError("sample.tokens must be a list or tuple")
    if (
        isinstance(response_length, bool)
        or not isinstance(response_length, int)
        or response_length < 0
        or response_length > len(tokens)
    ):
        raise TokenAccountingError(
            "sample.response_length must be between zero and len(sample.tokens)"
        )

    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        loss_mask_tokens = response_length
    else:
        if not isinstance(loss_mask, (list, tuple)) or len(loss_mask) != response_length:
            raise TokenAccountingError("sample.loss_mask length must equal sample.response_length")
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or value not in (0, 1)
            for value in loss_mask
        ):
            raise TokenAccountingError("sample.loss_mask must contain only numeric 0/1 values")
        loss_mask_tokens = int(sum(loss_mask))
    if bool(getattr(sample, "remove_sample", False)):
        loss_mask_tokens = 0

    metadata = getattr(sample, "metadata", None) or {}
    if not isinstance(metadata, dict):
        raise TokenAccountingError("sample.metadata must be a mapping")
    raw_role = metadata.get("role")
    role = raw_role if raw_role in ("actor", "environment") else "unknown"
    is_pad = bool(metadata.get("spade_is_pad"))
    sequence_tokens = len(tokens)
    counts = {
        "samples": 1,
        "prompt": sequence_tokens - response_length,
        "response": response_length,
        "loss_mask": loss_mask_tokens,
        "sequence": sequence_tokens,
    }
    return role, is_pad, counts


def rollout_token_accounting_metrics(
    grouped_samples: Iterable[Iterable[Any]],
) -> dict[str, int]:
    """Return additive metrics for exactly the samples sent to Slime.

    ``real`` excludes inert samples marked ``metadata.spade_is_pad``;
    ``padded`` contains only those samples; and ``total`` is their sum.  Every
    population also has actor, environment, and unknown-role components, making
    every aggregate independently re-derivable from the emitted metrics.
    """

    counts = _empty_counts()
    group_counts = {population: {role: 0 for role in _ROLES} for population in _POPULATIONS}
    for raw_group in grouped_samples:
        group = list(raw_group)
        if not group:
            raise TokenAccountingError("final sample groups must not be empty")
        group_identities: set[tuple[str, bool]] = set()
        for sample in group:
            role, is_pad, sample_counts = _sample_counts(sample)
            group_identities.add((role, is_pad))
            population = "padded" if is_pad else "real"
            for target_population in (population, "total"):
                for target_role in (role, "all"):
                    for measure, value in sample_counts.items():
                        counts[target_population][target_role][measure] += value
        if len(group_identities) != 1:
            raise TokenAccountingError(
                "all samples in an episode group must share role and padding status"
            )
        role, is_pad = next(iter(group_identities))
        population = "padded" if is_pad else "real"
        for target_population in (population, "total"):
            for target_role in (role, "all"):
                group_counts[target_population][target_role] += 1

    metrics: dict[str, int] = {}
    for population in _POPULATIONS:
        for role in _ROLES:
            for measure in _MEASURES:
                metrics[
                    (
                        f"rollout/tokens/{population}/{role}/{measure}_tokens"
                        if measure != "samples"
                        else f"rollout/tokens/{population}/{role}/samples"
                    )
                ] = counts[population][role][measure]
            metrics[f"rollout/groups/{population}/{role}/episode_groups"] = group_counts[
                population
            ][role]
    return metrics


__all__ = [
    "TokenAccountingError",
    "rollout_source_topology_metrics",
    "rollout_token_accounting_metrics",
]
