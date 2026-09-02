#!/usr/bin/env python3
"""Materialize or verify sealed static pools for the SPADE learner assay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make direct invocation work from any current directory without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spade.core.learner_branch_pools import (
    LearnerPoolError,
    PoolBundle,
    load_learner_pool_manifest,
    materialize_learner_pools,
)
from spade.slime.branch_assay import (
    SlimeAssayError,
    SlimeAssayPlan,
    load_slime_assay_plan,
    materialize_slime_assay_plan,
    validate_slime_assay_results,
)


def _summary(bundle: PoolBundle) -> dict[str, object]:
    pools = bundle.manifest["pools"]
    return {
        "status": "verified",
        "root": str(bundle.root.resolve()),
        "manifest_path": str(bundle.manifest_path.resolve()),
        "manifest_digest": bundle.manifest["manifest_digest"],
        "schedule_id": bundle.schedule_id,
        "coverage_forced_dir": str(bundle.coverage_forced_dir.resolve()),
        "redundant_historical_dir": str(bundle.redundant_historical_dir.resolve()),
        "heldout_v4_dir": (str(bundle.heldout_v4_dir.resolve()) if bundle.heldout_v4_dir else None),
        "pool_counts": {name: value["game_count"] for name, value in sorted(pools.items())},
        "execution_contract": bundle.manifest["execution_contract"],
    }


def _slime_plan_summary(plan: SlimeAssayPlan) -> dict[str, object]:
    return {
        "status": "verified",
        "backend": plan.manifest["backend"],
        "execution_state": plan.manifest["execution_state"],
        "evidence_state": plan.manifest["evidence_state"],
        "root": str(plan.root.resolve()),
        "manifest_path": str(plan.manifest_path.resolve()),
        "plan_digest": plan.manifest["plan_digest"],
        "pool_manifest_digest": plan.manifest["pool_manifest_digest"],
        "num_pairs": plan.manifest["num_pairs"],
        "num_runs": plan.manifest["num_runs"],
        "num_rollouts_per_run": plan.manifest["num_rollouts_per_run"],
        "paired_seeds": plan.manifest["paired_seeds"],
        "compute_gate": plan.manifest["compute_gate"],
        "rollout_topology_gate": plan.manifest["rollout_topology_gate"],
        "claim_scope": plan.manifest["claim_scope"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only sealed-pool preparation. This tool makes no provider calls "
            "and does not launch learner training."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize", help="validate sealed source evidence and write a new pool bundle"
    )
    materialize.add_argument("--actor-plan", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--source-intent", type=Path)
    materialize.add_argument("--v4-run-dir", type=Path)
    materialize.add_argument(
        "--without-heldout",
        action="store_true",
        help="omit the disjoint v4 heldout pool (training pools remain unchanged)",
    )

    verify = subparsers.add_parser(
        "verify", help="revalidate a materialized manifest and exact file inventory"
    )
    verify.add_argument("--bundle", type=Path, required=True)

    plan_slime = subparsers.add_parser(
        "plan-slime",
        help="write 12 sealed, counterbalanced Slime launch specs for six paired seeds",
    )
    plan_slime.add_argument("--bundle", type=Path, required=True)
    plan_slime.add_argument("--output-dir", type=Path, required=True)
    plan_slime.add_argument("--remote-pool-root", required=True)
    plan_slime.add_argument("--remote-output-root", required=True)
    plan_slime.add_argument("--hf-checkpoint", required=True)
    plan_slime.add_argument("--hf-checkpoint-digest", required=True)
    plan_slime.add_argument("--reference-checkpoint", required=True)
    plan_slime.add_argument("--reference-checkpoint-digest", required=True)
    plan_slime.add_argument("--runtime-image", required=True)
    plan_slime.add_argument("--runtime-image-digest", required=True)
    plan_slime.add_argument("--spade-source-revision", required=True)
    plan_slime.add_argument("--source-root", type=Path, default=Path.cwd())
    plan_slime.add_argument("--num-rollouts", type=int, default=16)
    plan_slime.add_argument("--token-tolerance", type=float, default=0.05)

    verify_slime = subparsers.add_parser(
        "verify-slime-plan", help="revalidate a sealed Slime assay plan and launch inventory"
    )
    verify_slime.add_argument("--plan", type=Path, required=True)

    validate_slime = subparsers.add_parser(
        "validate-slime-results",
        help="check self-supplied receipt consistency and emit an unconditional PLAN-ONLY HOLD",
    )
    validate_slime.add_argument("--plan", type=Path, required=True)
    validate_slime.add_argument("--results", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "materialize":
            bundle = materialize_learner_pools(
                args.actor_plan,
                args.output_dir,
                source_intent_path=args.source_intent,
                v4_run_dir=args.v4_run_dir,
                include_heldout=not args.without_heldout,
            )
        elif args.command == "verify":
            bundle = load_learner_pool_manifest(args.bundle, verify_files=True)
        elif args.command == "plan-slime":
            plan = materialize_slime_assay_plan(
                args.bundle,
                args.output_dir,
                remote_pool_root=args.remote_pool_root,
                remote_output_root=args.remote_output_root,
                hf_checkpoint=args.hf_checkpoint,
                hf_checkpoint_digest=args.hf_checkpoint_digest,
                reference_checkpoint=args.reference_checkpoint,
                reference_checkpoint_digest=args.reference_checkpoint_digest,
                runtime_image=args.runtime_image,
                runtime_image_digest=args.runtime_image_digest,
                spade_source_revision=args.spade_source_revision,
                source_root=args.source_root,
                num_rollouts=args.num_rollouts,
                token_tolerance=args.token_tolerance,
            )
        elif args.command == "verify-slime-plan":
            plan = load_slime_assay_plan(args.plan, verify_files=True)
        else:
            validation = validate_slime_assay_results(args.plan, args.results)
    except (LearnerPoolError, SlimeAssayError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.command in {"materialize", "verify"}:
        output = _summary(bundle)
    elif args.command in {"plan-slime", "verify-slime-plan"}:
        output = _slime_plan_summary(plan)
    else:
        output = validation
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
