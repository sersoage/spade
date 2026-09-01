#!/usr/bin/env python3
"""Run the fixed, offline constant-zero CWA shadow-archive smoke.

The source selections are read only to recover the externally anchored game
bytes.  They are never imported or executed.  The command has no AGY, provider,
network, reward, or active-selection integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spade.core.witness_archive_shadow import (  # noqa: E402
    ShadowArchiveLedger,
    ValidatedWitnessCohort,
    WitnessArchiveShadowError,
    _pretty_json,
    _read_json,
    load_validated_cwa_run,
)


QUALITY_POLICY_ID = "constant-zero-integration-smoke/v1"
LINEAGE_POLICY_ID = "singleton-environment-digest/v1"
SELECTION_SCHEMA = "spade-agy-selected-candidate/v1"
SUMMARY_SCHEMA = "spade-counterfactual-witness-shadow-smoke/v1"


def _source_digest(value: object) -> str:
    content = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _load_source_bytes(
    selection_root: Path,
    cohort: ValidatedWitnessCohort,
) -> dict[str, bytes]:
    if not selection_root.is_absolute():
        raise WitnessArchiveShadowError("source_selections_dir must be absolute")
    if selection_root.is_symlink() or not selection_root.is_dir():
        raise WitnessArchiveShadowError("source selections directory is missing or unsafe")
    expected_names = {f"{item.cluster_id}.json" for item in cohort.evidence}
    observed_names: set[str] = set()
    for path in selection_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise WitnessArchiveShadowError(f"unsafe source selection artifact: {path}")
        observed_names.add(path.name)
    if observed_names != expected_names:
        raise WitnessArchiveShadowError("source selection inventory differs from the CWA cohort")

    sources: dict[str, bytes] = {}
    expected_fields = {
        "schema_version",
        "plan_digest",
        "cluster_id",
        "candidate_id",
        "candidate_role",
        "skill",
        "difficulty",
        "code",
        "code_digest",
        "environment_name",
        "environment_digest",
        "qualification_digest",
        "probes",
        "hints",
        "selection_digest",
    }
    for evidence in cohort.evidence:
        selection = _read_json(
            selection_root / f"{evidence.cluster_id}.json",
            f"source selection {evidence.cluster_id}",
        )
        if set(selection) != expected_fields or selection["schema_version"] != SELECTION_SCHEMA:
            raise WitnessArchiveShadowError(
                f"source selection {evidence.cluster_id} differs from its schema"
            )
        body = {key: value for key, value in selection.items() if key != "selection_digest"}
        if selection["selection_digest"] != _source_digest(body):
            raise WitnessArchiveShadowError(
                f"source selection {evidence.cluster_id} self-digest mismatch"
            )
        expected_binding: Mapping[str, Any] = {
            "plan_digest": evidence.source_plan_digest,
            "cluster_id": evidence.cluster_id,
            "candidate_id": evidence.candidate_id,
            "skill": evidence.skill,
            "difficulty": evidence.difficulty,
            "environment_name": evidence.environment_name,
            "environment_digest": evidence.environment_digest,
            "qualification_digest": evidence.qualification_digest,
        }
        if any(selection.get(key) != value for key, value in expected_binding.items()):
            raise WitnessArchiveShadowError(
                f"source selection {evidence.cluster_id} breaks its CWA binding"
            )
        code = selection["code"]
        if not isinstance(code, str):
            raise WitnessArchiveShadowError(f"source selection {evidence.cluster_id} lacks code")
        source = code.encode("utf-8", errors="strict")
        if (
            _source_digest(code) != selection["code_digest"]
            or _bytes_digest(source) != evidence.environment_digest
        ):
            raise WitnessArchiveShadowError(
                f"source selection {evidence.cluster_id} game bytes drifted"
            )
        sources[evidence.cluster_id] = source
    return sources


def run_constant_zero_smoke(
    *,
    run_dir: Path,
    expected_aggregate_digest: str,
    source_selections_dir: Path,
    ledger_dir: Path,
) -> dict[str, Any]:
    """Replay the fixed integration smoke into the shadow-only archive ledger."""
    cohort = load_validated_cwa_run(run_dir, expected_aggregate_digest)
    sources = _load_source_bytes(source_selections_dir, cohort)
    ledger = ShadowArchiveLedger(
        ledger_dir,
        cohort=cohort,
        quality_policy_id=QUALITY_POLICY_ID,
        lineage_policy_id=LINEAGE_POLICY_ID,
    )
    actions: list[str] = []
    with tempfile.TemporaryDirectory(
        prefix="spade-witness-shadow-games-", dir=ledger_dir.parent
    ) as temporary:
        temporary_root = Path(temporary)
        for evidence in cohort.evidence:
            game_file = temporary_root / f"{evidence.cluster_id}.py"
            game_file.write_bytes(sources[evidence.cluster_id])
            receipt = ledger.consider(
                event_id=f"constant-zero:{evidence.cluster_id}",
                evidence=evidence,
                game_file=game_file,
                quality_score=0.0,
                lineage=(evidence.environment_digest,),
            )
            actions.append(receipt.action)
    summary = ledger.summary()
    return {
        "schema_version": SUMMARY_SCHEMA,
        "aggregate_digest": cohort.aggregate_digest,
        "quality_policy_id": QUALITY_POLICY_ID,
        "lineage_policy_id": LINEAGE_POLICY_ID,
        "event_count": summary.event_count,
        "cell_count": summary.cell_count,
        "action_counts": dict(summary.action_counts),
        "actions": actions,
        "head_event_digest": summary.head_event_digest,
        "post_state_digest": summary.post_state_digest,
        "provider_calls": 0,
        "learner_updates": 0,
        "game_source_executions": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-aggregate-digest", required=True)
    parser.add_argument("--source-selections-dir", required=True, type=Path)
    parser.add_argument("--ledger-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = run_constant_zero_smoke(
            run_dir=arguments.run_dir,
            expected_aggregate_digest=arguments.expected_aggregate_digest,
            source_selections_dir=arguments.source_selections_dir,
            ledger_dir=arguments.ledger_dir,
        )
    except WitnessArchiveShadowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_pretty_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
