"""Fail-closed shadow ingestion for sealed counterfactual-witness evidence.

This module deliberately has no active-selection interface.  It validates a
completed CWA run from an external aggregate digest, binds each certificate to
caller-supplied game bytes and archive policies, and records archive decisions
in an immutable hash-chained ledger.  It never executes game source and never
calls a model, provider, reward service, or network API.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

from spade.core.counterfactual_witness import (
    WITNESS_SCHEMA_VERSION,
    WitnessProbe,
    mutation_catalog_digest,
)
from spade.core.witness_archive import (
    ArchiveDecision,
    ArchiveEntry,
    BehaviorDescriptor,
    CounterfactualWitnessArchive,
)


PLAN_SCHEMA = "spade-counterfactual-witness-plan/v1"
CLUSTER_RESULT_SCHEMA = "spade-counterfactual-witness-cluster-result/v1"
AGGREGATE_SCHEMA = "spade-counterfactual-witness-aggregate/v1"
PROTOCOL_ID = "spade-counterfactual-witness-falsification/v1"
LEDGER_SCHEMA = "spade-counterfactual-witness-shadow-ledger/v1"
EVENT_SCHEMA = "spade-counterfactual-witness-shadow-event/v1"
AUTHORIZED_SOURCE_PLAN_DIGEST = (
    "sha256:8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc"
)
AUTHORIZED_SOURCE_COHORT_DIGEST = (
    "sha256:161353ebd4454516e3379414444323dd13aeab95640eb130ec7414f23876b84b"
)
AUTHORIZED_CLUSTER_COUNT = 18

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
_EVENT_NAME_RE = re.compile(r"([0-9]{6})-([0-9a-f]{64})\.json")


class WitnessArchiveShadowError(RuntimeError):
    """The sealed evidence or immutable shadow ledger is unsafe or inconsistent."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_digest(value: object, where: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise WitnessArchiveShadowError(f"{where} must be a lowercase sha256 digest")
    return value


def _require_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WitnessArchiveShadowError(f"{where} must be non-empty canonical text")
    return value


def _require_component(value: object, where: str) -> str:
    text = _require_text(value, where)
    if _COMPONENT_RE.fullmatch(text) is None:
        raise WitnessArchiveShadowError(f"{where} is not a safe path component")
    return text


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WitnessArchiveShadowError(f"JSON object repeats key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise WitnessArchiveShadowError(f"non-finite JSON number {value!r} is forbidden")


def _reject_symlink_ancestors(path: Path) -> None:
    candidate = path
    while True:
        if candidate.is_symlink():
            raise WitnessArchiveShadowError(f"symlinked paths are forbidden: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _safe_regular_file(path: Path, where: str) -> None:
    _reject_symlink_ancestors(path)
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise WitnessArchiveShadowError(f"missing {where}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise WitnessArchiveShadowError(f"{where} is not a regular file: {path}")


def _decode_json(content: bytes, where: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WitnessArchiveShadowError(f"invalid JSON at {where}") from exc
    if not isinstance(value, dict):
        raise WitnessArchiveShadowError(f"JSON leaf at {where} must be an object")
    return value


def _read_json(path: Path, where: str) -> dict[str, Any]:
    _safe_regular_file(path, where)
    content = path.read_bytes()
    value = _decode_json(content, str(path))
    if content != _pretty_json(value):
        raise WitnessArchiveShadowError(f"{where} is not canonical pretty JSON: {path}")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        raise WitnessArchiveShadowError(f"{where} fields differ from its sealed schema")


def _sealed_body(value: Mapping[str, Any], digest_key: str, where: str) -> None:
    claimed = _require_digest(value.get(digest_key), f"{where} {digest_key}")
    body = {key: item for key, item in value.items() if key != digest_key}
    if claimed != _digest(body):
        raise WitnessArchiveShadowError(f"{where} self-digest mismatch")


def _descriptor(value: object, where: str) -> BehaviorDescriptor:
    expected = {
        "action_format",
        "oracle_depth_bin",
        "reset_seed_diversity_bin",
        "invalid_reward_bin",
        "invalid_end_state",
        "recovery_success",
        "trace_order_divergent",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise WitnessArchiveShadowError(f"{where} fields differ from descriptor schema")
    try:
        return BehaviorDescriptor(**value)
    except (TypeError, ValueError) as exc:
        raise WitnessArchiveShadowError(f"invalid {where}") from exc


def _read_game_digest(path_value: Path | str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        raise WitnessArchiveShadowError("game_file must be an absolute path")
    _safe_regular_file(path, "game file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WitnessArchiveShadowError(f"cannot safely open game file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise WitnessArchiveShadowError("game file descriptor is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
            raise WitnessArchiveShadowError("game file changed while its bytes were verified")
        return _bytes_digest(b"".join(chunks))
    finally:
        os.close(descriptor)


def _quality_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WitnessArchiveShadowError("quality_score must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise WitnessArchiveShadowError("quality_score must be finite")
    return score


def _lineage(value: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not value:
        raise WitnessArchiveShadowError("lineage must contain identifiers")
    lineage = tuple(value)
    if any(not isinstance(item, str) or not item or item != item.strip() for item in lineage):
        raise WitnessArchiveShadowError("lineage identifiers must be non-empty canonical text")
    if len(set(lineage)) != len(lineage):
        raise WitnessArchiveShadowError("lineage identifiers must be unique")
    return lineage


@dataclass(frozen=True)
class ValidatedWitnessEvidence:
    """One evidence item reachable from an externally anchored passing aggregate."""

    aggregate_digest: str
    plan_digest: str
    source_plan_digest: str
    source_cohort_digest: str
    schedule_ordinal: int
    cluster_id: str
    cluster_result_digest: str
    candidate_id: str
    skill: str
    difficulty: str
    environment_name: str
    environment_digest: str
    qualification_digest: str
    descriptor: BehaviorDescriptor
    witness_digest: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "aggregate_digest": self.aggregate_digest,
            "plan_digest": self.plan_digest,
            "source_plan_digest": self.source_plan_digest,
            "source_cohort_digest": self.source_cohort_digest,
            "schedule_ordinal": self.schedule_ordinal,
            "cluster_id": self.cluster_id,
            "cluster_result_digest": self.cluster_result_digest,
            "candidate_id": self.candidate_id,
            "skill": self.skill,
            "difficulty": self.difficulty,
            "environment_name": self.environment_name,
            "environment_digest": self.environment_digest,
            "qualification_digest": self.qualification_digest,
            "behavior_descriptor": self.descriptor.to_dict(),
            "witness_digest": self.witness_digest,
        }

    def bind(
        self,
        game_file: Path | str,
        quality_score: float,
        lineage: Sequence[str],
        *,
        quality_policy_id: str,
        lineage_policy_id: str,
    ) -> ArchiveEntry:
        """Verify game bytes and bind caller-owned archive policy inputs."""
        quality_policy = _require_text(quality_policy_id, "quality_policy_id")
        lineage_policy = _require_text(lineage_policy_id, "lineage_policy_id")
        score = _quality_score(quality_score)
        bound_lineage = _lineage(lineage)
        game_digest = _read_game_digest(game_file)
        if game_digest != self.environment_digest:
            raise WitnessArchiveShadowError(
                f"game bytes do not match sealed environment {self.cluster_id}"
            )
        return ArchiveEntry(
            environment_digest=self.environment_digest,
            witness_digest=self.witness_digest,
            # Do not persist or expose the caller's filesystem path.
            game_file=f"verified-game@{game_digest}",
            skill=self.skill,
            difficulty=self.difficulty,
            descriptor=self.descriptor,
            quality_score=score,
            lineage=bound_lineage,
            metadata={
                "aggregate_digest": self.aggregate_digest,
                "plan_digest": self.plan_digest,
                "cluster_id": self.cluster_id,
                "cluster_result_digest": self.cluster_result_digest,
                "candidate_id": self.candidate_id,
                "qualification_digest": self.qualification_digest,
                "game_bytes_digest": game_digest,
                "quality_policy_id": quality_policy,
                "lineage_policy_id": lineage_policy,
            },
        )


@dataclass(frozen=True)
class ValidatedWitnessCohort:
    """The validated archive-facing projection of one passing CWA run."""

    aggregate_digest: str
    plan_digest: str
    source_plan_digest: str
    source_cohort_digest: str
    evidence: Tuple[ValidatedWitnessEvidence, ...]

    def __post_init__(self) -> None:
        if len(self.evidence) != AUTHORIZED_CLUSTER_COUNT:
            raise WitnessArchiveShadowError("validated CWA cohort must contain 18 evidence items")
        cluster_ids = [item.cluster_id for item in self.evidence]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise WitnessArchiveShadowError("validated CWA cohort repeats a cluster id")
        for ordinal, item in enumerate(self.evidence, start=1):
            if (
                item.schedule_ordinal != ordinal
                or item.aggregate_digest != self.aggregate_digest
                or item.plan_digest != self.plan_digest
                or item.source_plan_digest != self.source_plan_digest
                or item.source_cohort_digest != self.source_cohort_digest
            ):
                raise WitnessArchiveShadowError("validated evidence differs from cohort identity")

    def _by_cluster(self) -> dict[str, ValidatedWitnessEvidence]:
        return {item.cluster_id: item for item in self.evidence}


def _validate_plan(plan: Mapping[str, Any]) -> None:
    _require_keys(
        plan,
        {
            "schema_version",
            "protocol_id",
            "experiment_id",
            "source",
            "output_root",
            "runtime_identity",
            "configuration",
            "family_split",
            "clusters",
            "analysis_role",
            "provider_calls",
            "learner_updates",
            "plan_digest",
        },
        "plan",
    )
    if (
        plan["schema_version"] != PLAN_SCHEMA
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["experiment_id"] != "spade-counterfactual-witness-v1"
        or plan["analysis_role"] != "offline-representation-falsification-only"
        or plan["provider_calls"] != 0
        or plan["learner_updates"] != 0
    ):
        raise WitnessArchiveShadowError("plan protocol identity is invalid")
    _sealed_body(plan, "plan_digest", "plan")
    source = plan["source"]
    if not isinstance(source, dict) or set(source) != {
        "run_dir",
        "plan_digest",
        "cohort_digest",
        "import_manifest_digest",
        "import_leaf_count",
    }:
        raise WitnessArchiveShadowError("plan source binding is malformed")
    if (
        source["plan_digest"] != AUTHORIZED_SOURCE_PLAN_DIGEST
        or source["cohort_digest"] != AUTHORIZED_SOURCE_COHORT_DIGEST
        or source["import_leaf_count"] != 321
    ):
        raise WitnessArchiveShadowError("plan does not bind the authorized source cohort")
    _require_digest(source["import_manifest_digest"], "plan import manifest digest")
    configuration = plan["configuration"]
    if not isinstance(configuration, dict):
        raise WitnessArchiveShadowError("plan configuration is malformed")
    _require_keys(
        configuration,
        {
            "action_format",
            "seeds",
            "max_turns",
            "operation_timeout_seconds",
            "repetitions",
            "witness_budget",
            "random_baseline_draws",
            "sandbox_operation_ceiling",
            "primary_recall_margin",
            "max_equivalent_false_rejection_rate",
            "minimum_training_recall",
            "minimum_safe_bank_heldout_recall",
        },
        "plan configuration",
    )
    if (
        configuration["action_format"] != "boxed"
        or configuration["seeds"] != [0, 1, 42]
        or configuration["max_turns"] != 5
        or configuration["repetitions"] != 2
        or configuration["primary_recall_margin"] != 0.15
        or configuration["max_equivalent_false_rejection_rate"] != 0.05
        or configuration["minimum_training_recall"] != 0.9
        or configuration["minimum_safe_bank_heldout_recall"] != 0.9
    ):
        raise WitnessArchiveShadowError("plan configuration differs from the CWA v1 protocol")
    clusters = plan["clusters"]
    if not isinstance(clusters, list) or len(clusters) != AUTHORIZED_CLUSTER_COUNT:
        raise WitnessArchiveShadowError("plan must schedule exactly 18 clusters")
    seen: set[str] = set()
    for ordinal, cluster in enumerate(clusters, start=1):
        if not isinstance(cluster, dict) or cluster.get("schedule_ordinal") != ordinal:
            raise WitnessArchiveShadowError("plan cluster schedule is malformed")
        _require_keys(
            cluster,
            {
                "cluster_id",
                "candidate_id",
                "skill",
                "difficulty",
                "environment_name",
                "code_digest",
                "environment_digest",
                "qualification_digest",
                "variants",
                "probes",
                "schedule_ordinal",
            },
            f"plan cluster {ordinal}",
        )
        cluster_id = _require_component(cluster.get("cluster_id"), "cluster id")
        if cluster_id in seen:
            raise WitnessArchiveShadowError("plan repeats a cluster id")
        seen.add(cluster_id)
        for key in ("candidate_id", "skill", "difficulty", "environment_name"):
            _require_text(cluster.get(key), f"cluster {cluster_id} {key}")
        for key in ("code_digest", "environment_digest", "qualification_digest"):
            _require_digest(cluster.get(key), f"cluster {cluster_id} {key}")
        if not isinstance(cluster.get("probes"), list) or not cluster["probes"]:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} has no probe catalog")
        if not isinstance(cluster.get("variants"), list) or not cluster["variants"]:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} has no mutation catalog")


def _validate_aggregate(aggregate: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    _require_keys(
        aggregate,
        {
            "schema_version",
            "protocol_id",
            "plan_digest",
            "source_plan_digest",
            "source_cohort_digest",
            "status",
            "metrics",
            "gates",
            "thresholds",
            "cluster_result_digests",
            "sandbox_operations_completed",
            "provider_calls",
            "learner_updates",
            "claim_boundary",
            "aggregate_digest",
        },
        "aggregate",
    )
    source = plan["source"]
    if (
        aggregate["schema_version"] != AGGREGATE_SCHEMA
        or aggregate["protocol_id"] != PROTOCOL_ID
        or aggregate["plan_digest"] != plan["plan_digest"]
        or aggregate["source_plan_digest"] != source["plan_digest"]
        or aggregate["source_cohort_digest"] != source["cohort_digest"]
        or aggregate["status"] != "pass"
        or aggregate["provider_calls"] != 0
        or aggregate["learner_updates"] != 0
    ):
        raise WitnessArchiveShadowError("aggregate is not a passing zero-provider CWA result")
    gates = aggregate["gates"]
    if (
        not isinstance(gates, dict)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise WitnessArchiveShadowError("passing aggregate must have every sealed gate true")
    config = plan["configuration"]
    expected_thresholds = {
        "primary_recall_margin": config["primary_recall_margin"],
        "max_equivalent_false_rejection_rate": config["max_equivalent_false_rejection_rate"],
        "minimum_training_recall": config["minimum_training_recall"],
        "minimum_safe_bank_heldout_recall": config["minimum_safe_bank_heldout_recall"],
    }
    if (
        aggregate["thresholds"] != expected_thresholds
        or aggregate["sandbox_operations_completed"] != config["sandbox_operation_ceiling"]
    ):
        raise WitnessArchiveShadowError("aggregate thresholds or operation count drifted")
    digests = aggregate["cluster_result_digests"]
    if not isinstance(digests, list) or len(digests) != AUTHORIZED_CLUSTER_COUNT:
        raise WitnessArchiveShadowError("aggregate must bind exactly 18 cluster results")
    for index, digest in enumerate(digests, start=1):
        _require_digest(digest, f"aggregate cluster digest {index}")
    _sealed_body(aggregate, "aggregate_digest", "aggregate")


def _validate_certificate(
    certificate: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    cluster: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    cluster_id = str(cluster["cluster_id"])
    _require_keys(
        certificate,
        {
            "schema_version",
            "environment_digest",
            "mutation_catalog_digest",
            "candidate_pool_digest",
            "selection",
            "probes",
            "expected_signatures",
            "metadata",
            "certificate_digest",
        },
        f"certificate {cluster_id}",
    )
    if (
        certificate["schema_version"] != WITNESS_SCHEMA_VERSION
        or certificate["environment_digest"] != cluster["environment_digest"]
        or certificate["mutation_catalog_digest"] != mutation_catalog_digest()
        or certificate["candidate_pool_digest"] != _digest(cluster["probes"])
        or certificate["selection"] != result["selection"]
    ):
        raise WitnessArchiveShadowError(f"certificate {cluster_id} binding is invalid")
    selection = certificate["selection"]
    if not isinstance(selection, dict):
        raise WitnessArchiveShadowError(f"certificate {cluster_id} selection is malformed")
    _require_keys(
        selection,
        {
            "selected_probe_ids",
            "safe_probe_ids",
            "rejected_control_breakers",
            "killed_train_mutants",
            "uncovered_train_mutants",
            "family_coverage",
            "budget",
            "algorithm",
        },
        f"certificate {cluster_id} selection",
    )
    if selection["algorithm"] != "safe-inverse-family-greedy-set-cover/v1":
        raise WitnessArchiveShadowError(f"certificate {cluster_id} selection algorithm drifted")
    selected_ids = selection.get("selected_probe_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise WitnessArchiveShadowError(f"certificate {cluster_id} selects no probes")
    catalog: dict[str, dict[str, Any]] = {}
    for raw_probe in cluster["probes"]:
        if not isinstance(raw_probe, dict) or set(raw_probe) != {
            "probe_id",
            "seed",
            "actions",
            "role",
        }:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} probe is malformed")
        try:
            probe = WitnessProbe(
                probe_id=raw_probe["probe_id"],
                seed=raw_probe["seed"],
                actions=tuple(raw_probe["actions"]),
                role=raw_probe["role"],
            )
        except (TypeError, ValueError) as exc:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} probe is invalid") from exc
        if probe.probe_id in catalog:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} repeats a probe")
        catalog[probe.probe_id] = probe.to_dict()
    if len(selected_ids) != len(set(selected_ids)) or any(
        item not in catalog for item in selected_ids
    ):
        raise WitnessArchiveShadowError(f"certificate {cluster_id} selected probes are invalid")
    expected_probes = [catalog[item] for item in selected_ids]
    if certificate["probes"] != expected_probes:
        raise WitnessArchiveShadowError(f"certificate {cluster_id} probe projection drifted")
    signatures = certificate["expected_signatures"]
    if not isinstance(signatures, dict) or set(signatures) != set(selected_ids):
        raise WitnessArchiveShadowError(f"certificate {cluster_id} signatures are malformed")
    for probe_id, digest in signatures.items():
        _require_digest(digest, f"certificate {cluster_id} signature {probe_id}")
    source = plan["source"]
    expected_metadata = {
        "plan_digest": plan["plan_digest"],
        "source_plan_digest": source["plan_digest"],
        "source_cohort_digest": source["cohort_digest"],
        "cluster_id": cluster_id,
        "candidate_id": cluster["candidate_id"],
        "qualification_digest": cluster["qualification_digest"],
        "analysis_role": plan["analysis_role"],
    }
    if certificate["metadata"] != expected_metadata:
        raise WitnessArchiveShadowError(f"certificate {cluster_id} metadata drifted")
    _sealed_body(certificate, "certificate_digest", f"certificate {cluster_id}")


def _validate_exact_inventory(directory: Path, expected: set[str], where: str) -> None:
    _reject_symlink_ancestors(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise WitnessArchiveShadowError(f"missing or unsafe {where}: {directory}")
    observed: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise WitnessArchiveShadowError(f"unexpected {where} artifact: {path}")
        observed.add(path.name)
    if observed != expected:
        raise WitnessArchiveShadowError(f"{where} inventory differs from the sealed schedule")


def load_validated_cwa_run(
    run_dir: Path | str,
    expected_aggregate_digest: str,
) -> ValidatedWitnessCohort:
    """Load the strict plan -> aggregate -> cluster -> certificate evidence closure."""
    external_anchor = _require_digest(expected_aggregate_digest, "external aggregate anchor")
    root = Path(run_dir)
    if not root.is_absolute():
        raise WitnessArchiveShadowError("run_dir must be absolute")
    _reject_symlink_ancestors(root)
    if root.is_symlink() or not root.is_dir():
        raise WitnessArchiveShadowError("run_dir must be an existing non-symlink directory")

    plan = _read_json(root / "plan.json", "CWA plan")
    _validate_plan(plan)
    aggregate = _read_json(root / "aggregate.json", "CWA aggregate")
    _validate_aggregate(aggregate, plan)
    if aggregate["aggregate_digest"] != external_anchor:
        raise WitnessArchiveShadowError("aggregate does not match the external digest anchor")

    expected_names = {f"{item['cluster_id']}.json" for item in plan["clusters"]}
    cluster_root = root / "clusters"
    certificate_root = root / "certificates"
    _validate_exact_inventory(cluster_root, expected_names, "cluster result")
    _validate_exact_inventory(certificate_root, expected_names, "certificate")

    evidence: list[ValidatedWitnessEvidence] = []
    for index, cluster in enumerate(plan["clusters"]):
        cluster_id = str(cluster["cluster_id"])
        result = _read_json(cluster_root / f"{cluster_id}.json", f"cluster result {cluster_id}")
        _require_keys(
            result,
            {
                "schema_version",
                "plan_digest",
                "cluster",
                "selection",
                "scores",
                "applicability",
                "proofpack_fixed_probe_ids",
                "random_baseline",
                "behavior_descriptor",
                "certificate",
                "trace_inventory",
                "provider_calls",
                "learner_updates",
                "cluster_result_digest",
            },
            f"cluster result {cluster_id}",
        )
        if (
            result["schema_version"] != CLUSTER_RESULT_SCHEMA
            or result["plan_digest"] != plan["plan_digest"]
            or result["cluster"] != cluster
            or result["provider_calls"] != 0
            or result["learner_updates"] != 0
        ):
            raise WitnessArchiveShadowError(f"cluster result {cluster_id} binding is invalid")
        _sealed_body(result, "cluster_result_digest", f"cluster result {cluster_id}")
        if result["cluster_result_digest"] != aggregate["cluster_result_digests"][index]:
            raise WitnessArchiveShadowError(f"aggregate does not bind cluster result {cluster_id}")
        descriptor = _descriptor(result["behavior_descriptor"], f"descriptor {cluster_id}")
        reference = result["certificate"]
        expected_reference = {
            "path": f"certificates/{cluster_id}.json",
            "digest": None,
        }
        if not isinstance(reference, dict) or set(reference) != {"path", "digest"}:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} certificate reference malformed")
        expected_reference["digest"] = _require_digest(
            reference["digest"], f"certificate reference {cluster_id}"
        )
        if reference != expected_reference:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} certificate path drifted")
        certificate = _read_json(
            certificate_root / f"{cluster_id}.json", f"certificate {cluster_id}"
        )
        _validate_certificate(certificate, plan=plan, cluster=cluster, result=result)
        if certificate["certificate_digest"] != reference["digest"]:
            raise WitnessArchiveShadowError(f"cluster {cluster_id} certificate digest drifted")
        evidence.append(
            ValidatedWitnessEvidence(
                aggregate_digest=external_anchor,
                plan_digest=plan["plan_digest"],
                source_plan_digest=plan["source"]["plan_digest"],
                source_cohort_digest=plan["source"]["cohort_digest"],
                schedule_ordinal=cluster["schedule_ordinal"],
                cluster_id=cluster_id,
                cluster_result_digest=result["cluster_result_digest"],
                candidate_id=cluster["candidate_id"],
                skill=cluster["skill"],
                difficulty=cluster["difficulty"],
                environment_name=cluster["environment_name"],
                environment_digest=cluster["environment_digest"],
                qualification_digest=cluster["qualification_digest"],
                descriptor=descriptor,
                witness_digest=certificate["certificate_digest"],
            )
        )
    return ValidatedWitnessCohort(
        aggregate_digest=external_anchor,
        plan_digest=plan["plan_digest"],
        source_plan_digest=plan["source"]["plan_digest"],
        source_cohort_digest=plan["source"]["cohort_digest"],
        evidence=tuple(evidence),
    )


def _decision_dict(decision: ArchiveDecision) -> dict[str, Any]:
    return {
        "action": decision.action,
        "cell_key": list(decision.cell_key),
        "accepted_digest": decision.accepted_digest,
        "demoted_digest": decision.demoted_digest,
        "evicted_digests": list(decision.evicted_digests),
    }


def _baseline_selection(
    archive: CounterfactualWitnessArchive, decision: ArchiveDecision
) -> dict[str, Any]:
    cell = archive.cells.get(decision.cell_key)
    if cell is None:
        raise WitnessArchiveShadowError("archive decision does not resolve to a post-state cell")
    return {
        "cell_key": list(decision.cell_key),
        "champion_environment_digest": cell.champion.environment_digest,
        "challenger_environment_digest": (
            cell.challenger.environment_digest if cell.challenger is not None else None
        ),
    }


def _entry_from_dict(value: object, evidence: ValidatedWitnessEvidence) -> ArchiveEntry:
    if not isinstance(value, dict) or set(value) != {
        "environment_digest",
        "witness_digest",
        "game_file",
        "skill",
        "difficulty",
        "descriptor",
        "quality_score",
        "lineage",
        "metadata",
    }:
        raise WitnessArchiveShadowError("ledger archive entry is malformed")
    descriptor = _descriptor(value["descriptor"], "ledger archive descriptor")
    if descriptor != evidence.descriptor:
        raise WitnessArchiveShadowError("ledger descriptor differs from validated evidence")
    metadata = value["metadata"]
    if not isinstance(metadata, dict):
        raise WitnessArchiveShadowError("ledger archive metadata is malformed")
    try:
        entry = ArchiveEntry(
            environment_digest=value["environment_digest"],
            witness_digest=value["witness_digest"],
            game_file=value["game_file"],
            skill=value["skill"],
            difficulty=value["difficulty"],
            descriptor=descriptor,
            quality_score=_quality_score(value["quality_score"]),
            lineage=_lineage(value["lineage"]),
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise WitnessArchiveShadowError("ledger archive entry is invalid") from exc
    return entry


@dataclass(frozen=True)
class ShadowArchiveReceipt:
    event_id: str
    sequence: int
    action: str
    event_digest: str
    post_state_digest: str
    resumed: bool


@dataclass(frozen=True)
class ShadowArchiveSummary:
    event_count: int
    cell_count: int
    action_counts: Mapping[str, int]
    head_event_digest: str
    post_state_digest: str


class ShadowArchiveLedger:
    """Append-only, exactly-once shadow replay of a validated witness cohort."""

    def __init__(
        self,
        ledger_dir: Path | str,
        *,
        cohort: ValidatedWitnessCohort,
        quality_policy_id: str,
        lineage_policy_id: str,
    ) -> None:
        self._root = Path(ledger_dir)
        if not self._root.is_absolute():
            raise WitnessArchiveShadowError("ledger_dir must be absolute")
        self._cohort = cohort
        self._quality_policy_id = _require_text(quality_policy_id, "quality_policy_id")
        self._lineage_policy_id = _require_text(lineage_policy_id, "lineage_policy_id")
        self._identity = {
            "schema_version": LEDGER_SCHEMA,
            "aggregate_digest": cohort.aggregate_digest,
            "plan_digest": cohort.plan_digest,
            "quality_policy_id": self._quality_policy_id,
            "lineage_policy_id": self._lineage_policy_id,
        }
        self._identity_digest = _digest(self._identity)
        self._prepare_root()
        with self._locked():
            self._reload()

    def _prepare_root(self) -> None:
        _reject_symlink_ancestors(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise WitnessArchiveShadowError("ledger root is not a safe directory")
        events = self._root / "events"
        events.mkdir(exist_ok=True)
        if events.is_symlink() or not events.is_dir():
            raise WitnessArchiveShadowError("ledger events path is not a safe directory")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_path = self._root / ".writer.lock"
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise WitnessArchiveShadowError("ledger writer lock is unsafe")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise WitnessArchiveShadowError("cannot safely open ledger writer lock") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise WitnessArchiveShadowError("ledger writer lock is not regular")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _validate_inventory(self) -> list[Path]:
        allowed = {".writer.lock", "events"}
        observed = {path.name for path in self._root.iterdir()}
        if observed != allowed:
            raise WitnessArchiveShadowError("ledger root contains an unknown or missing artifact")
        events_root = self._root / "events"
        if events_root.is_symlink() or not events_root.is_dir():
            raise WitnessArchiveShadowError("ledger events directory is unsafe")
        paths: list[Path] = []
        for path in events_root.iterdir():
            if (
                path.is_symlink()
                or not path.is_file()
                or _EVENT_NAME_RE.fullmatch(path.name) is None
            ):
                raise WitnessArchiveShadowError(f"unknown or unsafe ledger event artifact: {path}")
            paths.append(path)
        return sorted(paths, key=lambda item: item.name)

    def _reload(self) -> None:
        archive = CounterfactualWitnessArchive()
        by_cluster = self._cohort._by_cluster()
        event_inputs: dict[str, tuple[str, ShadowArchiveReceipt]] = {}
        previous = self._identity_digest
        post_state_digest = _digest(archive.to_dict())
        action_counts: Counter[str] = Counter()
        paths = self._validate_inventory()
        for expected_sequence, path in enumerate(paths, start=1):
            match = _EVENT_NAME_RE.fullmatch(path.name)
            assert match is not None
            if int(match.group(1)) != expected_sequence:
                raise WitnessArchiveShadowError("ledger event sequence is not contiguous")
            event = _read_json(path, f"ledger event {expected_sequence}")
            _require_keys(
                event,
                {
                    "schema_version",
                    "ledger_identity_digest",
                    "sequence",
                    "event_id",
                    "previous_event_digest",
                    "input",
                    "input_digest",
                    "decision",
                    "actual_baseline_selection",
                    "post_state",
                    "post_state_digest",
                    "event_digest",
                },
                f"ledger event {expected_sequence}",
            )
            if (
                event["schema_version"] != EVENT_SCHEMA
                or event["ledger_identity_digest"] != self._identity_digest
                or event["sequence"] != expected_sequence
                or event["previous_event_digest"] != previous
            ):
                raise WitnessArchiveShadowError("ledger chain identity or sequence is invalid")
            event_id = _require_text(event["event_id"], "ledger event id")
            if event_id in event_inputs:
                raise WitnessArchiveShadowError("ledger records an event id more than once")
            input_value = event["input"]
            if not isinstance(input_value, dict) or set(input_value) != {
                "evidence",
                "archive_entry",
            }:
                raise WitnessArchiveShadowError("ledger event input is malformed")
            if event["input_digest"] != _digest(input_value):
                raise WitnessArchiveShadowError("ledger event input digest mismatch")
            evidence_value = input_value["evidence"]
            if not isinstance(evidence_value, dict):
                raise WitnessArchiveShadowError("ledger evidence identity is malformed")
            cluster_id = evidence_value.get("cluster_id")
            evidence = by_cluster.get(cluster_id) if isinstance(cluster_id, str) else None
            if evidence is None or evidence_value != evidence.identity_dict():
                raise WitnessArchiveShadowError("ledger evidence is outside its anchored cohort")
            entry = _entry_from_dict(input_value["archive_entry"], evidence)
            expected_metadata = {
                "aggregate_digest": evidence.aggregate_digest,
                "plan_digest": evidence.plan_digest,
                "cluster_id": evidence.cluster_id,
                "cluster_result_digest": evidence.cluster_result_digest,
                "candidate_id": evidence.candidate_id,
                "qualification_digest": evidence.qualification_digest,
                "game_bytes_digest": evidence.environment_digest,
                "quality_policy_id": self._quality_policy_id,
                "lineage_policy_id": self._lineage_policy_id,
            }
            if (
                entry.environment_digest != evidence.environment_digest
                or entry.witness_digest != evidence.witness_digest
                or entry.game_file != f"verified-game@{evidence.environment_digest}"
                or entry.skill != evidence.skill
                or entry.difficulty != evidence.difficulty
                or entry.metadata != expected_metadata
            ):
                raise WitnessArchiveShadowError("ledger archive entry breaks its policy binding")
            decision = archive.consider(entry)
            expected_decision = _decision_dict(decision)
            post_state = archive.to_dict()
            expected_post_digest = _digest(post_state)
            baseline = _baseline_selection(archive, decision)
            if (
                event["decision"] != expected_decision
                or event["actual_baseline_selection"] != baseline
                or event["post_state"] != post_state
                or event["post_state_digest"] != expected_post_digest
            ):
                raise WitnessArchiveShadowError("ledger event differs from deterministic replay")
            _sealed_body(event, "event_digest", f"ledger event {expected_sequence}")
            if event["event_digest"].removeprefix("sha256:") != match.group(2):
                raise WitnessArchiveShadowError("ledger event filename does not match its digest")
            receipt = ShadowArchiveReceipt(
                event_id=event_id,
                sequence=expected_sequence,
                action=decision.action,
                event_digest=event["event_digest"],
                post_state_digest=expected_post_digest,
                resumed=True,
            )
            event_inputs[event_id] = (event["input_digest"], receipt)
            action_counts[decision.action] += 1
            previous = event["event_digest"]
            post_state_digest = expected_post_digest
        self._archive = archive
        self._event_inputs = event_inputs
        self._head_digest = previous
        self._post_state_digest = post_state_digest
        self._action_counts = action_counts

    def _write_event(self, value: Mapping[str, Any]) -> None:
        sequence = int(value["sequence"])
        digest_suffix = str(value["event_digest"]).removeprefix("sha256:")
        path = self._root / "events" / f"{sequence:06d}-{digest_suffix}.json"
        content = _pretty_json(value)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise WitnessArchiveShadowError("immutable ledger event path already exists") from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def consider(
        self,
        *,
        event_id: str,
        evidence: ValidatedWitnessEvidence,
        game_file: Path | str,
        quality_score: float,
        lineage: Sequence[str],
    ) -> ShadowArchiveReceipt:
        """Record one shadow decision, or return its exact prior receipt on resume."""
        normalized_event_id = _require_text(event_id, "event_id")
        expected = self._cohort._by_cluster().get(evidence.cluster_id)
        if expected is None or evidence != expected:
            raise WitnessArchiveShadowError("evidence is not a member of this ledger's cohort")
        entry = evidence.bind(
            game_file,
            quality_score,
            lineage,
            quality_policy_id=self._quality_policy_id,
            lineage_policy_id=self._lineage_policy_id,
        )
        input_value = {
            "evidence": evidence.identity_dict(),
            "archive_entry": entry.to_dict(),
        }
        input_digest = _digest(input_value)
        with self._locked():
            self._reload()
            existing = self._event_inputs.get(normalized_event_id)
            if existing is not None:
                if existing[0] != input_digest:
                    raise WitnessArchiveShadowError(
                        "event id already exists with conflicting policy or evidence input"
                    )
                return existing[1]
            decision = self._archive.consider(entry)
            post_state = self._archive.to_dict()
            post_state_digest = _digest(post_state)
            body = {
                "schema_version": EVENT_SCHEMA,
                "ledger_identity_digest": self._identity_digest,
                "sequence": len(self._event_inputs) + 1,
                "event_id": normalized_event_id,
                "previous_event_digest": self._head_digest,
                "input": input_value,
                "input_digest": input_digest,
                "decision": _decision_dict(decision),
                "actual_baseline_selection": _baseline_selection(self._archive, decision),
                "post_state": post_state,
                "post_state_digest": post_state_digest,
            }
            event = {**body, "event_digest": _digest(body)}
            self._write_event(event)
            self._reload()
            _, receipt = self._event_inputs[normalized_event_id]
            return ShadowArchiveReceipt(
                event_id=receipt.event_id,
                sequence=receipt.sequence,
                action=receipt.action,
                event_digest=receipt.event_digest,
                post_state_digest=receipt.post_state_digest,
                resumed=False,
            )

    def summary(self) -> ShadowArchiveSummary:
        """Return only audit counts and digests; no active archive selection."""
        with self._locked():
            self._reload()
            return ShadowArchiveSummary(
                event_count=len(self._event_inputs),
                cell_count=len(self._archive),
                action_counts=dict(sorted(self._action_counts.items())),
                head_event_digest=self._head_digest,
                post_state_digest=self._post_state_digest,
            )


__all__ = [
    "ShadowArchiveLedger",
    "ShadowArchiveReceipt",
    "ShadowArchiveSummary",
    "ValidatedWitnessCohort",
    "ValidatedWitnessEvidence",
    "WitnessArchiveShadowError",
    "load_validated_cwa_run",
]
