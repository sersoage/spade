"""Offline materialization of sealed learner-assay game pools.

This module imports no provider outcomes and executes no environment source.  It
validates the sealed coverage-forced actor plan and candidate evidence, then
writes paired basename-aligned training pools plus an environment-disjoint v4
heldout pool.  The output root is write-once and every byte is manifest-bound.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from spade.slime.static_pool import DEFAULT_SCHEDULE_ID, SCHEDULE_SCHEMA


MANIFEST_SCHEMA = "spade-learner-branch-pools/v1"
SEALED_INTENT_DIGEST = "sha256:df1a06c7fb854d5267ec4d1e41cd44c1ffd229018bf0f5a0dcde512fa47c4c09"
SEALED_ACTOR_PLAN_DIGEST = "sha256:fc7989bfffb0851363137fe450e94cf11c8a4658b104f2f2729c08e98d843c2a"
SEALED_LEARNER_POOL_MANIFEST_DIGEST = (
    "sha256:c10b3be0acd42df8b4bf3b695c02eab0cdc83678e699f8e860890e70b2783b1d"
)
V4_PLAN_DIGEST = "sha256:8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc"
V4_COHORT_DIGEST = "sha256:161353ebd4454516e3379414444323dd13aeab95640eb130ec7414f23876b84b"

TRAINING_STRATA = (
    "c001-pattern-recognition-medium",
    "c003-mathematical-reasoning-medium",
    "c004-mathematical-reasoning-hard",
    "c005-logical-deduction-medium",
    "c006-logical-deduction-hard",
    "c007-strategic-planning-medium",
)
HELDOUT_V4_STRATA = (
    "c002-pattern-recognition-hard",
    "c008-strategic-planning-hard",
    "c009-spatial-reasoning-medium",
    "c010-spatial-reasoning-hard",
    "c011-causal-inference-medium",
    "c012-causal-inference-hard",
    "c013-memory-recall-medium",
    "c014-memory-recall-hard",
    "c015-optimization-medium",
    "c016-optimization-hard",
    "c017-language-understanding-medium",
    "c018-language-understanding-hard",
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


class LearnerPoolError(ValueError):
    """Sealed source evidence or a materialized bundle is inconsistent."""


@dataclass(frozen=True)
class PoolBundle:
    root: Path
    manifest_path: Path
    schedule_id: str
    coverage_forced_dir: Path
    redundant_historical_dir: Path
    heldout_v4_dir: Path | None
    manifest: Mapping[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LearnerPoolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, where: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LearnerPoolError(f"{where} must be a regular non-symlinked file: {path}")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LearnerPoolError(f"cannot read {where}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LearnerPoolError(f"{where} must contain a JSON object")
    return value


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
        raise LearnerPoolError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sealed(value: Mapping[str, Any], field: str, where: str) -> str:
    observed = value.get(field)
    if not isinstance(observed, str) or _DIGEST_RE.fullmatch(observed) is None:
        raise LearnerPoolError(f"{where}.{field} is not a lowercase sha256 digest")
    body = {key: item for key, item in value.items() if key != field}
    if observed != _digest(body):
        raise LearnerPoolError(f"{where}.{field} does not bind its artifact")
    return observed


def _safe_id(value: object, where: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise LearnerPoolError(f"{where} is not a safe identifier")
    return value


def _validate_code(value: Mapping[str, Any], where: str) -> bytes:
    code = value.get("code")
    if not isinstance(code, str) or not code:
        raise LearnerPoolError(f"{where}.code is empty")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise LearnerPoolError(f"{where}.code does not parse: {exc}") from exc
    encoded = code.encode("utf-8")
    if value.get("code_digest") != _digest(code):
        raise LearnerPoolError(f"{where}.code_digest mismatch")
    if value.get("environment_digest") != _bytes_digest(encoded):
        raise LearnerPoolError(f"{where}.environment_digest mismatch")
    return encoded


def _candidate_path(run_dir: Path, reference: Mapping[str, Any]) -> Path:
    candidate_id = _safe_id(reference.get("candidate_id"), "candidate reference id")
    expected = f"candidate-evidence/{candidate_id}.json"
    if reference.get("path") != expected:
        raise LearnerPoolError(f"candidate reference path is not canonical: {candidate_id}")
    path = run_dir / expected
    try:
        path.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise LearnerPoolError(f"candidate reference escapes actor-plan directory: {path}") from exc
    return path


def _validate_candidate(
    value: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bytes:
    where = f"candidate {reference.get('candidate_id')}"
    if value.get("schema_version") != "spade-coverage-forced-candidate-evidence/v1":
        raise LearnerPoolError(f"{where} schema differs")
    if value.get("intent_digest") != SEALED_INTENT_DIGEST:
        raise LearnerPoolError(f"{where} intent binding differs")
    _sealed(value, "evidence_digest", where)
    for key in ("candidate_id", "stratum_id", "source_arm", "evidence_digest"):
        if value.get(key) != reference.get(key):
            raise LearnerPoolError(f"{where} {key} differs from actor-plan reference")
    if reference.get("source_arm") not in {"v3", "v4", "challenger"}:
        raise LearnerPoolError(f"{where} source arm differs")

    encoded = _validate_code(value, where)
    qualification = value.get("qualification")
    if not isinstance(qualification, dict) or qualification.get("passed") is not True:
        raise LearnerPoolError(f"{where} lacks a positive qualification")
    if qualification.get("environment_digest") != value.get("environment_digest"):
        raise LearnerPoolError(f"{where} qualification environment differs")
    _sealed(qualification, "qualification_digest", f"{where}.qualification")
    if reference.get("qualification_digest") != qualification.get("qualification_digest"):
        raise LearnerPoolError(f"{where} qualification reference differs")

    viability = value.get("one_turn_viability")
    if not isinstance(viability, dict):
        raise LearnerPoolError(f"{where} lacks one-turn viability")
    _sealed(viability, "viability_digest", f"{where}.one_turn_viability")
    if (
        reference.get("viability_digest") != viability.get("viability_digest")
        or viability.get("horizon") != 1
        or any(
            viability.get("seeds", {}).get(seed, {}).get("reward_positive_success") is not True
            for seed in ("0", "42")
        )
    ):
        raise LearnerPoolError(f"{where} one-turn viability differs")

    cwa = value.get("cwa")
    if not isinstance(cwa, dict):
        raise LearnerPoolError(f"{where} lacks CWA evidence")
    _sealed(cwa, "evidence_digest", f"{where}.cwa")
    if (
        reference.get("cwa_evidence_digest") != cwa.get("evidence_digest")
        or cwa.get("candidate_id") != value.get("candidate_id")
        or cwa.get("environment_digest") != value.get("environment_digest")
        or cwa.get("eligible") is not True
    ):
        raise LearnerPoolError(f"{where} CWA binding differs")

    hints = value.get("hints")
    if not isinstance(hints, dict) or set(hints) != {"0", "42"}:
        raise LearnerPoolError(f"{where} hint inventory differs")
    expected_hints = {seed: _digest(hints[seed]) for seed in ("0", "42")}
    if reference.get("hint_digests") != expected_hints:
        raise LearnerPoolError(f"{where} hint reference differs")
    return encoded


def _validate_actor_plan(actor_plan_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    actor_plan = _read_json(actor_plan_path, "actor plan")
    digest = _sealed(actor_plan, "actor_plan_digest", "actor plan")
    if digest != SEALED_ACTOR_PLAN_DIGEST:
        raise LearnerPoolError(f"actor plan digest is not the authorized fc798 seal: {digest}")
    if actor_plan.get("intent_digest") != SEALED_INTENT_DIGEST:
        raise LearnerPoolError("actor plan is not bound to the authorized df1a intent")
    if actor_plan.get("schema_version") != "spade-coverage-forced-actor-plan/v1":
        raise LearnerPoolError("actor plan schema differs")

    references = actor_plan.get("candidate_evidence")
    if not isinstance(references, list) or len(references) != 18:
        raise LearnerPoolError("actor plan must bind exactly 18 candidate evidence files")
    reference_by_id: dict[str, Mapping[str, Any]] = {}
    evidence_by_id: dict[str, bytes] = {}
    for reference in references:
        if not isinstance(reference, dict):
            raise LearnerPoolError("candidate evidence reference is not an object")
        candidate_id = _safe_id(reference.get("candidate_id"), "candidate reference id")
        if candidate_id in reference_by_id:
            raise LearnerPoolError(f"duplicate candidate reference: {candidate_id}")
        evidence_path = _candidate_path(actor_plan_path.parent, reference)
        evidence = _read_json(evidence_path, f"candidate evidence {candidate_id}")
        evidence_by_id[candidate_id] = _validate_candidate(evidence, reference)
        reference_by_id[candidate_id] = reference

    portfolios = actor_plan.get("portfolios")
    if not isinstance(portfolios, list) or len(portfolios) != len(TRAINING_STRATA):
        raise LearnerPoolError("actor plan must contain exactly six training portfolios")
    if tuple(item.get("stratum_id") for item in portfolios) != TRAINING_STRATA:
        raise LearnerPoolError("actor plan training stratum order differs")
    for portfolio in portfolios:
        stratum = str(portfolio["stratum_id"])
        challenger = portfolio.get("challenger_id")
        displaced = portfolio.get("displaced_historical_id")
        retained = portfolio.get("retained_historical_id")
        if (
            portfolio.get("differs") is not True
            or set(portfolio.get("candidate_ids", [])) != {challenger, displaced, retained}
            or set(portfolio.get("coverage_forced", [])) != {challenger, retained}
            or set(portfolio.get("redundant_historical", [])) != {displaced, retained}
        ):
            raise LearnerPoolError(f"portfolio topology differs: {stratum}")
        for candidate_id in (challenger, displaced, retained):
            if candidate_id not in reference_by_id:
                raise LearnerPoolError(f"portfolio references unknown candidate: {candidate_id}")
            if reference_by_id[candidate_id].get("stratum_id") != stratum:
                raise LearnerPoolError(f"portfolio candidate crosses strata: {candidate_id}")
    return actor_plan, evidence_by_id


def _validate_source_intent(path: Path) -> dict[str, Any]:
    intent = _read_json(path, "source intent")
    digest = _sealed(intent, "intent_digest", "source intent")
    if digest != SEALED_INTENT_DIGEST:
        raise LearnerPoolError(f"source intent is not the authorized df1a seal: {digest}")
    return intent


def _validate_v4_sources(
    intent: Mapping[str, Any],
    override_run_dir: Path | None,
) -> tuple[Path, dict[str, bytes], dict[str, dict[str, Any]], Mapping[str, Any]]:
    sources = intent.get("sources")
    matches = [
        item for item in sources or [] if isinstance(item, dict) and item.get("label") == "v4"
    ]
    if len(matches) != 1:
        raise LearnerPoolError("source intent must bind exactly one v4 source")
    source = matches[0]
    if (
        source.get("plan_digest") != V4_PLAN_DIGEST
        or source.get("cohort_digest") != V4_COHORT_DIGEST
        or _DIGEST_RE.fullmatch(str(source.get("manifest_digest"))) is None
    ):
        raise LearnerPoolError("v4 source manifest identity differs")
    run_dir = override_run_dir or Path(str(source.get("run_dir", "")))
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise LearnerPoolError(f"v4 run directory is unavailable: {run_dir}")

    plan = _read_json(run_dir / "plan.json", "v4 plan")
    if _sealed(plan, "plan_digest", "v4 plan") != V4_PLAN_DIGEST:
        raise LearnerPoolError("v4 plan digest differs")
    cohort = _read_json(run_dir / "cohort-lock.json", "v4 cohort lock")
    if _sealed(cohort, "cohort_digest", "v4 cohort lock") != V4_COHORT_DIGEST:
        raise LearnerPoolError("v4 cohort digest differs")
    if cohort.get("plan_digest") != V4_PLAN_DIGEST:
        raise LearnerPoolError("v4 cohort is bound to a different plan")
    cohort_entries = cohort.get("selections")
    if not isinstance(cohort_entries, list) or len(cohort_entries) != 18:
        raise LearnerPoolError("v4 cohort lock must bind exactly 18 selections")
    cohort_by_cluster = {
        _safe_id(item.get("cluster_id"), "v4 cohort cluster id"): item
        for item in cohort_entries
        if isinstance(item, dict)
    }
    if len(cohort_by_cluster) != 18:
        raise LearnerPoolError("v4 cohort contains duplicate or malformed clusters")

    code_by_cluster: dict[str, bytes] = {}
    metadata_by_cluster: dict[str, dict[str, Any]] = {}
    for cluster_id in HELDOUT_V4_STRATA:
        locked = cohort_by_cluster.get(cluster_id)
        if locked is None:
            raise LearnerPoolError(f"v4 cohort omits heldout cluster: {cluster_id}")
        selection = _read_json(
            run_dir / "selections" / f"{cluster_id}.json",
            f"v4 selection {cluster_id}",
        )
        selection_digest = _sealed(selection, "selection_digest", f"v4 selection {cluster_id}")
        if (
            selection.get("schema_version") != "spade-agy-selected-candidate/v1"
            or selection.get("plan_digest") != V4_PLAN_DIGEST
            or selection.get("cluster_id") != cluster_id
            or selection_digest != locked.get("selection_digest")
            or selection.get("candidate_id") != locked.get("candidate_id")
            or selection.get("code_digest") != locked.get("code_digest")
            or selection.get("qualification_digest") != locked.get("qualification_digest")
        ):
            raise LearnerPoolError(f"v4 selection differs from cohort lock: {cluster_id}")
        encoded = _validate_code(selection, f"v4 selection {cluster_id}")
        code_by_cluster[cluster_id] = encoded
        metadata_by_cluster[cluster_id] = {
            "candidate_id": selection["candidate_id"],
            "source_arm": "v4",
            "code_digest": selection["code_digest"],
            "environment_digest": selection["environment_digest"],
            "qualification_digest": selection["qualification_digest"],
            "selection_digest": selection_digest,
        }
    return run_dir, code_by_cluster, metadata_by_cluster, source


def _entry(
    *,
    basename: str,
    stratum_id: str,
    slot_role: str,
    candidate_id: str,
    source_arm: str,
    code: bytes,
    evidence_digest: str | None = None,
    qualification_digest: str | None = None,
    selection_digest: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "basename": basename,
        "stratum_id": stratum_id,
        "slot_role": slot_role,
        "candidate_id": candidate_id,
        "source_arm": source_arm,
        "size_bytes": len(code),
        "code_digest": _digest(code.decode("utf-8")),
        "environment_digest": _bytes_digest(code),
    }
    if evidence_digest is not None:
        value["evidence_digest"] = evidence_digest
    if qualification_digest is not None:
        value["qualification_digest"] = qualification_digest
    if selection_digest is not None:
        value["selection_digest"] = selection_digest
    return value


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _bundle_from_manifest(root: Path, manifest: Mapping[str, Any]) -> PoolBundle:
    pools = manifest["pools"]
    heldout = pools.get("heldout_v4")
    return PoolBundle(
        root=root,
        manifest_path=root / "learner-pools-manifest.json",
        schedule_id=str(manifest["schedule"]["schedule_id"]),
        coverage_forced_dir=root / str(pools["coverage_forced"]["relative_dir"]),
        redundant_historical_dir=root / str(pools["redundant_historical"]["relative_dir"]),
        heldout_v4_dir=(root / str(heldout["relative_dir"])) if heldout else None,
        manifest=manifest,
    )


def materialize_learner_pools(
    actor_plan_path: Path | str,
    output_dir: Path | str,
    *,
    source_intent_path: Path | str | None = None,
    v4_run_dir: Path | str | None = None,
    include_heldout: bool = True,
) -> PoolBundle:
    """Validate the sealed sources and create a new immutable pool bundle."""

    actor_path = Path(actor_plan_path)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise LearnerPoolError(f"output directory already exists; refusing overwrite: {output}")
    actor_plan, evidence_by_id = _validate_actor_plan(actor_path)
    intent_path = (
        Path(source_intent_path)
        if source_intent_path is not None
        else actor_path.parent.parent / "intent.json"
    )
    intent = _validate_source_intent(intent_path)

    heldout_payload: (
        tuple[Path, dict[str, bytes], dict[str, dict[str, Any]], Mapping[str, Any]] | None
    ) = None
    if include_heldout:
        heldout_payload = _validate_v4_sources(
            intent,
            Path(v4_run_dir) if v4_run_dir is not None else None,
        )

    schedule_id = (
        "spade-learner-branch-"
        f"{SEALED_INTENT_DIGEST.removeprefix('sha256:')[:12]}-"
        f"{SEALED_ACTOR_PLAN_DIGEST.removeprefix('sha256:')[:12]}-v1"
    )
    if schedule_id == DEFAULT_SCHEDULE_ID:
        raise AssertionError("learner assay must not use the compatibility schedule id")

    pool_codes: dict[str, dict[str, bytes]] = {
        "coverage_forced": {},
        "redundant_historical": {},
    }
    pool_entries: dict[str, list[dict[str, Any]]] = {
        "coverage_forced": [],
        "redundant_historical": [],
    }
    references = {item["candidate_id"]: item for item in actor_plan["candidate_evidence"]}
    for stratum_index, portfolio in enumerate(actor_plan["portfolios"]):
        stratum = str(portfolio["stratum_id"])
        slots = (
            (
                f"game_{stratum_index * 2:03d}_swap.py",
                "swap",
                str(portfolio["challenger_id"]),
                str(portfolio["displaced_historical_id"]),
            ),
            (
                f"game_{stratum_index * 2 + 1:03d}_retained.py",
                "retained",
                str(portfolio["retained_historical_id"]),
                str(portfolio["retained_historical_id"]),
            ),
        )
        for basename, slot_role, treatment_id, control_id in slots:
            for pool_name, candidate_id in (
                ("coverage_forced", treatment_id),
                ("redundant_historical", control_id),
            ):
                reference = references[candidate_id]
                code = evidence_by_id[candidate_id]
                pool_codes[pool_name][basename] = code
                pool_entries[pool_name].append(
                    _entry(
                        basename=basename,
                        stratum_id=stratum,
                        slot_role=slot_role,
                        candidate_id=candidate_id,
                        source_arm=str(reference["source_arm"]),
                        code=code,
                        evidence_digest=str(reference["evidence_digest"]),
                        qualification_digest=str(reference["qualification_digest"]),
                    )
                )

    pools: dict[str, Any] = {
        name: {
            "relative_dir": name,
            "purpose": (
                "coverage-witness-curriculum"
                if name == "coverage_forced"
                else "redundant-history-control"
            ),
            "game_count": len(entries),
            "entries": entries,
        }
        for name, entries in pool_entries.items()
    }

    source_artifacts: dict[str, Any] = {
        "intent_digest": SEALED_INTENT_DIGEST,
        "actor_plan_digest": SEALED_ACTOR_PLAN_DIGEST,
    }
    if heldout_payload is not None:
        _, heldout_codes, heldout_metadata, v4_source = heldout_payload
        heldout_entries: list[dict[str, Any]] = []
        pool_codes["heldout_v4"] = {}
        for index, stratum in enumerate(HELDOUT_V4_STRATA):
            basename = f"game_{index:03d}_heldout.py"
            code = heldout_codes[stratum]
            meta = heldout_metadata[stratum]
            pool_codes["heldout_v4"][basename] = code
            heldout_entries.append(
                _entry(
                    basename=basename,
                    stratum_id=stratum,
                    slot_role="heldout",
                    candidate_id=str(meta["candidate_id"]),
                    source_arm="v4",
                    code=code,
                    qualification_digest=str(meta["qualification_digest"]),
                    selection_digest=str(meta["selection_digest"]),
                )
            )
        pools["heldout_v4"] = {
            "relative_dir": "heldout_v4",
            "purpose": "environment-disjoint-heldout-endpoint",
            "game_count": len(heldout_entries),
            "entries": heldout_entries,
        }
        training_environment_digests = {
            entry["environment_digest"]
            for name in ("coverage_forced", "redundant_historical")
            for entry in pools[name]["entries"]
        }
        heldout_environment_digests = {entry["environment_digest"] for entry in heldout_entries}
        if len(heldout_environment_digests) != len(HELDOUT_V4_STRATA):
            raise LearnerPoolError("heldout v4 selections do not contain 12 unique environments")
        if training_environment_digests & heldout_environment_digests:
            raise LearnerPoolError("heldout v4 environments overlap a training environment")
        source_artifacts["heldout_v4"] = {
            "plan_digest": V4_PLAN_DIGEST,
            "cohort_digest": V4_COHORT_DIGEST,
            "source_manifest_digest": v4_source["manifest_digest"],
            "source_manifest_leaf_count": v4_source["manifest_leaf_count"],
            "selection_digests": {
                stratum: heldout_metadata[stratum]["selection_digest"]
                for stratum in HELDOUT_V4_STRATA
            },
        }

    body = {
        "schema_version": MANIFEST_SCHEMA,
        "immutability": "write-once-root;read-only-files;digest-verified-exact-inventory",
        "execution_contract": {
            "allowed_causal_assay_backends": [
                "slime-megatron-sglang",
                "tinker-qwen3-8b-lora",
            ],
            "backend_claims": "separate;no-cross-backend-equivalence-claim",
            "learner_rollout_model": "current-branch-trained-checkpoint-via-selected-backend",
            "heldout_evaluation_model": "final-branch-trained-checkpoint-via-selected-backend",
            "agy_google_role": "sealed-environment-hint-design-provenance-only",
            "external_actor_substitution": "forbidden",
        },
        "source_artifacts": source_artifacts,
        "training_strata": list(TRAINING_STRATA),
        "heldout_strata": list(HELDOUT_V4_STRATA) if include_heldout else [],
        "schedule": {
            "schema_version": SCHEDULE_SCHEMA,
            "schedule_id": schedule_id,
            "slot_key": "basename",
            "paired_training_pool_size": 12,
        },
        "pools": pools,
    }
    manifest = {**body, "manifest_digest": _digest(body)}
    if include_heldout and manifest["manifest_digest"] != SEALED_LEARNER_POOL_MANIFEST_DIGEST:
        raise LearnerPoolError(
            "materialized learner pool no longer matches the authorized df1a/fc798/v4 identity"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755)
    try:
        for pool_name, files in pool_codes.items():
            pool_dir = output / pool_name
            pool_dir.mkdir(mode=0o755)
            for basename, content in files.items():
                destination = pool_dir / basename
                _write_new(destination, content)
                destination.chmod(0o444)
        manifest_path = output / "learner-pools-manifest.json"
        _write_new(manifest_path, (_canonical_json(manifest) + "\n").encode("utf-8"))
        manifest_path.chmod(0o444)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return load_learner_pool_manifest(output, verify_files=True)


def load_learner_pool_manifest(
    manifest_or_root: Path | str,
    *,
    verify_files: bool = True,
) -> PoolBundle:
    """Load a learner-pool manifest and optionally verify its exact inventory."""

    supplied = Path(manifest_or_root)
    manifest_path = (
        supplied
        if supplied.name == "learner-pools-manifest.json"
        else supplied / "learner-pools-manifest.json"
    )
    root = manifest_path.parent
    manifest = _read_json(manifest_path, "learner pool manifest")
    digest = _sealed(manifest, "manifest_digest", "learner pool manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise LearnerPoolError("learner pool manifest schema differs")
    if manifest.get("execution_contract") != {
        "allowed_causal_assay_backends": [
            "slime-megatron-sglang",
            "tinker-qwen3-8b-lora",
        ],
        "backend_claims": "separate;no-cross-backend-equivalence-claim",
        "learner_rollout_model": "current-branch-trained-checkpoint-via-selected-backend",
        "heldout_evaluation_model": "final-branch-trained-checkpoint-via-selected-backend",
        "agy_google_role": "sealed-environment-hint-design-provenance-only",
        "external_actor_substitution": "forbidden",
    }:
        raise LearnerPoolError(
            "learner execution contract drifted or permits an external actor substitution"
        )
    if manifest.get("source_artifacts", {}).get("intent_digest") != SEALED_INTENT_DIGEST:
        raise LearnerPoolError("learner pool manifest intent differs")
    if manifest.get("source_artifacts", {}).get("actor_plan_digest") != SEALED_ACTOR_PLAN_DIGEST:
        raise LearnerPoolError("learner pool manifest actor plan differs")
    if tuple(manifest.get("training_strata", ())) != TRAINING_STRATA:
        raise LearnerPoolError("learner pool training strata differ")
    schedule = manifest.get("schedule")
    if (
        not isinstance(schedule, dict)
        or schedule.get("schema_version") != SCHEDULE_SCHEMA
        or schedule.get("slot_key") != "basename"
        or schedule.get("schedule_id") in (None, "", DEFAULT_SCHEDULE_ID)
        or schedule.get("paired_training_pool_size") != 12
    ):
        raise LearnerPoolError("learner pool schedule differs")
    pools = manifest.get("pools")
    if not isinstance(pools, dict) or not {"coverage_forced", "redundant_historical"}.issubset(
        pools
    ):
        raise LearnerPoolError("learner pool manifest omits a training arm")
    has_heldout = "heldout_v4" in pools
    if has_heldout and digest != SEALED_LEARNER_POOL_MANIFEST_DIGEST:
        raise LearnerPoolError("learner pool manifest is not the authorized df1a/fc798/v4 artifact")
    expected_heldout_strata = HELDOUT_V4_STRATA if has_heldout else ()
    if tuple(manifest.get("heldout_strata", ())) != expected_heldout_strata:
        raise LearnerPoolError("learner pool heldout strata differ")
    if set(TRAINING_STRATA) & set(expected_heldout_strata):
        raise LearnerPoolError("learner pool training and heldout strata overlap")

    treatment_names: list[str] = []
    control_names: list[str] = []
    training_environment_digests: set[str] = set()
    heldout_environment_digests: set[str] = set()
    expected_root_entries = {"learner-pools-manifest.json"}
    for pool_name, pool in pools.items():
        if pool_name not in {"coverage_forced", "redundant_historical", "heldout_v4"}:
            raise LearnerPoolError(f"unknown learner pool: {pool_name}")
        if not isinstance(pool, dict) or pool.get("relative_dir") != pool_name:
            raise LearnerPoolError(f"learner pool path differs: {pool_name}")
        entries = pool.get("entries")
        if not isinstance(entries, list) or pool.get("game_count") != len(entries):
            raise LearnerPoolError(f"learner pool count differs: {pool_name}")
        expected_count = 12
        if len(entries) != expected_count:
            raise LearnerPoolError(f"learner pool must contain exactly 12 games: {pool_name}")
        names = [item.get("basename") for item in entries if isinstance(item, dict)]
        if len(names) != len(entries) or len(set(names)) != len(entries):
            raise LearnerPoolError(f"learner pool basenames differ: {pool_name}")
        if any(
            not isinstance(name, str) or not name.startswith("game_") or not name.endswith(".py")
            for name in names
        ):
            raise LearnerPoolError(f"learner pool contains an unsafe basename: {pool_name}")
        if pool_name == "coverage_forced":
            treatment_names = names
        elif pool_name == "redundant_historical":
            control_names = names
        entry_environment_digests = {
            item.get("environment_digest") for item in entries if isinstance(item, dict)
        }
        if len(entry_environment_digests) != len(entries):
            raise LearnerPoolError(f"learner pool environment digests differ: {pool_name}")
        if pool_name == "heldout_v4":
            heldout_environment_digests = entry_environment_digests
        else:
            training_environment_digests.update(entry_environment_digests)
        expected_root_entries.add(pool_name)

        if verify_files:
            pool_dir = root / pool_name
            if pool_dir.is_symlink() or not pool_dir.is_dir():
                raise LearnerPoolError(f"learner pool directory is unavailable: {pool_name}")
            observed = {path.name for path in pool_dir.iterdir()}
            if observed != set(names):
                raise LearnerPoolError(f"learner pool inventory differs: {pool_name}")
            for entry in entries:
                path = pool_dir / str(entry["basename"])
                if path.is_symlink() or not path.is_file():
                    raise LearnerPoolError(f"learner pool entry is not a regular file: {path}")
                raw = path.read_bytes()
                if (
                    len(raw) != entry.get("size_bytes")
                    or _bytes_digest(raw) != entry.get("environment_digest")
                    or _digest(raw.decode("utf-8")) != entry.get("code_digest")
                ):
                    raise LearnerPoolError(f"learner pool entry digest differs: {path}")
                if stat.S_IMODE(path.stat().st_mode) & 0o222:
                    raise LearnerPoolError(f"learner pool entry is writable: {path}")

    if treatment_names != control_names:
        raise LearnerPoolError("paired training pool basename slots do not align")
    if training_environment_digests & heldout_environment_digests:
        raise LearnerPoolError("heldout environments overlap a training environment")
    if verify_files:
        observed_root = {path.name for path in root.iterdir()}
        if observed_root != expected_root_entries:
            raise LearnerPoolError("learner pool root inventory differs")
        if stat.S_IMODE(manifest_path.stat().st_mode) & 0o222:
            raise LearnerPoolError("learner pool manifest is writable")

    _ = digest
    return _bundle_from_manifest(root, manifest)
