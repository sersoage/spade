from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spade.core.witness_qd_proxy import (
    COVERAGE_CHALLENGER_DESCRIPTOR,
    STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR,
    ProxyCandidate,
    PairedOutcome,
    counterbalanced_pairs,
    lock_all_portfolios,
    portfolio_quality_diagnostics,
)
from tools import run_spade_witness_qd_proxy as pilot


def _sha(index: int) -> str:
    return f"sha256:{index:064x}"


def _engine(tmp_path: Path) -> pilot._Engine:
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("provider boundary was crossed")

    intent = {
        "intent_digest": _sha(1),
        "shared_ledger_root": str(tmp_path / "ledger"),
        "runtime_identity": {"agy_executable_digest": _sha(2)},
        "configuration": {
            "llm_timeout_seconds": 10.0,
            "qualification_timeout_seconds": 1.0,
            "maximum_stratum_absolute_quality_gap": 0.125,
            "maximum_mean_absolute_quality_gap": 0.0625,
        },
    }
    dependencies = SimpleNamespace(llm_call=forbidden, client_or_bin=tmp_path / "agy")
    return pilot._Engine(intent, b"{}", tmp_path / "run", dependencies)


def _request(
    engine: pilot._Engine,
    *,
    purpose: dict | None = None,
    prompt: str = "sealed prompt",
    system: str = "sealed system",
) -> dict:
    selected_purpose = purpose or {
        "phase": "challenger-design",
        "stratum_id": "c001",
        "candidate_id": "c001--challenger",
        "attempt": 1,
    }
    body = {
        "schema_version": pilot.CALL_REQUEST_SCHEMA,
        "intent_digest": engine.intent["intent_digest"],
        "actor_plan_digest": None,
        "call_id": pilot.base._call_id(selected_purpose),
        "local_ordinal": 1,
        "global_ordinal": pilot.PRIOR_CHARGED_CALLS + 1,
        "purpose": selected_purpose,
        "provider": "agy",
        "model": pilot.DESIGN_MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "runtime_identity_digest": pilot._digest(engine.intent["runtime_identity"]),
        "agy_executable_digest": engine.intent["runtime_identity"]["agy_executable_digest"],
        "prompt": prompt,
        "prompt_digest": pilot._digest(prompt),
        "system": system,
        "system_digest": pilot._digest(system),
        "timeout_seconds": 10.0,
        "workdir_policy": "fresh-temporary-directory-per-call",
        "reservation_status": "reserved-before-spawn",
        "reserved_at_utc": pilot.base._utc_now(),
    }
    return {**body, "request_digest": pilot._digest(body)}


def test_preactor_request_remains_valid_after_actor_plan_seal(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _request(engine)
    assert engine._validate_call_request(request) == request

    engine.run_dir.mkdir(parents=True)
    pilot.base._write_json(engine.run_dir / "actor-plan.json", {"deliberately": "opaque"})
    assert engine._validate_call_request(request) == request


def test_ambiguous_request_is_never_replayed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _request(engine)
    request_path = engine.run_dir / "calls" / request["call_id"] / "request.json"
    pilot.base._write_json(request_path, request)

    with pytest.raises(pilot.AmbiguousProviderCall, match="will not replay"):
        asyncio.run(
            engine.call(
                purpose=request["purpose"],
                prompt=request["prompt"],
                system=request["system"],
            )
        )


def test_total_208_cap_fails_before_provider_spawn(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    for ordinal in range(1, pilot.NEW_CALL_CAP + 1):
        path = engine.run_dir / "calls" / f"prior-{ordinal:03d}" / "request.json"
        pilot.base._write_json(path, {"charged": ordinal})

    with pytest.raises(pilot.CallCapExceeded, match="208"):
        asyncio.run(
            engine.call(
                purpose={
                    "phase": "challenger-design",
                    "stratum_id": "c001",
                    "candidate_id": "c001--challenger",
                    "attempt": 1,
                },
                prompt="no spawn",
            )
        )


def test_sealed_budget_arithmetic_has_37_call_headroom() -> None:
    assert pilot.PAIR_COUNT == 36
    assert pilot.PRE_ACTOR_CALL_CEILING == 36
    assert pilot.ACTOR_CALL_CEILING == 172
    assert pilot.NEW_CALL_CAP == 208
    assert pilot.PRIOR_CHARGED_CALLS + pilot.NEW_CALL_CAP == 413
    assert pilot.AUTHORIZED_GLOBAL_CALL_CAP - 413 == 37


def test_failure_category_is_recomputed_not_trusted(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    request = _request(engine)
    timestamp = request["reserved_at_utc"]
    body = {
        "schema_version": pilot.CALL_RESULT_SCHEMA,
        "intent_digest": engine.intent["intent_digest"],
        "call_id": request["call_id"],
        "local_ordinal": request["local_ordinal"],
        "global_ordinal": request["global_ordinal"],
        "request_digest": request["request_digest"],
        "status": "error",
        "failure_category": "fatal_transport",
        "exception_type": "LiveEvalError",
        "error": "agy returned an empty response",
        "exit_status": None,
        "response": None,
        "response_digest": None,
        "started_at_utc": timestamp,
        "finished_at_utc": timestamp,
        "duration_seconds": 0.0,
    }
    tampered = {**body, "result_digest": pilot._digest(body)}
    with pytest.raises(pilot.ProxyExperimentError, match="not derivable"):
        engine._validate_call_result(tampered, request)


def test_artifact_tree_rejects_symlink(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.run_dir.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (engine.run_dir / "escape.json").symlink_to(target)
    with pytest.raises(pilot.ProxyExperimentError, match="symlink"):
        engine._validate_tree()


def _actor_plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    strata = (1, 3, 4, 5, 6, 7)
    intent = {
        "intent_digest": _sha(10),
        "strata": [{"stratum_id": f"c{index:03d}"} for index in strata],
        "configuration": {
            "maximum_stratum_absolute_quality_gap": 0.125,
            "maximum_mean_absolute_quality_gap": 0.0625,
        },
        "analysis_gates": {"sealed": True},
    }
    candidates = []
    references = []
    grouped = {}
    for stratum in strata:
        stratum_id = f"c{stratum:03d}"
        grouped[stratum_id] = []
        for offset, (arm, quality) in enumerate(
            (("v3", 0.75), ("v4", 1.0), ("challenger", 1.0)),
            start=1,
        ):
            candidate_id = f"{stratum_id}--{arm}"
            grouped[stratum_id].append(candidate_id)
            serial = stratum * 10 + offset
            evidence = {
                "intent_digest": intent["intent_digest"],
                "stratum_id": stratum_id,
                "candidate_id": candidate_id,
                "source_arm": arm,
                "environment_digest": _sha(serial),
                "evidence_digest": _sha(1_000 + serial),
                "qualification": {"qualification_digest": _sha(2_000 + serial)},
                "cwa": {
                    "evidence_digest": _sha(3_000 + serial),
                    "quality_score": quality,
                    "descriptor": dict(
                        COVERAGE_CHALLENGER_DESCRIPTOR
                        if arm == "challenger"
                        else STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR
                    ),
                },
                "one_turn_viability": {"viability_digest": _sha(4_000 + serial)},
                "hints": {"0": {"hint": "h0"}, "42": {"hint": "h42"}},
            }
            relative = f"candidate-evidence/{candidate_id}.json"
            pilot.base._write_json(tmp_path / relative, evidence)
            references.append(
                {
                    "stratum_id": stratum_id,
                    "candidate_id": candidate_id,
                    "source_arm": arm,
                    "path": relative,
                    "evidence_digest": evidence["evidence_digest"],
                    "qualification_digest": evidence["qualification"]["qualification_digest"],
                    "cwa_evidence_digest": evidence["cwa"]["evidence_digest"],
                    "viability_digest": evidence["one_turn_viability"]["viability_digest"],
                    "hint_digests": {
                        seed: pilot._digest(evidence["hints"][seed]) for seed in ("0", "42")
                    },
                }
            )
            candidates.append(
                ProxyCandidate.create(
                    candidate_id=candidate_id,
                    stratum_id=stratum_id,
                    source_arm=arm,
                    quality_score=quality,
                    descriptor=evidence["cwa"]["descriptor"],
                    environment_digest=evidence["environment_digest"],
                    evidence_digest=evidence["cwa"]["evidence_digest"],
                )
            )
    monkeypatch.setattr(
        pilot._Engine,
        "_validate_candidate_evidence",
        staticmethod(lambda value: value),
    )
    schedule = list(
        counterbalanced_pairs({key: tuple(value) for key, value in grouped.items()}, pilot.SEEDS)
    )
    locks = lock_all_portfolios(candidates)
    portfolios = [item.to_dict() for item in locks]
    body = {
        "schema_version": pilot.ACTOR_PLAN_SCHEMA,
        "protocol_id": pilot.PROTOCOL_ID,
        "intent_digest": intent["intent_digest"],
        "chronology": "sealed-after-candidates-cwa-hints-before-actor",
        "provider": "agy",
        "model": pilot.ACTOR_MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "candidate_evidence": references,
        "portfolios": portfolios,
        "portfolio_quality_diagnostics": portfolio_quality_diagnostics(locks),
        "pair_schedule": schedule,
        "pair_schedule_digest": pilot._digest(schedule),
        "attempt_policy": {
            "whole_pair_retries": True,
            "waves_1_and_2": "all-unresolved",
            "wave_3": "only-if-unresolved-after-wave-2-at-most-14",
            "retryable_failure_categories": ["empty_response", "provider_timeout"],
            "nonempty_parser_misses_are_zero_reward_and_not_retried": True,
            "environment_runtime_integrity_failures_are_fatal": True,
        },
        "actor_call_ceiling": pilot.ACTOR_CALL_CEILING,
        "success_rule": "reward-positive-even-if-simultaneously-truncated",
        "analysis_gates": intent["analysis_gates"],
        "analysis_interpretation": (
            "quality-matched coverage-forced portfolio-swap association; exact 3!^6 "
            "label-permutation and 2^6 sign-flip sensitivity analyses under strong "
            "exchangeability/symmetry assumptions; neither is design-based"
        ),
    }
    return intent, {**body, "actor_plan_digest": pilot._digest(body)}


def test_actor_plan_recomputes_portfolios_from_candidate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, actor_plan = _actor_plan_fixture(tmp_path, monkeypatch)
    assert pilot.validate_actor_plan(actor_plan, intent, run_dir=tmp_path) == actor_plan

    tampered = deepcopy(actor_plan)
    tampered["portfolios"][0] = {
        **tampered["portfolios"][0],
        "coverage_forced": tampered["portfolios"][0]["redundant_historical"],
    }
    body = {key: value for key, value in tampered.items() if key != "actor_plan_digest"}
    tampered["actor_plan_digest"] = pilot._digest(body)
    with pytest.raises(pilot.ProxyExperimentError, match="portfolios differ"):
        pilot.validate_actor_plan(tampered, intent, run_dir=tmp_path)


def test_persisted_actor_reward_is_replayed_not_trusted(tmp_path: Path) -> None:
    class Env:
        def reset(self, seed=None):
            return "locked observation", {}

        def step(self, _action):
            return "post", 0.0, False, False, {}

    class Target:
        def instantiate(self):
            return Env()

    engine = object.__new__(pilot._ActorEngine)
    engine.intent = {"intent_digest": _sha(20)}
    engine.actor_plan = {"actor_plan_digest": _sha(21)}
    engine.run_dir = tmp_path
    engine.config = {"qualification_timeout_seconds": 1.0}
    engine.dependencies = SimpleNamespace(target_factory=lambda *_args, **_kwargs: Target())
    candidate_id = "c001--v3"
    engine._candidate_by_id = {
        candidate_id: {
            "code": "class FakeEnv: pass",
            "probes": {"0": {"observation": "locked observation"}},
            "hints": {"0": {"hint": "strategy"}},
        }
    }
    pair = {
        "pair_id": "pair-001",
        "pair_ordinal": 0,
        "stratum_id": "c001",
        "candidate_id": candidate_id,
        "seed": 0,
        "arm_order": ["unhinted", "hinted"],
    }
    purpose = {
        "phase": "actor",
        "actor_plan_digest": engine.actor_plan["actor_plan_digest"],
        "pair_id": pair["pair_id"],
        "pair_ordinal": pair["pair_ordinal"],
        "pair_attempt": 1,
        "stratum_id": pair["stratum_id"],
        "candidate_id": pair["candidate_id"],
        "seed": pair["seed"],
        "arm": "unhinted",
        "turn": 1,
        "horizon": 1,
    }
    call_id = pilot.base._call_id(purpose)
    pilot.base._write_json(tmp_path / "calls" / call_id / "request.json", {})
    pilot.base._write_json(tmp_path / "calls" / call_id / "result.json", {})
    raw = "\\boxed{answer}"
    clean = pilot.live.extract_clean_action(raw, "boxed")
    body = {
        "schema_version": "spade-coverage-forced-arm/v1",
        "intent_digest": engine.intent["intent_digest"],
        "actor_plan_digest": engine.actor_plan["actor_plan_digest"],
        "pair_id": pair["pair_id"],
        "pair_ordinal": pair["pair_ordinal"],
        "pair_attempt": 1,
        "stratum_id": pair["stratum_id"],
        "candidate_id": pair["candidate_id"],
        "seed": 0,
        "arm": "unhinted",
        "call_id": call_id,
        "raw_response": raw,
        "raw_response_digest": pilot._digest(raw),
        "parser_miss": False,
        "clean_action": clean,
        "pre_observation": "locked observation",
        "pre_observation_digest": pilot._digest("locked observation"),
        "post_observation": "post",
        "post_observation_digest": pilot._digest("post"),
        "raw_reward": 0.0,
        "binary_return": 0.0,
        "terminated": False,
        "truncated": False,
        "success_rule": "reward-positive-even-if-simultaneously-truncated",
    }
    good = {**body, "arm_digest": pilot._digest(body)}
    engine._validate_call_request = lambda _value, _expected=None: {}
    engine._validate_call_result = lambda _value, _request: {
        "status": "success",
        "response": raw,
    }
    engine._proofpack_call = lambda operation, *args, **kwargs: operation(*args, **kwargs)
    assert engine._validate_arm(good, pair=pair, attempt=1, arm="unhinted") == good

    changed_body = {**body, "raw_reward": 1.0, "binary_return": 1.0}
    changed = {**changed_body, "arm_digest": pilot._digest(changed_body)}
    with pytest.raises(pilot.ProxyExperimentError, match="deterministic replay"):
        engine._validate_arm(changed, pair=pair, attempt=1, arm="unhinted")


def test_design_prompt_matches_v4_difficulty_clause_and_bounds_output(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    medium = {
        "stratum_id": "c001",
        "challenger_candidate_id": "c001--challenger",
        "skill": "logic",
        "difficulty": "medium",
    }
    hard = {**medium, "difficulty": "hard"}
    assert (
        "Medium difficulty: require 2-3 dependent reasoning steps and one unique boxed answer."
        in engine._design_prompt(medium, "")
    )
    assert (
        "Hard difficulty: require 4-6 dependent reasoning steps, plausible distractors, and one "
        "unique boxed answer." in engine._design_prompt(hard, "")
    )
    raw = "```python\n" + ("# x\n" * 121) + "```"
    assessment = engine._assess_design_response(raw, "c001--challenger", "c001")
    assert assessment["status"] == "rejected"
    assert assessment["source_nonblank_line_count"] == 121
    assert "120 nonblank lines" in assessment["reason"]


def test_design_attempt_is_terminal_only_after_exact_cell_and_quality_match(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    historical = tuple(
        ProxyCandidate.create(
            candidate_id=f"c001--{arm}",
            stratum_id="c001",
            source_arm=arm,
            quality_score=0.75,
            descriptor=STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR,
            environment_digest=_sha(index),
            evidence_digest=_sha(100 + index),
        )
        for index, arm in enumerate(("v3", "v4"), start=1)
    )
    engine._historical_proxy_candidates = lambda _stratum_id: historical
    qualification = {
        "environment_digest": _sha(3),
        "qualification_digest": _sha(103),
    }
    probes = {"0": {"observation": "a"}, "42": {"observation": "b"}}
    viability = {"viability_digest": _sha(203)}
    engine._qualification = lambda *_args: qualification
    engine._probe_candidate = lambda *_args: probes
    engine._one_turn_viability = lambda *_args: viability

    descriptor = dict(STANDARD_REDUNDANT_HISTORICAL_DESCRIPTOR)

    def cwa_evidence(_candidate):
        return {
            "descriptor": descriptor,
            "quality_score": 1.0,
            "evidence_digest": _sha(303),
        }

    engine._cwa_evidence = cwa_evidence
    raw = "```python\nclass PuzzleEnv:\n    pass\n```"
    rejected = engine._assess_design_response(raw, "c001--challenger", "c001")
    assert rejected["status"] == "rejected"
    assert rejected["scientific_preview"] is None
    assert "descriptor" in rejected["feedback_for_next_attempt"]

    descriptor = dict(COVERAGE_CHALLENGER_DESCRIPTOR)
    accepted = engine._assess_design_response(raw, "c001--challenger", "c001")
    assert accepted["status"] == "coverage_eligible"
    assert accepted["scientific_preview"]["absolute_quality_gap"] == 0.125


def test_portfolio_identity_uses_stable_cwa_digest_not_later_hint_bytes(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    evidence = {
        "candidate_id": "c001--challenger",
        "stratum_id": "c001",
        "source_arm": "challenger",
        "environment_digest": _sha(3),
        "evidence_digest": _sha(900),
        "cwa": {
            "quality_score": 1.0,
            "descriptor": dict(COVERAGE_CHALLENGER_DESCRIPTOR),
            "evidence_digest": _sha(303),
        },
    }
    before = engine._proxy_candidate(evidence)
    after = engine._proxy_candidate({**evidence, "evidence_digest": _sha(1)})
    assert before == after
    assert before.evidence_digest == _sha(303)


def test_design_phase_resume_does_not_require_prior_challenger_hints(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    strata = (1, 3, 4, 5, 6, 7)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in strata
    ]
    for index in strata:
        for arm in ("v3", "v4"):
            pilot.base._write_json(engine._candidate_path(f"c{index:03d}--{arm}"), {"ok": True})
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda value, _request: value
    engine._validate_design_attempt = lambda value, **_kwargs: value
    for ordinal, index in enumerate((1, 3), start=1):
        purpose = {
            "phase": "challenger-design",
            "stratum_id": f"c{index:03d}",
            "candidate_id": f"c{index:03d}--challenger",
            "attempt": 1,
        }
        call_id = pilot.base._call_id(purpose)
        pilot.base._write_json(
            engine.run_dir / "calls" / call_id / "request.json",
            {"call_id": call_id, "local_ordinal": ordinal, "purpose": purpose},
        )
        pilot.base._write_json(engine.run_dir / "calls" / call_id / "result.json", {})
        pilot.base._write_json(
            engine.run_dir
            / "challenger-generation"
            / f"c{index:03d}--challenger"
            / "attempt-01.json",
            {
                "status": "coverage_eligible",
                "call_id": call_id,
                "feedback_for_next_attempt": "",
            },
        )
    engine._validate_pre_actor_call_inventory(require_closed=False)


def test_any_hint_request_first_revalidates_complete_global_quality_gate(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in (1, 3, 4, 5, 6, 7)
    ]
    purpose = {
        "phase": "challenger-hint",
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
    }
    call_id = pilot.base._call_id(purpose)
    pilot.base._write_json(
        engine.run_dir / "calls" / call_id / "request.json",
        {"call_id": call_id, "local_ordinal": 1, "purpose": purpose},
    )
    pilot.base._write_json(engine.run_dir / "calls" / call_id / "result.json", {})
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda value, _request: value

    def quality_gate():
        raise RuntimeError("complete quality gate invoked")

    engine._validate_complete_pre_hint_quality_gate = quality_gate
    with pytest.raises(RuntimeError, match="complete quality gate invoked"):
        engine._validate_pre_actor_call_inventory(require_closed=False)


def test_terminal_design_and_hint_attempts_reject_later_leaves(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    stratum = {
        "stratum_id": "c001",
        "challenger_candidate_id": "c001--challenger",
        "skill": "logic",
        "difficulty": "medium",
    }
    design_root = engine.run_dir / "challenger-generation" / "c001--challenger"
    pilot.base._write_json(
        design_root / "attempt-01.json", {"status": "coverage_eligible", "code": "x"}
    )
    pilot.base._write_json(design_root / "attempt-02.json", {"status": "rejected"})
    engine._validate_design_attempt = lambda value, **_kwargs: value
    with pytest.raises(pilot.ProxyExperimentError, match="after terminal eligibility"):
        asyncio.run(engine._generate_challenger(stratum))

    hint_root = engine.run_dir / "challenger-hints" / "c001--challenger" / "0"
    accepted = {"status": "accepted", "hint": "strategy"}
    pilot.base._write_json(hint_root / "attempt-01.json", accepted)
    pilot.base._write_json(hint_root / "attempt-02.json", {"status": "leaked"})
    engine._validate_hint_attempt = lambda value, **_kwargs: value
    with pytest.raises(pilot.ProxyExperimentError, match="after terminal acceptance"):
        asyncio.run(
            engine._lock_hint(
                candidate_id="c001--challenger",
                seed=0,
                observation="obs",
                solution="answer",
            )
        )


def test_hint_attempt_binds_locked_probe_digests(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    purpose = {
        "phase": "challenger-hint",
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
    }
    body = {
        "schema_version": "spade-coverage-forced-hint-attempt/v1",
        "intent_digest": engine.intent["intent_digest"],
        "candidate_id": "c001--challenger",
        "seed": 0,
        "attempt": 1,
        "call_id": pilot.base._call_id(purpose),
        "status": "accepted",
        "reason": "nonleaking",
        "hint": "strategy",
        "hint_digest": pilot._digest("strategy"),
        "observation_digest": pilot._digest("different observation"),
        "solution_digest": pilot._digest("answer"),
        "feedback_for_next_attempt": "",
    }
    leaf = {**body, "attempt_digest": pilot._digest(body)}
    with pytest.raises(pilot.ProxyExperimentError, match="probe binding"):
        engine._validate_hint_attempt(
            leaf,
            candidate_id="c001--challenger",
            seed=0,
            observation="locked observation",
            solution="answer",
            attempt=1,
            feedback="",
        )


def test_preactor_inventory_rejects_arbitrary_attempt_paths(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in (1, 3, 4, 5, 6, 7)
    ]
    pilot.base._write_json(
        engine.run_dir / "challenger-generation" / "junk" / "attempt-99.json",
        {"call_id": "laundered"},
    )
    with pytest.raises(pilot.ProxyExperimentError, match="noncanonical candidate"):
        engine._validate_pre_actor_call_inventory(require_closed=False)


def test_proofpack_boundary_revalidates_before_and_after_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    checks = 0

    def check(_identity):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise pilot.ProxyExperimentError("runtime drift")

    monkeypatch.setattr(pilot, "_validate_runtime_identity", check)
    operation_calls = 0

    def operation():
        nonlocal operation_calls
        operation_calls += 1
        return "result"

    with pytest.raises(pilot.ProxyExperimentError, match="runtime drift"):
        engine._proofpack_call(operation)
    assert operation_calls == 1
    assert checks == 2


def test_actor_preflight_rejects_calls_after_unmaterialized_terminal_result(
    tmp_path: Path,
) -> None:
    engine = object.__new__(pilot._ActorEngine)
    engine.run_dir = tmp_path
    engine.actor_plan = {
        "actor_plan_digest": _sha(31),
        "pair_schedule": [
            {
                "pair_id": "pair-001",
                "pair_ordinal": 0,
                "stratum_id": "c001",
                "candidate_id": "c001--v3",
                "seed": 0,
                "arm_order": ["unhinted", "hinted"],
            }
        ],
    }
    engine._candidate_by_id = {
        "c001--v3": {
            "probes": {"0": {"observation": "obs"}},
            "hints": {"0": {"hint": "strategy"}},
        }
    }
    engine._scientific_reaudit_complete = True
    engine._validate_pre_actor_call_inventory = lambda **_kwargs: None
    engine._validate_call_request = lambda value, _expected=None: value
    engine._validate_call_result = lambda value, _request: value
    for ordinal, arm in enumerate(("unhinted", "hinted"), start=1):
        purpose = {
            "phase": "actor",
            "actor_plan_digest": engine.actor_plan["actor_plan_digest"],
            "pair_id": "pair-001",
            "pair_ordinal": 0,
            "pair_attempt": 1,
            "stratum_id": "c001",
            "candidate_id": "c001--v3",
            "seed": 0,
            "arm": arm,
            "turn": 1,
            "horizon": 1,
        }
        call_id = pilot.base._call_id(purpose)
        pilot.base._write_json(
            tmp_path / "calls" / call_id / "request.json",
            {"call_id": call_id, "local_ordinal": ordinal, "purpose": purpose},
        )
        pilot.base._write_json(
            tmp_path / "calls" / call_id / "result.json",
            {
                "status": "error",
                "failure_category": "empty_response",
            },
        )
    with pytest.raises(pilot.ProxyExperimentError, match="multiple actor calls"):
        engine._preflight_actor_state()


def test_actor_preflight_rejects_postseal_orphaned_preactor_call(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.intent["strata"] = [
        {
            "stratum_id": f"c{index:03d}",
            "challenger_candidate_id": f"c{index:03d}--challenger",
            "skill": "logic",
            "difficulty": "medium",
        }
        for index in (1, 3, 4, 5, 6, 7)
    ]
    for source_arm in ("v3", "v4"):
        pilot.base._write_json(
            engine._candidate_path(f"c001--{source_arm}"), {"sealed": source_arm}
        )
    request = _request(engine)
    request_path = engine.run_dir / "calls" / request["call_id"] / "request.json"
    pilot.base._write_json(request_path, request)
    timestamp = request["reserved_at_utc"]
    result_body = {
        "schema_version": pilot.CALL_RESULT_SCHEMA,
        "intent_digest": engine.intent["intent_digest"],
        "call_id": request["call_id"],
        "local_ordinal": request["local_ordinal"],
        "global_ordinal": request["global_ordinal"],
        "request_digest": request["request_digest"],
        "status": "success",
        "failure_category": None,
        "exception_type": None,
        "error": None,
        "exit_status": 0,
        "response": "```python\nclass PuzzleEnv: pass\n```",
        "response_digest": pilot._digest("```python\nclass PuzzleEnv: pass\n```"),
        "started_at_utc": timestamp,
        "finished_at_utc": timestamp,
        "duration_seconds": 0.0,
    }
    pilot.base._write_json(
        request_path.parent / "result.json",
        {**result_body, "result_digest": pilot._digest(result_body)},
    )

    actor = object.__new__(pilot._ActorEngine)
    actor.intent = engine.intent
    actor.config = engine.config
    actor.run_dir = engine.run_dir
    actor.dependencies = engine.dependencies
    actor.actor_plan = {"actor_plan_digest": _sha(40), "pair_schedule": []}
    actor._candidate_by_id = {}
    actor._scientific_reaudit_complete = True
    with pytest.raises(pilot.ProxyExperimentError, match="orphaned pre-actor calls"):
        actor._preflight_actor_state()


def test_final_inventory_roots_nested_aggregate_and_rejects_model_lock(tmp_path: Path) -> None:
    engine = object.__new__(pilot._ActorEngine)
    engine.run_dir = tmp_path / "run"
    engine.intent = {"shared_ledger_root": str(tmp_path / "ledger")}
    engine.run_dir.mkdir(parents=True)
    engine.ledger_root.mkdir(parents=True)
    nested = engine.run_dir / "nested" / "aggregate.json"
    pilot.base._write_json(nested, {"must": "be rooted"})
    inventory = engine._complete_evidence_inventory()
    assert any(item["path"] == "nested/aggregate.json" for item in inventory)

    (engine.run_dir / "model.lock").write_text("forbidden", encoding="utf-8")
    with pytest.raises(pilot.ProxyExperimentError, match="model.lock"):
        engine._complete_evidence_inventory()
    (engine.run_dir / "model.lock").unlink()
    (engine.ledger_root / "model.lock").write_text("also forbidden", encoding="utf-8")
    with pytest.raises(pilot.ProxyExperimentError, match="model.lock"):
        engine._complete_evidence_inventory()


def test_aggregate_recomputes_outcomes_and_analysis_and_refuses_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_intent, actor_plan = _actor_plan_fixture(tmp_path, monkeypatch)
    engine = object.__new__(pilot._ActorEngine)
    engine.run_dir = tmp_path
    engine.actor_plan = actor_plan
    gates = {
        "pooled_unhinted_min": 0.10,
        "pooled_unhinted_max": 0.90,
        "minimum_discordant_pairs": 8,
        "minimum_coverage_forced_delta": 0.10,
        "one_sided_label_permutation_alpha": 0.05,
        "maximum_first_attempt_exogenous_failure_rate": 0.15,
    }
    engine.intent = {
        "intent_digest": fixture_intent["intent_digest"],
        "shared_ledger_root": str(tmp_path / "ledger"),
        "analysis_role": "exploratory",
        "claim_exclusions": ["no-causal-claim"],
        "analysis_gates": gates,
        "budget": {"governance_scope": "external-context"},
    }
    outcomes: list[PairedOutcome] = []
    for portfolio in actor_plan["portfolios"]:
        for candidate_id in portfolio["candidate_ids"]:
            for seed in (0, 42):
                if candidate_id == portfolio["challenger_id"]:
                    unhinted, hinted = 0.0, 1.0
                elif candidate_id == portfolio["displaced_historical_id"]:
                    unhinted, hinted = 1.0, 0.0
                else:
                    unhinted = hinted = 0.0
                outcomes.append(
                    PairedOutcome(
                        candidate_id=candidate_id,
                        stratum_id=portfolio["stratum_id"],
                        seed=seed,
                        unhinted=unhinted,
                        hinted=hinted,
                    )
                )
    engine._paired_outcomes = lambda _resolutions: outcomes
    engine._complete_evidence_inventory = lambda: []
    locks = tuple(
        pilot.LockedPortfolios(
            stratum_id=item["stratum_id"],
            candidate_ids=tuple(item["candidate_ids"]),
            coverage_forced=tuple(item["coverage_forced"]),
            redundant_historical=tuple(item["redundant_historical"]),
            challenger_id=item["challenger_id"],
            retained_historical_id=item["retained_historical_id"],
            displaced_historical_id=item["displaced_historical_id"],
            coverage_forced_quality=item["coverage_forced_quality"],
            redundant_historical_quality=item["redundant_historical_quality"],
            signed_quality_gap=item["signed_quality_gap"],
            absolute_quality_gap=item["absolute_quality_gap"],
        )
        for item in actor_plan["portfolios"]
    )
    analysis = pilot.analyze_proxy_pilot(locks, outcomes).to_dict()
    resolutions = [{"resolution_digest": _sha(index)} for index in range(pilot.PAIR_COUNT)]
    body = {
        "schema_version": pilot.AGGREGATE_SCHEMA,
        "protocol_id": pilot.PROTOCOL_ID,
        "intent_digest": engine.intent["intent_digest"],
        "actor_plan_digest": actor_plan["actor_plan_digest"],
        "analysis_role": engine.intent["analysis_role"],
        "claim_exclusions": engine.intent["claim_exclusions"],
        "provider": "agy",
        "model": pilot.ACTOR_MODEL,
        "backend_identity_attested": False,
        "route_authority": "requested-route-only",
        "estimand": (
            "association over locked realized environment plus source-specific hint packages; "
            "environment effects are not isolated from hint quality, epoch, or source arm"
        ),
        "resolution_digests": [item["resolution_digest"] for item in resolutions],
        "portfolios": actor_plan["portfolios"],
        "portfolio_quality_diagnostics": actor_plan["portfolio_quality_diagnostics"],
        "outcomes": [pilot._plain(item) for item in outcomes],
        "analysis": analysis,
        "analysis_interpretation": (
            "exploratory quality-matched coverage-forced portfolio-swap association; exact "
            "label-permutation and sign-flip sensitivity analyses under strong assumptions; "
            "neither is design-based and there is no causal or learner-improvement claim"
        ),
        "new_charged_calls": 0,
        "global_charged_calls": pilot.PRIOR_CHARGED_CALLS,
        "authorized_global_call_cap": pilot.AUTHORIZED_GLOBAL_CALL_CAP,
        "global_budget_governance_scope": "external-context",
        "evidence_inventory": [],
        "evidence_root_digest": pilot._digest([]),
        "assay_decision": "not-run-not-applicable",
        "release_authorized": False,
        "model_lock_status": "absent",
    }
    value = {**body, "aggregate_digest": pilot._digest(body)}
    engine._validate_aggregate(value, resolutions)

    tampered = deepcopy(value)
    tampered["analysis"]["coverage_forced_delta"] = -1.0
    tampered_body = {key: item for key, item in tampered.items() if key != "aggregate_digest"}
    tampered["aggregate_digest"] = pilot._digest(tampered_body)
    with pytest.raises(pilot.ProxyExperimentError, match="deterministic recomputation"):
        engine._validate_aggregate(tampered, resolutions)

    extra = {**deepcopy(value), "promotion_authorized": True}
    extra_body = {key: item for key, item in extra.items() if key != "aggregate_digest"}
    extra["aggregate_digest"] = pilot._digest(extra_body)
    with pytest.raises(pilot.ProxyExperimentError, match="fields/schema"):
        engine._validate_aggregate(extra, resolutions)

    released = deepcopy(value)
    released["release_authorized"] = True
    released_body = {key: item for key, item in released.items() if key != "aggregate_digest"}
    released["aggregate_digest"] = pilot._digest(released_body)
    with pytest.raises(pilot.ProxyExperimentError, match="identity/release boundary"):
        engine._validate_aggregate(released, resolutions)
