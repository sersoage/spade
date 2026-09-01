# SPADE assurance integration

SPADE can require ProofPack qualification while generating environments and can persist live
evaluation evidence through Assay. Both integrations are opt-in because SPADE supports Python
3.10+, whereas the current `proofpack-env` and `assay` packages require Python 3.12+.

## Trust boundary

ProofPack executes candidate environment source in a fresh, time-bounded OS sandbox while running
the V0–V4 qualification ladder. It fails closed if the compatible API or required isolation backend
is unavailable. Qualification checks the configured action format, rollout horizon, and all
configured seeds.

Passing qualification is evidence about those checks; it does not turn source code into trusted
Python. SPADE's standard training environments are still loaded by its native runtime. Operators
must treat that runtime as a separate trusted-code boundary or add an appropriate production
sandbox. The live runner avoids that boundary by using ProofPack's replay-backed session proxy for
every reset, solution, and step operation.

## Enable the generation gate

The core async, corpus-grounded, batched, Slime, Tinker, and legacy generation paths all carry the
same settings:

```python
from spade.core.types import SpadeConfig

config = SpadeConfig(
    use_proofpack_qualification=True,
    proofpack_seeds=[0, 1, 42],
    proofpack_timeout_seconds=5.0,
    action_format="boxed",
    max_turns=20,
)
```

When enabled, SPADE checks for the compatible qualifier before requesting generation from a model.
Each candidate must then pass ProofPack before it can enter the generated-game pool. A missing or
older ProofPack, an empty seed set, an invalid timeout, an infrastructure failure, or any failed
clause rejects the candidate.

For Slime, use the corresponding flags:

```text
--spade-use-proofpack-qualification
--spade-proofpack-seeds 0 1 42
--spade-proofpack-timeout-seconds 5
```

## Run the live smoke

Use one Python 3.12 environment containing `assay`, `proofpack-env`, and all of their runtime
dependencies. A sibling source checkout lets the runner discover the packages, but does not install
their dependencies. For the sibling workspaces used here, a concrete development setup is:

```bash
uv sync --project ../proofpack
uv pip install --python ../proofpack/.venv/bin/python --editable ../assay
ASSURANCE_PYTHON=../proofpack/.venv/bin/python
```

An independently managed Python 3.12 environment with the released packages installed is equally
valid.

```bash
"$ASSURANCE_PYTHON" tools/run_live_spade_eval.py \
  --provider agy \
  --skill "Deterministic Two-Step Arithmetic Reasoning" \
  --max-turns 5 \
  --design-attempts 3
```

The runner never loads `~/.env`. API-backed providers may receive `--api-key` or an explicitly
selected `--env-file`. The `agy` subprocess runs in an empty temporary directory with its own
sandbox flag and a bounded timeout; unrelated API-secret environment variables are not forwarded.

The two arms use the same qualified seed. Model prose is reduced to the final action before
`step()`, and rollouts continue until termination, truncation, or the configured horizon. The hint
sees only the initial observation, never environment source or `solution()`, and a basic
explicit-answer leakage heuristic rejects direct answer phrases. It is not a semantic proof that
all possible paraphrased leakage is absent.

## Artifacts and promotion semantics

Each invocation creates a unique run directory under `.assay/spade-live/`:

```text
<run-id>/
├── environment.py
├── proofpack-qualification.json
├── live-trace.json
└── assay/
    ├── evidence/
    └── spade/<certification-digest>/
        ├── curriculum-manifest.json
        ├── observations.json
        ├── evaluation.json
        └── certification.json
```

The curriculum manifest binds SHA-256 digests of both the ProofPack receipt and complete live trace.
The outer files remain writable, but any replacement is detectable because its digest no longer
matches the content-addressed Assay bundle; conflicting rewrites of the inner bundle are refused.
The default Evidence v2 adapter records the declared task-level dependence and aggregate returns; it
does not invent token, cost, or provider execution facts that the subscription CLI does not expose.

The smoke contains one independent environment cluster. Assay therefore records
`decision_reason="insufficient_clusters"`, sets `promoted=false`, and emits no `model.lock`, even if
the hinted rollout beats the unhinted rollout. A positive statistical signal requires at least four
declared independent environment clusters, nonzero identified between-cluster uncertainty, a chosen
effect threshold, and significance. That statistical result alone is not release approval; lock
emission has separate registration and operational evidence requirements.

## Multi-environment `agy` experiments

`run_live_spade_eval.py` is a one-cluster smoke harness. Re-running it creates separate one-cluster
bundles; it does not aggregate independent clusters and must not be reported as a larger experiment.
Do not wrap the current command in a shell loop as a substitute for an experiment runner.

A calibration pilot uses 18 separately scheduled, exact-byte-distinct selected environments: one
medium and one hard environment for each of SPADE's nine cognitive skills. Each environment is
evaluated at the three ProofPack-qualified seeds `[0, 1, 42]`, with paired hinted and unhinted
episodes from the same explicitly requested `agy` model route and a five-turn horizon. The
environment is treated as the independent cluster; the three seeds are within-environment
repetitions. Exact-byte distinction does not by itself prove causal independence. All environments
and hints are locked before actor rollouts, arm order is counterbalanced, and all 18 clusters feed
one Assay decision. Pilot data must remain separate from any later confirmatory study.

The multi-environment runner implements that protocol. First create a local run-integrity plan. This
seal is not witnessed preregistration and does not establish release authority:

```bash
SPADE_AGY_RUN_ROOT="$(pwd)/.assay/spade-experiments/runs"
"$ASSURANCE_PYTHON" tools/run_spade_agy_experiment.py pilot-plan \
  --output .assay/spade-experiments/pilot-plan.json \
  --output-root "$SPADE_AGY_RUN_ROOT" \
  --experiment-id spade-agy-pilot-v1 \
  --model '<explicit-agy-model>' \
  --total-call-cap 450
```

The generated plan contains 18 analysis clusters and 27 candidate slots, including nine
same-skill/difficulty hard-stratum reserves that can only be selected before the cohort lock. It
binds the requested model route, canonical absolute run-output root, source revisions, runner and
`agy` executable digests, CLI/runtime identity, qualification/evaluation seeds, attempts, timeouts,
schedules, statistical settings, and call cap. `agy` does not attest the resolved backend model, so
the plan explicitly records only a requested route.

Validate the plan without creating a run directory, constructing a live `agy` client, or spending a
provider call:

```bash
"$ASSURANCE_PYTHON" tools/run_spade_agy_experiment.py run \
  --plan .assay/spade-experiments/pilot-plan.json \
  --output-root "$SPADE_AGY_RUN_ROOT"
```

Execution is deliberately double opt-in. Run this only after separately authorizing the sealed cap:

```bash
"$ASSURANCE_PYTHON" tools/run_spade_agy_experiment.py run \
  --plan .assay/spade-experiments/pilot-plan.json \
  --output-root "$SPADE_AGY_RUN_ROOT" \
  --execute \
  --acknowledge-call-cap 450
```

The canonical output root is part of the plan digest and an alternate root is rejected before live
dependencies or calls are reached. Within that sealed root, the run directory is derived from the
experiment ID and plan digest. A single-writer lock, conflict-checked call reservations/results,
execution-start runtime/source checks, deterministic assignments, and semantic environment replay
make completed leaves resumable.
An invocation reserved before a crash but lacking a durable result is charged and treated as
ambiguous; it is never silently replayed. Every `agy` call uses a fresh empty temporary directory.
Nested symlinks are rejected. Only a complete cohort and outcome schedule can reach the sole
aggregate Assay write, whose full artifact subtree is inventoried and revalidated on resume.

This is a local integrity protocol, not an external uniqueness or preregistration witness. An
operator who controls the filesystem can create and reseal a different plan/root, remove prior local
state, or copy artifacts outside this protocol. Publish or witness the plan digest before a
confirmatory run, and route any release claim through Assay's `assay-experiment/v1` registration and
authorization controls. The calibration pilot itself never authorizes release.

The 18-cluster design needs at least 180 `agy` CLI launches and can use substantially more when
generation retries, hint rewrites, reserve selection, or multi-turn episodes occur; the sealed
protocol maximum is 783 CLI launches. The 450 limit is a hard cap for the plan's sealed local run
root, not a token, cost, or backend-request claim. The runner stops incomplete if the cap is reached
and never sends a favorable subset to Assay. No attempted pilot has completed. A confirmatory stage
should only be sized from variance in a complete locked pilot, receive its own authorization, and
never pool pilot outcomes. Neither stage can authorize `model.lock` through this integration.

### Incomplete Google Pro v3 baseline and prompt treatment

The requested-route `gemini-3.1-pro-high` v3 baseline run was attempted and failed before cohort
lock. It reserved 35 `agy` calls: 14 designer calls and 21 hint calls. Five designer calls returned
the sealed 180-second timeout. Seven clusters were selected, but the remaining strategic-planning
hard primary and reserve slots could not qualify: their completed candidates failed ProofPack V0
for the disallowed identifiers `system` and `modules`, while the other attempts timed out. No cohort
lock, actor outcome, aggregate Assay request, or Assay decision exists. This is an incomplete run,
not baseline performance evidence, and it must not be resumed or pooled with a later experiment.

The next treatment changes only `DESIGNER_PROMPT`. It tells the designer that ProofPack-disallowed
identifiers are forbidden even when used as ordinary variables, attributes, or aliases, names
`system` and `modules` plus dangerous reflection, dynamic-execution, filesystem, and process
capabilities, and caps the requested implementation at 120 nonblank lines and 8,000 characters.
The cap is intended to reduce overlong generations that can encounter the 180-second boundary; it
does not establish why any call timed out or guarantee that a later call will complete. The
environment interface, skill and difficulty instructions, sealed schedule, qualification gate, and
analysis remain unchanged. No success or improvement claim is warranted until a new separately
sealed run completes its full cohort and Assay decision.

### Prospective v5 outcome-only replay

The v4 prompt-treatment run produced a complete 18-cluster pre-outcome cohort, then stopped
fail-closed at call 89 when the first unhinted `c003` actor call returned an empty AGY response.
It contains 12 completed actor outcomes (six `1.0`/`1.0` pairs across the ordered `c001` and
`c002` prefix), one failed outcome, and no Assay request, Assay decision, or `model.lock`. Those
partial outcomes are descriptive ceiling evidence only and are not imported or pooled below.

The dedicated
`run_spade_agy_outcome_replay.py` runner can import it without importing a v4 actor call, actor
outcome, ledger, or Assay artifact. Plan creation requires the authorized historical identities:

```bash
SPADE_AGY_V4_RUN="$(pwd)/.assay/spade-experiments/runs/\
spade-agy-pilot-designer-prompt-treatment-google-pro-v4-\
8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc"
SPADE_AGY_V5_RUN_ROOT="$(pwd)/.assay/spade-experiments/runs"
"$ASSURANCE_PYTHON" tools/run_spade_agy_outcome_replay.py outcome-replay-plan \
  --source-run "$SPADE_AGY_V4_RUN" \
  --expected-source-plan-digest \
    sha256:8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc \
  --expected-source-cohort-digest \
    sha256:161353ebd4454516e3379414444323dd13aeab95640eb130ec7414f23876b84b \
  --output .assay/spade-experiments/outcome-replay-v5-plan.json \
  --output-root "$SPADE_AGY_V5_RUN_ROOT" \
  --experiment-id spade-agy-v4-cohort-outcome-replay-v5 \
  --total-call-cap 264
```

The import binds the complete designer/hint call prefix and validated candidate, qualification,
probe, hint, selection, and cohort leaves. Unknown candidate leaves, orphan calls, path escapes,
symlinks, and actor/outcome/Assay imports fail closed. The imported bytes become the local resume
authority, so later actor additions to the historical run cannot enter v5.

The evaluation horizon is derived only from locked probe solutions, before considering any v4
actor result: a boxed list uses one turn per item, a multiline string one turn per nonblank line,
and a scalar one turn. All three seeds must agree. For this cohort,
`c008-strategic-planning-hard` has horizon 5 and the other 17 clusters have horizon 1. Every
environment is revalidated under the original five-turn ProofPack V0-V4 contract. Separately, a
twice-per-seed deterministic oracle replay establishes actor-horizon viability; that receipt is not
represented as full ProofPack qualification.

The 54 `(cluster, seed)` pairs retain their counterbalanced arm order. Each has at most two attempt
slots. Only an exact empty response or either exact AGY timeout form may open attempt 2. A failure
discards the entire first pair attempt; both arms restart from the locked seed and are never mixed
across attempts. Any nonempty response is consumed as model behavior, including a malformed boxed
action. Generic nonzero exits, process-start errors, ambiguous reservations, local environment
failures, and every other error are terminal. Durable results are reused on resume without another
provider call.

The worst-case ceiling is `2 attempts × 2 arms × 66 horizon turns = 264` calls. With 178 previously
charged calls, the sealed maximum is 442 of the authorized global 450, leaving eight calls of
headroom. Validate or execute with an exact acknowledgement:

```bash
"$ASSURANCE_PYTHON" tools/run_spade_agy_outcome_replay.py run \
  --plan .assay/spade-experiments/outcome-replay-v5-plan.json \
  --output-root "$SPADE_AGY_V5_RUN_ROOT"

"$ASSURANCE_PYTHON" tools/run_spade_agy_outcome_replay.py run \
  --plan .assay/spade-experiments/outcome-replay-v5-plan.json \
  --output-root "$SPADE_AGY_V5_RUN_ROOT" \
  --execute \
  --acknowledge-call-cap 264
```

Assay is unreachable until all 54 same-attempt pair resolutions and exactly 108 selected outcomes
exist. Its request and task metadata bind the source plan/cohort/import, both offline assurance
receipts, pair-resolution manifest, and pre-Assay ledger. A physical `model.lock` is rejected even
if an integration object reports no lock. This remains exploratory post-v4 calibration; implementing
the runner does not mean v5 has launched or that the prompt treatment improves outcomes.

## Verification

Run the focused SPADE integration tests from this repository:

```bash
python -m pytest -q \
  tests/test_proofpack_bridge.py \
  tests/test_game_utils.py \
  tests/test_live_spade_eval.py \
  tests/test_spade_agy_experiment.py \
  tests/test_spade_agy_outcome_replay.py
python tools/run_live_spade_eval.py --help
python tools/run_spade_agy_experiment.py --help
python tools/run_spade_agy_outcome_replay.py --help
```

ProofPack and Assay maintain their own focused suites and repository-level `make verify` commands.
Their design documents are `docs/design/spade-environment-qualification.md` in ProofPack and
`docs/design/spade-autocurriculum-certification.md` in Assay.
