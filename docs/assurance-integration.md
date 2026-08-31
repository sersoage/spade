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

A proposed calibration pilot uses 18 independently generated environments: one medium and one hard
environment for each of SPADE's nine cognitive skills. Each environment is evaluated at the three
ProofPack-qualified seeds `[0, 1, 42]`, with paired hinted and unhinted episodes from the same
explicitly pinned `agy` model and a five-turn horizon. The environment is the independent cluster;
the three seeds are within-environment repetitions. All environments and hints are locked before
actor rollouts, arm order is counterbalanced, and all 18 clusters feed one Assay decision. Pilot data
must remain separate from any later confirmatory study.

That pilot is not supported by the current CLI. Before execution, the runner needs a sealed schedule,
multi-seed evaluation, deterministic counterbalancing, pre-outcome reserve handling, idempotent
resume, per-call provenance, a hard invocation cap, and one aggregate Assay write. It must bind the
schedule, repository revisions, runtime and sandbox details, explicit model request, and all leaf
artifact digests. The subscription CLI does not expose trustworthy token or cost records, so the
evidence must not claim them.

The 18-cluster design needs at least 180 `agy` invocations and can use substantially more when
generation retries, hint rewrites, or multi-turn episodes occur. Use a separately authorized pilot
cap of 450 invocations and stop incomplete if that cap is reached. A confirmatory stage should only
be sized from the locked pilot variance, receive its own authorization, and never pool pilot
outcomes. Neither stage can authorize `model.lock` through this integration.

## Verification

Run the focused SPADE integration tests from this repository:

```bash
python -m pytest -q \
  tests/test_proofpack_bridge.py \
  tests/test_game_utils.py \
  tests/test_live_spade_eval.py
python tools/run_live_spade_eval.py --help
```

ProofPack and Assay maintain their own focused suites and repository-level `make verify` commands.
Their design documents are `docs/design/spade-environment-qualification.md` in ProofPack and
`docs/design/spade-autocurriculum-certification.md` in Assay.
