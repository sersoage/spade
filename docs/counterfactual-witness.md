# Counterfactual witness certificates

SPADE's structural validation and ProofPack V0-V4 qualification establish a bounded set of
properties for generated executable environments. They do not establish that two source files are
behaviorally different, or that a small replay panel identifies the reward, termination, parser,
state, and observation semantics that matter for curriculum diversity.

The counterfactual witness experiment is an offline falsification test for a stronger environment
identity. It generates deterministic source variants, executes them only in ProofPack's isolated
worker, and selects a small set of replay probes that distinguishes admitted semantic mutants while
remaining invariant to admitted behavior-preserving controls.

This is a development experiment, not a learner-improvement result. The held-out operators use
different transformations but share observable channels with the training catalog, and the held-out
control uses the same wrapper mechanism as a training control. A pass is therefore an exploratory,
selection-blind operator holdout—not independent semantic generalization, causal improvement, or
release evidence.

## Core contract

`spade.core.counterfactual_witness` is pure Python 3.10-compatible code. It parses and rewrites ASTs
but never compiles or executes generated source. It provides:

- twelve deterministic variants per environment: three admitted equivalent controls, five training
  semantic families, and four operator-held-out semantic families;
- a bounded probe grammar over the three locked seeds, including reset, oracle, malformed,
  well-formed-wrong, recovery, ordering, repetition, and oracle-prefix relations;
- canonical non-textual trace signatures;
- equivalent-control-safe, inverse-family greedy set cover; and
- content-bound witness certificates with deterministic verification.

Generated wrapper variants rename the original concrete environment to a non-`Env` base and expose
exactly one concrete `*Env` entrypoint. ProofPack remains the only component that executes source.

`spade.core.witness_archive` supplies an optional quality-diversity archive. Cells are partitioned
by skill and difficulty before behavioral coordinates, so one skill cannot evict another. Missing
recovery or reversed-order probes are represented as `unmeasured`, not false. Each cell retains one
quality champion and one lineage-disjoint challenger, with explicit demotion and eviction records.

## Sealed 18-environment falsification

`tools/run_counterfactual_witness_experiment.py` consumes only the authorized, allowlisted
pre-outcome evidence from the preserved Google-v4 cohort:

- source plan `sha256:8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc`;
- cohort `sha256:161353ebd4454516e3379414444323dd13aeab95640eb130ec7414f23876b84b`;
- 18 environments: medium and hard for each of nine cognitive skills; and
- seeds `[0, 1, 42]`, boxed actions, and a five-turn ProofPack boundary.

No historical actor outcome, Assay result, or AGY response is an input. The runner has no callable
provider or Assay boundary. This also means AGY's current Google-only model availability is
irrelevant to this offline stage.

Plan creation seals the source and cohort digests, mutation/probe catalog, source/runtime bytes,
output root, two deterministic repetitions, analysis thresholds, and exact planned sandbox-operation
ceiling. The current catalog has 12 variants plus the base and 444 probes across the 18 environments,
for 11,544 isolated replays.

```bash
ASSURANCE_PYTHON=/path/to/pinned/proofpack-venv/bin/python
SOURCE_RUN=/absolute/path/to/the/authorized-v4-run
CWA_ROOT=/absolute/path/to/counterfactual-witness-results

"$ASSURANCE_PYTHON" tools/run_counterfactual_witness_experiment.py plan \
  --source-run "$SOURCE_RUN" \
  --output-root "$CWA_ROOT" \
  --output "$CWA_ROOT/plan.json"

# Validation only; creates no run and executes no candidate source.
"$ASSURANCE_PYTHON" tools/run_counterfactual_witness_experiment.py run \
  --plan "$CWA_ROOT/plan.json"

# Offline ProofPack execution; still zero AGY/provider calls and zero learner updates.
"$ASSURANCE_PYTHON" tools/run_counterfactual_witness_experiment.py run \
  --plan "$CWA_ROOT/plan.json" \
  --execute
```

Every sandbox operation has an immutable request written before execution and a bound result written
afterward. A request without a result is ambiguous and never replayed. Trace leaves, certificates,
cluster reports, and the aggregate are content-bound; unknown files, symlinked paths, runtime/source
drift, conflicting resume bytes, nondeterminism, and non-behavioral worker errors fail closed.
ProofPack's exact canonical early-terminal response is retained as a behavioral observation for this
audited boxed cohort. Other errors remain fatal.

## Metrics and gates

The report preserves raw mutant denominators and separately identifies mutants observable by the
complete training-control-safe probe bank. This matters because a source mutation can be
observationally equivalent for a particular environment—for example, collapsing reset seeds where
all locked seeds already expose the same behavior.

The exploratory gate requires all of:

- training-mutant macro recall at least `0.90`;
- operator-held-out recall at least `0.15` above 512 deterministic, cost-matched random draws from
  the same training-control-safe bank;
- admitted held-out-control false rejection at most `0.05`; and
- full safe-bank held-out killability at least `0.90`.

The report also compares against a cost-matched, qualification-inspired fixed ordering that cycles
oracle, well-formed-wrong, reset, malformed, and blank probes within each seed. It is not a claim
that one selected witness has the same total operation budget as full V0-V4. The report records the
full safe-bank upper bound, applicable-mutant recall, selected probe count, behavioral descriptor,
certificate digest, every trace digest, and the exact zero-provider/zero-learner boundary.

Passing permits only the next engineering step: use certificates and skill/difficulty-partitioned
archive decisions in shadow curriculum selection, then run a separate same-checkpoint,
compute-matched Slime learner assay. Failing means revise or abandon the witness representation;
neither outcome can be promoted to an improvement claim by itself.
