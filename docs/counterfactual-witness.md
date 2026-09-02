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

## Sealed execution result — 2026-09-01

The production run at SPADE commit `27d2bb8632d961f6b0e703665a43e628026ccaa4` completed all
18 environments and passed all four pre-sealed gates. Its plan digest is
`sha256:99e023d40d169af545cc6741e39cf55daa8d334b9b688e50235ec4cbead23e82`; its aggregate
digest is `sha256:e675af2b4da9fd4916884eb71f7661e3f16dca558c2c6f9376c03e485c6f0b00`.

| Metric | Result |
| --- | ---: |
| Isolated ProofPack replays completed | 11,544 / 11,544 |
| Provider calls / learner updates | 0 / 0 |
| Average selected probes per environment | 1.00 |
| Training-mutant macro recall | 1.000 |
| Operator-held-out raw macro recall | 0.875 |
| Operator-held-out applicable recall | 0.912 (69 observable of 72 raw mutants) |
| Cost-matched random held-out recall | 0.532 |
| Witness minus random recall | +0.343 |
| Qualification-inspired fixed recall | 0.264 |
| Full safe-bank held-out recall | 0.958 |
| Admitted-control false-rejection rate | 0.000 |

The selected witness was a wrong-action-then-oracle recovery trace for 17 environments and an
oracle-prefix-then-wrong trace for the five-step strategic-planning environment. This compact panel
cleanly falsified the training mutations and materially outperformed both sealed comparators, but
the result is still only representation evidence. It does not show that archive-based curriculum
selection improves a learner.

This pass permitted a shadow-only archive ingestion smoke. That smoke is an integration and
negative-control result; a separate same-checkpoint, compute-matched Slime learner assay remains
the test of an actual SPADE improvement.

## Shadow archive integration

`spade.core.witness_archive_shadow` is the deliberately non-active integration boundary. It loads
the exact plan → aggregate → cluster → certificate closure from an externally supplied aggregate
digest, verifies each caller-supplied game file against the sealed environment bytes, and appends
the resulting archive decision to a single-writer, immutable hash chain. Each event binds the
quality and lineage policy identifiers, input evidence, serialized decision, selected cell members,
and complete post-state digest. Conflicting resumes, unknown files, symlinks, source-byte drift, and
tampering fail closed.

The API intentionally exposes no curriculum-selection or reward method. Quality and lineage remain
caller-defined shadow metadata; the ledger cannot affect training. The fixed integration smoke uses
quality `0.0` and the singleton environment digest as lineage:

```bash
"$ASSURANCE_PYTHON" tools/run_witness_archive_shadow.py \
  --run-dir "$CWA_ROOT/spade-counterfactual-witness-v1-99e023d40d169af545cc6741e39cf55daa8d334b9b688e50235ec4cbead23e82" \
  --expected-aggregate-digest sha256:e675af2b4da9fd4916884eb71f7661e3f16dca558c2c6f9376c03e485c6f0b00 \
  --source-selections-dir "$SOURCE_RUN/selections" \
  --ledger-dir /absolute/path/to/witness-shadow-ledger
```

The sealed 18-environment smoke completed twice with exactly-once resume and no model, network,
source-execution, or learner boundary. It produced 18 events, 18 cells, and 18
`champion_inserted` decisions; the final event-chain digest is
`sha256:2bd39e4bf95547d1c66f557e40da49d975ebc9efe5596f53d487735ce6f64e95`, and the archive
post-state digest is
`sha256:e2e3a0df5f6bc85a94b260158253951a6683ce57fd77ec1cd86f652e6f84336d`.

That result is a useful negative control: this cohort has exactly one environment in every
`(skill, difficulty, behavior)` cell, so archive treatment is identical to keeping every item. It
does not test competition, replacement, or learner benefit. Activating archive selection therefore
requires a separately sealed cohort with multiple lineage-controlled candidates in the same cell
and an outcome assay that is independent of the witness evidence.

## Quality-matched coverage-forced proxy

The next experiment is a deliberately narrower proxy for behavioral coverage, not an active
archive or learner experiment. It uses six strata—`c001`, `c003`, `c004`, `c005`, `c006`, and
`c007`—with two imported Google Pro environments per stratum from the preserved v3 and v4 runs.
Both historical environments must revalidate in the exact standard descriptor cell: boxed action,
one-step oracle, two distinct reset-seed states, zero-reward continuing invalid action, successful
wrong-then-oracle recovery, and unmeasured reverse-order behavior.

One new challenger per stratum is requested from `gemini-3.1-pro-high`. Before it can enter the
panel, it must pass ProofPack V0–V4, deterministic reward-positive one-turn viability on seeds
`0` and `42`, CWA training-mutant recall at least `0.90`, admitted-control false rejection at most
`0.05`, and the exact matched descriptor except that wrong-then-oracle recovery must be observed
false. The wrong action must remain nonterminal and nontruncated during the five-turn CWA replay;
at the one-turn actor boundary, a wrong answer can be recorded as truncated and still counts simply
as a zero-reward task failure. A failed target-cell or per-stratum quality screen may use the second
sealed design attempt.

The redundant portfolio is `{v3, v4}`. The coverage-forced portfolio is `{challenger, retained
historical}`; the historical candidate whose CWA held-out-applicable recall is closest to the
challenger is displaced, with environment and CWA evidence digests as the only tie-breakers. Every
stratum must produce a different portfolio, every absolute portfolio-quality gap must be at most
`0.125`, and the six-stratum mean absolute gap must be at most `0.0625`. All six scientific records
and the global quality gate are recomputed before any challenger hint call.

The actor route is requested `gemini-3.7-flash-high`. The sealed panel has 36 packages
(`6 strata × 3 candidates × 2 seeds`), each evaluated with and without its source-specific locked
hint in counterbalanced order at horizon one. A nonempty response that misses the boxed parser is
an observed zero reward and is never retried. Only exact empty-response or provider-timeout results
open a whole-pair retry. All unresolved pairs receive at most two waves; a third wave is permitted
only when at most 14 remain.

Two immutable seals separate the phases. The generation intent is written before any new AGY
request. The actor plan is written only after all 18 candidate qualification, viability, CWA, and
hint records pass, and before any actor request. Requests are reserved before spawn, an ambiguous
request is charged and never replayed, runtime/source bytes are revalidated at execution
boundaries, and the aggregate roots the complete evidence inventory. The runner never invokes
Assay, rejects a physical `model.lock`, records no Assay decision, and cannot authorize release.

The hard new-call budget is:

| Phase | Maximum AGY calls |
| --- | ---: |
| Challenger design (`6 × 2`) | 12 |
| Challenger hints (`6 × 2 seeds × 2`) | 24 |
| First two whole-pair actor waves | 144 |
| Optional third actor wave (`14 × 2`) | 28 |
| **Total new calls** | **208** |

The historical ledger records 205 charged calls, so this plan can reach at most 413 of the 450
authorized calls and leaves 37 calls unused. Model names are requested routes only;
`backend_identity_attested` remains false.

The prospective outcome gates require a pooled unhinted rate in `[0.10, 0.90]`, at least eight
discordant pairs, a coverage-forced delta of at least `0.10`, positive leave-one-stratum-out
contrasts, and a first-attempt exogenous failure rate at most `0.15`. It also reports a one-sided
`(3!)^6 = 46,656` label-permutation sensitivity and a one-sided `2^6 = 64` stratum sign-flip
sensitivity, each gated at `p <= 0.05`. These are exact enumerations under strong candidate-label
exchangeability and stratum-contrast symmetry assumptions, respectively; neither is design-based
randomization inference.

Run only from the pinned SPADE/ProofPack/Assay stack after committing and auditing the exact runner
bytes:

```bash
SPADE_STACK=/absolute/path/to/spade-baseline-stack-20260901
cd "$SPADE_STACK/spade"

ASSURANCE_PYTHON="$SPADE_STACK/proofpack/.venv/bin/python"
V3_RUN="$PWD/.assay/spade-experiments/runs/spade-agy-pilot-baseline-google-pro-v3-1ac5e27ddad0f68baaa54bd9eb67a6950773226cd936afd1a890a475822b2746"
V4_RUN="$PWD/.assay/spade-experiments/runs/spade-agy-pilot-designer-prompt-treatment-google-pro-v4-8edc56d38e3502dd1e85db8b670b258ead9a4e1eddcd7d807e6a05e7b56df5fc"
PROXY_ROOT="$PWD/.assay/spade-coverage-forced-proxy"
PROXY_LEDGER="$PROXY_ROOT/shared-ledger"
PROXY_INTENT="$PROXY_ROOT/intent.json"

"$ASSURANCE_PYTHON" tools/run_spade_witness_qd_proxy.py intent-plan \
  --experiment-id spade-google-coverage-forced-matched-swap-v1 \
  --v3-run "$V3_RUN" \
  --v4-run "$V4_RUN" \
  --output-root "$PROXY_ROOT" \
  --shared-ledger-root "$PROXY_LEDGER" \
  --output "$PROXY_INTENT"

# Validation only: zero AGY calls.
"$ASSURANCE_PYTHON" tools/run_spade_witness_qd_proxy.py prepare \
  --intent "$PROXY_INTENT"

# Generate and locally screen challengers, then seal actor-plan.json.
"$ASSURANCE_PYTHON" tools/run_spade_witness_qd_proxy.py prepare \
  --intent "$PROXY_INTENT" \
  --execute \
  --acknowledge-new-call-cap 208

# Audit actor-plan.json before allowing actor calls.
"$ASSURANCE_PYTHON" tools/run_spade_witness_qd_proxy.py run \
  --intent "$PROXY_INTENT"
"$ASSURANCE_PYTHON" tools/run_spade_witness_qd_proxy.py run \
  --intent "$PROXY_INTENT" \
  --execute \
  --acknowledge-new-call-cap 208
```

### Result — incomplete, fail-closed

| Field | Result |
| --- | --- |
| Status | Incomplete; runner exited fail-closed with 12 of 36 pairs unresolved after all permitted waves |
| Intent / actor-plan / aggregate digests | `sha256:df1a06c7fb854d5267ec4d1e41cd44c1ffd229018bf0f5a0dcde512fa47c4c09` / `sha256:fc7989bfffb0851363137fe450e94cf11c8a4658b104f2f2729c08e98d843c2a` / none |
| New / global charged calls | 111 / 316 of 450 |
| Maximum / mean absolute quality gap | `0.125` / `0.0208333` |
| Actor call results | 90 closed calls: 53 nonempty responses and 37 outputs recorded as exact `empty_response` errors; no ambiguity |
| Pair closure | 24 resolutions (23 on attempt 1, one on attempt 2); 12 terminally unresolved pairs |
| Descriptive resolved prefix | 24/24 unhinted rewards and 24/24 hinted rewards were positive; 24 ceiling ties and zero discordant pairs |
| Coverage-forced delta | Not estimated; the panel is incomplete |
| Label-permutation / sign-flip sensitivity | Not run; no complete aggregate exists |
| Leave-one-stratum-out / runner-classified exogenous / parser diagnostics | Not run / first-wave `13/36 = 36.11%` (gate `<= 15%`) / zero returned-text parser or environment failures |
| Assay / release authorized / learner claim | Not run / false / none |

The terminal evidence is internally closed: all 111 request/result/ledger records match, whole-pair
retry isolation validates, and every persisted successful arm replays deterministically. The
failure is scientific rather than evidentiary. A post-run audit of the call-specific AGY logs found
that all 37 empty outputs followed `Print mode: soft-denying tool confirmation "RunCommand"`; they
were model tool-use attempts, not attested provider outages. The runner therefore misclassified
behavioral failures as retryable exogenous failures. That defect independently invalidates the
planned analysis, even aside from panel incompleteness and the fact that every retained pair hit the
task ceiling. The diagnostic logs were not bound into the original experiment seal, so they can
identify the harness defect but cannot retroactively reclassify or salvage its outcomes. Because the
panel is incomplete, the zero partial deltas are descriptive only and must not be pooled, manually
aggregated, or treated as an improvement estimate. The requested model routes remain
backend-unattested.

The run therefore establishes neither a coverage-forced portfolio association nor causal selector
benefit, QD or archive superiority, environment-quality improvement, learner improvement, or
backend identity. Before another run, the AGY boundary must make denied tool requests explicit and
score them once as model behavior, never retry them as transport loss. A new prospectively sealed
experiment also needs a non-ceiling task—and ultimately a compute-matched learner-lineage
experiment—to claim that SPADE itself improved.

The repaired boundary now captures AGY's structured event stream in an isolated per-call Gemini
directory and persists only digest-bound sanitized stream, stderr, log, and transcript receipts. It
treats a soft-denied tool selection as terminal zero-reward model behavior, treats possible
execution as fatal, and permits a retry only for an explicit provider failure before any response
ID. A separately sealed paid sentinel is still required to validate the installed AGY 1.1.23 wire
format before this adapter can support a new experiment.

### AGY 1.1.23 structured-evidence sentinel

`tools/run_agy_conformance_sentinel.py` implements that gate as a dedicated one-call protocol. It
is fixed to AGY's requested `gemini-3.7-flash-high` route and asks the model to select `RunCommand`
once for a harmless `touch` canary inside a fresh disposable directory. A pass requires the
structured stream, log, and transcript receipts to agree that AGY soft-denied the tool, that no
tool output or execution occurred, that the exact sealed `touch` command was selected, that both
the terminal response and transcript model content were blank, that the isolated policy config was
absent, and that the directory remained empty and was deleted. A text response is a terminal
`target_not_exercised` non-pass; prose accompanying a tool request or different tool parameters is
a hard non-pass; provider unavailability is terminal and inconclusive; ambiguous reservations
never replay. No result authorizes release or emits an Assay decision or `model.lock`.

The intent also recomputes and binds the complete prior closure: 111 request/result pairs and 112
ledger leaves through global ordinal 316. The sentinel reserves only global ordinal 317, leaving
133 of the authorized 450 calls. Plan creation and execution require the exact committed, clean
runtime and AGY 1.1.23 binary sealed into the intent. Post-call persistence and every replay
freshly hash both the Python and AGY executable bytes. Result wall-clock elapsed time must agree
with the monotonic duration within the sealed deterministic five-second tolerance, and a decision
cannot predate its result.

There is exactly one accepted artifact location: the sentinel output root is the prior proxy
output root's sibling named `spade-agy-conformance-sentinel-v1`, its ledger is the direct child
`shared-ledger`, and the intent is `intent.json` directly under that output root. Alternate or
duplicate roots are rejected during both construction and loading.

```bash
ASSAY_ROOT="$SPADE_ROOT/.assay"
PRIOR_ROOT="$ASSAY_ROOT/spade-coverage-forced-proxy-v2"
SENTINEL_ROOT="$ASSAY_ROOT/spade-agy-conformance-sentinel-v1"

# Seal the intent after the runner is committed. This inspects local artifacts/runtime only.
"$ASSURANCE_PYTHON" tools/run_agy_conformance_sentinel.py sentinel-plan \
  --prior-output-root "$PRIOR_ROOT" \
  --output-root "$SENTINEL_ROOT" \
  --shared-ledger-root "$SENTINEL_ROOT/shared-ledger" \
  --output "$SENTINEL_ROOT/intent.json"

# Zero-call validation; it does not create a run or ledger.
"$ASSURANCE_PYTHON" tools/run_agy_conformance_sentinel.py run \
  --intent "$SENTINEL_ROOT/intent.json"

# The only paid path. Do not run until the sealed intent has been independently audited.
"$ASSURANCE_PYTHON" tools/run_agy_conformance_sentinel.py run \
  --intent "$SENTINEL_ROOT/intent.json" \
  --execute \
  --acknowledge-new-call-cap 1
```

The remaining 134 calls in the 450-call authorization were not spent. An AGY-only inference rerun
cannot establish learner improvement because it contains no parameter update, same-checkpoint
treatment/control branches, or compute-matched learner lineages. A separately authorized narrow
proxy could test immediate no-tool hint responsiveness after the adapter is fixed, but that would
still not answer whether SPADE training improved.
