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

The repaired boundary captures AGY's structured event stream in an isolated per-call Gemini
directory and persists bounded canonical stream, stderr, log, transcript, policy-transition, and
sandbox-invocation receipts. These local hashes establish deterministic self-consistency, not
authentication or tamper-proof provenance. The generic adapter still treats any possible tool
execution—including a tool error with output—as fatal. Only the sentinel-specific decision may
recognize the exact calibrated soft-denial shape; generic tool-error handling is not weakened.

### Terminal AGY 1.1.23 sentinel

The v1 sentinel reserved and spent global call 317. During that call AGY replaced its sealed
1.1.23 executable with 1.1.24, so the runner correctly failed closed before writing `result.json`.
Its terminal artifact state is an immutable request, matching global-0317 ledger entry, and four
sanitized stream/stderr/log/transcript receipts, with no result or decision. That request is
ambiguous under the protocol and must never be replayed. The receipts are useful only for
prospective calibration: they show the raw `run_command` name, the canonical
`{"CommandLine":"touch SPADE_AGY_SENTINEL_TOOL_MUST_NOT_EXECUTE"}` parameter shape, two response
IDs, an ACTIVE-to-ERROR tool transition, a log soft denial, no tool output, and no transcript tool
execution. They do not retroactively make v1 pass.

### Terminal AGY 1.1.24 sentinel v2

V2 reserved and spent global call 318 and then closed durably as failed. It observed the calibrated
tool-denial wire shape, but its isolated config changed from absent to created during the call. An
endpoint-only final snapshot cannot prove which bytes a same-UID child consumed earlier, so the
runner correctly rejected `policy_config_changed_during_call`. V2 is terminal and must never be
replayed. V3 binds the exact 13-leaf v2 closure—including its zero-byte writer lock, request,
ledger entry, four v1 evidence receipts, result, workdir observation, and decision—with anchor
`sha256:7afe62457048313a081374572a324de4e18b309eb13730feebc1cfd03ddbaaf2`.

### Prospective AGY 1.1.24 structured-evidence sentinel v3

`tools/run_agy_conformance_sentinel.py` implements a distinct one-call protocol at global ordinal
319. Its only accepted location is the closed v2 root's sibling
`spade-agy-conformance-sentinel-v3`; alternate roots and ledgers fail closed. Any durable global-319
reservation is terminal under every disposition and is never replayed.

V3 precreates exactly this public 76-byte config before AGY starts:

```json
{"userSettings":{"globalPermissionGrants":{"allow":[],"deny":[],"ask":[]}}}
```

Its SHA-256 is
`293b65b15673320856ca9061c64b55a99dd0fe495f1a4ba2af25ed9c71391a72`.
The Gemini root and config directory must be current-UID mode `0700`; the regular config file must
be current-UID mode `0600`, `nlink=1`, and retain the same device, inode, size, times, and exact
bytes through process-group reap. Parsing is strict UTF-8 JSON with duplicate-key and non-finite
constant rejection, a 4 KiB limit, exact root/settings/grant-bucket inventories, bounded arrays,
and all `allow`, `deny`, and `ask` arrays empty. This attests a fixed file with no explicit grants
and AGY's `no shared config permissions` log report. It does not independently establish AGY's
default-policy semantics or prove which bytes AGY interpreted.

The paid invocation is wrapped by the pinned `/usr/bin/sandbox-exec` binary
(`sha256:7fa7df193f26e32cc740e38d55eae13b31a9a98165b9fd9d03473a96e5b37284`)
and a byte-exact parameterized profile
(`sha256:e5e5567b476d9186847ad8f14e066912ab8d0c68af3d8cb89d8b0a564f6f37c9`).
The profile denies fork, denies exec except the pinned AGY pathname, denies all writes globally,
then allows workdir-subpath and `/dev/null` writes before specifically denying the workdir/Gemini/
config identities and canonical `/private/tmp` ancestors. It binds literal `CONFIG_DIR` and
`CONFIG_FILE` denials as well as the config subtree. `stdin` is `/dev/null`, non-stdio file
descriptors are closed, a fresh session/process group is used, and `TMPDIR` is the private
`<workdir>/tmp`. AGY and sandbox bytes are rechecked immediately before spawn and after reap.

Before any global-319 request or ledger entry is written, a no-AGY/no-network local self-probe runs
the same profile with the sealed Python executable as the sole exec exception. It requires
workdir/tmp/app-data writes to succeed and requires `EPERM` or `EACCES` for config mutation,
unlink, creation, chmod and renames; workdir/Gemini renames; inward and outward hardlinks; symlink
mutation; outside writes; fork; and non-pinned exec. It then verifies the exact config transition,
deletes its private artifacts, and revalidates an empty current-UID mode-`0700`, `nlink=2` direct
child of `/private/tmp`. Its sanitized receipt is bound into the eventual request. The adapter
constructs final argv, environment, cwd, stdin/stdout/stderr, FD and session options first; derives
an invocation receipt from those actual values; and invokes a synchronous reservation callback
immediately before subprocess creation. A deterministic local setup or probe failure therefore
does not consume global 319.

New calls emit structured evidence v3 and sanitized log v3; exact legacy v1 and prospective v2
readers remain frozen for replay. New receipts persist no raw log/transcript, private Gemini path,
AGY-generated config value, or raw private-config digest. The fixed public empty-grants payload and
its digest are intentionally sealed. Receipt digests are recomputable local consistency checks,
not signatures.

The v3 decision can pass only when every calibrated denial signal agrees: two distinct response
IDs; exact raw `run_command` and parameter receipt; ACTIVE then ERROR with error absent then
present; no tool output, approval, subagent trace, transcript execution, prose, canary, surviving
descendant, capture failure, explicit provider-failure marker, or detected nested-sandbox failure;
the expected `RunCommand` log denial; a blank successful terminal result; the exact unchanged
empty-grants transition; the preflight receipt; actual-derived sandbox invocation receipt; exact
runtime bytes; and durable workdir observation followed by deletion.

Threat limits are sealed in intent, result, and decision: the parent/single-writer process is
trusted and no hostile concurrent same-UID process is assumed; `(allow default)` plus the explicit
rules confines process creation and file writes but does not isolate reads, network, or Mach
services; pathname and pre/post hashes cannot exclude a transient same-UID executable replacement;
and a pass is one requested-route adapter observation, not backend identity, model quality,
experimental effect, learner improvement, or release evidence. Even a pass only becomes eligible
for separate downstream review. It always records
`future_paid_google_experiments_authorized=false`, `release_authorized=false`, no Assay decision,
and no `model.lock`.

```bash
ASSAY_ROOT="$SPADE_ROOT/.assay"
PRIOR_ROOT="$ASSAY_ROOT/spade-agy-conformance-sentinel-v2"
SENTINEL_ROOT="$ASSAY_ROOT/spade-agy-conformance-sentinel-v3"

# Seal only after the runner and adapter are committed and independently audited.
"$ASSURANCE_PYTHON" tools/run_agy_conformance_sentinel.py sentinel-plan \
  --prior-output-root "$PRIOR_ROOT" \
  --output-root "$SENTINEL_ROOT" \
  --shared-ledger-root "$SENTINEL_ROOT/shared-ledger" \
  --output "$SENTINEL_ROOT/intent.json"

# Zero-call validation; it creates no run or ledger.
"$ASSURANCE_PYTHON" tools/run_agy_conformance_sentinel.py run \
  --intent "$SENTINEL_ROOT/intent.json"

# Prospective paid v3 path. Do not run without a separate post-audit authorization.
"$ASSURANCE_PYTHON" tools/run_agy_conformance_sentinel.py run \
  --intent "$SENTINEL_ROOT/intent.json" \
  --execute \
  --acknowledge-new-call-cap 1
```

Global usage is 318/450 before v3; 132 calls remain. The single global-319 sentinel is
scientifically interpretable for the narrow adapter-conformance question only after independent
review of the frozen implementation and intent; if separately authorized and spawned, it leaves
131. This document and any v3 result do not authorize that call or any later paid call. An AGY-only
inference rerun cannot establish learner improvement because it contains no parameter update,
same-checkpoint treatment/control branches, or compute-matched learner lineages.
