# Tinker learner branch assay

This protocol is a preparatory, plan-only HOLD for an exploratory learner comparison. Its topology
would train both arms from the same asserted optimizer-bearing Tinker state, hold submitted training
compute and randomness fixed within each pair, and evaluate each final branch checkpoint directly.
It cannot support a causal learner claim: this Tinker SDK exposes bare state/checkpoint paths but no
authenticated artifact digest or signed provider receipt, and the runner cannot authenticate the
source of a user-entered assignment seed. Live execution is disabled in this protocol version. It
does not use AGY as the learner, rollout actor, or evaluator.
AGY Google artifacts are provenance for the already sealed environments and hints only. No new AGY
call is part of this protocol.

The assay is implemented but has not been sealed or run. No Tinker, AGY, or training call was made
while adding it.

## Exact question and claim boundary

The treatment is the immutable 12-game `coverage_forced` pool. The control is the immutable,
basename-aligned 12-game `redundant_historical` pool. Six independent stochastic training-run pairs
start from one common zero-update Qwen3-8B LoRA state, including its optimizer state. Within a pair,
the two arms receive identical environment and sampling seeds. Six independent fair coin flips assign
the curricula to slots; execution order is separately counterbalanced. The user-entered assignment
seed is separate from the stochastic learner/evaluation seed: changing only the assignment seed
cannot change any pair, training, or held-out sampling seed. Do not call this v1 design randomized:
even if a user reports drawing the seed independently before outcomes, the runner cannot authenticate
that source or chronology.

The endpoint is the paired difference in mean terminal reward on 12 environment-disjoint, qualified
v4 games (`c002` and `c008` through `c018`), with eight paired seeds per game and no hints. Each
checkpoint would be sampled by Tinker from that same branch's final weights at update 16. The
two-sided sign-flip sensitivity analysis enumerates all `2^6 = 64` label flips under an explicit
within-pair exchangeability assumption. Its numerical threshold is `p <= 0.05` and an average reward
gain of at least `0.05`; it is not a design-randomization distribution.

A favorable sensitivity result is descriptive only and never emits a causal learner decision,
`supports_narrow_claim`, promotion, release, or a model lock. Six pairs are the discreteness minimum
that can attain a two-sided sign-flip value below 0.05, not a variance-based power guarantee.

## Fixed topology and resource ceilings

Each branch performs 16 updates. Every update collects exactly eight two-turn-capped trajectories
from each of the 12 games: 96 complete episode datums. Missing, duplicated, provider-error, or
substituted trajectories fail the whole pair; there is no favorable-outcome filtering or upsampling.
Rewards are normalized within each eight-trajectory game group. Every episode datum is right-padded
to 4,096 next-token positions, with zero advantage on padding.

This gives exact submitted training accounting:

- 393,216 positions per update;
- 6,291,456 positions per branch;
- 75,497,472 positions across 12 branches;
- 192 forward/backward calls and 192 optimizer-step calls in the complete assay.

This is an exact submitted-position and update-count match, not a claim that Tinker's private server
uses identical physical FLOPs. The two arms can still differ in non-padding token mix.

The hard logical sampling ceiling is 48,397 calls: at most two actor turns for each of 18,432
training episodes, at most the native ten turns for each of 1,152 held-out episodes, one restored-state
canary per branch, and one common-state canary. Each call permits at most 1,024 completion tokens.
One semaphore shared by both simultaneously executing branches caps pair-wide sampling concurrency
at 16. SDK-internal transport attempts are not separately observable, so
the ceiling is on logical `sample_async` operations, not HTTP retries.

## Evidence and failure policy

The intent binds the actor plan, immutable pool manifest, SPADE source files, pinned Tinker cookbook
gitlink and renderer files, Qwen3-8B model ID, renderer, optimizer, seeds, schedules, endpoint, analysis,
and resource ceilings. The planned common-state request would be persisted before state creation. Its
local receipt would bind the SDK-returned URI, SDK version, supported-model inventory digest, and a
deterministic zero-step token canary. These are local-process self-attestations, not authenticated
proof that the remote object contains a particular optimizer state or checkpoint. The installed SDK
exposes no provider-signed content digest or artifact receipt.

The endpoint is fixed to
`https://tinker.thinkingmachines.dev/services/tinker-prod`; live commands expose no endpoint override,
and the intent, common-state request/result/receipt, and every branch runtime attestation bind it.
The runtime seal also binds the Python executable/version, the Tinker, PyTorch, and NumPy wheel
`RECORD`, `METADATA`, versions and package entrypoints, plus the pinned cookbook and SPADE runtime
source files. Missing distribution records fail before sealing or live access.

Each completed pair contains exactly two request files, two execution receipts, two final-checkpoint
held-out scores, and one pair completion seal. Execution receipts bind all 16 update audits and all 16
optimizer-bearing/sampler checkpoint receipts; the pair seal roots both execution and score digests.
The aggregate roots all 12 execution digests, all 12 score digests, and all six pair seals. Unknown,
orphaned, symlinked, partial, out-of-order, or extra artifacts fail validation.

Each held-out outcome binds the public parsed action sequence, a digest of every raw response-token
sequence, any terminal parser-failure turn, and a trajectory digest. Offline validation re-executes
those actions from the sealed environment seed and rejects any reward, termination, truncation, turn,
or topology mismatch. It does not persist private reasoning text. Every train and held-out operation
reads a digest-checked game into an immutable in-memory snapshot, then rehashes its path afterward.
The complete pool and all lazily imported runtime sources are revalidated before and after each
branch, so mid-run source drift makes the reserved pair terminal rather than mixing environments.

There is intentionally no retry after a common-state or pair reservation. A crash or provider failure
after a pair reservation makes the experiment terminal. This v1 protocol does not authorize resealing
the same comparison after partial learner outcomes; a later protocol would have to disclose and account
for the failed attempt. Intent sealing writes a source-derived canonical lineage lock, and validation
rejects every unknown, alternate, or symlinked artifact in that root. The runner refuses another
intent for that actor-plan/pool lineage even before common-state creation. This is repository-local
protection only: copying or cloning the repository can bypass it, so a future live protocol needs an
external write-once authorization registry. Local
SDK/import/renderer checks and a read-only capability check occur before
pair reservation. A common-state failure occurs before learner outcomes, but its persisted ambiguity
still forbids reuse of that run root.

Generated environments were previously qualified by ProofPack V0-V4. ProofPack is not the rollout
actor and is not substituted for the learned checkpoint. The runtime preserves episode-local Python
and NumPy global RNG state around generated-environment operations so concurrent pairs cannot corrupt
their paired seeds.

## Deployment blockers

There is no authorized live sequence for this protocol version. `prepare-base` and `run-pair` fail
closed even with `--allow-live` and a credential. A successor protocol would require, at minimum:

- explicit user authorization for the projected Tinker spend;
- `TINKER_API_KEY` in the live process environment;
- Python 3.11 or newer with the Tinker SDK, PyTorch, and the pinned cookbook installed;
- the `tinker-cookbook` submodule initialized at the repository gitlink;
- the live capability receipt explicitly lists `Qwen/Qwen3-8B`;
- adequate Tinker quota for the sealed logical-call and token ceilings;
- the canonical source-derived `.assay/spade-tinker-learner-<actor>-<pool>-v1` root absent and enough
  durable storage for its write-once intent, lineage lock, checkpoints, and evidence.

`Qwen/Qwen3-4B-Instruct-2507` is deliberately not accepted: it is retired. The fixed default is the
currently supported `Qwen/Qwen3-8B`, whose renderer exists in the pinned cookbook. If the service no
longer advertises it, this protocol fails before training rather than silently substituting a model.

## Offline preparation

Materialize the shared pools from the authorized `df1a...` intent, `fc798...` actor plan, and v4
cohort. The output directory must not already exist:

```bash
python tools/run_spade_learner_branch_assay.py materialize \
  --actor-plan <coverage-forced-run>/actor-plan.json \
  --source-intent <coverage-forced-root>/intent.json \
  --output-dir <fresh-absolute-pool-root>

python tools/run_spade_learner_branch_assay.py verify \
  --bundle <fresh-absolute-pool-root>
```

The expected bundle has 12 treatment games, 12 control games, and 12 held-out games. Its schedule ID
is `spade-learner-branch-df1a06c7fb85-fc7989bfffb0-v1`. The materializer verifies all candidate,
qualification, selection, code, environment, and source-manifest digests and proves held-out
environment-digest disjointness.

After installing the pinned runtime, a user may record an assignment seed and a separate stochastic
sampling seed to exercise and validate the offline plan. The v1 seal records that assignment
provenance is unauthenticated and permits only assumption-based sensitivity analysis. The command prints the only
accepted output root if a different path is supplied; the intent must be `<canonical-root>/intent.json`:

```bash
python tools/run_spade_tinker_branch_assay.py seal-intent \
  --actor-plan <coverage-forced-run>/actor-plan.json \
  --pool-manifest <fresh-absolute-pool-root>/learner-pools-manifest.json \
  --output-root <repo>/.assay/spade-tinker-learner-<actor>-<pool>-v1 \
  --intent <repo>/.assay/spade-tinker-learner-<actor>-<pool>-v1/intent.json \
  --assignment-seed <recorded-integer> \
  --sampling-seed <independent-stochastic-seed>

python tools/run_spade_tinker_branch_assay.py validate \
  --intent <fresh-absolute-intent-path>/intent.json
```

Both commands above make zero provider calls. Once the intent and lineage lock are sealed, this
protocol cannot reseal that source comparison. Any changed source/runtime requires an explicitly new
protocol and disclosure; it cannot silently create a second expensive run under another output path.

## Live execution status

HOLD. The CLI retains `prepare-base` and `run-pair` only as tested adapter seams, but its live guard
unconditionally rejects both. Do not invoke Tinker for this v1 plan. A successor must add externally
authenticated, prospectively acquired assignment entropy; provider-authenticated optimizer-state and
checkpoint identities; and an external one-run lineage authorization that survives repository clones.

The separate Slime assay is documented in `docs/slime-learner-branch-assay.md`. It is a replication
path with a different backend, not interchangeable evidence for this exploratory Tinker plan.
