# Controlled marginal teachability

Status: research design and offline integrity primitives, not a sealed confirmatory plan. The
implementation can construct and validate schedules and evidence records; it does not execute
learner updates or claim that the intervention improves training. A powered, externally witnessed
multi-run study is still required.

## Motivation

SPADE rewards an Environment Designer using the difference between Reasoning Agent returns with
and without a privileged hint. In the published protocol, the two arms use independently seeded
environment resets and the reward is a difference of arm means. The current implementation also
resets the ordinary actor without an explicit seed, derives hinted seeds with Python's
process-randomized `hash()`, and subtracts the resulting unpaired means.

That score mixes the hint intervention with environment-instance variation. More importantly, even
a perfectly estimated hint effect answers only this question:

> Does a strategy hint help the current actor solve this environment now?

It does not answer the question a curriculum designer ultimately cares about:

> Does training on this environment improve the actor on unseen, related tasks?

The first question is useful as a cheap frontier screen. The second requires a controlled training
intervention.

## Novelty boundary

The broad ideas are established prior art. PAIRED and later unsupervised-environment-design work use
regret to target learning frontiers; MBeED/MBeDED reward teachers using marginal post-training
benefit and combine it with behavioral diversity; PACE uses provisional learner updates; and QD
archives are well established. Relevant primary sources include:

- [SPADE](https://arxiv.org/abs/2608.19197)
- [PAIRED](https://arxiv.org/abs/2012.02096)
- [MBeED/MBeDED](https://ojs.aaai.org/index.php/AAAI/article/view/34008)
- [PACE](https://arxiv.org/abs/2605.01358)
- [Prioritized Level Replay](https://arxiv.org/abs/2010.03934)
- [ACCEL](https://arxiv.org/abs/2203.01302)
- [OMNI-EPIC](https://arxiv.org/abs/2405.15568)

The narrower candidate contribution proposed here is a controlled estimator for
ProofPack-qualified, SPADE-compatible executable interactive environments:

1. two learner clones start from the exact same checkpoint;
2. treatment and control receive equal optimizer, example, and token budgets;
3. the treatment micro-update uses the candidate environment while the primary control uses a
   draw from one frozen matched-environment distribution;
4. both are evaluated without hints under common random numbers;
5. evaluation is split from construction and training at the source/template/family level, not
   merely by integer seed, and includes both same-family and sibling-task holdouts; and
6. a lower confidence bound across independent outer training replicates supplies curriculum
   quality, with uncertain or invalid trials abstaining.

Do not describe this as the first learning-progress reward, first causal teachability score, or
first behavioral UED archive. The candidate claim is the controlled design and its application to
LLM-written executable environments; it remains unsubstantiated until the MBeED comparison and a
broader literature review are complete.

## Stage 1: paired hint-effect prerequisite

Before branching learners, fix the cheaper frontier measurement.

For every accepted environment, derive a sealed even-sized pair schedule from SHA-256 domain
separation over the run seed, rollout, exact environment digest, and pair index. Use one locked hint
for every pair; the resulting estimand is the effect of that specific hint, not the average quality
of the hint generator. Each pair runs hinted and unhinted arms through the same code path on fresh
environment instances with the same explicit environment and policy-sampling seeds. Exactly half
the pairs are hint-first and half unhinted-first.

Persist the hint, its generation prompt/model/decoding metadata, a semantic leakage-check receipt,
the success-scoring code digest, and the schedule before the first outcome. Both arms must begin
from identical raw observations, and each result must bind its full rollout/trace digest. Exogenous
provider outages and ambiguous executions make a pair missing; they are never imputed as task
failures. Candidate- or hint-induced overlength, parser, resource, or runtime failures are quality
failures rather than ignorable missingness. A normal terminal loss or time-limit truncation remains
a valid zero outcome. Persist every attempted slot, and do not replace one after any outcome from
that slot is visible.

For binary success, record the paired table:

- `n10`: hinted success, unhinted failure;
- `n01`: unhinted success, hinted failure;
- ties; and
- `ATE_success = (n10 - n01) / K`.

Use the exact one-sided McNemar tail and Holm correction over every precommitted environment in the
sealed hypothesis family, including incomplete and invalid entries in the multiplicity count. Fix
`K` from a power calculation before execution and verify that rejection is mathematically possible
at the first Holm threshold; the current four-play default is not adequate for such a family.

McNemar tests the zero-effect symmetry null. Requiring the observed `ATE_success` to meet an
operational floor is therefore only a screening rule; it does not show that the true effect exceeds
that floor. Any confirmatory floor claim must instead require a multiplicity-adjusted one-sided
lower confidence bound for the paired risk difference to exceed the precommitted floor. Strong
evidence in the negative direction is a separate `harmful`/quality-reject outcome, not abstention.
Reserve abstention for incomplete or statistically uncertain evidence, and omit abstained proposer
samples from the loss rather than assigning zero and then centering it.

This stage improves attribution but is not evidence of learning gain.

## Stage 2: controlled marginal teachability

Define the primary control distribution `Q` before candidate scoring. `Q` is a frozen pool of
ProofPack-qualified alternative environments matched on skill, action format, horizon, and sealed
baseline-success bin, while excluding the candidate's source/template/lineage family. For candidate
`e` and independent outer replicate `r`, draw `C_r ~ Q` before outcomes, start two learners from
checkpoint `theta_r`, and also evaluate the unchanged checkpoint:

```text
treatment: theta_r --equal-budget update on e-----------> theta_r^T
control:   theta_r --equal-budget update on C_r ~ Q-----> theta_r^C
baseline:  theta_r -------------------------------------> theta_r
                                                         |
                              +-- sealed unhinted evaluation panel
```

The primary estimand is the finite-panel average of `Y(update(e)) - Y(update(C_r))`, with
`C_r ~ Q`. A no-op/sham update answers a different question and may be reported only as a separately
powered secondary contrast. Superiority to the matched control is insufficient by itself: the
treatment must also improve over the unchanged checkpoint, and a separately sealed regression
panel must rule out material catastrophic forgetting.

The branch manifests must bind:

- base-checkpoint digest and actor/runtime revisions;
- optimizer and scheduler configuration digests;
- exact optimizer-step, training-example, and token budgets;
- candidate and control batch digests;
- identical initial optimizer state, training seed schedule, minibatch order, and compatible PRNG
  streams, with any nondeterministic-kernel deviation recorded;
- train, score-construction, calibration, and confirmation panel identities and seeds;
- common-random-number schedule and evaluation order;
- environment, ProofPack receipt, and lineage/family digests;
- content-addressed execution receipts containing realized—not only planned—step, example, and
  token counts plus input/output checkpoint digests; and
- complete treatment/control/baseline outcomes bound to those output checkpoint digests.

Commit the panel-construction algorithm, frozen source pools, relationship taxonomy, and split rule
before candidate generation; seal concrete candidate-specific panel digests after cohort lock and
before scoring. A same-family holdout shares the declared task family but uses a distinct instance,
source artifact, template realization, and seed. A sibling holdout belongs to a distinct task family
under the sealed relationship taxonomy. Score-construction, calibration, and final-confirmation
panels must otherwise be disjoint by source/template/task family as well as seed, and their payloads
remain inaccessible to the designer. Evaluation prompts contain no privileged hint. All three
checkpoints execute the same evaluation schedule. Any base-checkpoint mismatch, unequal realized
budget, missing required outcome, runtime drift, or artifact mismatch invalidates the replicate.
Predeclare which infrastructure failures are exogenous; intervention-induced OOM, overlength, or
instability counts against candidate quality and cannot trigger replacement.

An independent outer replicate means a separately initialized upstream learner run and seed, not a
different checkpoint taken from the same training trajectory. Within each replicate, normalize
returns using a sealed task-specific rule, average units within sibling family, and weight sibling
families equally. Compute same-family and sibling-task contrasts separately. Repeated environments
or seeds from one trained checkpoint reduce measurement noise but are not independent training
replicates.

The primary quality rule requires precommitted one-sided 95% lower confidence bounds on both the
mean sibling-task treatment-minus-control contrast and the mean sibling-task treatment-minus-
baseline gain across independent outer replicates. A noninferiority bound on the regression panel
is an additional gate. These are claims about fixed sealed panels unless the design separately
samples enough task families for population inference. Same-family gain is a secondary mechanism
check. Before any run, a variance pilot must seal the outer replicate count `R`, confidence-interval
method, effect and noninferiority floors, alpha allocation, retry rules, and call/token caps. A
candidate receives positive quality only when every bound passes; negative evidence is harmful,
while incompleteness or an inconclusive bound abstains.

### Current implementation scope

`spade.core.paired_teachability` is an offline integrity layer. It constructs content-addressed
pair/family plans, validates paired outcomes, records screening decisions, and validates declared
controlled-branch plans and caller-attested execution receipts for internal consistency. Its
controlled estimates summarize one outer replicate only. They are descriptive, do not weight
multiple sibling families for population inference, and do not implement the baseline/regression
gates or the outer-replicate confidence bound above. The module also does not generate hints,
execute a trainer, attest provider identity, prove that a receipt is truthful, or prove that a plan
predated its outcomes. Backend integration, a trusted runner, and pre-outcome anchoring in an
append-only external witness are separate prerequisites.

## Falsifiable comparison

First freeze one candidate cohort and three mutually disjoint family/template panels: one for
constructing each score, one for measuring the larger-update calibration target, and one held back
for final confirmation. Then score every candidate using four methods:

1. a commit- and config-pinned canonical full SPADE credit, with raw unpaired hint regret also
   reported as a component;
2. paired/counterbalanced hint effect;
3. a commit- and tuning-procedure-pinned MBeED marginal-benefit implementation; and
4. controlled marginal teachability.

Independently perform a larger candidate-specific update and measure realized unhinted gain on the
calibration panel. Precommit a family-clustered paired bootstrap or permutation analysis for the
dependent correlations, including tie handling and `k`. The primary calibration endpoint is paired
improvement in rank correlation with realized sibling-task gain. Secondary endpoints are top-k
positive-gain precision, estimator variance, abstention rate, and realized gain per learner token.

The claim fails if controlled marginal teachability does not beat MBeED—not merely current SPADE—on
out-of-wave gain prediction with a paired confidence interval. Its transfer interpretation fails if
the effect vanishes on sibling tasks.

Seal the number of candidates and outer runs, all retry/stopping rules, and alpha allocation across
the calibration, Stage 2 selection, and curriculum hypotheses; outcome-dependent waves or reserve
substitution are not allowed. Only after calibration should a curriculum study randomize training
runs between SPADE credit and controlled marginal teachability and evaluate once on the untouched
confirmation panel. A behavioral quality-diversity archive is a subsequent orthogonal factor,
producing a `2 x 2` credit-by-archive experiment. Archive coverage without unseen-family return per
compute is not an improvement.

## Role of `agy`

`agy` can provide Google-model environment generation, hint generation, and requested-route
evaluation. It cannot clone or update the underlying subscription model, so an `agy`-only run can
test qualification yield, paired hint effects, leakage, and environment diversity, but cannot
establish controlled marginal teachability or training improvement. Until `agy` can attest the
resolved model/checkpoint and executed sampling seed, its evaluation results remain descriptive
rather than evidence of a fixed policy identity.

The full experiment therefore needs a trainable learner backend. Slime is the nearer candidate,
while `agy` may remain the sealed Environment Designer or requested-route external evaluation
panel. Tinker is currently blocked as a baseline until its regret call path, required reward inputs,
and hinted token-input path are repaired and verified. Before using either trainer for a claim, the
baseline must have deterministic seed schedules, exact checkpoint/state resume, equal arm budgets,
and working reward plumbing across the chosen backend.

## Release boundary

These artifacts are research evidence. A positive statistical signal does not authorize a model
release, emit a `model.lock`, or establish causal independence between generated environment
families. Release claims remain subject to an externally witnessed experiment plan and Assay's
separate registration and authorization controls.
