# Slime learner branch assay

This plan-only harness specifies a future causal comparison between the sealed coverage/witness curriculum and its
redundant-history control. It is a Slime Qwen3-8B experiment plan; it does not claim equivalence with the
separate Tinker assay. AGY Google artifacts supply only the already sealed environment, hint, and
design provenance. Training rollouts must come from the current Slime checkpoint through its in-job
SGLang server, and heldout evaluation must come from each run's final Slime checkpoint through a
dedicated SGLang server.

The local preparation commands are offline and do not start training, invoke AGY, or contact a
provider. The plan is explicitly in `plan-only-hold` state because no artifact-backed trusted job
runner, receipt collector, or heldout evaluator has been implemented.

```bash
PYTHONPATH=. python tools/run_spade_learner_branch_assay.py materialize \
  --actor-plan /path/to/sealed-fc798/actor-plan.json \
  --output-dir /path/to/new/learner-pools

PYTHONPATH=. python tools/run_spade_learner_branch_assay.py plan-slime \
  --bundle /path/to/new/learner-pools \
  --output-dir /path/to/new/slime-plan \
  --remote-pool-root /scratch/spade-assay/pools \
  --remote-output-root /scratch/spade-assay/runs \
  --hf-checkpoint /scratch/models/Qwen3-8B \
  --hf-checkpoint-digest sha256:<verified-tree-manifest-digest> \
  --reference-checkpoint /scratch/models/Qwen3-8B_torch_dist \
  --reference-checkpoint-digest sha256:<verified-tree-manifest-digest> \
  --runtime-image registry.example/spade@sha256:<image-digest> \
  --runtime-image-digest sha256:<image-digest> \
  --spade-source-revision <clean-40-hex-commit> \
  --source-root .
```

Do not substitute a path hash, mutable image tag, or placeholder digest. The two checkpoint digests
must identify independently verified artifact manifests, and the source revision must contain the
critical-file bytes recorded in the plan. `verify-slime-plan` rechecks the immutable 12-spec
inventory before it is copied to the eight-GPU Linux/CUDA worker.

Each launch spec contains the exact environment and `train_spade_slime` argv for one job. The
schedule has six paired seeds and alternates which arm occupies the first job slot. This is
deterministic counterbalancing, not randomized treatment assignment. Each job has 16 rollouts of 12
games by 16 trajectories, or 3,072 requested actor episodes. The strict launcher flag aborts before
an optimizer step unless every rollout has exactly 192 real actor episode groups and zero padding,
play failure, or filtered trajectory.

Heldout evaluation covers all 12 environment-disjoint v4 games with 16 predeclared replicates per
game. Every replicate seals a distinct environment seed and sampling seed shared across arms. The
request parameters, action extraction, termination, truncation, and success semantics are part of
the plan. Success is derived as `terminated && terminal_reward > 0`; a receipt cannot supply an
independent `won` assertion.

The current validator can check the internal consistency of a manually assembled receipt in the
schema exercised by `tests/test_slime_branch_assay.py`:

```bash
PYTHONPATH=. python tools/run_spade_learner_branch_assay.py validate-slime-results \
  --plan /path/to/slime-plan \
  --results /path/to/slime-results.json
```

Receipt checking fails closed on any checkpoint-chain break, unchanged final checkpoint, external actor, missing/extra schema field,
seed drift, incomplete rollout, padding, or pair whose real sequence-token or loss-mask-token count
differs by more than 5%. It also requires nonzero action/loss-mask training tokens and cross-checks
zero episode-group populations against zero sample and token counters. It may report the predeclared
sign-flip calculation for inspection, but the interpretation is exploratory only. That test assumes
paired differences are exchangeable under the sharp null; it is not described as design-based
because job-slot assignment is not randomized.

The validator always returns `hold-no-trusted-artifact-collector` and `causal_evidence:
not-established`; it has no code path that can authorize an improvement claim. Generating, testing,
or manually satisfying this harness is not evidence that the learner improvement works. A future
trusted, artifact-backed runner/collector/evaluator must be implemented and audited before the hold
can be lifted.
