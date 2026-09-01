<h1 align="center">SPADE &#9824;</h1>

<h2 align="center"> Self-Play in Adaptive Synthetic Executable Environments </h2>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/spade-rl/spade" target="_blank"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-spade--rl-000000?logo=github&logoColor=white&color=000000"/></a>
  <a href="https://huggingface.co/spade-rl" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-fcd022?color=fcd022&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2608.19197" target="_blank"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-2608.19197-b31b1b?logo=arxiv&logoColor=white"/></a>
  <a href="LICENSE" target="_blank"><img alt="License"
    src="https://img.shields.io/badge/License-MIT-green.svg"/></a>
</div>

<p align="center">
  <a href="https://huggingface.co/collections/spade-rl"><b>&#129303; Models &amp; Data</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/papers/2608.19197"><b>&#9824; SPADE on Hugging Face</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2608.19197"><b>&#128196; Paper</b></a>
</p>

## Updates

* 20/08/2026: &#127881; We release our [paper](https://arxiv.org/abs/2608.19197) on arXiv, together with the SPADE checkpoints (4B / 8B / 30B-A3B, games and tool use), the grounding corpora, and the generated environments on [Hugging Face](https://huggingface.co/spade-rl).
* 11/08/2026: &#127881; We release our self-play codebase and the static GPT-5.5 environment corpus.

## Introduction

Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches draw their reward signal from fixed, hand-built pools of environments that stop adapting once the learner masters them.

We introduce SPADE, a self-play framework where a single language model learns in two roles: an **environment designer that writes complete multi-turn environments as executable Python with `reset()` and `step()` interfaces, and a reasoning agent that learns by acting in them**. The designer is trained with **hint-based regret**, the gap between the agent's return with and without a privileged hint, which steers generation toward environments at the agent's capability frontier while keeping them feasible. Through this loop, SPADE generates an **_adaptive curriculum_** that keeps moving with the learner instead of saturating.

Applying SPADE to Qwen3 models at 4B, 8B, and 30B-A3B scale in two settings, cognitive games and multi-turn tool use, we observe the designer produce progressively harder, more interactive environments and the agent improve on held-out math, science, code, and procedural-reasoning benchmarks past the saturation point of fixed-environment baselines. These results suggest that making environment design itself a learnable component is a promising direction for open-ended self-improvement.

## Architecture

<p align="center"><img src="assets/spade_framework.png" width="90%" /></p>

SPADE trains one shared policy that plays both roles. Each cycle, the environment designer samples grounding context from a pretraining corpus and an environment memory, then writes a complete executable environment with a privileged hint; generated code passes structural and runtime validation before entering the training pool. The reasoning agent plays each environment with and without the hint: task return trains the agent role, hint-based regret trains the designer role, and per-role advantage normalization keeps the joint update stable. The backend-independent orchestration lives in `spade/core/`; distributed training uses the Slime/SGLang integration under `spade/slime/` (SGLang inference, Megatron-LM policy updates, Ray orchestration), with a Tinker integration under `spade/tinker/`.

## Supported Models

SPADE is model-agnostic: the training loop only needs a chat-capable policy, so any
model your backend can fine-tune works. The table lists the popular families; **bold**
entries are the ones we train in the paper.

| Model family | Models |
|---|---|
| **Qwen3** | **`Qwen/Qwen3-4B-Instruct-2507`**, **`Qwen/Qwen3-8B`**, **`Qwen/Qwen3-30B-A3B-Instruct-2507`**, `Qwen/Qwen3-32B` |
| **Qwen3.5** | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-35B-A3B` |
| **GPT-OSS** | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` |
| **Nemotron** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| **GLM** | `GLM-5.2`, `GLM-5.3` |

For the complete set, see the [Miles model list](https://miles.radixark.com/docs/models)
for full fine-tuning and the [Tinker model docs](https://tinker-docs.thinkingmachines.ai/tinker/models/)
for Tinker-backed training.

## Usage

### Installation
```bash
# clone codebase with pinned submodules (slime, tinker-cookbook)
git clone --recurse-submodules git@github.com:spade-rl/spade.git && cd spade

# prepare environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10 through 3.12 are supported; `python -m pip install -e ".[dev]"` adds test and lint tooling. GEM support on Python 3.12.1+ needs `python -m pip install --ignore-requires-python gem-llm`; see [`cmd/README.md`](cmd/README.md) for details.

### Training

The launchers read paths and credentials from the environment:

```bash
export MODEL_ROOT=/path/to/model/checkpoints     # HF checkpoints + Megatron conversions
export WORKSPACE_DIR=/path/to/spade/workspace    # external eval data (aime-*/, bfcl/)
export CORPUS_FILE=/path/to/grounding.jsonl      # designer grounding corpus (adaptive recipes)
export WANDB_API_KEY=...                         # Weights & Biases logging
export WANDB_ENTITY=your-wandb-entity
```

```bash
bash cmd/games/train_spade_30b.sh
```

This training script runs SPADE games self-play for 400 rollouts on a single 8-GPU node, training both roles of Qwen3-30B-A3B-Instruct with GRPO. The full paper matrix is organized by setting:

| Setting | Models | Commands |
|---|---|---|
| SPADE games | 4B, 8B, 30B-A3B | `cmd/games/train_spade_{4b,8b,30b}.sh` |
| Fixed-env GRPO, GPT-5.5 corpus | 4B, 8B, 30B-A3B | `cmd/games/train_fixed_gpt55_{4b,8b,30b}.sh` |
| Fixed-env RLVE | 4B, 8B, 30B-A3B | `cmd/games/train_fixed_rlve_{4b,8b,30b}.sh` |
| SPADE tool use | 4B, 8B, 30B-A3B | `cmd/tool_use/train_spade_{4b,8b,30b}.sh` |
| Paper ablations | 30B-A3B | `cmd/ablations/*.sh` |

The fixed-env GRPO recipes need no `CORPUS_FILE`: they train on the released [static GPT-5.5 corpus](https://huggingface.co/datasets/spade-rl/SPADE-Environment-Pool-GPT5.5-Games) (7,872 validated Python environments across six cognitive skills, pinned revision with per-environment SHA-256 checksums, Apache-2.0). Checkpoint and data prerequisites are documented in [`cmd/README.md`](cmd/README.md#prerequisites), and the setting-specific READMEs under `cmd/games/`, `cmd/tool_use/`, and `cmd/ablations/` document the corpora and overrides each recipe expects.

### Evaluation

`eval_offline/` scores a trained checkpoint, or an OpenAI-compatible endpoint, against the paper's benchmark suites:

```bash
# run the offline benchmark suites
python -m eval_offline.run_offline_eval --help

# format the results
python -m eval_offline.render_table --help
```

The runner needs the `[eval]` extra and per-benchmark data setup; both, plus the benchmark matrix, are documented in [`eval_offline/README.md`](eval_offline/README.md). `eval_configs/` is separate: those YAML files drive the in-loop evaluations the training launchers run during a job.

## ProofPack Qualification & Assay Evidence

SPADE has an optional assurance path for generated cognitive and tool-use environments:

* **ProofPack V0–V4 qualification** checks source policy, deterministic multi-seed reset,
  oracle solvability, a no-agent control, and a planted invalid action. Enable it with
  `SpadeConfig(use_proofpack_qualification=True)`. A requested gate fails closed when a
  compatible ProofPack install or its OS isolation backend is unavailable.
* **Assay persistence and decision support** writes a content-addressed curriculum manifest,
  observations, a statistical decision, and Evidence Schema v2 dependence records. Assay
  requires at least four declared independent environment clusters and identified uncertainty
  before reporting a positive statistical signal. That signal is not release approval; a real
  `model.lock` has additional registration and operational-evidence gates.

SPADE itself remains compatible with Python 3.10+. The optional ProofPack and Assay packages
require Python 3.12+, so use a Python 3.12 environment for the assurance path. Qualification is
isolated; SPADE's ordinary training environment runtime remains a separate, trusted native-code
boundary. See [the assurance guide](docs/assurance-integration.md) for configuration, threat
boundaries, artifact schemas, and verification commands.

### Live Evaluation Runner

The live smoke performs designer synthesis, isolated ProofPack qualification, observation-only
hint generation, paired multi-turn rollouts, and native Assay artifact persistence. With one
environment it is deliberately evidence-producing but non-promotional: the expected decision is
`insufficient_clusters`, and no `model.lock` is emitted.

```bash
# ASSURANCE_PYTHON must be a Python 3.12 environment with both proofpack-env
# and assay (including their runtime dependencies) installed.
ASSURANCE_PYTHON=/path/to/assurance-venv/bin/python

# Uses the existing Google Antigravity CLI subscription; no API key is copied.
"$ASSURANCE_PYTHON" tools/run_live_spade_eval.py \
  --provider agy \
  --skill "Graph Theory and Shortest Path"

# OpenAI-compatible providers are also supported with an explicit key or --env-file.
"$ASSURANCE_PYTHON" tools/run_live_spade_eval.py \
  --provider openrouter \
  --model deepseek/deepseek-chat \
  --skill "Combinatorial Knapsack and Dynamic Programming"
```

Every run receives its own directory under `.assay/spade-live/`, including the generated source,
ProofPack receipt, complete paired trace, Assay manifest, decision, certification, and Evidence v2
ledger. The manifest binds the receipt and trace digests. The CLI returns nonzero on dependency,
generation, qualification, leakage, rollout, or persistence failure.

For an offline test of stronger behavioral environment identity, see the
[counterfactual witness guide](docs/counterfactual-witness.md). Its sealed runner mutates and
replays the preserved 18-environment cohort entirely inside ProofPack isolation, compares a compact
witness certificate with cost-matched random and fixed-probe baselines, and makes zero AGY calls or
learner updates. A passing result is an exploratory operator-holdout signal only; it is not evidence
that learner performance improved.

This command is intentionally a one-cluster smoke harness. Repeating it does not produce an
aggregate multi-environment experiment. The
[assurance guide](docs/assurance-integration.md#multi-environment-agy-experiments) documents the
sealed 18-cluster `agy` pilot runner, its dry-run-first workflow, invocation budget, resume and
provenance controls, sealed output root, and the explicit execution acknowledgement required before
provider calls.

## Tinker Training

SPADE also supports training with [Thinking Machines](https://thinkingmachines.ai/tinker)' **Tinker** distributed training framework through the integration under `spade/tinker/`.

### Quick Start

```bash
# Install with Tinker dependencies (Python 3.11+)
python -m pip install -e ".[tinker]"
```

### Supported Models

| Model | Launchers |
|-------|-----------|
| Qwen3-4B-Instruct | `cmd/tinker/qwen3_4b_instruct/` |
| Qwen3-8B | `cmd/tinker/qwen3_8b/` |
| Qwen3-8B-Base | `cmd/tinker/qwen3_8b_base/` |
| Qwen3-30B-A3B-Instruct | `cmd/tinker/qwen3_30b_instruct/` |
| GPT-OSS-20B | `cmd/tinker/gpt_oss_20b/` |

The full set of Tinker-trainable models is listed in the [Tinker model docs](https://tinker-docs.thinkingmachines.ai/tinker/models/). See [`cmd/tinker/README.md`](cmd/tinker/README.md) for setup, launchers, and advanced usage. For more information on the Tinker framework, see the [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) repository.

## Citation

If you find our work useful for your research, please consider citing:
```bibtex
@article{liu2026spade,
  title={SPADE: Self-Play in Adaptive Synthetic Executable Environments},
  author={Liu, Bo and Yu, Simon and Jiang, Yiding and Qu, Ao and Zhao, Andrew and Liu, Zichen and Kim, Junsu and Zhou, Zijian and Kim, Seungone and Ren, Tongzheng and Liu, Mickel and Yu, Hanfei and Chen, Zhaorun and Shi, Weiyan and Liang, Paul Pu and Zettlemoyer, Luke and Choi, Yejin and Jaques, Natasha},
  journal={arXiv preprint arXiv:2608.19197},
  year={2026}
}
```

## Acknowledgement
* The distributed RL training is implemented with [Slime](https://github.com/THUDM/slime), pairing [SGLang](https://github.com/sgl-project/sglang) inference with Megatron-LM policy updates, and informed by the [Miles](https://github.com/radixark/miles) team's RL post-training framework.
* We thank [Thinking Machines](https://thinkingmachines.ai/tinker) for the Tinker framework and [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), an alternative training backend.
* We thank [Modal](https://modal.com) for compute and model serving during development.
* The evaluation stack builds on RLVE, the Berkeley Function Calling Leaderboard, and PRIME evaluation utilities.
* The base models are from [Qwen3](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).

SPADE source code is released under the [MIT License](LICENSE); datasets, vendored code, and adapted evaluation components retain their respective licenses as documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
