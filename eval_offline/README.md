# Offline evaluation

The offline driver evaluates a local Hugging Face checkpoint or an existing OpenAI-compatible endpoint using the paper's benchmark suites.

## Install

```bash
# editable checkout
python -m pip install --ignore-requires-python -e ".[eval]"

# published package
pip install --ignore-requires-python "spade[eval]"
```

`--ignore-requires-python` is required: the `[eval]` extra pulls tau2-bench from git and tau2 pins `requires-python >=3.12,<3.14`, which otherwise blocks installation on the Python 3.10 Slime training environment. See the comment on the `eval` extra in `pyproject.toml`.

LiveCodeBench, tau2-bench, BFCL, and ACEBench also require their official upstream repositories or datasets. Configure those paths before selecting the corresponding suite.

## Data setup

- **AIME.** The suite reads `${WORKSPACE_DIR}/aime-<year>/aime-<year>.jsonl` (default workspace `/workspace/spade-workspace`); a `path_template` in the suite config overrides it. Years without a file are skipped with a warning. Preprocess the jsonls with `cmd/eval/prepare_aime_datasets.py`, which appends the step-by-step `\boxed{}` instruction to each user message.
- **GPQA-Diamond.** Loads `Idavidrein/gpqa` (config `gpqa_diamond`) from the Hub. The dataset is gated: accept its terms on the dataset page with your account, then export `HF_TOKEN` before running.
- **LiveCodeBench.** `WORKSPACE_DIR=/path/to/workspace bash scripts/setup_lcb.sh` clones the pinned official LiveCodeBench runner into `${WORKSPACE_DIR}/LiveCodeBench-official` and downloads the `release_v6` `test6.jsonl` into `${WORKSPACE_DIR}/livecodebench`. Export `LCB_OFFICIAL_DIR` to the clone and `LCB_V6_JSONL` to the downloaded jsonl, and install the official package so `lcb_runner` is importable (`pip install --ignore-requires-python git+https://github.com/LiveCodeBench/LiveCodeBench.git`).
- **tau2-bench.** Needs `OPENROUTER_API_KEY` or `OPENAI_API_KEY` for the user simulator; without either, the suite logs a warning and is skipped. The git install does not ship tau2's data files; clone tau2-bench separately and set `TAU2_DATA_DIR`.
- **ACEBench.** Set `ACEBENCH_DIR` to an ACEBench checkout.

## Run

Evaluate a local checkpoint on the games benchmarks:

```bash
python -m eval_offline.run_offline_eval \
  --ckpt /path/to/checkpoint \
  --config eval_offline/configs/games.yaml \
  --output-dir /path/to/results
```

Evaluate an existing endpoint on the tool-use benchmarks:

```bash
python -m eval_offline.run_offline_eval \
  --base-url http://localhost:8000 \
  --served-model-name qwen3 \
  --config eval_offline/configs/tool_use.yaml \
  --output-dir /path/to/results
```

`games.yaml` is the paper's evaluation protocol:

| Benchmark | Protocol |
|---|---|
| AIME 2025/2026 | Avg@32 |
| GPQA-Diamond | Avg@10 |
| LiveCodeBench-v6 | Pass@1, full 175-problem release_v6 |
| Reasoning-Gym (4 categories) | Avg@8, 100-task hard split |

GPQA-Diamond and LiveCodeBench-v6 follow the official Qwen3 evaluation settings: the GPQA sample count (Avg@10) per the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), and the non-thinking sampling parameters and benchmark versions from the Qwen3 model cards. AIME 2025/2026 runs Avg@32 at the same sampling parameters. Reasoning-Gym uses the per-task *hard* configurations of the [reasoning_gym paper](https://arxiv.org/abs/2505.24760) (Appendix A.3), with the four-category rollup (Math / Algorithmic / Cognition / Logic) following [Golden Goose](https://arxiv.org/abs/2601.22975). We thank the authors of these works for the protocols and reference numbers this evaluation builds on.

`tool_use.yaml` covers BFCL v4, tau2-bench, and ACEBench. Files beginning with `_` are component configurations used by `tool_use.yaml`; `games.yaml` is fully self-contained.

`tool_use.yaml` pins the BFCL model handle to the 4B checkpoint (`Qwen/Qwen3-4B-Instruct-2507-FC`). Set the `BFCL_MODEL_HANDLE` environment variable (it takes precedence over the config value) to the `bfcl_eval` handle matching the model under test when evaluating 8B or 30B; otherwise those results are generated and labelled as 4B.

Use `--suites` to select a subset, `--max-concurrent` to cap request concurrency, and `--no-wandb` to keep results local. Each run writes `results.json` plus suite-specific artifacts.

## Render a paper row

```bash
python -m eval_offline.render_table \
  --results /path/to/results/results.json \
  --rq RQ1 \
  --method-name SPADE \
  --format markdown
```

Run `python -m eval_offline.run_offline_eval --help` for all server and sampling options.
