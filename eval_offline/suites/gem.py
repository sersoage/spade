"""GEM suite adapter for the retained Reasoning-Gym evaluations.

Reads the same YAML schema as eval_configs/gem_eval_*.yaml, builds an
OfflineModelAdapter, calls GemEvaluator.evaluate_all, writes scores.json.

Config (either form):
    suites:
      gem:
        defaults: {...}      # inline gem_eval schema (preferred: single-file
        tasks: [...]         # protocol, nothing hidden in an include)
    suites:
      gem:
        config_path: path/to/gem_eval.yaml   # external include, still supported

Output:
    <out>/scores.json   — flat metric dict (gem_eval/... keys)
    <out>/raw_result.json — full GemEvalResult (per-task results, errors)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.gem")


def _resolve_config_path(p: str) -> Path:
    if not p:
        raise ValueError("gem suite needs `config_path`.")
    pp = Path(p)
    if pp.is_absolute() and pp.is_file():
        return pp
    for root in ("/workspace", os.getcwd()):
        candidate = Path(root) / p
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"gem config_path {p!r} not found")


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    from spade.core.eval.gem_evaluator import GemEvaluator
    from spade.core.eval.gem_tasks import load_gem_eval_config
    from eval_offline.model_adapter_shim import OfflineModelAdapter

    if cfg.get("tasks"):
        # Inline form: the suite config IS the gem_eval schema. Materialise it
        # to a temp file so the shared loader stays the single parsing path.
        import tempfile, yaml as _yaml
        inline = {"gem_eval": {k: cfg[k] for k in ("defaults", "tasks") if k in cfg}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            _yaml.safe_dump(inline, tf)
            config_path = Path(tf.name)
        logger.info("[gem] using inline task config (%d tasks)", len(cfg["tasks"]))
    else:
        config_path = _resolve_config_path(cfg.get("config_path", ""))
        logger.info("[gem] loading config from %s", config_path)
    defaults, task_specs = load_gem_eval_config(str(config_path))

    if not task_specs:
        logger.warning("[gem] no tasks in config — nothing to run")
        return {"gem_n_tasks": 0}

    # Fail fast on missing external prerequisites. The asset-backed envs load
    # their data in __init__ (GPQA pulls the gated Idavidrein/gpqa dataset,
    # LiveCodeBench needs LCB_OFFICIAL_DIR and the release_v6 jsonl), and a
    # task whose episodes all error would otherwise be scored as a
    # real-looking 0.0 win rate.
    import gem as _gem
    for task_id in dict.fromkeys(spec.task_id for spec in task_specs):
        try:
            _gem.make(task_id)
        except Exception as e:
            raise RuntimeError(
                f"[gem] cannot construct env {task_id!r}: {e}. Fix the missing "
                "prerequisite (eval_offline/README.md, Data setup) and rerun."
            ) from e

    adapter = OfflineModelAdapter(client, model_path=client.model)

    max_concurrent = cfg.get("max_concurrent", defaults.max_concurrent)
    evaluator = GemEvaluator(model=adapter, max_concurrent=max_concurrent)

    logger.info(
        "[gem] running %d tasks (defaults episodes=%d max_turns=%d, "
        "max_concurrent=%d)",
        len(task_specs), defaults.episodes, defaults.max_turns, max_concurrent,
    )
    eval_result = asyncio.run(evaluator.evaluate_all(task_specs))

    try:
        (out_dir / "raw_result.json").write_text(
            json.dumps(asdict(eval_result), indent=2, default=str)
        )
    except Exception as e:  # pragma: no cover
        logger.warning("[gem] could not serialize raw result: %s", e)

    # A task with zero completed episodes has no score; refusing to emit one
    # keeps a runtime failure (dead endpoint, mid-run asset loss) from
    # rendering as 0.0 in the paper table. raw_result.json above still holds
    # the per-episode errors for debugging.
    failed = [
        f"{r.task_id}{r.metric_suffix} ({r.errors} errors)"
        for r in eval_result.per_task_results
        if r.num_episodes == 0
    ]
    if failed:
        raise RuntimeError(
            "[gem] every episode failed for: " + ", ".join(failed)
            + ". No scores.json written; see raw_result.json for details."
        )

    metrics: dict[str, Any] = dict(eval_result.to_metrics_dict(prefix="gem_eval"))
    (out_dir / "scores.json").write_text(json.dumps(metrics, indent=2))

    return metrics
