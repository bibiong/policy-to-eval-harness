"""Orchestration: dataset → model responses → judge labels → scored table."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from .dataset import Item, build_dataset, write_dataset
from .judge import SimulatedJudge, decision_from_behavior, heuristic_label
from .providers import build_provider
from .taxonomy import load_taxonomy


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_judge(judge_cfg: dict):
    """Instantiate the judge described by a config's `judge:` block."""
    if judge_cfg["provider"] == "simulated":
        return SimulatedJudge(**judge_cfg.get("params", {}))
    from .judge import LLMJudge

    return LLMJudge(build_provider(judge_cfg), name=judge_cfg.get("name", "llm-judge"))


def _score_one(judge, item: Item, response: str, model: str, error: str | None) -> dict:
    """Apply the judge to a single response and build the scored row."""
    if error:
        behavior, confidence, reason = "error", 0.0, error[:80]
    elif isinstance(judge, SimulatedJudge):
        j = judge.judge(item, response, model=model)
        behavior, confidence, reason = j.label, j.confidence, j.reason
    else:
        j = judge.judge(item, response)
        behavior, confidence, reason = j.label, j.confidence, j.reason

    return {
        "model": model,
        "item_id": item.item_id,
        "seed_id": item.seed_id,
        "category": item.category,
        "subcategory": item.subcategory,
        "framing": item.framing,
        "expected": item.expected,
        "edge": item.edge,
        "behavior": behavior,
        "decision": decision_from_behavior(behavior),
        "judge_confidence": confidence,
        "judge_reason": reason,
        "heuristic_behavior": heuristic_label(response).label,
        "response_chars": len(response),
    }


def _response_records(provider, items: list[Item], workers: int) -> list[dict]:
    def one(item: Item) -> dict:
        started = time.time()
        try:
            text, error = provider.generate(item.prompt, item), None
        except Exception as exc:  # network/rate-limit failures are recorded, not dropped
            text, error = "", f"{type(exc).__name__}: {exc}"
        return {
            "model": provider.name,
            "simulated": provider.simulated,
            "item_id": item.item_id,
            "response": text,
            "error": error,
            "latency_s": round(time.time() - started, 3),
        }

    if workers > 1 and not provider.simulated:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, items))
    return [one(item) for item in items]


def run(
    config_path: str | Path = "configs/models.yaml",
    out_dir: str | Path = "results/demo_sim",
    taxonomy_path: str | Path = "policy/taxonomy.yaml",
    seeds_path: str | Path = "data/seeds/prompts_seed.yaml",
    limit: int | None = None,
    workers: int = 8,
) -> Path:
    config = load_config(config_path)
    taxonomy = load_taxonomy(taxonomy_path)
    items = build_dataset(taxonomy, seeds_path)
    if limit:
        items = items[:limit]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataset(items, out_dir / "dataset.jsonl")

    judge = build_judge(config.get("judge", {"provider": "simulated"}))

    by_id = {item.item_id: item for item in items}
    rows: list[dict] = []
    raw_path = out_dir / "responses.jsonl"

    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for spec in config["models"]:
            provider = build_provider(spec)
            print(f"  → {provider.name} ({len(items)} items)")
            for record in _response_records(provider, items, workers):
                raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                item = by_id[record["item_id"]]
                row = _score_one(
                    judge, item, record["response"], provider.name, record["error"]
                )
                rows.append({**row, "simulated": provider.simulated})

    scored = pd.DataFrame(rows)
    errors = int((scored["behavior"] == "error").sum())
    if errors:
        print(f"  ! {errors} generation errors excluded from scoring")
    scored = scored[scored["behavior"] != "error"]
    scored.to_csv(out_dir / "scored.csv", index=False)

    manifest = {
        "taxonomy_version": taxonomy.version,
        "n_items": len(items),
        "n_models": len(config["models"]),
        "models": [m.get("name", m.get("model")) for m in config["models"]],
        "judge": judge.name,
        "judge_simulated": getattr(judge, "simulated", False),
        "any_simulated_model": bool(scored["simulated"].any()),
        "all_simulated": bool(scored["simulated"].all()),
        "generation_errors": errors,
        "config_sha256": hashlib.sha256(
            Path(config_path).read_bytes()
        ).hexdigest()[:16],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir}/scored.csv ({len(scored)} rows)")
    return out_dir


def rejudge(
    run_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path | None = None,
) -> Path:
    """Re-score an existing run's saved responses with a different judge.

    Generation is the expensive step and the judge is the cheap one, but the
    judge is also the part you are most likely to want to change — swapping a
    local judge for a frontier one, or re-running after editing the rubric.
    Coupling them would mean paying for generation again to change a label.

    Reads `responses.jsonl` from `run_dir`, applies the judge from
    `config_path`'s `judge:` block, and writes a complete new run directory that
    `p2e report` can consume unchanged. The original run is not modified.
    """
    run_dir = Path(run_dir)
    config = load_config(config_path)
    judge = build_judge(config.get("judge", {"provider": "simulated"}))

    out_dir = Path(out_dir) if out_dir else run_dir.parent / f"{run_dir.name}_rejudged"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = {
        d["item_id"]: Item(**d)
        for d in (
            json.loads(line)
            for line in (run_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    records = [
        json.loads(line)
        for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"  re-judging {len(records)} responses with {judge.name}")
    rows = []
    for n, record in enumerate(records, 1):
        item = items[record["item_id"]]
        row = _score_one(judge, item, record["response"], record["model"], record["error"])
        rows.append({**row, "simulated": record.get("simulated", False)})
        if n % 200 == 0:
            print(f"    {n}/{len(records)}")

    scored = pd.DataFrame(rows)
    errors = int((scored["behavior"] == "error").sum())
    scored = scored[scored["behavior"] != "error"]
    scored.to_csv(out_dir / "scored.csv", index=False)

    # Carry the inputs across so the new directory is self-contained.
    for name in ("dataset.jsonl", "responses.jsonl"):
        (out_dir / name).write_text(
            (run_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "judge": judge.name,
            "judge_simulated": getattr(judge, "simulated", False),
            "generation_errors": errors,
            "rejudged_from": str(run_dir),
            "original_judge": manifest.get("judge"),
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir}/scored.csv ({len(scored)} rows)")
    return out_dir
