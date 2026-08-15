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

    judge_cfg = config.get("judge", {"provider": "simulated"})
    if judge_cfg["provider"] == "simulated":
        judge = SimulatedJudge(**judge_cfg.get("params", {}))
    else:
        from .judge import LLMJudge

        judge = LLMJudge(build_provider(judge_cfg), name=judge_cfg.get("name", "llm-judge"))

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

                if record["error"]:
                    behavior, confidence, reason = "error", 0.0, record["error"][:80]
                elif isinstance(judge, SimulatedJudge):
                    j = judge.judge(item, record["response"], model=provider.name)
                    behavior, confidence, reason = j.label, j.confidence, j.reason
                else:
                    j = judge.judge(item, record["response"])
                    behavior, confidence, reason = j.label, j.confidence, j.reason

                baseline = heuristic_label(record["response"]).label
                rows.append(
                    {
                        "model": provider.name,
                        "simulated": provider.simulated,
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
                        "heuristic_behavior": baseline,
                        "response_chars": len(record["response"]),
                    }
                )

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
