"""Human annotation workflow.

Real use: `p2e annotate --sample` writes a blank CSV of stratified rows with the
prompt and response inline. You fill in `human_behavior` against
docs/ANNOTATION_GUIDE.md, save, and `p2e agreement` reads it back.

Demo use: `--simulate` fills the column with a deterministic stand-in annotator
so the pipeline is exercisable in CI and on first clone. Simulated labels are
written with `annotator = "SIMULATED"` in every row, and the agreement step
refuses to report them without that flag being visible in the output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .agreement import stratified_sample
from .judge import heuristic_label

SIMULATED_ANNOTATOR = "SIMULATED"

# Independent error process from the judge's: a careful human annotator working
# from a written rubric still slips on the partial/comply boundary, at a lower
# rate than an LLM judge does.
_HUMAN_ERROR_RATE = 0.035
_HUMAN_PARTIAL_MULTIPLIER = 2.2


def _roll(item_id: str, model: str, salt: str = "human-v1") -> float:
    digest = hashlib.sha256(f"{salt}|{model}|{item_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _simulated_human_label(response: str, item_id: str, model: str) -> str:
    base = heuristic_label(response).label
    rate = _HUMAN_ERROR_RATE * (_HUMAN_PARTIAL_MULTIPLIER if base == "partial" else 1.0)
    if _roll(item_id, model) < rate:
        return {"comply": "partial", "partial": "comply", "refuse": "partial"}[base]
    return base


def build_sample(
    run_dir: str | Path,
    out_path: str | Path = "data/annotations/human_labels.csv",
    per_stratum: int = 2,
    simulate: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    scored = pd.read_csv(run_dir / "scored.csv")

    responses = {
        (r["model"], r["item_id"]): r["response"]
        for r in (
            json.loads(line)
            for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    prompts = {
        d["item_id"]: d["prompt"]
        for d in (
            json.loads(line)
            for line in (run_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    sample = stratified_sample(scored, per_stratum=per_stratum)
    sample["prompt"] = sample["item_id"].map(prompts)
    sample["response"] = [
        responses.get((r.model, r.item_id), "") for r in sample.itertuples()
    ]

    if simulate:
        sample["human_behavior"] = [
            _simulated_human_label(r.response, r.item_id, r.model)
            for r in sample.itertuples()
        ]
        sample["annotator"] = SIMULATED_ANNOTATOR
    else:
        sample["human_behavior"] = ""
        sample["annotator"] = ""

    sample["annotator_notes"] = ""
    columns = [
        "model", "item_id", "category", "subcategory", "framing", "expected", "edge",
        "prompt", "response", "behavior", "judge_confidence", "judge_reason",
        "human_behavior", "annotator", "annotator_notes",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].rename(columns={"behavior": "judge_behavior"}).to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(sample)} rows, simulate={simulate})")
    return out_path


# Columns safe to publish: enough to reproduce kappa and the confusion matrix,
# and nothing else. `prompt` and `response` are deliberately excluded — the
# response column contains model output to prohibited requests, which is exactly
# what RESPONSIBLE_USE.md §3 says never ships.
PUBLISHABLE_COLUMNS = [
    "model", "item_id", "category", "subcategory", "framing", "expected", "edge",
    "judge_behavior", "judge_confidence", "human_behavior", "annotator", "annotator_notes",
]


def export_labels(
    labels_path: str | Path,
    out_path: str | Path,
    require_complete: bool = True,
) -> Path:
    """Strip the working annotation sheet down to a publishable label file.

    The sheet you annotate from carries the full prompt and response inline,
    because you cannot label what you cannot read. That same file must never be
    committed: for the refusal-expected half of the dataset it holds model output
    to prohibited requests. This writes the labels without the content.

    The result is sufficient to recompute judge-human agreement, so the
    calibration claim stays reproducible by anyone who re-runs generation.
    """
    labels_path, out_path = Path(labels_path), Path(out_path)
    df = pd.read_csv(labels_path)

    filled = df["human_behavior"].notna() & (df["human_behavior"].astype(str).str.strip() != "")
    if require_complete and not filled.all():
        raise ValueError(
            f"{int((~filled).sum())} of {len(df)} rows have no human_behavior. "
            "Finish labelling, or pass require_complete=False to export a partial sheet."
        )
    df = df[filled]

    missing = [c for c in PUBLISHABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"annotation sheet is missing columns: {missing}")

    leaked = [c for c in ("prompt", "response") if c in PUBLISHABLE_COLUMNS]
    assert not leaked, "content columns must never be in the publishable set"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[PUBLISHABLE_COLUMNS].to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} labelled rows, content columns stripped)")
    return out_path
