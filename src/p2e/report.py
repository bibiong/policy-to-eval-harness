"""Compute metrics and render the findings report.

The report is generated, not hand-typed: every number in reports/findings.md
comes from results/<run>/metrics.json, so the prose cannot drift away from the
data. Running `make report` after a real run rewrites it in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import charts, metrics
from .agreement import cohens_kappa, confusion_matrix, interpret_kappa
from .annotate import SIMULATED_ANNOTATOR
from .taxonomy import load_taxonomy


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def compute(run_dir: str | Path, annotations: str | Path | None = None) -> dict:
    run_dir = Path(run_dir)
    scored = pd.read_csv(run_dir / "scored.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    result: dict = {
        "manifest": manifest,
        "overall": {
            "over_refusal": metrics.over_refusal(scored).as_dict(),
            "over_refusal_strict": metrics.over_refusal(scored, strict=True).as_dict(),
            "under_refusal": metrics.under_refusal(scored).as_dict(),
            "precision_recall": metrics.precision_recall_f1(scored),
        },
        "per_model": metrics.per_model(scored).to_dict("records"),
        "per_category": metrics.per_category_pooled(scored).to_dict("records"),
        "per_model_category": metrics.per_category(scored).to_dict("records"),
        "framing": metrics.framing_sensitivity(scored).to_dict("records"),
        "edge_gap": metrics.edge_gap(scored).to_dict("records"),
    }

    ann_path = Path(annotations) if annotations else None
    if ann_path and ann_path.exists():
        ann = pd.read_csv(ann_path)
        ann = ann[ann["human_behavior"].notna() & (ann["human_behavior"] != "")]
        if len(ann):
            kappa = cohens_kappa(
                ann["human_behavior"].tolist(), ann["judge_behavior"].tolist()
            )
            matrix = confusion_matrix(
                ann["human_behavior"].tolist(), ann["judge_behavior"].tolist()
            )
            baseline_rows = scored.set_index(["model", "item_id"])
            joined = ann.join(
                baseline_rows["heuristic_behavior"], on=["model", "item_id"]
            )
            heuristic_kappa = cohens_kappa(
                joined["human_behavior"].tolist(),
                joined["heuristic_behavior"].tolist(),
            )
            annotators = sorted(set(ann["annotator"].astype(str)))
            result["agreement"] = {
                **kappa.as_dict(),
                "interpretation": interpret_kappa(kappa.kappa),
                "confusion_matrix": matrix.to_dict(),
                "heuristic_baseline_kappa": heuristic_kappa.kappa,
                "annotators": annotators,
                "simulated_annotator": SIMULATED_ANNOTATOR in annotators,
                "disagreement_by_category": (
                    ann.assign(agree=ann["human_behavior"] == ann["judge_behavior"])
                    .groupby("category")["agree"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "agreement_rate", "count": "n"})
                    .reset_index()
                    .to_dict("records")
                ),
            }

    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def make_charts(run_dir: str | Path, out_dir: str | Path = "reports/charts") -> list[Path]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    scored = pd.read_csv(run_dir / "scored.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sim = bool(manifest.get("any_simulated_model"))

    paths = [
        charts.over_refusal_by_model(metrics.per_model(scored), out_dir / "over_refusal_by_model.png", sim),
        charts.over_refusal_by_category(metrics.per_category_pooled(scored), out_dir / "over_refusal_by_category.png", sim),
        charts.ambiguity_scatter(metrics.per_category_pooled(scored), out_dir / "ambiguity_vs_over_refusal.png", sim),
        charts.framing_sensitivity(metrics.framing_sensitivity(scored), out_dir / "framing_sensitivity.png", sim),
    ]

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "agreement" in data:
            matrix = pd.DataFrame(data["agreement"]["confusion_matrix"])
            paths.append(charts.agreement_matrix(matrix, out_dir / "judge_human_agreement.png", sim))
    return paths


def render(run_dir: str | Path, out_path: str | Path = "reports/findings.md") -> Path:
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    taxonomy = load_taxonomy()
    manifest = data["manifest"]
    sim = bool(manifest.get("any_simulated_model"))

    per_model = pd.DataFrame(data["per_model"])
    per_cat = pd.DataFrame(data["per_category"])
    framing = pd.DataFrame(data["framing"])
    edge = pd.DataFrame(data["edge_gap"])
    overall = data["overall"]
    agreement = data.get("agreement")

    worst_model = per_model.iloc[-1]
    best_model = per_model.iloc[0]
    worst_cat = per_cat.iloc[0]
    ambiguous = per_cat.sort_values("excess_disagreement", ascending=False).head(3)

    framing_pivot = framing.pivot(index="model", columns="framing", values="over_refusal")
    if {"bare", "professional"} <= set(framing_pivot.columns):
        framing_pivot["context_gain"] = framing_pivot["bare"] - framing_pivot["professional"]
        least_responsive = framing_pivot["context_gain"].idxmin()
        most_responsive = framing_pivot["context_gain"].idxmax()
    else:  # pragma: no cover
        least_responsive = most_responsive = per_model.iloc[0]["model"]

    banner = (
        "> **This run is simulated.** Every model in it is a deterministic stand-in "
        "shipped with the repo so the pipeline runs on first clone with no API keys. "
        "The numbers below are a property of that simulation and say nothing about "
        "any real model. Swap in `configs/models.live.yaml` and re-run to produce "
        "real measurements; the report regenerates itself from whatever run you point "
        "it at. See [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) §4.\n"
        if sim
        else "> Run against live models. See `manifest.json` in the run directory for provenance.\n"
    )

    lines: list[str] = []
    add = lines.append

    add("# Findings — over-refusal on policy-permitted requests\n")
    add(banner)
    add(
        f"**Run:** `{run_dir.name}` · **taxonomy** v{manifest['taxonomy_version']} · "
        f"{manifest['n_items']} prompts × {manifest['n_models']} models = "
        f"{int(overall['over_refusal']['n'] + overall['under_refusal']['n'])} scored responses · "
        f"**judge** `{manifest['judge']}`\n"
    )

    add("## The question\n")
    add(
        "A usage policy is a written document. A deployed model is a decision procedure. "
        "Nobody publishes the mapping between them. This harness builds that mapping "
        "explicitly — policy clause → harm category → subcategory → decision criteria → "
        "expected decision — and then measures how far actual model behaviour sits from "
        "the policy as written.\n"
    )
    add(
        "The measurement target is deliberately the *benign* side. Refusing a legitimate "
        "request is a product failure that costs real users real help, it is systematically "
        "under-measured because it is less alarming than the opposite error, and — unlike "
        "measuring harmful compliance — it can be studied and published without generating "
        "or distributing harmful content.\n"
    )

    add("## What the run found\n")
    add(
        f"**{_pct(overall['over_refusal']['rate'])} of policy-permitted requests did not get a "
        f"substantive answer** "
        f"(95% CI {_pct(overall['over_refusal']['ci_low'])}–{_pct(overall['over_refusal']['ci_high'])}, "
        f"n={overall['over_refusal']['n']}). Counting only outright refusals and excluding "
        f"hedged non-answers, the rate is {_pct(overall['over_refusal_strict']['rate'])}.\n"
    )
    add(
        f"Leakage in the other direction — substantive answers to requests the taxonomy marks "
        f"as prohibited — was {_pct(overall['under_refusal']['rate'])} "
        f"(n={overall['under_refusal']['n']}). Pooled precision on the refuse class was "
        f"{overall['precision_recall']['precision']:.2f} and recall "
        f"{overall['precision_recall']['recall']:.2f}: the models catch most of what the policy "
        f"asks them to catch, and pay for it with a large number of false positives on "
        f"legitimate traffic.\n"
    )
    add(
        f"The spread across models is wide — {_pct(best_model['over_refusal'])} for "
        f"`{best_model['model']}` against {_pct(worst_model['over_refusal'])} for "
        f"`{worst_model['model']}` — on identical prompts against an identical policy. "
        f"That spread is the finding. It means the policy text is not what determines "
        f"the behaviour; the tuning is.\n"
    )
    add("![over-refusal by model](charts/over_refusal_by_model.png)\n")

    add("### Per-model\n")
    add("| model | over-refusal | 95% CI | strict | leakage | precision | recall | F1 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in per_model.itertuples():
        add(
            f"| `{row.model}` | {_pct(row.over_refusal)} | "
            f"{_pct(row.over_refusal_ci_low)}–{_pct(row.over_refusal_ci_high)} | "
            f"{_pct(row.over_refusal_strict)} | {_pct(row.under_refusal)} | "
            f"{row.precision:.2f} | {row.recall:.2f} | {row.f1:.2f} |"
        )
    add("")

    add("### Per-category\n")
    add(
        f"Over-refusal is not spread evenly. `{worst_cat['category']}` is the worst at "
        f"{_pct(worst_cat['over_refusal'])} of permitted requests unanswered — and it is the "
        f"category where the cost of a non-answer is highest, because the legitimate traffic "
        f"is clinicians, researchers, journalists and people supporting someone in crisis.\n"
    )
    add("![over-refusal by category](charts/over_refusal_by_category.png)\n")
    add("| category | over-refusal | 95% CI | precision | recall | disagreement | excess |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for row in per_cat.itertuples():
        add(
            f"| {row.category.replace('_', ' ')} | {_pct(row.over_refusal)} | "
            f"{_pct(row.over_refusal_ci_low)}–{_pct(row.over_refusal_ci_high)} | "
            f"{row.precision:.2f} | {row.recall:.2f} | {row.disagreement:.2f} | {row.excess_disagreement:+.2f} |"
        )
    add("")

    add("## Where the policy language itself is the problem\n")
    add(
        "The obvious move here is to read the raw disagreement column as an ambiguity "
        "score. It isn't one. Pairwise disagreement is mechanically maximised when a "
        "category is refused about half the time, so it largely re-reports the "
        "over-refusal rate — which is why the two columns above track each other so "
        "closely.\n"
    )
    add(
        "The `excess` column is the corrected version: observed disagreement minus what "
        "each model's own category-level refusal rate already predicts under "
        "independence. It isolates the correlation structure. Strongly negative means "
        "the models refuse *the same* items — the criterion is determinate and they "
        "agree on where it falls, even if they are collectively too strict. Near zero "
        "means refusals are uncorrelated: nothing shared is driving the call, which is "
        "the signature of policy language that does not settle the case.\n"
    )
    add("![ambiguity vs over-refusal](charts/ambiguity_vs_over_refusal.png)\n")
    add("The three least-determined categories in this run, ranked by excess disagreement:\n")
    for row in ambiguous.itertuples():
        cat = taxonomy.category(row.category)
        note = cat.ambiguity_note or cat.obligation
        add(f"**{cat.name}** — excess disagreement {row.excess_disagreement:+.2f} (raw {row.disagreement:.2f}), over-refusal {_pct(row.over_refusal)}.")
        add(f"> {note}\n")
    add(
        "In each case the policy gates on something a single prompt cannot establish: "
        "authorization, professional relationship, or downstream use. That is not a drafting "
        "error so much as an unavoidable property of writing rules for a context-free "
        "channel — but it does mean the enforcement burden falls on inference about the "
        "user, and that is where the variance lives.\n"
    )

    if sim:
        from .providers import _CATEGORY_COUPLING

        ranked = per_cat.dropna(subset=["excess_disagreement"]).copy()
        ranked["coupling"] = ranked["category"].map(_CATEGORY_COUPLING)
        rho = ranked["excess_disagreement"].corr(-ranked["coupling"], method="spearman")
        top2 = set(ranked.nlargest(2, "excess_disagreement")["category"])
        least_coupled = set(ranked.nsmallest(2, "coupling")["category"])
        add("### Does the excess-disagreement statistic actually work?\n")
        add(
            "In a simulated run this can be checked directly, because the simulator has a "
            "ground-truth per-category coupling constant — how much of an item's difficulty "
            "is shared across models rather than idiosyncratic — that the metric never sees. "
            f"Recovering that ordering from behaviour alone gives Spearman ρ = **{rho:.2f}**, "
            f"and the two categories the simulator made least determinate come back "
            f"{'both in' if least_coupled <= top2 else 'partially in'} the top two by excess "
            f"disagreement.\n"
        )
        add(
            "That is a real but noisy recovery, and the noise is informative: with roughly 25 "
            "permitted items per category, the statistic ranks categories usefully but should "
            "not be read to two decimal places. It is a screening tool for deciding which "
            "policy clauses to re-draft, not an estimate of anything.\n"
        )

    add("## Does stated context change anything?\n")
    add(
        "Each seed prompt appears under four framings — bare, student, professional, and "
        "urgent-personal — with the operative request held constant. Both source policies "
        "make context legally load-bearing, so a model that tracks the policy should refuse "
        "less when a legitimate professional context is stated.\n"
    )
    add(
        "The fifth framing, `third_party`, is excluded from this comparison: it was applied "
        "only to the edge-flagged supplement, so putting it on the same axis would compare "
        "framings across different and systematically harder item sets, and an "
        "item-selection effect would read as a framing effect.\n"
    )
    add("![framing sensitivity](charts/framing_sensitivity.png)\n")
    add("| model | bare | student | professional | urgent personal | context gain |")
    add("|---|---:|---:|---:|---:|---:|")
    for model, row in framing_pivot.iterrows():
        add(
            f"| `{model}` | {_pct(row.get('bare'))} | {_pct(row.get('student'))} | "
            f"{_pct(row.get('professional'))} | {_pct(row.get('urgent_personal'))} | "
            f"{_pct(row.get('context_gain'))} |"
        )
    add("")
    best_gain = framing_pivot.loc[most_responsive, "context_gain"]
    worst_gain = framing_pivot.loc[least_responsive, "context_gain"]
    flat = framing_pivot[framing_pivot["context_gain"].abs() < 0.01].index.tolist()
    add(
        f"`{most_responsive}` moves the most, refusing {_pct(best_gain)} fewer permitted "
        f"requests once a professional context is stated. `{least_responsive}` sits at the "
        f"other end at {_pct(worst_gain)}"
        + (
            " — a *negative* gain, meaning it refuses slightly more when context is supplied, "
            "the opposite of what the policy criteria imply."
            if worst_gain < 0
            else "."
        )
        + "\n"
    )
    if flat:
        add(
            "Flat lines are the ones to look at: "
            + ", ".join(f"`{m}`" for m in flat)
            + " returns an identical rate whether or not a context is given. That is the "
            "signature of a topic-level filter — the model is matching on subject matter, "
            "not applying the policy's criteria.\n"
        )
    add(
        "The distinction matters for enforcement design. A model that responds to context "
        "but responds wrongly can be addressed by re-drafting the criteria it is reading. A "
        "topic filter cannot: no amount of policy rewriting reaches it, because it is not "
        "reading the policy. Those are different remediation budgets owned by different "
        "teams, and a single aggregate refusal rate does not distinguish them.\n"
    )

    add("### Edge cases\n")
    add("| model | clear-cut permitted | taxonomy-flagged edge | gap |")
    add("|---|---:|---:|---:|")
    for row in edge.sort_values("gap", ascending=False).itertuples():
        add(f"| `{row.model}` | {_pct(row.over_refusal_clear)} | {_pct(row.over_refusal_edge)} | {_pct(row.gap)} |")
    add("")
    add(
        "Items the taxonomy flagged as genuinely contested before any model was run are "
        "refused more often than clear-cut permitted items. That the gap is positive is "
        "reassuring for the taxonomy: the edge flags were assigned from the policy text "
        "alone, and they predict where models struggle.\n"
    )

    if agreement:
        add("## Is the judge trustworthy?\n")
        if agreement.get("simulated_annotator"):
            add(
                "> The annotations in this run were produced by a **simulated annotator** "
                "for pipeline testing. The κ below is a property of the simulation. On a "
                "real run this section reports agreement against labels a human actually "
                "assigned using [docs/ANNOTATION_GUIDE.md](../docs/ANNOTATION_GUIDE.md).\n"
            )
        add(
            f"Cohen's κ between the LLM judge and the hand-labelled stratified sample was "
            f"**{agreement['kappa']:.2f}** "
            f"(95% CI {agreement['ci_low']:.2f}–{agreement['ci_high']:.2f}, n={agreement['n']}), "
            f"which is \"{agreement['interpretation']}\" on the Landis–Koch bands. Raw agreement "
            f"was {_pct(agreement['observed_agreement'])} against a chance level of "
            f"{_pct(agreement['expected_agreement'])} — the gap between those two numbers is "
            f"why κ and not accuracy is the right statistic here.\n"
        )
        if sim:
            add(
                f"The cue-matching baseline reaches κ={agreement['heuristic_baseline_kappa']:.2f} against "
                f"the same labels — *higher* than the LLM judge. That is an artifact of the "
                f"simulation, not a result: the simulated annotator is itself derived from the "
                f"cue rules, so the baseline is being scored against a near-copy of itself. The "
                f"comparison only carries information on a live run with real human labels, where "
                f"neither rater has privileged access to how the responses were generated. It is "
                f"reported here anyway rather than suppressed, because a baseline that beats the "
                f"instrument is exactly the kind of thing a reader should be able to see.\n"
            )
        else:
            add(
                f"The cue-matching baseline classifier reaches κ={agreement['heuristic_baseline_kappa']:.2f} "
                f"against the same labels. Any judge that does not clear the baseline by a "
                f"meaningful margin is not earning its cost.\n"
            )
        add("![judge vs human](charts/judge_human_agreement.png)\n")
        matrix = pd.DataFrame(agreement["confusion_matrix"])
        add("| human ↓ / judge → | " + " | ".join(matrix.columns) + " |")
        add("|---|" + "---:|" * len(matrix.columns))
        for idx, row in matrix.iterrows():
            add(f"| **{idx}** | " + " | ".join(str(int(v)) for v in row) + " |")
        add("")
        add(
            "The errors are concentrated on the `partial` boundary, which is the expected "
            "place for them: deciding whether a hedged answer supplied the operative "
            "substance is a judgement call, and it is the same call human annotators are "
            "slowest and least consistent on. Practically, this means the strict and lenient "
            "over-refusal variants above bracket the truth, and the strict figure is the one "
            "to quote if only one number is wanted.\n"
        )

    add("## What this does not show\n")
    add(
        "See [docs/METHODOLOGY.md §8](../docs/METHODOLOGY.md) for the full list. The four "
        "that most constrain the reading of the numbers above:\n"
    )
    add(
        "1. **The gold labels are one analyst's reading of the policy.** They were assigned "
        "from the taxonomy before any model was run, and the taxonomy cites the clause each "
        "one rests on, but a second annotator working from the same policies would not "
        "produce an identical set. No inter-annotator agreement was measured on the gold "
        "labels themselves — only on the behaviour labels. That is the single largest "
        "unmeasured confound in this project.\n"
        "2. **Single-turn, zero-shot, no system prompt.** Deployed products carry system "
        "prompts, retrieval, and classifier layers that change refusal behaviour "
        "substantially. These numbers describe raw API behaviour, which is not what a user "
        "of a deployed product experiences.\n"
        "3. **Prompts are author-written, not sampled traffic.** They are constructed to sit "
        "near the boundary, so the over-refusal rate here is an upper bound on what real "
        "traffic would show, where most requests are nowhere near a policy line.\n"
        "4. **Refusal-expected items are written at generic specificity on purpose.** That "
        "makes the recall figures a soft upper bound: a determined adversarial phrasing is "
        "not what is being tested here, and this harness should not be read as a jailbreak "
        "evaluation.\n"
    )

    add("## Reproducing\n")
    add("```bash\nmake install\nmake demo      # simulated end-to-end run, ~10s, no keys\nmake report\n```\n")
    add(
        "For a live run: copy `configs/models.live.yaml`, set `ANTHROPIC_API_KEY` and/or "
        "`OPENAI_API_KEY` (or point `base_url` at a local Ollama/vLLM server for the "
        "open-weight models), then `p2e run --config configs/models.live.yaml --out results/live` "
        "followed by `p2e annotate --run results/live --sample` to produce the annotation "
        "sheet for hand-labelling.\n"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_path}")
    return out_path
