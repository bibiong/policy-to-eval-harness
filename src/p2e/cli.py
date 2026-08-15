"""Command line interface: p2e <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import annotate, report, run_eval
from .dataset import build_dataset, write_dataset
from .taxonomy import load_taxonomy


def _cmd_taxonomy(args: argparse.Namespace) -> int:
    tax = load_taxonomy(args.taxonomy)
    print(f"taxonomy v{tax.version} — {len(tax.categories)} categories")
    for cat in tax.categories:
        subs = ", ".join(f"{s.id}[{s.expected}]" for s in cat.subcategories)
        print(f"  {cat.id:24s} {len(cat.subcategories)} subcats  refs={','.join(cat.policy_refs)}")
        print(f"    {subs}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    tax = load_taxonomy(args.taxonomy)
    items = build_dataset(tax, args.seeds)
    path = write_dataset(items, args.out)
    benign = sum(1 for i in items if i.expected == "comply")
    print(f"built {len(items)} items ({benign} permitted / {len(items) - benign} prohibited) → {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    print(f"running {args.config}")
    run_eval.run(
        config_path=args.config,
        out_dir=args.out,
        taxonomy_path=args.taxonomy,
        seeds_path=args.seeds,
        limit=args.limit,
        workers=args.workers,
    )
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    annotate.build_sample(
        run_dir=args.run,
        out_path=args.out,
        per_stratum=args.per_stratum,
        simulate=args.simulate,
    )
    if not args.simulate:
        print("  fill in the `human_behavior` column using docs/ANNOTATION_GUIDE.md, then run `p2e report`")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    data = report.compute(args.run, annotations=args.annotations)
    report.make_charts(args.run, out_dir=args.charts)
    report.render(args.run, out_path=args.out)
    over = data["overall"]["over_refusal"]
    print(f"  over-refusal: {over['rate'] * 100:.1f}% ({over['k']}/{over['n']})")
    if "agreement" in data:
        print(f"  judge–human κ: {data['agreement']['kappa']:.2f} (n={data['agreement']['n']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p2e", description=__doc__)
    parser.add_argument("--taxonomy", default="policy/taxonomy.yaml")
    parser.add_argument("--seeds", default="data/seeds/prompts_seed.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("taxonomy", help="validate and print the policy taxonomy")
    p.set_defaults(func=_cmd_taxonomy)

    p = sub.add_parser("build", help="build the prompt dataset")
    p.add_argument("--out", default="results/dataset.jsonl")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("run", help="generate responses and judge them")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--out", default="results/demo_sim")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("annotate", help="build the human-annotation sample")
    p.add_argument("--run", default="results/demo_sim")
    p.add_argument("--out", default="data/annotations/human_labels.csv")
    p.add_argument("--per-stratum", type=int, default=2)
    p.add_argument("--sample", action="store_true", help="(default) emit a blank sheet to fill in")
    p.add_argument("--simulate", action="store_true", help="fill labels with a simulated annotator")
    p.set_defaults(func=_cmd_annotate)

    p = sub.add_parser("report", help="compute metrics, draw charts, render findings")
    p.add_argument("--run", default="results/demo_sim")
    p.add_argument("--annotations", default="data/annotations/human_labels.csv")
    p.add_argument("--charts", default="reports/charts")
    p.add_argument("--out", default="reports/findings.md")
    p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.taxonomy).exists():
        print(f"error: {args.taxonomy} not found — run from the repo root", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
