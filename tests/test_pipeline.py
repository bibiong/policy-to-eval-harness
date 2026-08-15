"""End-to-end pipeline tests, plus regression locks on the shipped run.

The headline numbers are asserted here on purpose. The whole pipeline is
deterministic, so any change that silently moves the over-refusal rate or κ
fails the build instead of quietly rewriting the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from p2e import report, run_eval
from p2e.annotate import build_sample
from p2e.dataset import Item
from p2e.providers import SIM_PROFILES, SimulatedProvider

RUN = Path("results/demo_sim")

# Locked to the shipped simulated run. If a deliberate change moves these,
# re-run `make demo report` and update them in the same commit as the change.
EXPECTED_OVER_REFUSAL = 0.268
EXPECTED_KAPPA = 0.73
TOLERANCE = 0.005


def item(item_id="cyb01__bare", category="cyber_dual_use", expected="comply", edge=False, framing="bare"):
    return Item(item_id, item_id.split("__")[0], category, "defensive_explanation", expected, edge, framing, "q?")


# ── simulator behaviour ─────────────────────────────────────────────────────────


def test_simulator_is_deterministic():
    a = SimulatedProvider("sim-open-8b")
    b = SimulatedProvider("sim-open-8b")
    assert a.generate("p", item()) == b.generate("p", item())


def test_simulator_profiles_differ():
    outputs = {SimulatedProvider(n).generate("p", item()) for n in SIM_PROFILES}
    assert len(outputs) > 1


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        SimulatedProvider("does-not-exist")


def test_simulator_refuses_prohibited_requests_far_more_often():
    provider = SimulatedProvider("sim-open-8b")
    benign = sum(
        provider._behavior(item(f"i{n}__bare", expected="comply")) == "comply" for n in range(200)
    )
    prohibited = sum(
        provider._behavior(item(f"i{n}__bare", expected="refuse")) == "comply" for n in range(200)
    )
    assert benign > prohibited * 4


def test_stated_context_reduces_refusal_for_context_sensitive_profiles():
    provider = SimulatedProvider("sim-api-frontier")
    bare = sum(provider._behavior(item(f"i{n}__bare", framing="bare")) != "comply" for n in range(300))
    prof = sum(
        provider._behavior(item(f"i{n}__professional", framing="professional")) != "comply"
        for n in range(300)
    )
    assert prof < bare


# ── full pipeline on a temp directory ───────────────────────────────────────────


def test_pipeline_runs_end_to_end(tmp_path):
    out = run_eval.run(config_path="configs/models.yaml", out_dir=tmp_path / "run", limit=40)
    for name in ("dataset.jsonl", "responses.jsonl", "scored.csv", "manifest.json"):
        assert (out / name).exists(), name

    scored = pd.read_csv(out / "scored.csv")
    assert len(scored) == 40 * 6
    assert set(scored["behavior"]) <= {"comply", "partial", "refuse"}
    assert scored["simulated"].all()

    build_sample(out, out_path=tmp_path / "labels.csv", per_stratum=2, simulate=True)
    data = report.compute(out, annotations=tmp_path / "labels.csv")
    assert 0.0 <= data["overall"]["over_refusal"]["rate"] <= 1.0
    assert data["agreement"]["simulated_annotator"] is True


def test_charts_and_report_are_written(tmp_path):
    out = run_eval.run(config_path="configs/models.yaml", out_dir=tmp_path / "run", limit=40)
    build_sample(out, out_path=tmp_path / "labels.csv", per_stratum=2, simulate=True)
    report.compute(out, annotations=tmp_path / "labels.csv")
    charts = report.make_charts(out, out_dir=tmp_path / "charts")
    assert len(charts) == 5 and all(p.exists() and p.stat().st_size > 1000 for p in charts)

    path = report.render(out, out_path=tmp_path / "findings.md")
    text = path.read_text(encoding="utf-8")
    assert "This run is simulated" in text
    assert "What this does not show" in text


def test_simulated_runs_are_flagged_everywhere(tmp_path):
    """A reader must not be able to mistake a simulated run for a real one."""
    out = run_eval.run(config_path="configs/models.yaml", out_dir=tmp_path / "run", limit=20)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["all_simulated"] is True
    assert pd.read_csv(out / "scored.csv")["simulated"].all()


# ── regression locks on the shipped run ─────────────────────────────────────────


@pytest.mark.skipif(not RUN.exists(), reason="run `make demo` first")
def test_shipped_run_has_the_expected_shape():
    scored = pd.read_csv(RUN / "scored.csv")
    assert len(scored) == 2400
    assert scored["model"].nunique() == 6
    assert scored["item_id"].nunique() == 400
    assert scored["category"].nunique() == 12


@pytest.mark.skipif(not RUN.exists(), reason="run `make demo` first")
def test_shipped_over_refusal_is_unchanged():
    data = json.loads((RUN / "metrics.json").read_text())
    assert data["overall"]["over_refusal"]["rate"] == pytest.approx(EXPECTED_OVER_REFUSAL, abs=TOLERANCE)


@pytest.mark.skipif(not RUN.exists(), reason="run `make demo` first")
def test_shipped_kappa_is_unchanged():
    data = json.loads((RUN / "metrics.json").read_text())
    assert data["agreement"]["kappa"] == pytest.approx(EXPECTED_KAPPA, abs=0.02)


@pytest.mark.skipif(not RUN.exists(), reason="run `make demo` first")
def test_baseline_comparison_is_reported_and_its_artifact_is_known():
    """On a SIMULATED run the cue baseline necessarily beats the LLM judge.

    The simulated annotator derives from the same cue rules the baseline uses, so
    the baseline is scored against a near-copy of itself. This test locks in that
    known artifact so it cannot be mistaken for a result, and so the day someone
    "fixes" the judge to beat it, the test forces them to notice why it happened.
    On a live run with real human labels the expected relation is the opposite.
    """
    data = json.loads((RUN / "metrics.json").read_text())
    assert data["manifest"]["all_simulated"] is True
    assert data["agreement"]["heuristic_baseline_kappa"] > data["agreement"]["kappa"]
    assert data["agreement"]["kappa"] > 0.6


@pytest.mark.skipif(not RUN.exists(), reason="run `make demo` first")
def test_models_are_ordered_as_the_profiles_intend():
    """The guarded profile must over-refuse more than the frontier profile —
    if this inverts, the simulator's tuning has drifted."""
    data = json.loads((RUN / "metrics.json").read_text())
    rates = {r["model"]: r["over_refusal"] for r in data["per_model"]}
    assert rates["sim-guarded-7b"] > rates["sim-api-frontier"] * 2


@pytest.mark.skipif(not RUN.exists(), reason="run `make demo` first")
def test_edge_items_are_refused_more_than_clear_cut_ones():
    """The edge flags were assigned from policy text before any run. They should
    predict where models struggle."""
    data = json.loads((RUN / "metrics.json").read_text())
    assert sum(r["gap"] for r in data["edge_gap"]) > 0
