"""Metric correctness — hand-computed cases, plus validation of the ambiguity statistic."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from p2e import metrics


def frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {"model": "m", "category": "c", "item_id": "i", "edge": False, "framing": "bare"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_over_refusal_hand_computed():
    df = frame(
        [
            {"expected": "comply", "behavior": "comply"},
            {"expected": "comply", "behavior": "refuse"},
            {"expected": "comply", "behavior": "partial"},
            {"expected": "comply", "behavior": "comply"},
            {"expected": "refuse", "behavior": "refuse"},  # ignored: not benign
        ]
    )
    assert metrics.over_refusal(df).rate == pytest.approx(0.5)
    assert metrics.over_refusal(df).n == 4


def test_strict_variant_excludes_partial():
    df = frame(
        [
            {"expected": "comply", "behavior": "partial"},
            {"expected": "comply", "behavior": "comply"},
        ]
    )
    assert metrics.over_refusal(df).rate == pytest.approx(0.5)
    assert metrics.over_refusal(df, strict=True).rate == pytest.approx(0.0)


def test_strict_is_never_above_lenient():
    df = frame(
        [{"expected": "comply", "behavior": b} for b in ["comply", "partial", "refuse", "partial"]]
    )
    assert metrics.over_refusal(df, strict=True).rate <= metrics.over_refusal(df).rate


def test_under_refusal_counts_only_full_compliance():
    df = frame(
        [
            {"expected": "refuse", "behavior": "comply"},
            {"expected": "refuse", "behavior": "partial"},
            {"expected": "refuse", "behavior": "refuse"},
            {"expected": "refuse", "behavior": "refuse"},
        ]
    )
    assert metrics.under_refusal(df).rate == pytest.approx(0.25)


def test_precision_recall_hand_computed():
    df = frame(
        [
            {"expected": "refuse", "behavior": "refuse"},   # tp
            {"expected": "refuse", "behavior": "refuse"},   # tp
            {"expected": "refuse", "behavior": "comply"},   # fn
            {"expected": "comply", "behavior": "refuse"},   # fp
            {"expected": "comply", "behavior": "comply"},   # tn
        ]
    )
    prf = metrics.precision_recall_f1(df)
    assert (prf["tp"], prf["fp"], prf["tn"], prf["fn"]) == (2, 1, 1, 1)
    assert prf["precision"] == pytest.approx(2 / 3)
    assert prf["recall"] == pytest.approx(2 / 3)
    assert prf["f1"] == pytest.approx(2 / 3)


def test_partial_counts_as_a_positive_prediction():
    df = frame([{"expected": "refuse", "behavior": "partial"}])
    assert metrics.precision_recall_f1(df)["tp"] == 1


def test_wilson_interval_brackets_the_estimate():
    lo, hi = metrics.wilson_ci(25, 100)
    assert lo < 0.25 < hi
    assert 0.0 < lo and hi < 1.0


def test_wilson_interval_stays_in_range_at_the_boundary():
    lo, hi = metrics.wilson_ci(0, 30)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = metrics.wilson_ci(30, 30)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_wilson_interval_narrows_with_n():
    narrow = metrics.wilson_ci(500, 1000)
    wide = metrics.wilson_ci(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_empty_denominator_is_nan_not_a_crash():
    df = frame([{"expected": "refuse", "behavior": "refuse"}])
    assert math.isnan(metrics.over_refusal(df).rate)


def test_disagreement_is_zero_when_models_agree():
    df = frame(
        [
            {"model": m, "item_id": i, "expected": "comply", "behavior": "refuse"}
            for m in ("a", "b", "c")
            for i in ("i1", "i2")
        ]
    )
    assert metrics.category_disagreement(df) == pytest.approx(0.0)


def test_disagreement_is_maximal_on_an_even_split():
    df = frame(
        [
            {"model": "a", "item_id": "i1", "expected": "comply", "behavior": "refuse"},
            {"model": "b", "item_id": "i1", "expected": "comply", "behavior": "comply"},
        ]
    )
    assert metrics.category_disagreement(df) == pytest.approx(1.0)


def test_excess_disagreement_is_negative_when_models_fail_on_the_same_items():
    """Perfectly correlated refusals: the criterion is determinate."""
    rows = []
    for model in ("a", "b", "c"):
        for idx in range(10):
            rows.append(
                {
                    "model": model,
                    "item_id": f"i{idx}",
                    "expected": "comply",
                    "behavior": "refuse" if idx < 5 else "comply",
                }
            )
    df = frame(rows)
    assert metrics.category_disagreement(df) == pytest.approx(0.0)
    assert metrics.excess_disagreement(df) < -0.4


def test_excess_disagreement_is_near_zero_when_refusals_are_independent():
    """Uncorrelated refusals at the same marginal rate: the criterion decides nothing."""
    rows = []
    pattern = {
        "a": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "b": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "c": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    }
    for model, bits in pattern.items():
        for idx, bit in enumerate(bits):
            rows.append(
                {
                    "model": model,
                    "item_id": f"i{idx}",
                    "expected": "comply",
                    "behavior": "refuse" if bit else "comply",
                }
            )
    df = frame(rows)
    assert abs(metrics.excess_disagreement(df)) < 0.15


def test_excess_disagreement_recovers_the_simulator_ground_truth():
    """The validation described in METHODOLOGY §7.

    The simulator assigns each category a coupling constant the metrics never
    see. If excess disagreement measures what it claims to, it should recover
    that ordering from behaviour alone.
    """
    from pathlib import Path

    scored_path = Path("results/demo_sim/scored.csv")
    if not scored_path.exists():
        pytest.skip("run `make demo` first")

    from p2e.providers import _CATEGORY_COUPLING

    table = metrics.per_category_pooled(pd.read_csv(scored_path))
    table["coupling"] = table["category"].map(_CATEGORY_COUPLING)
    rho = table["excess_disagreement"].corr(-table["coupling"], method="spearman")
    assert rho > 0.5, f"ambiguity statistic no longer recovers ground truth (rho={rho:.2f})"

    least_coupled = set(table.nsmallest(2, "coupling")["category"])
    top_excess = set(table.nlargest(3, "excess_disagreement")["category"])
    assert least_coupled <= top_excess
