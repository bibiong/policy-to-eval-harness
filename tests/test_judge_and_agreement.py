"""Judge parsing, determinism, and the agreement statistics."""

from __future__ import annotations

import pytest
from p2e.agreement import cohens_kappa, confusion_matrix, interpret_kappa, stratified_sample
from p2e.dataset import Item
from p2e.judge import (
    SimulatedJudge,
    decision_from_behavior,
    heuristic_label,
    parse_judgment,
)


def item(item_id: str = "x__bare", expected: str = "comply") -> Item:
    return Item(item_id, "x", "cyber_dual_use", "defensive_explanation", expected, False, "bare", "q?")


# ── judge output parsing ────────────────────────────────────────────────────────


def test_parses_clean_json():
    j = parse_judgment('{"label":"comply","confidence":0.9,"reason":"answered"}')
    assert j.label == "comply" and j.confidence == pytest.approx(0.9)


def test_parses_json_wrapped_in_prose():
    j = parse_judgment('Sure!\n```json\n{"label":"refuse","confidence":0.8}\n```\nDone.')
    assert j.label == "refuse"


def test_rejects_an_invalid_label():
    with pytest.raises(ValueError, match="invalid judge label"):
        parse_judgment('{"label":"maybe","confidence":0.5}')


def test_rejects_output_with_no_json():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_judgment("I think the model refused.")


def test_label_case_and_whitespace_are_normalised():
    assert parse_judgment('{"label":"  REFUSE ","confidence":0.5}').label == "refuse"


# ── behaviour → decision collapse ───────────────────────────────────────────────


def test_partial_collapses_to_refuse():
    assert decision_from_behavior("partial") == "refuse"
    assert decision_from_behavior("refuse") == "refuse"
    assert decision_from_behavior("comply") == "comply"


# ── heuristic baseline ──────────────────────────────────────────────────────────


def test_heuristic_detects_refusal():
    assert heuristic_label("I'm not able to help with that request.").label == "refuse"


def test_heuristic_detects_hedge():
    assert heuristic_label("I can speak to this only in general terms.").label == "partial"


def test_heuristic_defaults_to_comply():
    assert heuristic_label("Here's a substantive walkthrough. First...").label == "comply"


# ── simulated judge determinism ─────────────────────────────────────────────────


def test_simulated_judge_is_deterministic():
    a, b = SimulatedJudge(), SimulatedJudge()
    text = "I can speak to this only in general terms."
    assert a.judge(item(), text, "m").label == b.judge(item(), text, "m").label


def test_simulated_judge_differs_across_models():
    """Judge error must not be perfectly correlated across models, or per-model
    over-refusal rates would share an identical error pattern."""
    judge = SimulatedJudge(error_rate=0.5)
    labels = {judge.judge(item(f"i{n}__bare"), "I can't assist with this one.", m).label
              for n in range(40) for m in ("a", "b")}
    assert len(labels) > 1


def test_simulated_judge_mostly_agrees_with_the_cue_baseline():
    judge = SimulatedJudge()
    text = "I'm not able to help with that request."
    agree = sum(judge.judge(item(f"i{n}__bare"), text, "m").label == "refuse" for n in range(100))
    assert agree > 80


# ── Cohen's kappa ───────────────────────────────────────────────────────────────


def test_kappa_is_one_on_perfect_agreement():
    labels = ["comply", "refuse", "partial"] * 10
    assert cohens_kappa(labels, labels, bootstrap=0).kappa == pytest.approx(1.0)


def test_kappa_is_about_zero_on_chance_agreement():
    a = ["comply", "refuse"] * 50
    b = ["comply", "comply", "refuse", "refuse"] * 25
    assert abs(cohens_kappa(a, b, bootstrap=0).kappa) < 0.15


def test_kappa_is_negative_on_systematic_disagreement():
    a = ["comply"] * 25 + ["refuse"] * 25
    b = ["refuse"] * 25 + ["comply"] * 25
    assert cohens_kappa(a, b, bootstrap=0).kappa < -0.5


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa(["comply"], ["comply", "refuse"], bootstrap=0)


def test_kappa_rejects_empty_input():
    with pytest.raises(ValueError, match="no annotated items"):
        cohens_kappa([], [], bootstrap=0)


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    a = ["comply"] * 30 + ["refuse"] * 20
    b = ["comply"] * 27 + ["refuse"] * 23
    first = cohens_kappa(a, b, bootstrap=300)
    second = cohens_kappa(a, b, bootstrap=300)
    assert first.ci_low == second.ci_low and first.ci_high == second.ci_high
    assert first.ci_low <= first.kappa <= first.ci_high


def test_confusion_matrix_totals_match_the_input():
    a = ["comply", "refuse", "partial", "comply"]
    b = ["comply", "partial", "partial", "refuse"]
    matrix = confusion_matrix(a, b)
    assert matrix.values.sum() == 4
    assert matrix.loc["comply", "comply"] == 1


def test_kappa_bands():
    assert interpret_kappa(0.85) == "almost perfect"
    assert interpret_kappa(0.65) == "substantial"
    assert interpret_kappa(-0.1) == "worse than chance"


# ── stratified sampling ─────────────────────────────────────────────────────────


def test_stratified_sample_covers_every_stratum():
    import pandas as pd

    rows = [
        {"model": m, "item_id": f"{c}-{b}-{m}", "category": c, "expected": e, "behavior": b}
        for m in ("m1", "m2")
        for c in ("cyber_dual_use", "medical_information")
        for e, b in (("comply", "comply"), ("comply", "refuse"), ("refuse", "refuse"))
    ]
    sample = stratified_sample(pd.DataFrame(rows), per_stratum=1)
    assert len(sample) == 6
    assert set(sample["behavior"]) == {"comply", "refuse"}


def test_stratified_sample_is_deterministic():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"model": "m", "item_id": f"i{n}", "category": "c", "expected": "comply", "behavior": "comply"}
            for n in range(50)
        ]
    )
    a = stratified_sample(df, per_stratum=5)["item_id"].tolist()
    b = stratified_sample(df, per_stratum=5)["item_id"].tolist()
    assert a == b


# ── .env loading ────────────────────────────────────────────────────────────────


def test_dotenv_parses_the_forms_that_appear_in_key_files(tmp_path, monkeypatch):
    from p2e.env import load_dotenv

    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        'export ANTHROPIC_API_KEY="sk-ant-quoted"\n'
        "OPENAI_API_KEY=sk-plain\n"
        "MALFORMED_LINE\n",
        encoding="utf-8",
    )
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    loaded = load_dotenv(path)
    import os

    assert set(loaded) == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-quoted"
    assert os.environ["OPENAI_API_KEY"] == "sk-plain"


def test_real_env_var_beats_the_file(tmp_path, monkeypatch):
    """An explicitly exported key must not be silently replaced by the file."""
    from p2e.env import load_dotenv

    path = tmp_path / ".env"
    path.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    assert load_dotenv(path) == []
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_missing_dotenv_is_not_an_error(tmp_path):
    from p2e.env import load_dotenv

    assert load_dotenv(tmp_path / "nope") == []
