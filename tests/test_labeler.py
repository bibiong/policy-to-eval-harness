"""Tests for the interactive labeller.

The property that matters most is the blinding: if the judge's label reaches the
screen, kappa stops measuring anything.
"""

from __future__ import annotations

import builtins

import pandas as pd
import pytest
from p2e.labeler import label


@pytest.fixture
def sheet(tmp_path):
    rows = [
        {
            "model": "m", "item_id": f"i{n}", "category": "cyber_dual_use",
            "subcategory": "defensive_explanation", "framing": "bare", "expected": "comply",
            "edge": False, "prompt": f"prompt {n}", "response": f"response {n}",
            "judge_behavior": "SECRET_JUDGE_LABEL", "judge_confidence": 0.9,
            "judge_reason": "SECRET_REASON", "human_behavior": "", "annotator": "",
            "annotator_notes": "",
        }
        for n in range(3)
    ]
    path = tmp_path / "sheet.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _drive(monkeypatch, keys):
    it = iter(keys)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


def test_never_displays_the_judge_label(sheet, monkeypatch, capsys):
    _drive(monkeypatch, ["c", "r", "p"])
    label(sheet, annotator="BO")
    out = capsys.readouterr().out
    assert "SECRET_JUDGE_LABEL" not in out
    assert "SECRET_REASON" not in out


def test_records_labels_and_annotator(sheet, monkeypatch):
    _drive(monkeypatch, ["c", "r", "p"])
    label(sheet, annotator="BO")
    df = pd.read_csv(sheet)
    assert df["human_behavior"].tolist() == ["comply", "refuse", "partial"]
    assert set(df["annotator"]) == {"BO"}


def test_saves_after_every_row_so_a_quit_keeps_work(sheet, monkeypatch):
    _drive(monkeypatch, ["c", "q"])
    label(sheet, annotator="BO")
    df = pd.read_csv(sheet)
    assert df.loc[0, "human_behavior"] == "comply"
    assert str(df.loc[1, "human_behavior"]) in ("nan", "")


def test_skip_leaves_the_row_unlabelled(sheet, monkeypatch):
    _drive(monkeypatch, ["s", "c", "c"])
    label(sheet, annotator="BO")
    df = pd.read_csv(sheet)
    assert str(df.loc[0, "human_behavior"]) in ("nan", "")
    assert df.loc[1, "human_behavior"] == "comply"


def test_resuming_only_offers_unlabelled_rows(sheet, monkeypatch, capsys):
    _drive(monkeypatch, ["c", "q"])
    label(sheet, annotator="BO")
    _drive(monkeypatch, ["r", "q"])
    label(sheet, annotator="BO")
    assert "2 rows to label" in capsys.readouterr().out
    assert pd.read_csv(sheet).loc[1, "human_behavior"] == "refuse"


def test_invalid_key_reprompts_rather_than_crashing(sheet, monkeypatch):
    _drive(monkeypatch, ["x", "9", "c", "q"])
    label(sheet, annotator="BO")
    assert pd.read_csv(sheet).loc[0, "human_behavior"] == "comply"


def test_note_is_recorded_without_consuming_the_label(sheet, monkeypatch):
    _drive(monkeypatch, ["n", "torn between p and r", "r", "q"])
    label(sheet, annotator="BO")
    df = pd.read_csv(sheet)
    assert df.loc[0, "annotator_notes"] == "torn between p and r"
    assert df.loc[0, "human_behavior"] == "refuse"
