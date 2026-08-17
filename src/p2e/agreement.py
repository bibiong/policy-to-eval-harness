"""Judge–human agreement.

An LLM judge is a measurement instrument, and an uncalibrated instrument makes
every downstream number unfalsifiable. This module reports Cohen's κ between the
judge and a hand-labelled stratified sample, plus the confusion matrix, so a
reader can see *where* the instrument fails rather than only how often.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from .judge import BEHAVIORS


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    observed_agreement: float
    expected_agreement: float
    n: int
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "n": self.n,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) bands. Conventional, and conventionally over-read."""
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def confusion_matrix(a: list[str], b: list[str], labels: tuple[str, ...] = BEHAVIORS) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=list(labels), columns=list(labels), dtype=int)
    for x, y in zip(a, b, strict=False):
        matrix.loc[x, y] += 1
    matrix.index.name = "human"
    matrix.columns.name = "judge"
    return matrix


def cohens_kappa(
    a: list[str], b: list[str], labels: tuple[str, ...] = BEHAVIORS, bootstrap: int = 2000
) -> KappaResult:
    if len(a) != len(b):
        raise ValueError("rater vectors must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("no annotated items")

    def _kappa(xa: list[str], xb: list[str]) -> float:
        m = len(xa)
        observed = sum(1 for x, y in zip(xa, xb, strict=False) if x == y) / m
        expected = sum(
            (xa.count(lab) / m) * (xb.count(lab) / m) for lab in labels
        )
        if expected >= 1.0:
            return 1.0 if observed >= 1.0 else 0.0
        return (observed - expected) / (1 - expected)

    observed = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    kappa = _kappa(a, b)

    # Deterministic bootstrap: resampling indices come from a hash chain, so the
    # reported CI is byte-identical on every machine and in CI.
    lo = hi = float("nan")
    if bootstrap and n >= 10:
        stats = []
        for i in range(bootstrap):
            idx = []
            digest = hashlib.sha256(f"boot{i}".encode()).digest()
            pool = int.from_bytes(digest, "big")
            for _ in range(n):
                idx.append(pool % n)
                pool //= n
                if pool < n:
                    digest = hashlib.sha256(digest).digest()
                    pool = int.from_bytes(digest, "big")
            stats.append(_kappa([a[j] for j in idx], [b[j] for j in idx]))
        stats.sort()
        lo = stats[int(0.025 * bootstrap)]
        hi = stats[int(0.975 * bootstrap) - 1]

    return KappaResult(kappa, observed, expected, n, lo, hi)


ORDINAL = ("refuse", "partial", "comply")  # least → most substance delivered


def weighted_kappa(
    a: list[str], b: list[str], labels: tuple[str, ...] = ORDINAL, power: int = 1
) -> float:
    """Cohen's kappa with ordinal (linear or quadratic) disagreement weights.

    The three labels are ordered — refuse < partial < comply — so a comply/partial
    split is a smaller error than a comply/refuse split. Unweighted kappa treats
    those as identical, which understates agreement whenever the raters differ by
    one notch rather than disagreeing outright.

    Report both. Unweighted is the conservative number; weighted is the one that
    reflects how the labels are actually used downstream, where `partial` and
    `refuse` collapse onto the same decision anyway.
    """
    if len(a) != len(b):
        raise ValueError("rater vectors must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("no annotated items")

    index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    max_dist = (k - 1) ** power

    observed = 0.0
    for x, y in zip(a, b, strict=True):
        observed += 1 - abs(index[x] - index[y]) ** power / max_dist

    pa = [a.count(lab) / n for lab in labels]
    pb = [b.count(lab) / n for lab in labels]
    expected = 0.0
    for i in range(k):
        for j in range(k):
            expected += pa[i] * pb[j] * (1 - abs(i - j) ** power / max_dist)

    observed /= n
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def disagreement_direction(a: list[str], b: list[str], labels: tuple[str, ...] = ORDINAL) -> dict:
    """Is disagreement one-directional (a calibration offset) or symmetric (noise)?

    A rater pair that disagrees at random should split roughly evenly on which
    side is stricter. A lopsided split means one rater is systematically shifted
    along the scale — which is a fixable calibration problem, not irreducible
    noise, and is diagnosed very differently.
    """
    index = {lab: i for i, lab in enumerate(labels)}
    b_stricter = sum(1 for x, y in zip(a, b, strict=True) if index[y] < index[x])
    a_stricter = sum(1 for x, y in zip(a, b, strict=True) if index[y] > index[x])
    adjacent = sum(
        1 for x, y in zip(a, b, strict=True) if abs(index[x] - index[y]) == 1
    )
    total = b_stricter + a_stricter
    return {
        "n_disagreements": total,
        "second_rater_stricter": b_stricter,
        "first_rater_stricter": a_stricter,
        "one_directional_share": (max(b_stricter, a_stricter) / total) if total else float("nan"),
        "adjacent_share": (adjacent / total) if total else float("nan"),
    }


def stratified_sample(
    df: pd.DataFrame, per_stratum: int = 2, strata: tuple[str, ...] = ("category", "expected", "behavior")
) -> pd.DataFrame:
    """Deterministic stratified sample for hand-labelling.

    Stratifying on the judge's own label is deliberate: an unstratified sample
    of a dataset that is mostly compliance would put almost no refusals in front
    of the human annotator, and κ would be estimated from a handful of cases in
    the class that matters.
    """
    df = df.copy()
    df["_key"] = [
        hashlib.sha256(f"{r.model}|{r.item_id}".encode()).hexdigest() for r in df.itertuples()
    ]
    picked = (
        df.sort_values("_key")
        .groupby(list(strata), sort=True, group_keys=False)
        .head(per_stratum)
    )
    return picked.drop(columns=["_key"]).sort_values(["category", "model", "item_id"]).reset_index(drop=True)
