"""Charts for the findings report.

Deliberately plain: no seaborn, no gradients, one idea per figure, and a
"SIMULATED RUN" stamp on every figure produced from simulated models so a chart
lifted out of the repo cannot be mistaken for a measurement of a real system.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1c1c1c"
ACCENT = "#c1440e"
MUTED = "#9a9a9a"
GRID = "#e3e3e3"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _stamp(fig, simulated: bool) -> None:
    if simulated:
        fig.text(
            0.995, 0.005, "SIMULATED RUN — not a measurement of any deployed model",
            ha="right", va="bottom", fontsize=7, color=ACCENT, alpha=0.85,
        )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def over_refusal_by_model(per_model_df: pd.DataFrame, out: Path, simulated: bool) -> Path:
    df = per_model_df.sort_values("over_refusal")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = range(len(df))
    ax.barh(list(y), df["over_refusal"] * 100, color=INK, height=0.55)
    ax.errorbar(
        df["over_refusal"] * 100, list(y),
        xerr=[
            (df["over_refusal"] - df["over_refusal_ci_low"]) * 100,
            (df["over_refusal_ci_high"] - df["over_refusal"]) * 100,
        ],
        fmt="none", ecolor=ACCENT, elinewidth=1.2, capsize=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("over-refusal rate on policy-permitted requests (%)")
    ax.set_title("Legitimate requests the model would not answer", loc="left", color=INK, fontsize=11)
    _style(ax)
    _stamp(fig, simulated)
    return _save(fig, out)


def over_refusal_by_category(cat_df: pd.DataFrame, out: Path, simulated: bool) -> Path:
    df = cat_df.sort_values("over_refusal")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    y = range(len(df))
    ax.barh(list(y), df["over_refusal"] * 100, color=INK, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels([c.replace("_", " ") for c in df["category"]])
    ax.set_xlabel("over-refusal rate, pooled across models (%)")
    ax.set_title("Where refusals land on permitted requests", loc="left", color=INK, fontsize=11)
    _style(ax)
    _stamp(fig, simulated)
    return _save(fig, out)


def ambiguity_scatter(cat_df: pd.DataFrame, out: Path, simulated: bool) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(
        cat_df["over_refusal"] * 100, cat_df["excess_disagreement"],
        s=60, color=INK, zorder=3,
    )
    for row in cat_df.itertuples():
        ax.annotate(
            row.category.replace("_", " "),
            (row.over_refusal * 100, row.excess_disagreement),
            textcoords="offset points", xytext=(7, 3), fontsize=8, color=INK,
        )
    ax.set_xlabel("over-refusal rate (%)")
    ax.set_ylabel("excess disagreement (observed − marginal baseline)")
    ax.set_title(
        "Toward 0 = refusals are uncorrelated across models;\nthe policy text is not settling the case",
        loc="left", color=INK, fontsize=10,
    )
    _style(ax)
    _stamp(fig, simulated)
    return _save(fig, out)


def framing_sensitivity(fs_df: pd.DataFrame, out: Path, simulated: bool) -> Path:
    order = ["bare", "student", "professional", "urgent_personal"]
    pivot = fs_df.pivot(index="framing", columns="model", values="over_refusal").reindex(
        [f for f in order if f in set(fs_df["framing"])]
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model] * 100, marker="o", linewidth=1.6, label=model)
    ax.set_ylabel("over-refusal rate (%)")
    ax.set_xlabel("how the same request was framed")
    ax.set_title("Does the model read stated context, or just the topic?", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax)
    _stamp(fig, simulated)
    return _save(fig, out)


def agreement_matrix(matrix: pd.DataFrame, out: Path, simulated: bool) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=matrix.values.max() or 1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("judge label")
    ax.set_ylabel("human label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(
                j, i, str(value), ha="center", va="center", fontsize=10,
                color="white" if value > matrix.values.max() * 0.55 else INK,
            )
    ax.set_title("Judge vs human on the annotated sample", loc="left", color=INK, fontsize=11)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=9)
    _stamp(fig, simulated)
    return _save(fig, out)
