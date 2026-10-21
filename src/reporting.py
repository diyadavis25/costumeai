"""Matplotlib figures for model evaluation (used by evaluate_model.py and the dashboard)."""
import numpy as np
from matplotlib.figure import Figure

INK, VIOLET, MINT, CORAL, AMBER, MUTED = "#15151F", "#5A3CF0", "#19B37F", "#FF6B57", "#F5A524", "#8E8BA3"
PALETTE = [VIOLET, MINT, AMBER, CORAL]
LINES = [VIOLET, MINT, AMBER, CORAL, "#2B8CFF", "#B04FD8", "#8E8BA3", INK]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.title.set_color(INK)


def confusion_fig(metrics, model=None):
    model = model or metrics["selected_model"]
    cm = np.array(metrics["models"][model]["test"]["confusion_matrix"])
    fig = Figure(figsize=(4.2, 3.6))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Purples")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15, fontweight="bold",
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Not compatible", "Compatible"])
    ax.set_yticks([0, 1], ["Not compatible", "Compatible"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test) - {model}", fontsize=10)
    fig.tight_layout()
    return fig


def _short(n):
    n = n.replace(" (XGBoost not installed)", "*")
    if n.startswith("Ensemble ("):
        return "Ensemble (top 3)"
    return n


def comparison_fig(metrics):
    names = list(metrics["models"])
    keys = ["accuracy", "precision", "recall", "f1"]
    fig = Figure(figsize=(7, max(3.6, 0.55 * len(names) + 1.4)))
    ax = fig.subplots()
    h = 0.2
    y = np.arange(len(names))
    for i, k in enumerate(keys):
        vals = [metrics["models"][n]["test"][k] for n in names]
        ax.barh(y + (i - 1.5) * h, vals, h, label=k.capitalize() if k != "f1" else "F1", color=PALETTE[i])
    ax.set_yticks(y, [_short(n) for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Score (axis starts at 0.5)")
    ax.set_title("Model comparison (test set)", fontsize=11)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def roc_fig(metrics):
    fig = Figure(figsize=(4.4, 3.6))
    ax = fig.subplots()
    for i, (n, m) in enumerate(metrics["models"].items()):
        ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], color=LINES[i % len(LINES)], lw=2,
                label=f"{_short(n)} (AUC {m['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    return fig


def importance_fig(metrics):
    items = metrics["feature_importance"][::-1]
    fig = Figure(figsize=(5.2, 3.8))
    ax = fig.subplots()
    ax.barh([i["feature"] for i in items], [i["importance"] for i in items], color=VIOLET)
    ax.set_title("What the model relies on (F1 drop)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    return fig


def score_hist_fig(scores):
    fig = Figure(figsize=(6, 2.4))
    ax = fig.subplots()
    ax.hist(scores, bins=np.arange(0, 105, 5), color=VIOLET, edgecolor="white")
    for x, c in ((50, AMBER), (65, MINT), (80, MINT)):
        ax.axvline(x, color=c, lw=1, ls="--")
    ax.set_xlabel("Compatibility score")
    ax.set_ylabel("Outfits")
    ax.set_title("Score distribution of all wardrobe combinations", fontsize=10)
    _style(ax)
    fig.tight_layout()
    return fig