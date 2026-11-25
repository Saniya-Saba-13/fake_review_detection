"""Matplotlib / Seaborn figures for the Model Performance and Dataset pages (ReviewGuard palette)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

VIOLET, ORANGE, EMERALD, ROSE, INK = "#6D28D9", "#F97316", "#059669", "#E11D48", "#1E1B2E"
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#D9D4EA", "grid.color": "#EEEAF7"})
# fonts that can draw Devanagari when installed (Windows ships Nirmala UI / Mangal); DejaVu Sans is the fallback
plt.rcParams["font.sans-serif"] = ["Nirmala UI", "Mangal", "Noto Sans Devanagari", "DejaVu Sans"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path).name


def plot_class_by_language(df, path):
    d = df.assign(Class=df["label"].map({0: "Genuine", 1: "Fake"}))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.countplot(data=d, x="language", hue="Class", palette={"Genuine": EMERALD, "Fake": ROSE}, ax=ax)
    ax.set_title("Training data after balancing: classes per language")
    ax.set_xlabel("")
    ax.set_ylabel("Reviews")
    return _save(fig, path)


def plot_platform_distribution(df, path):
    d = df.assign(Class=df["label"].map({0: "Genuine", 1: "Fake"}))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.countplot(data=d, x="platform", hue="Class", palette={"Genuine": EMERALD, "Fake": ROSE}, ax=ax)
    ax.set_title("Training data: reviews per platform label")
    ax.set_xlabel("")
    ax.set_ylabel("Reviews")
    return _save(fig, path)


def plot_model_comparison(rows, path):
    best = {}
    for r in rows:
        if r["model"] not in best or r["val_f1"] > best[r["model"]]["val_f1"]:
            best[r["model"]] = r
    names = list(best)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.bar(x - 0.2, [best[n]["val_f1"] for n in names], 0.4, label="F1 (fake class)", color=VIOLET)
    ax.bar(x + 0.2, [best[n]["val_roc_auc"] for n in names], 0.4, label="ROC-AUC", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation comparison (best setting per algorithm)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                xticklabels=["Genuine", "Fake"], yticklabels=["Genuine", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out test set)")
    return _save(fig, path)


def plot_roc(y_true, y_score, auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr, tpr, color=VIOLET, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (held-out test set)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_language_metrics(by_language, path):
    langs = [k for k, v in by_language.items() if v.get("n", 0) > 0]
    x = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - 0.27, [by_language[k]["f1"] for k in langs], 0.27, label="F1 (fake class)", color=VIOLET)
    ax.bar(x, [by_language[k]["recall"] for k in langs], 0.27, label="Recall (fakes caught)", color=ORANGE)
    ax.bar(x + 0.27, [by_language[k]["fpr"] for k in langs], 0.27, label="False-positive rate (genuine flagged)", color=ROSE)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test results per language")
    ax.legend(loc="upper center", fontsize=8, ncol=1)
    return _save(fig, path)


def plot_top_features(names, weights, is_word, path, n=12):
    if weights is None:
        return None
    # the PNG shows Latin-script words only (always renderable); Hindi words are listed in the HTML table on the page
    idx = np.array([i for i in np.where(is_word)[0] if names[i][6:].isascii()])
    if len(idx) < 2 * n:
        return None
    order = idx[np.argsort(weights[idx])]
    sel = list(order[:n]) + list(order[-n:])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh([names[i][6:] for i in sel], weights[sel], color=[EMERALD] * n + [ROSE] * n)
    ax.set_title("Most influential words (rose = fake, green = genuine)")
    ax.set_xlabel("Model weight")
    return _save(fig, path)
