"""Train, compare and register the multilingual fake-review classifier.

Usage (PowerShell):
    python build_dataset.py --demo        # once: SYNTHETIC demo data (or ingest your own labelled files first)
    python train_model.py                 # verified data if there is enough, otherwise the demo data (flagged)
    python train_model.py --data-mode all --no-balance

Pipeline: labelled reviews (database) -> unified frame -> split 70/15/15 (stratified by language x label) BEFORE any fitting
-> TF-IDF word + character n-grams (handles Devanagari and Hinglish spelling variants) -> Logistic Regression / Multinomial
Naive Bayes / Linear SVM / Random Forest compared on the VALIDATION split -> best model evaluated once on the TEST split,
overall and per language and per platform -> saved as a new model version and made active.
The TF-IDF vocabulary is learned inside the Pipeline from the training split only (no leakage). Platform and language are
NOT model inputs, so the model cannot learn "Hindi = fake" or "Meesho = genuine" from the metadata.
Product: if at least 20% of the labelled rows have a product, the product group and the review-vs-product fit are added as extra
tokens (product-aware model); otherwise the product is used only by the transparent product check in predict.py.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

import config
import db
import hinglish_augment
import hinglish_support
import model_store
from build_dataset import build_training_frame
from ingest import IngestError
from multilingual import TOKEN_PATTERN, detect_language, normalize_for_model
from predict import explanation_weights
from sentiment import analyze_sentiment


def make_pipeline(clf):
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2), min_df=2,
                                 max_df=0.95, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.95, sublinear_tf=True, max_features=30000)),
    ])
    return Pipeline([("features", features), ("clf", clf)])


def candidate_models():
    return [
        ("Logistic Regression", lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=2000), [{"C": 0.5}, {"C": 2.0}, {"C": 8.0}]),
        ("Multinomial Naive Bayes", lambda alpha: MultinomialNB(alpha=alpha), [{"alpha": 0.1}, {"alpha": 0.5}, {"alpha": 1.0}]),
        ("Linear SVM (calibrated)", lambda C: CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", dual="auto"), cv=3),
         [{"C": 0.1}, {"C": 0.5}, {"C": 1.0}]),
        ("Random Forest", lambda **kw: RandomForestClassifier(class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1, **kw),
         [{"n_estimators": 200, "min_samples_leaf": 1}, {"n_estimators": 200, "min_samples_leaf": 2}]),
    ]


def overall_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)), "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)), "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)), "confusion_matrix": cm.tolist(),
        "support": {"genuine": int((y_true == 0).sum()), "fake": int((y_true == 1).sum())},
    }


def group_metrics(frame, y_pred, y_score, col):
    """Metrics per value of `col` (language / platform). `reliable` = enough reviews of both classes to trust the numbers."""
    out = {}
    y_pred, y_score = np.asarray(y_pred), np.asarray(y_score)
    for key in sorted(frame[col].unique()):
        idx = np.where(frame[col].to_numpy() == key)[0]
        yt, yp, ys = frame["label"].to_numpy()[idx], y_pred[idx], y_score[idx]
        n_gen, n_fake = int((yt == 0).sum()), int((yt == 1).sum())
        fp, fn = int(((yt == 0) & (yp == 1)).sum()), int(((yt == 1) & (yp == 0)).sum())
        out[key] = {
            "n": int(len(idx)), "n_genuine": n_gen, "n_fake": n_fake, "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)), "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)), "fpr": fp / n_gen if n_gen else 0.0, "fnr": fn / n_fake if n_fake else 0.0,
            "roc_auc": float(roc_auc_score(yt, ys)) if n_gen and n_fake else None,
            "predicted_fake_rate": float(yp.mean()), "actual_fake_rate": float(yt.mean()),
            "reliable": bool(len(idx) >= 30 and n_gen >= 5 and n_fake >= 5),
        }
    return out


HINGLISH_EVAL_PATH = config.DATA_DIR / "hinglish_eval.csv"
HINGLISH_MIN_F1, HINGLISH_MAX_FPR = 0.75, 0.10      # below this, Hinglish stays "not very sure" even with enough test reviews (a model that knows no Hinglish word but has the rule layer scores F1 ~0.67, false-positive rate ~0.2)


def score_like_app(pipe, texts):
    """Fake probability for raw texts exactly as predict.py computes it for a review without product: language detection, English
    bridge for Hinglish/Hindi, model, then the capped spam-pattern nudge."""
    langs = [detect_language(t)["language"] for t in texts]
    cleaned = []
    for t, lg in zip(texts, langs):
        c = normalize_for_model(t)
        cleaned.append((c + " " + hinglish_support.english_bridge(c, lg)[0]).strip())
    probs = pipe.predict_proba(cleaned)[:, 1]
    out = []
    for p, t, lg in zip(probs, texts, langs):
        p2, _ = hinglish_support.adjust_probability(float(p), hinglish_support.spam_patterns(t, lg))
        out.append(p2)
    return np.array(out)


def hinglish_eval(pipe, path=HINGLISH_EVAL_PATH):
    """Measure the model on the hand-written Hinglish reviews (never used for training). Returns {} when the file is missing.
    `dev` was used while tuning the converter and word lists; `test` was only looked at for the final numbers."""
    path = Path(path)
    if not path.exists():
        return {}
    ev = pd.read_csv(path, encoding="utf-8-sig")
    out = {}
    for split in ("dev", "test"):
        sub = ev[ev["split"] == split]
        if sub.empty:
            continue
        y = (sub["label"].str.lower() == "fake").astype(int).to_numpy()
        score = score_like_app(pipe, sub["review"].tolist())
        pred = (score >= 0.5).astype(int)
        g = group_metrics(pd.DataFrame({"label": y, "language": "Hinglish"}), pred, score, "language")["Hinglish"]
        base = np.array([hinglish_support.adjust_probability(0.45, hinglish_support.spam_patterns(t, detect_language(t)["language"]))[0]
                         for t in sub["review"]])             # "no Hinglish knowledge at all" + the rule layer: what the model must beat
        bp = (base >= 0.5).astype(int)
        g["rules_only_baseline"] = {"f1": float(f1_score(y, bp, zero_division=0)), "fpr": float(((y == 0) & (bp == 1)).sum() / max(1, (y == 0).sum()))}
        g["source"] = f"hand-written Hinglish reviews ({path.name}, split={split})"
        g["reliable"] = bool(g["reliable"] and g["f1"] >= HINGLISH_MIN_F1 and g["fpr"] <= HINGLISH_MAX_FPR)
        out[split] = g
    return out


def sentiment_check(frame, y_pred):
    """Do genuine reviews with negative sentiment get flagged as fake more often? (they must not be)"""
    labels = frame["model_text"].map(lambda t: analyze_sentiment(t)["label"]).to_numpy()
    y, y_pred = frame["label"].to_numpy(), np.asarray(y_pred)
    out = {}
    for s in ("Positive", "Negative", "Neutral", "Mixed"):
        m = (labels == s) & (y == 0)
        if m.any():
            out[s] = {"genuine_reviews": int(m.sum()), "flagged_as_fake": int(y_pred[m].sum()), "flagged_rate": float(y_pred[m].mean())}
    return out


def split_frame(df, seed=config.RANDOM_STATE):
    key = df["language"].astype(str) + "|" + df["label"].astype(str)
    stratify = key if key.value_counts().min() >= 6 else df["label"]
    holdout = config.TEST_SIZE + config.VAL_SIZE
    tr, rest = train_test_split(df, test_size=holdout, stratify=stratify, random_state=seed)
    key_r = rest["language"].astype(str) + "|" + rest["label"].astype(str)
    strat_r = key_r if key_r.value_counts().min() >= 2 else rest["label"]
    va, te = train_test_split(rest, test_size=config.TEST_SIZE / holdout, stratify=strat_r, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def choose_rows(db_path, data_mode):
    """auto: verified rows if there are enough, otherwise everything (demo). Returns (include_demo, mode_used)."""
    verified = db.labelled_summary(db_path)["verified"]
    if data_mode == "real" or (data_mode == "auto" and verified >= config.MIN_ROWS_FOR_TRAINING):
        return False, "verified data only"
    return True, "all stored labelled data" + (" (no verified data yet, so demo data is used)" if verified < config.MIN_ROWS_FOR_TRAINING else "")


def train(db_path=config.DB_PATH, models_dir=config.MODELS_DIR, images_dir=config.IMAGES_DIR, data_mode="auto", balance=True,
          make_plots=True, activate=True, verbose=True, progress=None, hinglish_per_class=2500):
    log = print if verbose else (lambda *a, **k: None)
    note = progress or (lambda msg: None)
    models_dir, images_dir = Path(models_dir), Path(images_dir)
    include_demo, mode_used = choose_rows(db_path, data_mode)
    note("Building dataset")
    log(f"[1/6] Building dataset from the database ({mode_used})")
    df, report = build_training_frame(db_path, include_demo=include_demo, balance=balance)
    counts = report["class_counts"]
    if report["rows_final"] < config.MIN_ROWS_FOR_TRAINING or min(counts.values()) < config.MIN_ROWS_PER_CLASS:
        raise IngestError(f"Not enough labelled data to train ({report['rows_final']} reviews, classes {counts}). "
                          f"Need at least {config.MIN_ROWS_FOR_TRAINING} reviews and {config.MIN_ROWS_PER_CLASS} per class.")
    is_demo = report["is_demo"]
    log(f"      {report['rows_final']} reviews, classes {counts}, languages {report['languages_after']}, platforms {report['platforms_after']}")

    log("[2/6] Splitting 70/15/15 (stratified by language x label) before fitting anything")
    tr, va, te = split_frame(df)
    log(f"      train={len(tr)} val={len(va)} test={len(te)}")
    aug_info = None
    if hinglish_per_class and not report["is_demo"]:
        aug = hinglish_augment.augment_training(tr, per_class=hinglish_per_class)
        if len(aug):
            tr = pd.concat([tr, aug], ignore_index=True)
            aug_info = {"rows": int(len(aug)), "per_class": int((aug["label"] == 0).sum()), "source": "English training rows only, converted to Hinglish by hinglish_augment.py",
                        "applied_to": "training split only (never validation or test)"}
            log(f"      + {len(aug)} Hinglish rows converted from English TRAINING rows (training split only; real Hinglish data does not exist)")

    note("Comparing models")
    log("[3/6] Comparing models on the validation split")
    fitted, rows = [], []
    for name, factory, grid in candidate_models():
        for params in grid:
            pipe = make_pipeline(factory(**params))
            pipe.fit(tr["clean_text"], tr["label"])
            score = pipe.predict_proba(va["clean_text"])[:, 1]
            m = overall_metrics(va["label"], (score >= 0.5).astype(int), score)
            rows.append({"model": name, "params": params, "val_accuracy": m["accuracy"], "val_precision": m["precision"],
                         "val_recall": m["recall"], "val_f1": m["f1"], "val_roc_auc": m["roc_auc"]})
            fitted.append(pipe)
            log(f"      {name:26s} {str(params):40s} F1(fake)={m['f1']:.3f} AUC={m['roc_auc']:.3f}")
    best_i = max(range(len(rows)), key=lambda i: (rows[i]["val_f1"], rows[i]["val_roc_auc"]))
    best, best_pipe = rows[best_i], fitted[best_i]
    log(f"[4/6] Selected {best['model']} {best['params']} ({config.SELECTION_CRITERION})")

    note("Evaluating")
    log("[5/6] Final evaluation on the untouched test split")
    score = best_pipe.predict_proba(te["clean_text"])[:, 1]
    pred = (score >= 0.5).astype(int)
    overall = overall_metrics(te["label"], pred, score)
    by_lang, by_plat = group_metrics(te, pred, score, "language"), group_metrics(te, pred, score, "platform")
    senti = sentiment_check(te, pred)
    h_eval = hinglish_eval(best_pipe)
    if h_eval.get("test"):          # the number the app uses for "can it judge Hinglish?": measured on hand-written Hinglish, not on converted rows
        by_lang["Hinglish"] = h_eval["test"]
        h = h_eval["test"]
        log(f"      Hinglish (hand-written test set, n={h['n']}): F1={h['f1']:.3f} recall={h['recall']:.3f} false-positive-rate={h['fpr']:.3f} reliable={h['reliable']}")
    log(f"      overall: accuracy={overall['accuracy']:.3f} precision={overall['precision']:.3f} recall={overall['recall']:.3f} "
        f"F1={overall['f1']:.3f} ROC-AUC={overall['roc_auc']:.3f}")
    for k, v in by_lang.items():
        log(f"      {k:9s}: n={v['n']:4d} F1={v['f1']:.3f} recall={v['recall']:.3f} false-positive-rate={v['fpr']:.3f}")

    version = model_store.new_version()
    vdir = model_store.version_dir(models_dir, version)
    vdir.mkdir(parents=True, exist_ok=True)
    feats = best_pipe.named_steps["features"]
    names = np.asarray(feats.get_feature_names_out())
    weights = explanation_weights(best_pipe, feats.transform(tr["clean_text"]).tocsr(), tr["label"].to_numpy())
    figures = []
    if make_plots:
        import visualize as viz
        images_dir.mkdir(parents=True, exist_ok=True)
        pre = f"{version}_"
        figures = [viz.plot_class_by_language(df, images_dir / (pre + "class_by_language.png")),
                   viz.plot_platform_distribution(df, images_dir / (pre + "platform_distribution.png")),
                   viz.plot_model_comparison(rows, images_dir / (pre + "model_comparison.png")),
                   viz.plot_confusion_matrix(overall["confusion_matrix"], images_dir / (pre + "confusion_matrix.png")),
                   viz.plot_roc(te["label"], score, overall["roc_auc"], images_dir / (pre + "roc_curve.png")),
                   viz.plot_language_metrics(by_lang, images_dir / (pre + "language_metrics.png"))]
        top = viz.plot_top_features(names, weights, np.array([n.startswith("word__") for n in names]), images_dir / (pre + "top_features.png"))
        if top:
            figures.append(top)

    top_words = None
    if weights is not None:
        wi = np.where(np.array([n.startswith("word__") for n in names]))[0]
        o = wi[np.argsort(weights[wi])]
        top_words = {"fake": [{"term": names[i][6:], "weight": round(float(weights[i]), 4)} for i in o[::-1][:15]],
                     "genuine": [{"term": names[i][6:], "weight": round(float(weights[i]), 4)} for i in o[:15]]}
    majority = max(counts.values()) / sum(counts.values())
    notes = [
        "Measured on a held-out test split that was never used for fitting or model selection.",
        "Platform and language are not model inputs; they are used only to report results per group.",
        f"Always predicting the majority class would score {majority:.1%} accuracy here, so precision, recall, F1 and ROC-AUC are reported too.",
        "Per-language numbers with few reviews are marked 'small sample' and should not be over-interpreted.",
        "Confidence = estimated probability of the predicted class; it is not guaranteed to be calibrated on new data.",
    ]
    if aug_info:
        notes.append(f"HINGLISH: {aug_info['rows']} training rows are English reviews converted to Hinglish (labels inherited from the English originals, training split only). "
                     "Hinglish accuracy is measured on a separate small set of hand-written Hinglish reviews, not on converted rows. "
                     "It shows the model uses Hinglish words sensibly; it is not proof of accuracy on real Hinglish reviews.")
    if is_demo:
        notes.insert(0, "DEMONSTRATION DATA: the training reviews are SYNTHETIC, assembled from templates by generate_multilingual_demo.py. The scores "
                        "only show that the pipeline works; they are NOT evidence of accuracy on real Meesho, Flipkart, Myntra or Amazon reviews.")
    weak = bool(df["label_method"].astype(str).str.startswith("weak_").any()) if "label_method" in df.columns else False
    if weak:
        notes.insert(0, "RULE-BASED LABELS: the fake/genuine labels come from rules (copy-pasted text, hype wording), not from human checking. The scores "
                        "below measure how well the model reproduces those rules; they are NOT proof of accuracy on real fake reviews.")
    metrics = {
        "weak_labels": weak, "product_aware": bool(report.get("product_aware")), "version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "is_demo_data": is_demo,
        "data_mode": mode_used, "cleaning_report": report, "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
        "selected_model": {"name": best["model"], "params": best["params"], "type": "sklearn"}, "selection_criterion": config.SELECTION_CRITERION,
        "validation_results": rows,
        "test_metrics": {"overall": overall, "by_language": by_lang, "by_platform": by_plat, "sentiment_check": senti},
        "hinglish_augmentation": aug_info, "hinglish_eval": h_eval,
        "majority_class_baseline_accuracy": majority, "top_words": top_words, "figures": figures, "notes": notes,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    log("[6/6] Saving as model version " + version)
    note("Saving")
    joblib.dump({"pipeline": best_pipe, "weights": weights}, vdir / "model.joblib", compress=3)
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    db.register_model(db_path, version, "sklearn: " + best["model"], vdir, is_demo, overall["f1"], overall["accuracy"],
                      f"{report['rows_final']} reviews, {mode_used}", activate=activate)
    if activate:
        model_store.set_active(models_dir, version)
    if is_demo:
        log("\n*** DEMO MODEL (synthetic data). Ingest verified, labelled reviews and run this script again for a meaningful model. ***")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train the multilingual fake-review model.")
    ap.add_argument("--data-mode", choices=["auto", "real", "all"], default="auto", help="auto: verified data if enough, else everything (demo)")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--hinglish-per-class", type=int, default=2500, help="Hinglish rows converted from English training rows, per class (0 = off)")
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args()
    try:
        train(args.db, data_mode=args.data_mode, balance=not args.no_balance, hinglish_per_class=args.hinglish_per_class)
    except IngestError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
