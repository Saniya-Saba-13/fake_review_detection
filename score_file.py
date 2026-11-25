"""Check EVERY review in a CSV / Excel file and save the answers next to your original columns.

Usage (PowerShell):
    python score_file.py "C:\\data\\my_reviews.csv"
    python score_file.py "C:\\data\\my_reviews.xlsx" --output answers.csv
    python score_file.py "C:\\data\\Dataset.csv" --text-column Summary --title-column Review --limit 50000

It finds the review column by name (review, review_text, text, comment, content, Summary ...). If your column has another name,
pass --text-column. New columns added to the right:
    verdict            Looks genuine / Looks suspicious / Not sure
    fake_probability   0 to 1 (model's estimate that the review is fake)
    confidence         0.5 to 1 (how sure the model is of its verdict)
    language           English / Hindi / Hinglish / Mixed / Unknown
    reason             why the verdict is "Not sure" (short, very short, or too little training data in that language)

This is an estimate, not proof. See the Accuracy page for how the model was trained.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from build_master_dataset import SYN, read_any
from multilingual import detect_language, normalize_for_model
from predict import FakeReviewPredictor, ModelNotFoundError, model_text


def pick(df, wanted, names):
    if wanted:
        if wanted not in df.columns:
            sys.exit(f"Column '{wanted}' not found. Columns in your file: {list(df.columns)}")
        return wanted
    low = {str(c).lower().strip(): c for c in df.columns}
    return next((low[n] for n in names if n in low), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("--output"); ap.add_argument("--text-column"); ap.add_argument("--title-column")
    ap.add_argument("--limit", type=int, help="only check the first N rows (quick test)")
    ap.add_argument("--batch", type=int, default=5000)
    a = ap.parse_args()

    df = read_any(a.input)
    if a.limit:
        df = df.head(a.limit)
    tcol = pick(df, a.text_column, SYN["review_text"])
    if tcol is None:
        sys.exit(f"I could not find the review column. Use --text-column. Columns in your file: {list(df.columns)}")
    hcol = pick(df, a.title_column, []) if a.title_column else None
    print(f"Checking {len(df):,} reviews from column '{tcol}'" + (f" (+ title '{hcol}')" if hcol else "") + " ...")

    try:
        pr = FakeReviewPredictor(config.MODELS_DIR, config.TOP_TERMS, config.LOW_CONFIDENCE_THRESHOLD).ensure_loaded()
    except ModelNotFoundError as exc:
        sys.exit(str(exc))
    low_conf = config.LOW_CONFIDENCE_THRESHOLD

    texts = df[tcol].fillna("").astype(str).tolist()
    titles = df[hcol].fillna("").astype(str).tolist() if hcol else [""] * len(texts)
    probs = np.zeros(len(texts)); langs, reasons = [""] * len(texts), [""] * len(texts)
    lang_cache = {}
    for s in range(0, len(texts), a.batch):
        e = min(s + a.batch, len(texts))
        full = [model_text(texts[i], titles[i]) for i in range(s, e)]
        cleaned = [normalize_for_model(t) for t in full]
        p, _ = pr.backend.fake_proba(cleaned)
        probs[s:e] = p
        for k, i in enumerate(range(s, e)):
            t = full[k]
            if t not in lang_cache:
                lang_cache[t] = detect_language(t)["language"]
            langs[i] = lang_cache[t]
            n_words = len(cleaned[k].split())
            conf = probs[i] if probs[i] >= 0.5 else 1 - probs[i]
            if not texts[i].strip():
                reasons[i] = "empty review"
            elif n_words < 4:
                reasons[i] = "very short review"
            elif pr.reliable_langs and langs[i] not in pr.reliable_langs:
                reasons[i] = f"too little training data in {langs[i]}"
            elif conf < low_conf:
                reasons[i] = "model is unsure"
        print(f"  {e:,} / {len(texts):,}", end="\r")
    conf = np.where(probs >= 0.5, probs, 1 - probs)
    unsure = np.array([r != "" for r in reasons])
    verdict = np.where(unsure, "Not sure", np.where(probs >= 0.5, "Looks suspicious", "Looks genuine"))
    out = df.copy()
    out["verdict"], out["fake_probability"], out["confidence"] = verdict, probs.round(4), conf.round(4)
    out["language"], out["reason"] = langs, reasons
    dest = Path(a.output) if a.output else Path(a.input).with_name(Path(a.input).stem + "_checked.csv")
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    vc = pd.Series(verdict).value_counts()
    print(f"\nDone. Saved to {dest}\n" + "\n".join(f"  {k:17s} {v:>9,}  ({v / len(out):.1%})" for k, v in vc.items()))
    print("Remember: Hindi / Hinglish are marked 'Not sure' until the model has enough real examples in them.")


if __name__ == "__main__":
    main()
