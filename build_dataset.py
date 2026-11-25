"""Build the unified, balanced training dataset from the labelled reviews stored in the database.

Usage examples (PowerShell):
    python build_dataset.py --demo                                  # SYNTHETIC demo: generate + ingest + build
    python build_dataset.py --ingest data\\raw\\meesho.csv --platform Meesho --label-method manual_annotation
    python build_dataset.py --stats                                 # just show what is stored
    python build_dataset.py                                         # rebuild data\\unified_training.csv

Cleaning: missing values and empty texts are rejected at ingestion, exact/normalised duplicates are removed (the same
text with conflicting labels is dropped), and the result is balanced per language by down-sampling the larger class, so
that no language is mostly 'fake' or mostly 'genuine' (this stops the model from learning "Hindi = fake").
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import config
import db
import hinglish_support
import product_context
from ingest import IngestError, ingest_labelled
from multilingual import normalize_for_model


def balance_by_language(df, seed=config.RANDOM_STATE):
    """Down-sample the larger class inside every language that has enough data. Returns (balanced_df, report)."""
    parts, report = [], {}
    for lang, g in df.groupby("language"):
        counts = g["label"].value_counts().to_dict()
        n0, n1 = counts.get(0, 0), counts.get(1, 0)
        if min(n0, n1) >= config.MIN_ROWS_PER_LANGUAGE_TO_BALANCE:
            n = min(n0, n1)
            g = pd.concat([g[g["label"] == 0].sample(n, random_state=seed), g[g["label"] == 1].sample(n, random_state=seed)])
            report[lang] = {"before": {"genuine": n0, "fake": n1}, "after": {"genuine": n, "fake": n}, "balanced": True}
        else:
            report[lang] = {"before": {"genuine": n0, "fake": n1}, "after": {"genuine": n0, "fake": n1}, "balanced": False,
                            "note": f"fewer than {config.MIN_ROWS_PER_LANGUAGE_TO_BALANCE} reviews in a class: kept as is (class weights are used instead)"}
        parts.append(g)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True), report


def build_training_frame(db_path=config.DB_PATH, include_demo=True, balance=True, write_csv=None):
    """Return (dataframe, report). Columns: platform, language, model_text, clean_text, label, rating, label_method, collection_date, is_demo."""
    rows = db.fetch_labelled(db_path, include_demo=include_demo)
    if not rows:
        raise IngestError("No labelled reviews in the database yet. Run `python build_dataset.py --demo` for the synthetic demo, or ingest your own labelled files.")
    df = pd.DataFrame(rows)
    report = {"rows_in_database": len(df), "platforms": df["platform"].value_counts().to_dict(), "languages_before": df["language"].value_counts().to_dict()}
    df["model_text"] = (df["review_title"].fillna("").str.strip() + " . " + df["review_text"]).str.strip(" .")
    df["clean_text"] = df["model_text"].map(normalize_for_model)
    df = df[df["clean_text"] != ""]
    # Hinglish/Hindi/mixed rows also get the English-meaning tokens (original text is kept), exactly as at prediction time.
    df["clean_text"] = [(c + " " + hinglish_support.english_bridge(c, lg)[0]).strip() for c, lg in zip(df["clean_text"], df["language"])]
    # Product-aware training: only when enough rows carry a product (otherwise the tokens would just be noise).
    has_prod = df["product"].notna() & (df["product"].astype(str).str.strip() != "") if "product" in df.columns else pd.Series(False, index=df.index)
    product_aware = bool(has_prod.mean() >= config.PRODUCT_AWARE_MIN_SHARE and len(df) and df.loc[has_prod, "product"].nunique() >= 2)
    if product_aware:
        toks = [product_context.model_tokens(p, t) if h else "" for p, t, h in zip(df["product"], df["model_text"], has_prod)]
        df["clean_text"] = (df["clean_text"] + " " + pd.Series(toks, index=df.index)).str.strip()
    before = len(df)
    df = df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)
    if balance:
        df, report["balancing"] = balance_by_language(df)
    report.update(product_aware=product_aware, rows_with_product=int(has_prod.sum()), rows_final=len(df), class_counts={"genuine": int((df["label"] == 0).sum()), "fake": int((df["label"] == 1).sum())},
                  languages_after=df["language"].value_counts().to_dict(), platforms_after=df["platform"].value_counts().to_dict(),
                  is_demo=bool(df["is_demo"].all()), has_demo=bool(df["is_demo"].any()))
    if write_csv:
        Path(write_csv).parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["clean_text"]).to_csv(write_csv, index=False, encoding="utf-8-sig")
    return df, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="generate + ingest the SYNTHETIC demo files (flagged is_demo)")
    ap.add_argument("--ingest", nargs="*", default=[], help="labelled .csv/.xlsx files to ingest")
    ap.add_argument("--platform", default="Other", help="default platform for files without a platform column")
    ap.add_argument("--label-method", help="how the labels were obtained (required unless the file has a label_method column)")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args()
    try:
        if args.demo:
            import generate_multilingual_demo as gen
            for platform, frame in gen.generate().items():
                path = config.DEMO_SOURCES_DIR / f"{platform.lower()}_demo_reviews.csv"
                config.DEMO_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                rep = ingest_labelled(args.db, path, path.name, platform, config.DEMO_LABEL_METHOD, is_demo=True)
                print(f"  demo {platform:9s}: read {rep['rows_read']}, added {rep['rows_added']}, duplicates {rep['duplicates']}, invalid {rep['invalid']}")
        for f in args.ingest:
            rep = ingest_labelled(args.db, f, Path(f).name, args.platform, args.label_method)
            print(f"  {f}: read {rep['rows_read']}, added {rep['rows_added']}, duplicates {rep['duplicates']}, "
                  f"conflicts {rep['conflicts']}, invalid {rep['invalid']} {rep['invalid_reasons'] or ''}")
        if args.stats:
            s = db.labelled_summary(args.db)
            print(f"labelled reviews: {s['total']} (verified {s['verified']}, synthetic demo {s['demo']}); labels {s['by_label']}")
            for key in ("by_platform", "by_language", "by_method"):
                print(f"  {key}: " + ", ".join(f"{r['name']}={r['total']}" for r in s[key]))
            return
        df, rep = build_training_frame(args.db, include_demo=True, balance=not args.no_balance, write_csv=config.UNIFIED_DATASET_PATH)
        print(f"\nunified dataset: {rep['rows_final']} reviews -> {config.UNIFIED_DATASET_PATH}")
        print(f"  classes {rep['class_counts']}, languages {rep['languages_after']}, platforms {rep['platforms_after']}")
        if rep["has_demo"]:
            print("  NOTE: contains SYNTHETIC demo rows (is_demo). Not real platform data.")
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
