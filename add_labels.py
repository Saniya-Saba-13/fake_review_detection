"""Add YOUR hand-labelled reviews and retrain the model in one command (this is how you teach it more Hinglish).

1. Open data/hinglish_labelling_sheet.csv (and/or data/labelling_sheet.csv) in Excel.
2. In the `label` column type  genuine  or  fake  for each review you are sure about. Leave it empty if unsure (those rows are skipped).
   Be strict: label `fake` only with real evidence (copy-pasted wording, obvious paid promotion, seller asking for 5 stars ...).
3. Save as CSV (UTF-8) and run (PowerShell):
       python add_labels.py "C:\\path\\hinglish_labelling_sheet_filled.csv"
   It stores the labels (label_method = manual_annotation), retrains, and prints the new scores per language.
The more Hinglish reviews you label (aim for 200+ fake and 200+ genuine), the more reliably the tool can judge Hinglish
instead of saying "Not very sure".
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import config
from build_master_dataset import read_any
from ingest import IngestError, ingest_labelled

MAP = {"genuine": "genuine", "real": "genuine", "g": "genuine", "0": "genuine", "or": "genuine",
       "fake": "fake", "f": "fake", "suspicious": "fake", "paid": "fake", "1": "fake", "cg": "fake"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+"); ap.add_argument("--no-train", action="store_true", help="only store the labels")
    a = ap.parse_args()
    frames = []
    for f in a.files:
        df = read_any(f)
        cols = {str(c).lower().strip(): c for c in df.columns}
        if "review" not in cols or "label" not in cols:
            sys.exit(f"{f}: the file needs a 'review' column and a 'label' column. Found: {list(df.columns)}")
        lab = df[cols["label"]].astype(str).str.strip().str.lower().map(MAP)
        kept = df[lab.notna()].copy()
        kept["label"] = lab[lab.notna()]
        print(f"{Path(f).name}: {len(df)} rows, {len(kept)} labelled ({(kept['label'] == 'fake').sum()} fake, {(kept['label'] == 'genuine').sum()} genuine), {len(df) - len(kept)} skipped (empty or unclear label)")
        frames.append(pd.DataFrame({"review": kept[cols["review"]], "label": kept["label"],
                                    "platform": kept[cols["platform"]] if "platform" in cols else "Other",
                                    "rating": kept[cols["rating"]] if "rating" in cols else "", "label_method": "manual_annotation"}))
    data = pd.concat(frames, ignore_index=True)
    if len(data) < 20 or data["label"].nunique() < 2:
        sys.exit("Label at least 20 reviews, with both genuine and fake examples, then run again.")
    tmp = config.DATA_DIR / "manual_labels_ingest.csv"
    data.to_csv(tmp, index=False, encoding="utf-8-sig")
    try:
        rep = ingest_labelled(config.DB_PATH, tmp, tmp.name, "Other", "manual_annotation")
    except IngestError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Stored {rep['rows_added']} labelled reviews (duplicates skipped: {rep['duplicates']}, invalid: {rep['invalid']}).")
    if a.no_train:
        return
    import train_model
    print("Retraining (a few minutes) ...")
    train_model.train(config.DB_PATH, data_mode="real", verbose=False)
    import json, model_store
    m = model_store.read_metrics(config.MODELS_DIR)
    print(f"New model {m['version']}: accuracy {m['test_metrics']['overall']['accuracy']:.1%} on the held-out test split.")
    for lang, v in m["test_metrics"]["by_language"].items():
        print(f"  {lang:9s} n={v['n']:5d}  F1={v['f1']:.2f}  recall={v['recall']:.2f}  false-positive rate={v['fpr']:.2f}" + ("" if v["reliable"] else "   (too few test reviews: still 'Not very sure')"))
    print("Restart the app (python app.py) to use the new model.")


if __name__ == "__main__":
    main()
