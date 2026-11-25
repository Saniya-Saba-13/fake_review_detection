"""Prepare REAL review exports (no fake/genuine labels inside them) for training, and build a sheet for human labelling.

None of the supported exports says which reviews are fake. This script therefore creates WEAK (rule-based) labels and
says so everywhere (label_method = weak_rules_v1). Read the rules below: they are the only "evidence" the model sees.

Suspicious (label 1) when a review matches at least one rule:
  R1 copy-paste   Flipkart-style: the same text (>= 8 words) appears under >= 3 different products.
                  Amazon / Meesho: the same text (>= 5 words) appears >= 2 times.
  R2 hype         5-star review (judged on the review body, not Flipkart's suggested headline tag) with >= 2 promotional phrases, or 1 promotional phrase plus >= 3 exclamation marks
                  (English, Hindi and Hinglish phrase lists from preprocessing.py).
Presumed genuine (label 0): unique text (appears once), >= 3 words, matches no rule.
Everything else (for example a text repeated only twice) is left out, because it cannot be called either way.

To stop the model from learning shortcuts, presumed-genuine reviews are sampled to match the suspicious ones in
language, length bucket and star-rating group. A held-out set of reviews (the labelling sheet) is kept out of training.

Usage:
    python prepare_real_data.py --amazon a1.csv a2.csv --meesho m.csv --flipkart f.csv
    python build_dataset.py --ingest data/real_weak_labelled.csv --label-method weak_rules_v1
    python train_model.py --data-mode real
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from multilingual import detect_language, normalize_for_model
from preprocessing import PROMO_PHRASES, PROMO_PHRASES_MULTI

RULES_VERSION = "weak_rules_v1"
SEED = 42
SHEET_SIZE = {"Meesho_nonEnglish": 200, "Meesho_English": 100, "Amazon": 150, "Flipkart": 150}


def _read(path, **kw):
    try:
        return pd.read_csv(path, on_bad_lines="skip", low_memory=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, on_bad_lines="skip", low_memory=False, encoding="latin-1", **kw)


def load_amazon(paths):
    df = pd.concat([_read(p) for p in paths], ignore_index=True)
    df["text"] = df["Review Content"].astype(str)
    df = df[~df["text"].str.contains("How customer reviews and ratings work", na=False)]
    df = df.drop_duplicates(subset=["Product URL", "text"])  # the two export files overlap
    return pd.DataFrame({"platform": "Amazon", "product": df["Product URL"], "rating": pd.to_numeric(df["Rating"], errors="coerce"),
                         "text": df["text"], "body": df["text"], "date": df["Date"].astype(str)})


def load_meesho(path):
    df = _read(path)
    return pd.DataFrame({"platform": "Meesho", "product": df["product_name"], "rating": pd.to_numeric(df["ratings"], errors="coerce"),
                         "text": df["comments"].astype(str), "body": df["comments"].astype(str), "date": df["date"].astype(str)})


def load_flipkart(path):
    df = _read(path, dtype=str)
    df = df[df["Rate"].isin(list("12345"))]
    text = (df["Review"].fillna("") + ". " + df["Summary"].fillna("")).str.strip(" .")
    return pd.DataFrame({"platform": "Flipkart", "product": df["Product_name"], "rating": df["Rate"].astype(int), "text": text, "body": df["Summary"].fillna(""), "date": ""})


def hype(text):
    low = str(text).lower()
    norm = normalize_for_model(low)
    promo = sum(p in low for p in PROMO_PHRASES) + sum(normalize_for_model(p) in norm for p in PROMO_PHRASES_MULTI)
    return promo, str(text).count("!")


def wbucket(w):
    return "1-4" if w <= 4 else "5-7" if w <= 7 else "8-15" if w <= 15 else "16-40" if w <= 40 else "40+"


def rgroup(r):
    return "low" if r <= 2 else "mid" if r == 3 else "high"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amazon", nargs="*", default=[]); ap.add_argument("--meesho"); ap.add_argument("--flipkart")
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    ap.add_argument("--flipkart-negatives", type=int, default=20000, help="how many unique Flipkart-style reviews to language-check as candidates")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    parts = []
    if a.amazon: parts.append(load_amazon(a.amazon))
    if a.meesho: parts.append(load_meesho(a.meesho))
    if a.flipkart: parts.append(load_flipkart(a.flipkart))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["rating"].between(1, 5) & df["text"].str.strip().ne("") & df["text"].ne("nan")].copy()
    df["rating"] = df["rating"].astype(int)
    uniq = pd.Series(df["text"].unique())
    keymap = dict(zip(uniq, uniq.map(lambda t: normalize_for_model(str(t)).strip().lower())))
    df["key"] = df["text"].map(keymap)
    df = df[df["key"] != ""]
    df["words"] = df["key"].str.split().str.len()
    g = df.groupby(["platform", "key"]).agg(rows=("key", "size"), prods=("product", "nunique"))
    df = df.join(g, on=["platform", "key"])
    hp = df["body"].map(hype)
    df["promo"], df["excl"] = hp.map(lambda x: x[0]), hp.map(lambda x: x[1])

    r1 = np.where(df["platform"] == "Flipkart", (df["words"] >= 8) & (df["prods"] >= 3), (df["words"] >= 5) & (df["rows"] >= 2))
    r2 = (df["rating"] == 5) & ((df["promo"] >= 2) | ((df["promo"] >= 1) & (df["excl"] >= 3)))
    df["rule"] = np.select([r1 & r2, r1, r2], ["R1+R2", "R1 copy-paste", "R2 hype"], default="")
    df["sus"] = df["rule"] != ""
    df["clean"] = (~df["sus"]) & (df["rows"] == 1) & (df["words"] >= 3)

    # one row per distinct text (the project removes duplicates anyway)
    pos = df[df["sus"]].drop_duplicates(["platform", "key"]).copy()
    neg_all = df[df["clean"]].copy()
    neg_all = pd.concat([neg_all[neg_all["platform"] != "Flipkart"], neg_all[neg_all["platform"] == "Flipkart"].sample(
        min(a.flipkart_negatives, int((neg_all["platform"] == "Flipkart").sum())), random_state=SEED)])

    # ---- the held-out labelling sheet (kept OUT of training) ----
    cand = df.drop_duplicates(["platform", "key"]).copy()
    cand = cand[cand["words"] >= 3]
    sheet_pool = pd.concat([cand[cand["sus"]].sample(frac=1, random_state=SEED), cand[~cand["sus"]].sample(frac=1, random_state=SEED)])
    lang_cache = {}

    def lang(t):
        if t not in lang_cache:
            lang_cache[t] = detect_language(t)["language"]
        return lang_cache[t]

    sheet = []
    for name, n in SHEET_SIZE.items():
        plat = name.split("_")[0]
        pool = cand[cand["platform"] == plat].sample(frac=1, random_state=SEED)
        if plat == "Meesho":
            pool = pool.assign(language=pool["text"].map(lang))
            pool = pool[(pool["language"] != "English") if name.endswith("nonEnglish") else (pool["language"] == "English")]
        n_sus = int(n * 0.4)
        take = pd.concat([pool[pool["sus"]].head(n_sus), pool[~pool["sus"]].head(n - n_sus)])
        sheet.append(take.head(n))
    sheet = pd.concat(sheet).drop_duplicates(["platform", "key"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    held_keys = set(zip(sheet["platform"], sheet["key"]))

    # ---- stratified weak-labelled training file ----
    pos = pos[[(p, k) not in held_keys for p, k in zip(pos["platform"], pos["key"])]].copy()
    neg_all = neg_all[[(p, k) not in held_keys for p, k in zip(neg_all["platform"], neg_all["key"])]].copy()
    for frame in (pos, neg_all):
        frame["language"] = frame["text"].map(lang)
        frame["stratum"] = frame["language"] + "|" + frame["words"].map(wbucket) + "|" + frame["rating"].map(rgroup)
    chosen_neg = []
    for s, gp in pos.groupby("stratum"):
        pool = neg_all[neg_all["stratum"] == s]
        if len(pool):
            chosen_neg.append(pool.sample(min(len(pool), len(gp)), random_state=SEED))
    neg = pd.concat(chosen_neg) if chosen_neg else neg_all.head(0)
    pos = pos[pos["stratum"].isin(set(neg["stratum"]))]
    # make each stratum exactly balanced
    bal = []
    for s, gp in pos.groupby("stratum"):
        n = min(len(gp), int((neg["stratum"] == s).sum()))
        bal.append(gp.sample(n, random_state=SEED)); bal.append(neg[neg["stratum"] == s].sample(n, random_state=SEED))
    train = pd.concat(bal)
    train = train.assign(label=np.where(train["sus"], 1, 0)).sample(frac=1, random_state=SEED)
    tr_out = pd.DataFrame({"review": train["text"], "label": train["label"].map({0: "genuine", 1: "fake"}), "platform": train["platform"],
                           "rating": train["rating"], "collection_date": "", "label_method": RULES_VERSION})
    tr_out.to_csv(out / "real_weak_labelled.csv", index=False, encoding="utf-8-sig")
    train[["platform", "language", "rating", "words", "rule", "rows", "prods", "text"]].to_csv(out / "real_weak_audit.csv", index=False, encoding="utf-8-sig")

    sheet["language"] = sheet["text"].map(lang)
    sh = pd.DataFrame({"id": np.arange(1, len(sheet) + 1), "platform": sheet["platform"], "rating": sheet["rating"], "language": sheet["language"],
                       "review": sheet["text"], "label": "", "notes": ""})
    sh.to_csv(out / "labelling_sheet.csv", index=False, encoding="utf-8-sig")
    # reference set for the "real-world flag rate" check: never trained on, never in the sheet
    used = set(zip(train["platform"], train["key"])) | held_keys
    ref = cand[[(p, k) not in used for p, k in zip(cand["platform"], cand["key"])]]
    ref = pd.concat([ref[ref["platform"] == pl].sample(min(1500, int((ref["platform"] == pl).sum())), random_state=SEED) for pl in ref["platform"].unique()])
    ref[["platform", "rating", "text", "rule"]].to_csv(out / "real_reference_sample.csv", index=False, encoding="utf-8-sig")

    rep = {"rules": RULES_VERSION, "pool_rows": int(len(df)), "rule_hits_by_platform": df[df["sus"]].groupby("platform")["key"].nunique().to_dict(),
           "unique_texts_by_platform": df.groupby("platform")["key"].nunique().to_dict(),
           "train_rows": int(len(train)), "train_by_class": train["label"].value_counts().to_dict(),
           "train_by_language_class": {f"{l}|{'fake' if c else 'genuine'}": int(n) for (l, c), n in train.groupby(["language", "label"]).size().items()},
           "train_by_platform": train["platform"].value_counts().to_dict(), "train_by_rule": train[train["sus"]]["rule"].value_counts().to_dict(),
           "sheet_rows": int(len(sh)), "sheet_by_platform_language": sh.groupby(["platform", "language"]).size().rename(lambda x: x).reset_index().values.tolist()}
    (out / "real_weak_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
