"""Collect, normalise and combine e-commerce review datasets into one master dataset (plus one CSV per platform).

What it does (and does not do)
  * `list`   prints the catalogue of sources found (data_collection/sources.json) with licence and verified record counts.
  * `fetch`  downloads the sources that allow scripted download (Kaggle, through the official `kaggle` command, which needs
             your own Kaggle API key). Every other source is printed with the manual steps. It never logs in for you,
             never bypasses CAPTCHAs / rate limits, and never scrapes a shopping site.
  * `build`  reads whatever is in raw/<source_id>/ (or files given with --file id=path), converts it to ONE schema, and writes
             master/*.csv, the data dictionary and three reports. Missing values stay empty: nothing is invented.

Processing
  exact-duplicate records removed; near-duplicates are FLAGGED (near_dup_group_id), not deleted; original text kept in
  review_text and a Hindi/Hinglish-safe normalised copy in text_normalized; language detected per review; the split
  (train/validation/test) is made per near-duplicate GROUP so near-identical reviews never sit in different splits.
  Authenticity labels (genuine / fake / unknown) are copied only from a source that provides them, with the source named.
  Verified-purchase is kept as its own column and is never turned into an authenticity label.

Usage (PowerShell):
    python build_master_dataset.py list
    python build_master_dataset.py fetch
    python build_master_dataset.py build --file user_flipkart_style=C:\\data\\Dataset.csv --file user_meesho_multilingual=C:\\data\\meesho.csv
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual import detect_language, normalize_for_model

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "master"
REPORTS = ROOT / "master" / "reports"
SOURCES = json.loads((ROOT / "data_collection" / "sources.json").read_text(encoding="utf-8"))["sources"]
PLATFORMS = ["Amazon", "Flipkart", "Myntra", "Meesho"]
SALT = "reviewguard-v1"  # reviewer names are replaced by a salted hash

COLUMNS = ["review_id", "platform_name", "marketplace_domain", "record_type", "product_id", "product_name", "product_category", "product_brand",
           "review_title", "review_text", "text_normalized", "review_text_translated", "rating", "review_date", "verified_purchase", "helpful_votes",
           "reviewer_id", "product_price", "seller_information", "review_language", "source_sentiment_label", "authenticity_label", "label_source",
           "source_dataset", "source_url", "dup_text_count", "near_dup_group_id", "split"]

DICTIONARY = {
    "review_id": "Unique id made as <source_dataset>:<row id>. Taken from the source's own id when it has one.",
    "platform_name": "Amazon, Flipkart, Myntra, Meesho or Other. Kept exactly as the source says.",
    "marketplace_domain": "Domain taken from the product URL (for example amazon.com or amazon.in). Empty if the source has no URL.",
    "record_type": "product_review (review of a product), app_review (review of the shopping app) or fake_benchmark (research set with authenticity labels).",
    "product_id": "Product id from the source (for example the Amazon ASIN read from the URL). Empty if not available.",
    "product_name": "Product title from the source.", "product_category": "Category from the source, if given.", "product_brand": "Brand from the source, if given.",
    "review_title": "Review headline. Empty when the source has none.",
    "review_text": "ORIGINAL review text (whitespace tidied only). Hindi, Hinglish and emojis are preserved.",
    "text_normalized": "Lower-cased copy with Devanagari and common Hinglish spelling variants normalised. Used for duplicate detection and models; review_text is never overwritten.",
    "review_text_translated": "Always empty here. No translation is performed; if you add translations, keep them in this column.",
    "rating": "Star rating 1 to 5 as given.", "review_date": "ISO date (YYYY-MM-DD) when the source date could be read, otherwise empty.",
    "verified_purchase": "True / False only when the source says so; empty otherwise. NOT an authenticity label.",
    "helpful_votes": "Number of helpful votes if given.", "reviewer_id": "Salted hash of the reviewer name. Empty when the source only has a placeholder name.",
    "product_price": "Price as a number (digits only) if the source has one; currency symbol was not readable in some files.", "seller_information": "Seller, only if the source has it.",
    "review_language": "English, Hindi, Hinglish, Mixed or Unknown, detected from the text by lexicon and script (approximate).",
    "source_sentiment_label": "Sentiment label that came with the source (not computed here). Not an authenticity label.",
    "authenticity_label": "genuine, fake or unknown. 'unknown' for every source that does not provide human or platform labels.",
    "label_source": "Where the authenticity label came from (source id and method). Empty when unknown.",
    "source_dataset": "Id of the dataset in data_collection/sources.json.", "source_url": "Product page URL from the source when present.",
    "dup_text_count": "How many records on the same platform have exactly the same normalised text (including this one).",
    "near_dup_group_id": "Reviews with the same words in any order, or the same first 80 characters, share a group id. Flagged, not deleted.",
    "split": "train / validation / test (70/15/15), assigned per near_dup_group_id so near-identical reviews never leak across splits.",
}


# ----------------------------------------------------------------------------- reading
def read_any(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def blank(df, n):
    out = pd.DataFrame(index=range(n), columns=COLUMNS)
    return out


def digits(s):
    d = re.sub(r"[^0-9.]", "", str(s or ""))
    return d or ""


def to_date(s, fmt=None):
    return pd.to_datetime(s, errors="coerce", format=fmt).dt.strftime("%Y-%m-%d")


def hash_name(s, placeholder=()):
    s = str(s or "").strip()
    if not s or s.lower() in ("nan", "meesho user", "anonymous") or re.fullmatch(r"customer \d+", s.lower()) or s in placeholder:
        return ""
    return hashlib.sha256((SALT + s.lower()).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- adapters: raw frame -> COLUMNS
def a_amazon_com(df, src):
    df = df[~df["Review Content"].astype(str).str.contains("How customer reviews and ratings work", na=False)].reset_index(drop=True)
    o = blank(df, len(df))
    url = df["Product URL"].astype(str)
    o["product_id"] = url.str.extract(r"/dp/([A-Z0-9]{10})")[0]
    o["marketplace_domain"] = url.str.extract(r"https?://(?:www\.)?([^/]+)")[0]
    o["product_name"] = df["Product Title"].astype(str).str.replace(r"^Amazon\.com:\s*", "", regex=True)
    o["review_title"] = df["Review Title"].where(~df["Review Title"].astype(str).str.contains("How customer reviews", na=False))
    o["review_text"] = df["Review Content"]
    o["rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    o["review_date"] = to_date(df["Date"])
    o["verified_purchase"] = df["Verified"].map({"Yes": True, "No": False})
    o["reviewer_id"] = df["Review Author"].map(hash_name)
    o["source_url"] = url
    o["review_id"] = [f"{src}:{i}" for i in range(len(df))]
    return o


def a_meesho(df, src):
    o = blank(df, len(df))
    o["review_id"] = [f"{src}:{i}" for i in df["id"]]
    o["product_name"] = df["product_name"]; o["product_category"] = df["category"]
    o["review_text"] = df["comments"]; o["rating"] = pd.to_numeric(df["ratings"], errors="coerce")
    o["review_date"] = to_date(df["date"], "%d-%b-%y")
    o["helpful_votes"] = pd.to_numeric(df["helpful"], errors="coerce")
    o["reviewer_id"] = df["reviewer"].map(hash_name)
    o["source_sentiment_label"] = df["mapped_sentiment"].astype(str).str.lower()
    return o


def a_flipkart(df, src):
    df = df[df["Rate"].isin(list("12345"))].reset_index(drop=True)  # the file repeats its header inside the data
    o = blank(df, len(df))
    o["review_id"] = [f"{src}:{i}" for i in range(len(df))]
    o["product_name"] = df["Product_name"]; o["product_price"] = df["Price"].map(digits)
    o["review_title"] = df["Review"]; o["review_text"] = df["Summary"]
    o["rating"] = pd.to_numeric(df["Rate"], errors="coerce")
    return o


SYN = {"review_text": ["review_text", "review", "reviewtext", "text", "comment", "comments", "content", "review content", "body", "content_text"],
       "review_title": ["review_title", "title", "summary", "headline", "review title"], "rating": ["rating", "ratings", "rate", "score", "stars", "star_rating"],
       "review_date": ["review_date", "date", "timestamp", "time", "at"], "product_name": ["product_name", "product", "title_product", "item_name", "name"],
       "product_id": ["product_id", "asin", "parent_asin", "item_id"], "verified_purchase": ["verified_purchase", "verified", "is_verified"],
       "helpful_votes": ["helpful_votes", "helpful_vote", "helpful", "thumbsupcount", "thumbs_up"], "product_category": ["category", "product_category", "main_category"],
       "product_brand": ["brand", "product_brand"], "reviewer_id": ["reviewer_id", "user_id", "reviewer", "author", "username"], "product_price": ["price", "product_price"],
       "seller_information": ["seller", "seller_information"]}
GENUINE = {"genuine", "or", "real", "truthful", "recommended", "legitimate", "0", "original"}
FAKE = {"fake", "cg", "deceptive", "spam", "filtered", "1", "computer-generated"}


def a_generic(df, src, labelled=False):
    cols = {c.lower().strip(): c for c in df.columns}
    o = blank(df, len(df))
    for target, names in SYN.items():
        for n in names:
            if n in cols:
                o[target] = df[cols[n]].values
                break
    if o["review_text"].isna().all():
        raise ValueError(f"{src}: no review-text column found (looked for {SYN['review_text']}). Columns are {list(df.columns)[:15]}")
    o["rating"] = pd.to_numeric(o["rating"], errors="coerce")
    o["review_date"] = pd.to_datetime(o["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    o["reviewer_id"] = o["reviewer_id"].map(hash_name)
    o["verified_purchase"] = o["verified_purchase"].map(lambda v: {"true": True, "yes": True, "false": False, "no": False, "y": True, "n": False}.get(str(v).strip().lower()))
    o["review_id"] = [f"{src}:{i}" for i in range(len(df))]
    if labelled:
        lab = next((cols[c] for c in ("label", "labels", "class", "is_fake", "fake", "authenticity") if c in cols), None)
        if lab is None:
            raise ValueError(f"{src}: no label column (label / class / is_fake ...).")
        v = df[lab].astype(str).str.strip().str.lower()
        o["authenticity_label"] = np.where(v.isin(GENUINE), "genuine", np.where(v.isin(FAKE), "fake", "unknown"))
    return o


ADAPTERS = {"amazon_com_export": a_amazon_com, "meesho_multilingual": a_meesho, "flipkart_style_csv": a_flipkart,
            "generic": a_generic, "labelled_generic": lambda d, s: a_generic(d, s, True)}


# ----------------------------------------------------------------------------- commands
def cmd_list(_a):
    for s in SOURCES:
        print(f"\n[{s['id']}]  {s['title']}\n   platform: {s['platform']}   type: {s['record_type']}   records: {s['records'] if s['records'] else 'not stated'}"
              f"   (checked: {s['verified']})\n   licence: {s['licence']}\n   access: {s['access']}   {s['url']}\n   labels: {s['labels']}\n   note: {s['notes']}")


def cmd_fetch(_a):
    kaggle = shutil.which("kaggle")
    for s in SOURCES:
        if s["access"] == "kaggle":
            if not kaggle:
                print(f"[{s['id']}] install the Kaggle CLI (pip install kaggle), put your API key in ~/.kaggle/kaggle.json, then re-run.\n   manual: {s['url']}")
                continue
            dest = RAW / s["id"]; dest.mkdir(parents=True, exist_ok=True)
            print(f"[{s['id']}] downloading {s['kaggle_slug']} ...")
            subprocess.run([kaggle, "datasets", "download", "-d", s["kaggle_slug"], "-p", str(dest), "--unzip"], check=False)
        elif s["access"] in ("manual_download", "huggingface", "email_request"):
            print(f"[{s['id']}] MANUAL: {s['access']}  {s['url']}  -> save the files into {RAW / s['id']}{'  (licence: ' + s['licence'] + ')'}")
        elif s["access"] == "commercial":
            print(f"[{s['id']}] COMMERCIAL: {s['url']}  (paid; read the terms). Not downloaded.")
    print("\nThen run: python build_master_dataset.py build")


def union_groups(keys_a, keys_b):
    parent = list(range(len(keys_a)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for keys in (keys_a, keys_b):
        seen = {}
        for i, k in enumerate(keys):
            if not k:
                continue
            if k in seen:
                ra, rb = find(i), find(seen[k])
                if ra != rb:
                    parent[ra] = rb
            else:
                seen[k] = i
    return np.array([find(i) for i in range(len(parent))])


def md_table(df):
    if df.empty:
        return "_none_\n"
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:,.0f}" if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and float(v).is_integer() else (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def cmd_build(a):
    files = {}
    for kv in a.file or []:
        k, _, v = kv.partition("=")
        files[k] = [Path(v)]
    parts, log = [], []
    by_id = {s["id"]: s for s in SOURCES}
    for sid, s in by_id.items():
        paths = files.get(sid) or (sorted(p for p in (RAW / sid).glob("**/*") if p.suffix.lower() in (".csv", ".xlsx", ".json")) if (RAW / sid).exists() else [])
        if not paths or s["adapter"] == "none":
            continue
        for p in paths:
            raw = read_any(p)
            o = ADAPTERS[s["adapter"]](raw, sid)
            o["source_dataset"] = sid
            o["record_type"] = s["record_type"]
            if o["platform_name"].isna().all():
                first = s["platform"].split(";")[0].strip()
                o["platform_name"] = first if first in PLATFORMS else "Other"
            if o["authenticity_label"].isna().all():
                o["authenticity_label"] = "unknown"
            else:
                o["label_source"] = f"{sid} (labels shipped with the source)"
            o["source_url"] = o["source_url"].fillna(s["url"]) if s["adapter"] == "generic" else o["source_url"]
            log.append((sid, p.name, len(raw), len(o)))
            parts.append(o)
            print(f"  {sid:28s} {p.name}: read {len(raw):,} rows -> {len(o):,} records")
    if not parts:
        sys.exit("No data found. Put files in raw/<source_id>/ or pass --file source_id=path (python build_master_dataset.py list shows the ids).")
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)

    # --- missing values & exact duplicates
    df["review_text"] = df["review_text"].astype(str).str.replace(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["review_text"].ne("") & df["review_text"].ne("nan")]
    n_empty = n0 - len(df)
    df["review_title"] = df["review_title"].where(df["review_title"].notna(), "")
    df["label_source"] = df["label_source"].where(df["label_source"].notna(), "")
    before = len(df)
    prod_key = df["product_id"].fillna(df["product_name"]).astype(str)
    df = df.loc[~pd.DataFrame({"p": df["platform_name"], "k": prod_key, "t": df["review_title"].astype(str), "x": df["review_text"], "r": df["rating"],
                               "d": df["review_date"], "u": df["reviewer_id"]}).duplicated()].reset_index(drop=True)
    n_exact = before - len(df)

    # --- normalise, language, near duplicates
    uniq = pd.Series(df["review_text"].unique())
    print(f"  normalising + language detection on {len(uniq):,} distinct texts ...")
    norm = dict(zip(uniq, uniq.map(lambda t: normalize_for_model(t).strip().lower())))
    lang = dict(zip(uniq, uniq.map(lambda t: detect_language(t)["language"])))
    df["text_normalized"] = df["review_text"].map(norm)
    df["review_language"] = df["review_text"].map(lang)
    df["dup_text_count"] = df.groupby(["platform_name", "text_normalized"])["text_normalized"].transform("size")
    toks = df["text_normalized"].str.split()
    raw_a = toks.map(lambda t: " ".join(sorted(set(t))) if len(t) >= 4 else "")
    raw_b = df["text_normalized"].map(lambda t: t[:80] if len(t) >= 40 else "")
    key_a = (df["platform_name"] + "|" + raw_a).where(raw_a != "", "")  # keys are per platform; short texts get no key
    key_b = (df["platform_name"] + "|" + raw_b).where(raw_b != "", "")
    root = union_groups(key_a.tolist(), key_b.tolist())  # rows with no key stay alone
    df["near_dup_group_id"] = ["g%d" % r for r in root]

    # --- leak-free split, decided per group
    h = df["near_dup_group_id"].map(lambda g: int(hashlib.md5(g.encode()).hexdigest(), 16) % 100)
    df["split"] = np.where(h < 70, "train", np.where(h < 85, "validation", "test"))
    df["review_text_translated"] = ""
    df = df[COLUMNS]

    OUT.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT / "master_reviews.csv", index=False, encoding="utf-8-sig")
    for p in PLATFORMS:
        df[df["platform_name"] == p].to_csv(OUT / f"{p.lower()}_reviews.csv", index=False, encoding="utf-8-sig")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "DATA_DICTIONARY.md").write_text("# Data dictionary (master_reviews.csv)\n\n| column | meaning |\n|---|---|\n" + "\n".join(f"| `{c}` | {DICTIONARY[c]} |" for c in COLUMNS) + "\n", encoding="utf-8")

    # --- reports
    pt = df.groupby(["platform_name", "record_type"]).size().unstack(fill_value=0)
    src = df.groupby(["source_dataset", "platform_name", "record_type"]).size().rename("records").reset_index().set_index("source_dataset")
    lab = df.groupby(["platform_name", "authenticity_label"]).size().unstack(fill_value=0)
    vp = df.groupby("platform_name")["verified_purchase"].apply(lambda s: pd.Series({"verified=True": int((s == True).sum()), "verified=False": int((s == False).sum()), "not stated": int(s.isna().sum())})).unstack()
    langs = df.groupby(["platform_name", "review_language"]).size().unstack(fill_value=0)
    langs_pct = (langs.div(langs.sum(axis=1), axis=0) * 100).round(1)
    miss = df.replace("", np.nan).groupby("platform_name").apply(lambda g: (g.isna().mean() * 100).round(1)).T
    miss.index.name = "column (% missing)"
    spl = df.groupby(["platform_name", "split"]).size().unstack(fill_value=0)
    rating = df.groupby(["platform_name", "rating"]).size().unstack(fill_value=0)
    log_df = pd.DataFrame(log, columns=["source", "file", "rows read", "records kept by adapter"]).set_index("source")
    (REPORTS / "source_report.md").write_text(
        "# Source report: product reviews vs shopping-app reviews\n\n"
        f"Processed {n0:,} records from the files found. Empty review text removed: {n_empty:,}. Exact duplicate records removed: {n_exact:,}. Final master: {len(df):,}.\n\n"
        "## Records per platform and type\n" + md_table(pt) + "\n## Records per source\n" + md_table(src) +
        "\n## Files read\n" + md_table(log_df) +
        "\nThe catalogue of ALL sources found (including those not downloaded yet) is in docs/DATASET_SOURCES_REPORT.md.\n"
        "`product_review` = reviews of products. `app_review` = reviews of the shopping apps (not usable for product-review fraud). `fake_benchmark` = research sets with authenticity labels.\n", encoding="utf-8")
    (REPORTS / "label_report.md").write_text(
        "# Label report: genuine, fake, unknown\n\nAuthenticity labels are only copied from sources that ship them. Everything else is `unknown`. "
        "Verified-purchase is shown separately and is not a label.\n\n## Authenticity label per platform\n" + md_table(lab) +
        "\n## Verified-purchase information per platform\n" + md_table(vp) +
        "\n## Label sources\n" + md_table(df[df["label_source"] != ""].groupby(["label_source", "authenticity_label"]).size().rename("records").reset_index().set_index("label_source")) , encoding="utf-8")
    (REPORTS / "quality_language_report.md").write_text(
        "# Quality and language report\n\n## Language distribution (% of reviews)\n" + md_table(langs_pct) + "\n## Language counts\n" + md_table(langs) +
        "\n## Star ratings\n" + md_table(rating) + "\n## Missing values (%)\n" + md_table(miss) +
        f"\n## Duplicates\n* Exact duplicate records removed: {n_exact:,}\n* Records whose text also appears elsewhere on the same platform: {(df['dup_text_count'] > 1).sum():,}\n"
        f"* Records in a near-duplicate group of 2+ reviews (flagged, not deleted): {(df.groupby('near_dup_group_id')['near_dup_group_id'].transform('size') > 1).sum():,}\n"
        "\n## Split sizes (grouped by near-duplicate group)\n" + md_table(spl) +
        "\nLanguage detection is a lexicon/script heuristic and can be wrong on very short or code-mixed text.\n", encoding="utf-8")
    print(f"\nDone: {len(df):,} records. Files in {OUT}\n" + pt.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list"); sub.add_parser("fetch")
    b = sub.add_parser("build"); b.add_argument("--file", action="append", help="source_id=path (repeatable)")
    a = ap.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
