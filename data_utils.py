"""Dataset loading, label encoding, cleaning and duplicate handling."""
from pathlib import Path

import pandas as pd

from preprocessing import clean_text

TEXT_COLUMNS = ["review_text", "text", "text_", "review", "reviewtext", "content", "comment"]
LABEL_COLUMNS = ["label", "class", "target", "is_fake"]

GENUINE_LABELS = {
    "genuine", "real", "original", "or", "truthful", "legit", "legitimate",
    "authentic", "not fake", "not_fake", "non-fake", "nonfake", "0",
}
FAKE_LABELS = {
    "fake", "cg", "computer-generated", "computer generated", "computer_generated",
    "deceptive", "spam", "fraud", "fraudulent", "untruthful", "1",
}


def find_column(columns, candidates):
    """Return the first column whose lower-cased name is in candidates."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def encode_labels(series: pd.Series):
    """Map label strings to 0 (genuine) / 1 (fake). Returns (encoded, unknown_values)."""
    normalised = series.astype(str).str.strip().str.lower()
    encoded = normalised.map(lambda v: 0 if v in GENUINE_LABELS else (1 if v in FAKE_LABELS else None))
    unknown = sorted(set(normalised[encoded.isna()]))
    return encoded, unknown


def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1")


def prepare_dataset(raw: pd.DataFrame):
    """Clean a raw dataframe. Returns (clean_df, report).

    clean_df columns: review_text, clean_text, label (0 genuine / 1 fake), [rating]
    """
    text_col = find_column(raw.columns, TEXT_COLUMNS)
    label_col = find_column(raw.columns, LABEL_COLUMNS)
    if text_col is None or label_col is None:
        raise ValueError(
            "Dataset needs a review-text column (one of "
            f"{TEXT_COLUMNS}) and a label column (one of {LABEL_COLUMNS}). "
            f"Found columns: {list(raw.columns)}"
        )
    rating_col = find_column(raw.columns, ["rating", "stars"])

    df = pd.DataFrame({"review_text": raw[text_col], "label_raw": raw[label_col]})
    if rating_col is not None:
        df["rating"] = raw[rating_col]

    report = {"rows_loaded": int(len(df))}

    # 1. missing values
    before = len(df)
    df["review_text"] = df["review_text"].astype("string").str.strip()
    df = df.dropna(subset=["review_text", "label_raw"])
    df = df[df["review_text"] != ""]
    report["missing_removed"] = int(before - len(df))

    # 2. label encoding
    encoded, unknown = encode_labels(df["label_raw"])
    if unknown:
        raise ValueError(
            f"Unrecognised label values: {unknown[:8]}. "
            f"Use genuine/fake (or OR/CG, 0/1). Accepted genuine: {sorted(GENUINE_LABELS)}; "
            f"accepted fake: {sorted(FAKE_LABELS)}"
        )
    df["label"] = encoded.astype(int)
    df = df.drop(columns=["label_raw"])

    # 3. text cleaning
    df["review_text"] = df["review_text"].astype(str)
    df["clean_text"] = df["review_text"].map(clean_text)
    before = len(df)
    df = df[df["clean_text"] != ""]
    report["empty_after_cleaning_removed"] = int(before - len(df))

    # 4. conflicting duplicates (same text, different labels) are ambiguous -> drop all
    n_labels = df.groupby("clean_text")["label"].transform("nunique")
    report["conflicting_duplicates_removed"] = int((n_labels > 1).sum())
    df = df[n_labels == 1]

    # 5. exact duplicates -> keep one (prevents the same text landing in train AND test)
    before = len(df)
    df = df.drop_duplicates(subset="clean_text", keep="first")
    report["duplicates_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    report["rows_final"] = int(len(df))
    report["class_counts"] = {
        "genuine": int((df["label"] == 0).sum()),
        "fake": int((df["label"] == 1).sum()),
    }
    return df, report
