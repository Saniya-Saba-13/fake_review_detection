"""Data ingestion: read CSV / Excel, validate, de-duplicate and store labelled reviews (or prepare unlabelled ones).

Nothing here talks to any marketplace. Data comes from files YOU provide (exports, public datasets, your own
annotation). Every labelled row must say HOW its label was obtained (`label_method`); a label is never inferred
from the platform name.
"""
import hashlib
import io
from datetime import date
from pathlib import Path

import pandas as pd

import config
import db
from data_utils import encode_labels, find_column
from multilingual import detect_language, normalize_for_model

TEXT_COLS = ["review_text", "review", "text", "text_", "comment", "content", "review_body"]
TITLE_COLS = ["review_title", "title", "headline", "summary"]
RATING_COLS = ["rating", "stars", "star_rating"]
PLATFORM_COLS = ["platform", "source_platform", "marketplace", "store", "source"]
LABEL_COLS = ["label", "class", "target", "is_fake"]
METHOD_COLS = ["label_method", "labelling_method", "label_source"]
PRODUCT_COLS = ["product", "product_name", "product_category", "item", "category"]
DATE_COLS = ["collection_date", "collected_at", "date", "review_date"]


class IngestError(ValueError):
    """The file cannot be used (wrong type, unreadable, missing required columns ...)."""


class UnsupportedFileType(IngestError):
    """The file extension is not .csv / .xlsx (reported to API clients as HTTP 415)."""


def normalize_platform(value, default="Other"):
    v = str(value or "").strip().lower()
    for name in config.PLATFORMS[:4]:
        if name.lower() in v:
            return name
    return default if default in config.PLATFORMS else "Other"


def text_hash(text) -> str:
    return hashlib.sha1(normalize_for_model(text).encode("utf-8")).hexdigest()


def read_table(source, filename: str, max_rows=config.MAX_INGEST_ROWS) -> pd.DataFrame:
    """Read .csv or .xlsx (bytes or path) into a string DataFrame."""
    name = (filename or "").lower()
    if name.endswith(".xls"):
        raise UnsupportedFileType("Old .xls files are not supported. Open the file in Excel and save it as .xlsx or .csv.")
    if not (name.endswith(".csv") or name.endswith(".xlsx")):
        raise UnsupportedFileType("Only .csv and .xlsx files are accepted.")
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    if not data or not data.strip():
        raise IngestError("The file is empty.")
    try:
        if name.endswith(".xlsx"):
            if data[:2] != b"PK":
                raise IngestError("This does not look like a valid .xlsx file.")
            df = pd.read_excel(io.BytesIO(data), dtype=str, engine="openpyxl", nrows=max_rows + 1)
        else:
            if b"\x00" in data:
                raise IngestError("This does not look like a text CSV file.")
            for enc in ("utf-8-sig", "utf-16", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding=enc, nrows=max_rows + 1)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                raise IngestError("Could not decode the CSV. Save it as UTF-8 (needed for Hindi text).")
    except IngestError:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise IngestError(f"Could not parse the file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - corrupt xlsx etc.
        raise IngestError(f"Could not read the file ({type(exc).__name__}). Is it a valid .csv/.xlsx?") from exc
    if len(df) > max_rows:
        raise IngestError(f"Too many rows (limit {max_rows}).")
    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


def _clean_rating(value):
    try:
        r = float(str(value).strip())
    except ValueError:
        return None
    return r if 1 <= r <= 5 else None


def _clean_date(value, default):
    if not str(value).strip():
        return default, False
    try:
        d = pd.to_datetime(str(value).strip(), errors="raise")
        return d.date().isoformat(), False
    except (ValueError, TypeError):
        return default, True


# ------------------------------------------------------------- labelled data
def prepare_labelled(df, filename="", default_platform="Other", label_method=None, is_demo=False):
    """Validate a labelled table. Returns (rows, report). Rows are ready for db.insert_labelled."""
    text_col, label_col = find_column(df.columns, TEXT_COLS), find_column(df.columns, LABEL_COLS)
    if text_col is None or label_col is None:
        raise IngestError(f"Labelled data needs a review-text column ({TEXT_COLS}) and a label column ({LABEL_COLS}). "
                          f"Found: {list(df.columns)}")
    title_col, rating_col = find_column(df.columns, TITLE_COLS), find_column(df.columns, RATING_COLS)
    plat_col, method_col, date_col = (find_column(df.columns, PLATFORM_COLS), find_column(df.columns, METHOD_COLS),
                                      find_column(df.columns, DATE_COLS))
    prod_col = find_column(df.columns, PRODUCT_COLS)
    if method_col is None and not (label_method or "").strip():
        raise IngestError("Say how the labels were obtained: add a 'label_method' column or fill in the label method field "
                          "(e.g. manual_annotation, moderator_removed, verified_purchase_heuristic). Labels are never assumed.")
    labels, unknown = encode_labels(df[label_col])
    report = {"rows_read": int(len(df)), "invalid": 0, "invalid_reasons": {}, "rating_cleared": 0, "date_defaulted": 0}

    def bad(reason):
        report["invalid"] += 1
        report["invalid_reasons"][reason] = report["invalid_reasons"].get(reason, 0) + 1

    today, rows, seen = date.today().isoformat(), [], set()
    for i in range(len(df)):
        text = str(df[text_col].iloc[i]).replace("\x00", "").strip()
        if not text:
            bad("empty review text")
            continue
        if len(text) > config.MAX_REVIEW_CHARS:
            bad("review longer than %d characters" % config.MAX_REVIEW_CHARS)
            continue
        lab = labels.iloc[i]
        if lab is None or pd.isna(lab):
            bad("unrecognised label")
            continue
        method = (str(df[method_col].iloc[i]).strip() if method_col else "") or (label_method or "").strip()
        if not method:
            bad("missing label_method")
            continue
        h = text_hash(text)
        if not normalize_for_model(text):
            bad("no usable words after cleaning")
            continue
        if h in seen:
            continue  # duplicate inside the same file: counted by the database step as duplicate of the first copy
        seen.add(h)
        rating = _clean_rating(df[rating_col].iloc[i]) if rating_col else None
        if rating_col and str(df[rating_col].iloc[i]).strip() and rating is None:
            report["rating_cleared"] += 1
        cdate, defaulted = _clean_date(df[date_col].iloc[i], today) if date_col else (today, False)
        report["date_defaulted"] += int(defaulted)
        platform = normalize_platform(df[plat_col].iloc[i], default_platform) if plat_col else normalize_platform(default_platform)
        rows.append({"platform": platform, "review_title": str(df[title_col].iloc[i]).strip() if title_col else "",
                     "review_text": text, "rating": rating, "label": int(lab), "label_method": method,
                     "collection_date": cdate, "source_file": filename,
                     "product": (str(df[prod_col].iloc[i]).strip()[:80] or None) if prod_col and not pd.isna(df[prod_col].iloc[i]) else None, "language": detect_language(text)["language"],
                     "text_hash": h, "is_demo": int(is_demo)})
    report["duplicates_in_file"] = report["rows_read"] - report["invalid"] - len(rows)
    return rows, report


def ingest_labelled(db_path, source, filename, default_platform="Other", label_method=None, is_demo=False, max_rows=config.MAX_INGEST_ROWS):
    """Read + validate + store labelled reviews. Returns a report dict."""
    df = read_table(source, filename, max_rows)
    rows, report = prepare_labelled(df, filename, default_platform, label_method, is_demo)
    if not rows:
        raise IngestError("No valid labelled rows found. Reasons: " + (str(report["invalid_reasons"]) or "n/a"))
    added, dup, conflicts = db.insert_labelled(db_path, rows)
    report.update(rows_added=added, duplicates=dup + report["duplicates_in_file"], conflicts=conflicts)
    db.record_ingestion(db_path, filename=filename, platform=default_platform, label_method=label_method or "(per-row)",
                        rows_read=report["rows_read"], rows_added=added, duplicates=report["duplicates"],
                        conflicts=conflicts, invalid=report["invalid"], is_demo=is_demo)
    return report


# ----------------------------------------------------------- unlabelled data
def prepare_unlabelled(df, default_platform="Other", max_rows=config.MAX_BULK_REVIEWS):
    """Rows of a table to analyse. Returns (items, skipped). item = {row, text, title, rating, platform}."""
    text_col = find_column(df.columns, TEXT_COLS)
    if text_col is None:
        raise IngestError("Required column missing. The file needs a review column named one of: " + ", ".join(TEXT_COLS)
                          + f". Found: {list(df.columns)}")
    if len(df) > max_rows:
        raise IngestError(f"Too many rows ({len(df)}). The limit is {max_rows}.")
    title_col, rating_col, plat_col = (find_column(df.columns, TITLE_COLS), find_column(df.columns, RATING_COLS),
                                       find_column(df.columns, PLATFORM_COLS))
    prod_col = find_column(df.columns, PRODUCT_COLS)
    items, skipped = [], []
    for i in range(len(df)):
        row = i + 1
        text = str(df[text_col].iloc[i]).replace("\x00", "").strip()
        if not text:
            skipped.append({"row": row, "reason": "empty review text"})
            continue
        if len(text) < config.MIN_REVIEW_CHARS or not any(c.isalpha() for c in text):
            skipped.append({"row": row, "reason": "review must contain some letters"})
            continue
        if len(text) > config.MAX_REVIEW_CHARS:
            skipped.append({"row": row, "reason": f"review longer than {config.MAX_REVIEW_CHARS} characters"})
            continue
        rating = None
        if rating_col and str(df[rating_col].iloc[i]).strip():
            rating = _clean_rating(df[rating_col].iloc[i])
            if rating is None:
                skipped.append({"row": row, "reason": f"rating '{df[rating_col].iloc[i]}' must be a number from 1 to 5"})
                continue
        platform = normalize_platform(df[plat_col].iloc[i], default_platform) if plat_col else normalize_platform(default_platform)
        items.append({"row": row, "text": text, "title": str(df[title_col].iloc[i]).strip() if title_col else "",
                      "rating": rating, "platform": platform,
                      "product": (str(df[prod_col].iloc[i]).strip()[:80] or None) if prod_col and not pd.isna(df[prod_col].iloc[i]) else None})
    return items, skipped
