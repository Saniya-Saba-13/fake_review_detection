"""SQLite storage (standard library only): analysed reviews, batches, labelled training data, model registry.

One connection per call (SQLite is fine for a local app); WAL mode lets the web app read while a retraining job writes.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    batch_id INTEGER,
    source_type TEXT NOT NULL,            -- manual | batch_csv | batch_xlsx | batch_json
    platform TEXT NOT NULL,
    language TEXT NOT NULL,
    review_title TEXT,
    review_text TEXT NOT NULL,
    rating REAL,
    product TEXT,
    prediction TEXT NOT NULL,             -- Likely Genuine | Likely Fake
    fake_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    low_confidence INTEGER NOT NULL,
    sentiment TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    model_version TEXT NOT NULL,
    top_terms TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_platform ON analyses(platform);
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, filename TEXT, platform TEXT,
    rows_read INTEGER, rows_analysed INTEGER, rows_skipped INTEGER
);
CREATE TABLE IF NOT EXISTS labelled_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL, review_title TEXT, review_text TEXT NOT NULL, rating REAL, product TEXT,
    label INTEGER NOT NULL,               -- 0 genuine, 1 fake
    label_method TEXT NOT NULL,           -- how the label was obtained (documented, never assumed from the platform)
    collection_date TEXT NOT NULL, source_file TEXT, language TEXT NOT NULL,
    text_hash TEXT NOT NULL UNIQUE, is_demo INTEGER NOT NULL DEFAULT 0, ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labelled_platform ON labelled_reviews(platform);
CREATE TABLE IF NOT EXISTS ingestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, filename TEXT, platform TEXT, label_method TEXT,
    rows_read INTEGER, rows_added INTEGER, duplicates INTEGER, conflicts INTEGER, invalid INTEGER, is_demo INTEGER
);
CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_type TEXT NOT NULL, path TEXT NOT NULL,
    is_demo INTEGER NOT NULL, is_active INTEGER NOT NULL DEFAULT 0, test_f1 REAL, test_accuracy REAL, summary TEXT
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        cols = {r[1] for r in con.execute("PRAGMA table_info(analyses)")}
        if "product" not in cols:      # upgrade databases created before the product field existed
            con.execute("ALTER TABLE analyses ADD COLUMN product TEXT")
        if "product" not in {r[1] for r in con.execute("PRAGMA table_info(labelled_reviews)")}:
            con.execute("ALTER TABLE labelled_reviews ADD COLUMN product TEXT")
        yield con
        con.commit()
    finally:
        con.close()


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------- analyses
def create_batch(path, filename, platform, rows_read):
    with connect(path) as con:
        cur = con.execute("INSERT INTO batches(created_at, filename, platform, rows_read) VALUES (?,?,?,?)",
                          (now(), filename, platform, rows_read))
        return cur.lastrowid


def finish_batch(path, batch_id, analysed, skipped):
    with connect(path) as con:
        con.execute("UPDATE batches SET rows_analysed=?, rows_skipped=? WHERE id=?", (analysed, skipped, batch_id))


def record_analyses(path, entries, source_type, batch_id=None):
    """entries: list of dicts (platform, language, review_text, ... prediction fields). Returns list of new ids."""
    ids = []
    with connect(path) as con:
        for e in entries:
            cur = con.execute(
                """INSERT INTO analyses(created_at, batch_id, source_type, platform, language, review_title, review_text, rating, product,
                   prediction, fake_probability, confidence, low_confidence, sentiment, sentiment_score, model_version, top_terms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now(), batch_id, source_type, e["platform"], e["language"], e.get("title"), e["review"], e.get("rating"), e.get("product"),
                 e["label"], e["fake_probability"], e["confidence"], int(e["low_confidence"]), e["sentiment"]["label"],
                 e["sentiment"]["score"], e["model_version"],
                 json.dumps({"fake": [t["term"] for t in e["top_fake_terms"]], "genuine": [t["term"] for t in e["top_genuine_terms"]]},
                            ensure_ascii=False)))
            ids.append(cur.lastrowid)
    return ids


def list_analyses(path, limit=200, offset=0, platform=None, label=None, language=None):
    where, args = [], []
    for col, val in (("platform", platform), ("prediction", label), ("language", language)):
        if val:
            where.append(f"{col} = ?")
            args.append(val)
    sql = "SELECT * FROM analyses" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ? OFFSET ?"
    with connect(path) as con:
        return _rows(con.execute(sql, (*args, int(limit), int(offset))))


def clear_analyses(path):
    with connect(path) as con:
        n = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        con.execute("DELETE FROM analyses")
        con.execute("DELETE FROM batches")
        return n


def analysis_stats(path):
    """Aggregates for the dashboard / results page (real counts from the database)."""
    with connect(path) as con:
        total = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        out = {"total": total, "fake": con.execute("SELECT COUNT(*) FROM analyses WHERE prediction='Likely Fake'").fetchone()[0]}
        out["genuine"] = total - out["fake"]
        out["low_confidence"] = con.execute("SELECT COUNT(*) FROM analyses WHERE low_confidence=1").fetchone()[0]
        for key, col in (("by_platform", "platform"), ("by_language", "language"), ("by_sentiment", "sentiment")):
            out[key] = _rows(con.execute(
                f"SELECT {col} AS name, COUNT(*) AS total, SUM(prediction='Likely Fake') AS fake FROM analyses GROUP BY {col} ORDER BY total DESC"))
        out["by_day"] = _rows(con.execute(
            "SELECT substr(created_at,1,10) AS day, COUNT(*) AS total, SUM(prediction='Likely Fake') AS fake FROM analyses "
            "GROUP BY day ORDER BY day DESC LIMIT 14"))[::-1]
        out["batches"] = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        return out


# ---------------------------------------------------------- labelled data
def insert_labelled(path, rows):
    """rows: list of dicts. Returns (added, duplicates, conflicts). Duplicate = same normalised text already stored."""
    added = dup = conflict = 0
    with connect(path) as con:
        for r in rows:
            existing = con.execute("SELECT label FROM labelled_reviews WHERE text_hash=?", (r["text_hash"],)).fetchone()
            if existing is not None:
                if existing["label"] != r["label"]:
                    conflict += 1
                else:
                    dup += 1
                continue
            con.execute(
                """INSERT INTO labelled_reviews(platform, review_title, review_text, rating, product, label, label_method, collection_date,
                   source_file, language, text_hash, is_demo, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["platform"], r.get("review_title"), r["review_text"], r.get("rating"), r.get("product"), r["label"], r["label_method"],
                 r["collection_date"], r.get("source_file"), r["language"], r["text_hash"], int(r.get("is_demo", 0)), now()))
            added += 1
    return added, dup, conflict


def record_ingestion(path, **kw):
    with connect(path) as con:
        con.execute("""INSERT INTO ingestions(created_at, filename, platform, label_method, rows_read, rows_added, duplicates,
                    conflicts, invalid, is_demo) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (now(), kw.get("filename"), kw.get("platform"), kw.get("label_method"), kw["rows_read"], kw["rows_added"],
                     kw["duplicates"], kw["conflicts"], kw["invalid"], int(kw.get("is_demo", 0))))


def fetch_labelled(path, include_demo=True, only_demo=False):
    where = "WHERE is_demo=1" if only_demo else ("" if include_demo else "WHERE is_demo=0")
    with connect(path) as con:
        return _rows(con.execute(f"SELECT * FROM labelled_reviews {where} ORDER BY id"))


def labelled_summary(path):
    with connect(path) as con:
        total = con.execute("SELECT COUNT(*) FROM labelled_reviews").fetchone()[0]
        out = {"total": total, "demo": con.execute("SELECT COUNT(*) FROM labelled_reviews WHERE is_demo=1").fetchone()[0]}
        out["verified"] = total - out["demo"]
        for key, col in (("by_platform", "platform"), ("by_language", "language"), ("by_method", "label_method")):
            out[key] = _rows(con.execute(
                f"SELECT {col} AS name, COUNT(*) AS total, SUM(label=1) AS fake, MIN(is_demo) AS demo FROM labelled_reviews GROUP BY {col} ORDER BY total DESC"))
        out["by_label"] = {"genuine": con.execute("SELECT COUNT(*) FROM labelled_reviews WHERE label=0").fetchone()[0],
                           "fake": con.execute("SELECT COUNT(*) FROM labelled_reviews WHERE label=1").fetchone()[0]}
        out["date_range"] = _rows(con.execute("SELECT MIN(collection_date) AS first, MAX(collection_date) AS last FROM labelled_reviews"))[0]
        out["ingestions"] = _rows(con.execute("SELECT * FROM ingestions ORDER BY id DESC LIMIT 15"))
        return out


# ---------------------------------------------------------- model registry
def register_model(path, version, model_type, model_path, is_demo, test_f1, test_accuracy, summary, activate=True):
    with connect(path) as con:
        if activate:
            con.execute("UPDATE model_versions SET is_active=0")
        con.execute("""INSERT OR REPLACE INTO model_versions(version, created_at, model_type, path, is_demo, is_active, test_f1,
                    test_accuracy, summary) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (version, now(), model_type, str(model_path), int(is_demo), int(activate), test_f1, test_accuracy, summary))


def list_models(path):
    with connect(path) as con:
        return _rows(con.execute("SELECT * FROM model_versions ORDER BY created_at DESC"))


def set_active_model(path, version):
    with connect(path) as con:
        if con.execute("SELECT 1 FROM model_versions WHERE version=?", (version,)).fetchone() is None:
            return False
        con.execute("UPDATE model_versions SET is_active=0")
        con.execute("UPDATE model_versions SET is_active=1 WHERE version=?", (version,))
        return True
