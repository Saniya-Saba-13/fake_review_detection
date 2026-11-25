"""Model registry on disk: models/versions/<version>/{model.joblib | transformer/, metrics.json} and models/active.json."""
import json
from datetime import datetime, timezone
from pathlib import Path

import config

ACTIVE_FILE = "active.json"


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def version_dir(models_dir, version) -> Path:
    return Path(models_dir) / "versions" / version


def set_active(models_dir, version):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    (Path(models_dir) / ACTIVE_FILE).write_text(json.dumps({"version": version}), encoding="utf-8")


def get_active(models_dir):
    p = Path(models_dir) / ACTIVE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        return None


def read_metrics(models_dir, version=None):
    version = version or get_active(models_dir)
    if not version:
        return None
    try:
        return json.loads((version_dir(models_dir, version) / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_versions(models_dir):
    """All versions found on disk (newest first) with a short summary."""
    root = Path(models_dir) / "versions"
    out = []
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            m = read_metrics(models_dir, d.name)
            if m:
                t = m["test_metrics"]["overall"]
                out.append({"version": d.name, "model_type": m["selected_model"]["name"], "is_demo": m["is_demo_data"],
                            "accuracy": t["accuracy"], "f1": t["f1"], "roc_auc": t.get("roc_auc"), "trained_at": m["trained_at_utc"],
                            "active": d.name == get_active(models_dir), "rows": m["cleaning_report"]["rows_final"]})
    return out
