"""Flask application: pages + JSON API for ReviewGuard (multilingual fake-review detection).

Every analysis is stored in SQLite (data/reviewguard.db) together with platform, language, sentiment, model version and
timestamp. Nothing here contacts a marketplace: reviews come from text the user types or files the user uploads.
"""
import io
import threading
import traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
import db
import ingest
import model_store
from ingest import IngestError, UnsupportedFileType, normalize_platform
from predict import DISCLAIMER, FakeReviewPredictor, ModelNotFoundError

# Honest integration status. No marketplace connection exists in this project.
PLATFORM_STATUS = [
    {"name": "Meesho", "live_api": False, "status": "Upload / export only",
     "note": "I found no official public review API. Reviews can be analysed from exports/datasets you provide (CSV/XLSX)."},
    {"name": "Flipkart", "live_api": False, "status": "Upload / export only",
     "note": "I found no official public API that returns product reviews. Third-party scraping services exist but may violate the platform's "
             "terms, so none is integrated. Upload exports or datasets you have the right to use."},
    {"name": "Myntra", "live_api": False, "status": "Upload / export only",
     "note": "I found no official public review API. Reviews can be analysed from exports/datasets you provide (CSV/XLSX)."},
    {"name": "Amazon", "live_api": False, "status": "Upload / export only",
     "note": "Amazon's Product Advertising API 5.0 does not return customer reviews (review retrieval was removed when version 5.0 replaced "
             "4.0 in March 2020) and I found no public review API in the Selling Partner API. Use review data you are entitled to use "
             "(your own seller/brand exports, licensed or public datasets) and upload it as CSV/XLSX."},
    {"name": "Other", "live_api": False, "status": "Manual / upload", "note": "Any other source: paste a review or upload a file."},
]
VERIFIED_ON = "October 2026 (web search; check each platform's current developer terms yourself)"


class ApiError(Exception):
    def __init__(self, message, status=400, details=None):
        super().__init__(message)
        self.message, self.status, self.details = message, status, details


class TrainingJob:
    """Runs one retraining at a time in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = {"status": "idle"}

    def snapshot(self):
        with self._lock:
            return dict(self.state)

    def start(self, fn):
        with self._lock:
            if self.state.get("status") == "running":
                raise ApiError("A training job is already running.", 409)
            self.state = {"status": "running", "message": "Starting", "started": db.now()}
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _run(self, fn):
        try:
            m = fn(progress=lambda msg: self._set(message=msg))
            o = m["test_metrics"]["overall"]
            self._set(status="done", message="Finished", finished=db.now(), version=m["version"], is_demo=m["is_demo_data"],
                      accuracy=o["accuracy"], f1=o["f1"], rows=m["cleaning_report"]["rows_final"])
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._set(status="error", message=str(exc), finished=db.now())


def validate_review_text(value, max_chars, min_chars):
    if not isinstance(value, str):
        raise ApiError("The review must be text.")
    text = value.replace("\x00", "").strip()
    if not text:
        raise ApiError("The review is empty. Please paste or type a review.")
    if len(text) < min_chars:
        raise ApiError(f"The review is too short (minimum {min_chars} characters).")
    if len(text) > max_chars:
        raise ApiError(f"The review is too long ({len(text)} characters). The limit is {max_chars}.", status=413)
    if not any(c.isalpha() for c in text):
        raise ApiError("The review must contain at least some letters.")
    return text


def parse_platform(value, default="Other"):
    if value in (None, ""):
        return default
    v = str(value).strip()
    for name in config.PLATFORMS:
        if name.lower() == v.lower():
            return name
    raise ApiError(f"Unknown platform '{v}'. Choose one of: {', '.join(config.PLATFORMS)}.")


def parse_product(value):
    """Optional product name. Accepts a pick-list item or free text (for products not in the list)."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ApiError("product must be text.")
    v = " ".join(value.replace("\x00", "").split())
    if len(v) > config.MAX_PRODUCT_CHARS:
        raise ApiError(f"The product name is too long. Keep it under {config.MAX_PRODUCT_CHARS} characters.")
    return v or None


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(config)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
    app.config.update({"DB_PATH": config.DB_PATH, "MODELS_DIR": config.MODELS_DIR, "IMAGES_DIR": config.IMAGES_DIR})
    if test_config:
        app.config.update(test_config)

    predictor = FakeReviewPredictor(app.config["MODELS_DIR"], config.TOP_TERMS, config.LOW_CONFIDENCE_THRESHOLD)
    app.extensions["predictor"] = predictor
    job = TrainingJob()
    app.extensions["training_job"] = job
    dbp = lambda: app.config["DB_PATH"]  # noqa: E731

    def get_predictor():
        try:
            return predictor.ensure_loaded()
        except ModelNotFoundError as exc:
            raise ApiError(str(exc), 503)

    def model_status():
        try:
            predictor.ensure_loaded()
            return {"loaded": True, "version": predictor.version, "is_demo": predictor.is_demo, "error": None}
        except ModelNotFoundError as exc:
            return {"loaded": False, "version": None, "is_demo": False, "error": str(exc)}

    def local_only():
        if request.remote_addr not in ("127.0.0.1", "::1", None):
            raise ApiError("This action is only allowed from the computer running the server (localhost).", 403)

    # ------------------------------------------------------------ pages
    @app.context_processor
    def inject():
        return {"model_ready": model_store.get_active(app.config["MODELS_DIR"]) is not None, "disclaimer": DISCLAIMER,
                "limits": {"maxChars": config.MAX_REVIEW_CHARS, "maxBulk": config.MAX_BULK_REVIEWS,
                           "maxUploadMb": config.MAX_UPLOAD_BYTES // (1024 * 1024), "platforms": config.PLATFORMS, "products": config.PRODUCTS}}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/single")
    def single_review():
        return render_template("single_review.html")

    @app.get("/bulk")
    def bulk_review():
        return render_template("bulk_review.html")

    @app.get("/history")
    def history_page():
        return render_template("history.html")

    @app.get("/about")
    def about():
        return render_template("about.html", platforms=PLATFORM_STATUS)

    @app.get("/download/sample-csv")
    def download_sample_csv():
        return send_file(config.SAMPLE_UPLOAD_MULTI_CSV, as_attachment=True, download_name="reviewguard_sample_reviews.csv")

    @app.get("/download/sample-xlsx")
    def download_sample_xlsx():
        return send_file(config.SAMPLE_UPLOAD_MULTI_XLSX, as_attachment=True, download_name="reviewguard_sample_reviews.xlsx")

    # -------------------------------------------------------------- API
    @app.get("/health")
    @app.get("/api/health")
    def health():
        st = model_status()
        return jsonify({"ok": True, "model_loaded": st["loaded"], "model_version": st["version"], "model_is_demo": st["is_demo"],
                        "model_error": st["error"]})

    @app.get("/api/platforms")
    def api_platforms():
        return jsonify({"ok": True, "platforms": PLATFORM_STATUS, "checked": VERIFIED_ON,
                        "live_integration_available": any(p["live_api"] for p in PLATFORM_STATUS)})

    def entry_for_db(item, result):
        return {**result, "review": item["text"], "title": item.get("title"), "rating": item.get("rating"),
                "platform": item.get("platform", "Other"), "product": item.get("product"), "language": result["language"]["language"]}

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "review" not in payload:
            raise ApiError('Send JSON like {"review": "your text", "platform": "Flipkart"}.')
        text = validate_review_text(payload["review"], config.MAX_REVIEW_CHARS, config.MIN_REVIEW_CHARS)
        platform = parse_platform(payload.get("platform"))
        rating = payload.get("rating")
        if rating not in (None, ""):
            try:
                rating = float(rating)
                if not 1 <= rating <= 5:
                    raise ValueError
            except (TypeError, ValueError):
                raise ApiError("rating must be a number from 1 to 5.")
        else:
            rating = None
        product = parse_product(payload.get("product"))
        item = {"text": text, "title": str(payload.get("title") or "")[:200], "platform": platform, "rating": rating, "product": product}
        result = get_predictor().predict_many([item])[0]
        [new_id] = db.record_analyses(dbp(), [entry_for_db(item, result)], "manual")
        return jsonify({"ok": True, "result": {**result, "id": new_id, "review": text, "product": product}})

    @app.post("/api/predict-bulk")
    def api_predict_bulk():
        upload = request.files.get("file")
        filename, source_type = None, "batch_json"
        if upload is not None:
            platform = parse_platform(request.form.get("platform"))
            filename = upload.filename or "upload"
            try:
                df = ingest.read_table(upload.read(), filename)
                items, skipped = ingest.prepare_unlabelled(df, platform)
            except IngestError as exc:
                raise ApiError(str(exc), 415 if isinstance(exc, UnsupportedFileType) else 400)
            source_type = "batch_xlsx" if filename.lower().endswith(".xlsx") else "batch_csv"
            rows_read = len(df)
        else:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
                raise ApiError('Send JSON like {"reviews": ["text 1", "text 2"], "platform": "Meesho"} or upload a .csv/.xlsx in the "file" field.')
            platform = parse_platform(payload.get("platform"))
            if len(payload["reviews"]) > config.MAX_BULK_REVIEWS:
                raise ApiError(f"Too many reviews ({len(payload['reviews'])}). The limit is {config.MAX_BULK_REVIEWS}.")
            items, skipped = [], []
            for i, v in enumerate(payload["reviews"], start=1):
                try:
                    items.append({"row": i, "text": validate_review_text(v, config.MAX_REVIEW_CHARS, config.MIN_REVIEW_CHARS),
                                  "title": "", "rating": None, "platform": platform})
                except ApiError as exc:
                    skipped.append({"row": i, "reason": exc.message})
            rows_read = len(payload["reviews"])
        if not items:
            raise ApiError("No valid reviews to analyse.", details={"skipped": skipped[:50]})
        predictor_ = get_predictor()
        results = predictor_.predict_many(items)
        batch_id = db.create_batch(dbp(), filename, platform, rows_read)
        ids = db.record_analyses(dbp(), [entry_for_db(i, r) for i, r in zip(items, results)], source_type, batch_id)
        db.finish_batch(dbp(), batch_id, len(items), len(skipped))
        out, fake = [], 0
        summary = {"by_language": {}, "by_platform": {}, "by_sentiment": {}}
        for aid, item, r in zip(ids, items, results):
            fake += r["label"] == "Likely Fake"
            for key, val in (("by_language", r["language"]["language"]), ("by_platform", item["platform"]), ("by_sentiment", r["sentiment"]["label"])):
                summary[key][val] = summary[key].get(val, 0) + 1
            out.append({"id": aid, "row": item["row"], "review": item["text"], "title": item["title"], "rating": item["rating"], "product": item.get("product"),
                        "product_check": r["product_check"]["level"], "platform": item["platform"], "label": r["label"], "display_label": r["display_label"],
                        "fake_probability": r["fake_probability"], "confidence": r["confidence"], "low_confidence": r["low_confidence"],
                        "language": r["language"], "sentiment": r["sentiment"], "model_version": r["model_version"],
                        "top_fake_terms": r["top_fake_terms"][:3], "top_genuine_terms": r["top_genuine_terms"][:3]})
        summary.update(total=len(out), fake=int(fake), genuine=len(out) - int(fake), low_confidence=sum(1 for r in results if r["low_confidence"]),
                       skipped=len(skipped), batch_id=batch_id)
        return jsonify({"ok": True, "summary": summary, "results": out, "skipped": skipped[:100], "disclaimer": DISCLAIMER})

    @app.get("/api/model-info")
    def api_model_info():
        get_predictor()
        return jsonify({"ok": True, "metrics": predictor.metadata, "version": predictor.version, "kind": predictor.kind})

    @app.get("/api/models")
    def api_models():
        return jsonify({"ok": True, "versions": model_store.list_versions(app.config["MODELS_DIR"]), "registry": db.list_models(dbp())})

    @app.post("/api/models/activate")
    def api_activate_model():
        local_only()
        payload = request.get_json(silent=True) or {}
        version = str(payload.get("version", ""))
        if not model_store.version_dir(app.config["MODELS_DIR"], version).exists():
            raise ApiError("Unknown model version.", 404)
        model_store.set_active(app.config["MODELS_DIR"], version)
        db.set_active_model(dbp(), version)
        return jsonify({"ok": True, "active": version})

    @app.get("/api/history")
    def api_history():
        try:
            limit = max(1, min(int(request.args.get("limit", 200)), config.MAX_RECENT_ANALYSES))
            offset = max(0, int(request.args.get("offset", 0)))
        except ValueError:
            raise ApiError("limit and offset must be integers.")
        items = db.list_analyses(dbp(), limit, offset, request.args.get("platform") or None, request.args.get("label") or None,
                                 request.args.get("language") or None)
        s = db.analysis_stats(dbp())
        return jsonify({"ok": True, "summary": {"total": s["total"], "fake": s["fake"], "genuine": s["genuine"]}, "items": items})

    @app.delete("/api/history")
    def api_history_clear():
        return jsonify({"ok": True, "removed": db.clear_analyses(dbp())})

    @app.get("/api/stats")
    def api_stats():
        st = model_status()
        return jsonify({"ok": True, "analyses": db.analysis_stats(dbp()), "labelled": db.labelled_summary(dbp()), "model": st})

    @app.get("/api/dataset-info")
    def api_dataset_info():
        return jsonify({"ok": True, "labelled": db.labelled_summary(dbp()), "platforms": PLATFORM_STATUS, "training": job.snapshot(),
                        "min_rows": config.MIN_ROWS_FOR_TRAINING, "min_rows_per_class": config.MIN_ROWS_PER_CLASS})

    @app.post("/api/ingest")
    def api_ingest():
        local_only()
        upload = request.files.get("file")
        if upload is None:
            raise ApiError('Upload a labelled .csv or .xlsx in the "file" field.')
        platform = parse_platform(request.form.get("platform"))
        method = (request.form.get("label_method") or "").strip()[:80] or None
        try:
            report = ingest.ingest_labelled(dbp(), upload.read(), upload.filename or "upload", platform, method)
        except IngestError as exc:
            raise ApiError(str(exc), 415 if isinstance(exc, UnsupportedFileType) else 400)
        return jsonify({"ok": True, "report": report})

    @app.post("/api/retrain")
    def api_retrain():
        local_only()
        payload = request.get_json(silent=True) or {}
        mode = payload.get("data_mode", "auto")
        if mode not in ("auto", "real", "all"):
            raise ApiError("data_mode must be auto, real or all.")
        balance = bool(payload.get("balance", True))
        import train_model

        def run(progress):
            return train_model.train(dbp(), app.config["MODELS_DIR"], app.config["IMAGES_DIR"], data_mode=mode, balance=balance,
                                     verbose=False, progress=progress)
        job.start(run)
        return jsonify({"ok": True, "training": job.snapshot()}), 202

    @app.get("/api/retrain/status")
    def api_retrain_status():
        return jsonify({"ok": True, "training": job.snapshot()})

    # ------------------------------------------------------------ errors
    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(ApiError)
    def on_api_error(exc):
        body = {"ok": False, "error": exc.message}
        if exc.details:
            body.update(exc.details)
        return jsonify(body), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def on_too_large(_exc):
        return jsonify({"ok": False, "error": f"Upload too large. Maximum size is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413

    @app.errorhandler(HTTPException)
    def on_http(exc):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": exc.description}), exc.code
        return render_template("error.html", code=exc.code, message=exc.description), exc.code

    @app.errorhandler(Exception)
    def on_unexpected(exc):
        app.logger.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Unexpected server error. See the terminal for details."}), 500
        return render_template("error.html", code=500, message="Unexpected server error."), 500

    return app


app = create_app()

def _free_port(start=5000, tries=11):
    import socket
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:      # nothing is listening there
                return port
    return start


if __name__ == "__main__":
    port = _free_port()
    if port != 5000:
        print(f"\nPort 5000 is already in use (an older copy may still be running). Using port {port} instead.")
    print(f"\nOpen this in your browser:  http://127.0.0.1:{port}/single\n")
    app.run(host="127.0.0.1", port=port, debug=False)
