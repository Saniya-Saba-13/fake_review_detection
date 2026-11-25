"""Load the active model version and produce predictions with language, sentiment and explanations.

The website never retrains while serving: it loads models/active.json -> models/versions/<version>/ and
reloads automatically when a newer version becomes active (after `python train_model.py` or a retrain job).
Two back-ends exist: scikit-learn bundles (model.joblib, always available) and optional transformer models
(train_transformer.py; need `pip install torch transformers`, UNTESTED in the build environment).
"""
import threading
from pathlib import Path

import joblib
import numpy as np

import config
import model_store
from multilingual import detect_language, normalize_for_model
import hinglish_support
import product_context
from preprocessing import clean_pasted_review, text_signals
from sentiment import analyze_sentiment

DISCLAIMER = (
    "This is a statistical estimate from writing patterns learned from a training dataset. It is not proof that a review is fake "
    "or genuine, and the model can be wrong. Writing in Hindi or Hinglish, or expressing a negative opinion, is never evidence of "
    "fraud. Do not use it as the sole basis for accusing anyone or for moderation decisions."
)


class ModelNotFoundError(RuntimeError):
    """No trained model is available (or it cannot be loaded)."""


def extract_weights(clf):
    """Per-feature weights (positive = towards 'fake') for linear / naive-Bayes models, or None."""
    if hasattr(clf, "calibrated_classifiers_"):
        ws = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                est = getattr(cc, "base_estimator", None)
            if est is not None and hasattr(est, "coef_"):
                ws.append(np.asarray(est.coef_).ravel())
        return np.mean(ws, axis=0) if ws else None
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel()
    if hasattr(clf, "feature_log_prob_"):
        classes = list(clf.classes_)
        return clf.feature_log_prob_[classes.index(1)] - clf.feature_log_prob_[classes.index(0)]
    return None


def explanation_weights(pipeline, X_train_features, y_train):
    """Weights used to explain predictions. Random Forests have no signed coefficients, so importance x direction
    (mean TF-IDF in fake reviews minus mean in genuine reviews) is used instead."""
    clf = pipeline.named_steps["clf"]
    w = extract_weights(clf)
    if w is not None:
        return w
    if hasattr(clf, "feature_importances_"):
        y = np.asarray(y_train)
        direction = np.asarray(X_train_features[y == 1].mean(axis=0)).ravel() - np.asarray(X_train_features[y == 0].mean(axis=0)).ravel()
        return clf.feature_importances_ * np.sign(direction) * 10
    return None


def model_text(text, title=""):
    """The text the model sees: review title (if any) + review text. Same rule as at training time."""
    return (f"{(title or '').strip()} . {text}").strip(" .") if (title or "").strip() else str(text)


class _SklearnBackend:
    def __init__(self, bundle):
        self.pipeline, self.weights = bundle["pipeline"], bundle.get("weights")
        feats = self.pipeline.named_steps["features"]
        self.names = np.asarray(feats.get_feature_names_out())
        self.is_word = np.array([n.startswith("word__") and not n[6:].startswith("zz") for n in self.names])

    def fake_proba(self, cleaned):
        X = self.pipeline.named_steps["features"].transform(cleaned).tocsr()
        clf = self.pipeline.named_steps["clf"]
        p = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
        return p, X

    def terms(self, row, top_n):
        if self.weights is None or row.nnz == 0:
            return [], []
        idx, vals = row.indices, row.data
        keep = self.is_word[idx]
        if not keep.any():
            return [], []
        idx, vals = idx[keep], vals[keep]
        contrib = vals * self.weights[idx]
        mx = float(np.max(np.abs(contrib))) or 1.0
        order = np.argsort(contrib)
        fake = [{"term": self.names[idx[j]][6:], "strength": int(round(100 * contrib[j] / mx))} for j in order[::-1] if contrib[j] > 0][:top_n]
        gen = [{"term": self.names[idx[j]][6:], "strength": int(round(100 * -contrib[j] / mx))} for j in order if contrib[j] < 0][:top_n]
        return fake, gen


class _TransformerBackend:
    """Optional Hugging Face back-end. Requires torch + transformers. Not exercised by the automated tests."""

    def __init__(self, directory):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelNotFoundError("This model version needs `pip install torch transformers`.") from exc
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).eval()

    def fake_proba(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                enc = self.tok(texts[i:i + 16], truncation=True, max_length=256, padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out), None

    def terms(self, text, top_n):
        """Occlusion: how much does the fake probability change when one word is removed?"""
        words = text.split()[:40]
        if len(words) < 2:
            return [], []
        base = self.fake_proba([text])[0][0]
        drops = self.fake_proba([" ".join(words[:i] + words[i + 1:]) for i in range(len(words))])[0]
        delta = base - drops
        mx = float(np.max(np.abs(delta))) or 1.0
        order = np.argsort(delta)
        fake = [{"term": words[j], "strength": int(round(100 * delta[j] / mx))} for j in order[::-1] if delta[j] > 0][:top_n]
        gen = [{"term": words[j], "strength": int(round(100 * -delta[j] / mx))} for j in order if delta[j] < 0][:top_n]
        return fake, gen


class FakeReviewPredictor:
    def __init__(self, models_dir=config.MODELS_DIR, top_n=config.TOP_TERMS, low_conf=config.LOW_CONFIDENCE_THRESHOLD):
        self.models_dir, self.top_n, self.low_conf = Path(models_dir), top_n, low_conf
        self.backend, self.version, self.metadata = None, None, {}
        self.kind = None
        self.reliable_langs = set()
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self.backend is not None

    @property
    def is_demo(self):
        return bool(self.metadata.get("is_demo_data", False))

    def load(self):
        version = model_store.get_active(self.models_dir)
        if not version:
            raise ModelNotFoundError("No trained model found. Run `python build_dataset.py --demo` and `python train_model.py` first "
                                     "(see the README), then restart or refresh.")
        vdir = model_store.version_dir(self.models_dir, version)
        try:
            if (vdir / "model.joblib").exists():
                self.backend, self.kind = _SklearnBackend(joblib.load(vdir / "model.joblib")), "sklearn"
            elif (vdir / "transformer").exists():
                self.backend, self.kind = _TransformerBackend(vdir / "transformer"), "transformer"
            else:
                raise ModelNotFoundError(f"Model files for version {version} are missing.")
        except ModelNotFoundError:
            self.backend = None
            raise
        except Exception as exc:  # noqa: BLE001 - corrupt file, library version mismatch ...
            self.backend = None
            raise ModelNotFoundError(f"Could not load model {version} ({exc}). Re-run `python train_model.py` in this environment.") from exc
        self.version = version
        self.metadata = model_store.read_metrics(self.models_dir, version) or {}
        by_lang = (self.metadata.get("test_metrics") or {}).get("by_language") or {}
        self.reliable_langs = {k for k, v in by_lang.items() if v.get("reliable")}
        return self

    def ensure_loaded(self):
        """Lazy load; reload when another version became active."""
        with self._lock:
            if self.backend is None or model_store.get_active(self.models_dir) != self.version:
                self.load()
            return self

    # ---------------------------------------------------------------- predict
    def predict_many(self, items):
        """items: list of str or dicts {text, title?, platform?, rating?}. Returns one result dict per item."""
        self.ensure_loaded()
        norm = [{"text": it} if isinstance(it, str) else dict(it) for it in items]
        pasted = []
        for n in norm:  # drop what the shopping site added around the review (date, 'Verified Purchase' ...)
            n["text"], info = clean_pasted_review(n["text"])
            pasted.append(info)
            if info["platform"] and n.get("platform") in (None, "", "Other"):
                n["platform"] = info["platform"]
            if info["rating"] and not n.get("rating"):
                n["rating"] = info["rating"]
        texts = [model_text(n["text"], n.get("title", "")) for n in norm]
        fits = [product_context.product_fit(n["text"], n.get("product")) for n in norm]
        aware = bool(self.metadata.get("product_aware"))     # only models trained with product tokens get them
        langs = [detect_language(t) for t in texts]
        bridges = [hinglish_support.english_bridge(normalize_for_model(t), lg["language"]) for t, lg in zip(texts, langs)]
        pats = [hinglish_support.spam_patterns(n["text"], lg["language"]) for n, lg in zip(norm, langs)]
        if self.kind == "sklearn":
            cleaned = [(normalize_for_model(t) + " " + b[0]).strip() for t, b in zip(texts, bridges)]
            if aware:
                cleaned = [(c + " " + product_context.model_tokens(n.get("product"), n["text"])).strip() for c, n in zip(cleaned, norm)]
            probs, X = self.backend.fake_proba(cleaned)
        else:
            probs, X = self.backend.fake_proba(texts)
        results = []
        for i, n in enumerate(norm):
            p_model = float(probs[i])
            p_fake, boosted = product_context.adjust_probability(p_model, fits[i])
            p_fake, pat_boost = hinglish_support.adjust_probability(p_fake, pats[i])
            is_fake = p_fake >= 0.5
            confidence = p_fake if is_fake else 1.0 - p_fake
            fake_terms, gen_terms = self.backend.terms(X[i] if self.kind == "sklearn" else texts[i], self.top_n)
            lang, senti, signals = langs[i], analyze_sentiment(texts[i]), list(text_signals(n["text"]))
            for d in pats[i]["details"]:
                signals.append({"name": "Repeated / stacked wording" if "repeated" in d else "Stacked hype wording", "detail": d})
            if fits[i]["level"] == product_context.MISMATCH:
                signals.append({"name": "Review does not match the product", "detail": fits[i]["message"]})
            n_words = len(normalize_for_model(texts[i]).split())
            low = confidence < self.low_conf or n_words < 4 or (not fake_terms and not gen_terms)
            lang_note = None
            if self.reliable_langs and lang["language"] not in self.reliable_langs:
                low = True
                lang_note = (f"The model has too little verified training data in {lang['language']}, so it cannot judge this review reliably. "
                             "Treat the answer as a guess.")
            label = "Likely Fake" if is_fake else "Likely Genuine"
            results.append({
                "label": label, "display_label": "Suspicious (likely fake)" if is_fake else "Likely genuine",
                "fake_probability": round(p_fake, 4), "genuine_probability": round(1 - p_fake, 4),
                "confidence": round(confidence, 4), "low_confidence": bool(low), "language_note": lang_note,
                "pasted_removed": pasted[i]["removed"], "detected_platform": pasted[i]["platform"], "detected_rating": pasted[i]["rating"],
                "language": lang, "sentiment": senti, "platform": n.get("platform", "Other"), "rating": n.get("rating"),
                "language_check": {"applied": lang["language"] in hinglish_support.NON_ENGLISH, "english_meanings_added": bridges[i][1],
                                   "pattern_details": pats[i]["details"], "raised_suspicion": bool(pat_boost)},
                "product": n.get("product"), "product_check": {**fits[i], "raised_suspicion": bool(boosted), "model_probability": round(p_model, 4)},
                "top_fake_terms": fake_terms, "top_genuine_terms": gen_terms, "signals": signals,
                "explanation": self._summary(label, confidence, fake_terms, gen_terms, low, n_words, lang, senti, fits[i], boosted, pats[i] if pat_boost else None),
                "disclaimer": DISCLAIMER, "model_version": self.version, "model_is_demo": self.is_demo,
            })
        return results

    def predict_one(self, text, **kw):
        return self.predict_many([{"text": text, **kw}])[0]

    @staticmethod
    def _summary(label, confidence, fake_terms, gen_terms, low, n_words, lang, senti, fit=None, boosted=False, pat=None):
        pct = int(round(confidence * 100))
        parts = [f"The model estimates this review is {label.lower()} (about {pct}% estimated probability)."]
        main, other = (fake_terms, gen_terms) if label == "Likely Fake" else (gen_terms, fake_terms)
        side, opposite = ("fake", "genuine") if label == "Likely Fake" else ("genuine", "fake")
        if main:
            parts.append(f"Words or phrases pushing towards '{side}': " + ", ".join(f"\"{t['term']}\"" for t in main[:4]) + ".")
        if other:
            parts.append(f"Terms pointing the other way ('{opposite}'): " + ", ".join(f"\"{t['term']}\"" for t in other[:3]) + ".")
        if not main and not other:
            parts.append("Few or none of the words were known to the model, so this result is close to a guess.")
        parts.append(f"Detected language: {lang['language']}. Sentiment: {senti['label'].lower()}. Language and sentiment are shown for context only; "
                     "negative, positive, Hindi or Hinglish wording is not evidence of a fake review.")
        if fit and fit.get("product"):
            parts.append("Product check (" + fit["product"] + "): " + (fit.get("message") or "no information") +
                         (" This raised the suspicion score a little." if boosted else " It did not change the score."))
        if pat:
            parts.append("Hinglish/Hindi check: " + " ".join(pat["details"]) + " This raised the suspicion score a little.")
        if n_words < 4:
            parts.append("The review is very short, which gives the model little to work with.")
        if low:
            parts.append("Confidence is low: treat this result with extra caution.")
        return " ".join(parts)
