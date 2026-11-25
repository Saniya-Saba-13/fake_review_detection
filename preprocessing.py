"""Text cleaning and simple, transparent text-signal heuristics.

`clean_text` is applied to every review *before* it reaches the scikit-learn
pipeline (both at training time and at prediction time), so the saved model
file contains only standard scikit-learn objects.

`text_signals` produces human-readable observations (short text, many
exclamation marks, promotional phrases ...). They are shown to the user as
context only. They are NOT used as proof of fraud and do not change the
model's prediction.
"""
import re

from multilingual import normalize_devanagari, normalize_for_model

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")
_NON_TEXT = re.compile(r"[^a-z0-9'!\s]")
_STRAY_QUOTE = re.compile(r"(?<![a-z])'|'(?![a-z])")
_SPACES = re.compile(r"\s+")

PROMO_PHRASES_MULTI = [
    # Hindi (Devanagari)
    "सबसे बढ़िया", "जरूर खरीदें", "पैसा वसूल", "पांच स्टार", "अभी ऑर्डर", "बेस्ट प्रोडक्ट", "सबको खरीदना चाहिए", "ज्यादा मत सोचिए",
    # Hinglish
    "paisa vasool", "must buy", "best product", "five star", "sabko kharidna", "abhi order", "zyada mat sochiye", "best ever",
]
PROMO_PHRASES = [
    "highly recommend", "must buy", "buy now", "best product ever", "five stars",
    "5 stars", "life changing", "changed my life", "do not think twice",
    "don't think twice", "100% satisfied", "worth every penny", "order now",
    "best ever", "perfect product", "amazing product", "trustworthy seller",
]


def clean_text(text) -> str:
    """Lower-case, strip HTML/URLs, squeeze repeated letters, keep words and '!'."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = str(text).lower()
    t = _HTML.sub(" ", t)
    t = _URL.sub(" urltoken ", t)
    t = _REPEAT.sub(r"\1\1", t)  # "soooo" -> "soo"
    t = t.replace("!", " ! ")
    t = _NON_TEXT.sub(" ", t)
    t = _STRAY_QUOTE.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def text_signals(raw_text: str) -> list:
    """Return a list of {'name', 'detail'} observations about the raw text."""
    text = (raw_text or "").strip()
    signals = []
    if not text:
        return signals

    words = re.findall(r"[A-Za-z0-9'\u0900-\u097F]+", text)
    n_words = len(words)

    if n_words < 8:
        signals.append({
            "name": "Very short review",
            "detail": f"Only {n_words} word(s). Short reviews carry little information, so any prediction is weak.",
        })

    exclam = text.count("!")
    if exclam >= 3:
        signals.append({
            "name": "Many exclamation marks",
            "detail": f"{exclam} exclamation marks. Enthusiastic genuine reviews also do this.",
        })

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 15:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.5:
            signals.append({
                "name": "Mostly capital letters",
                "detail": "Heavy use of capitals is sometimes seen in spam, but also in emotional reviews.",
            })

    lowered = text.lower()
    found = [p for p in PROMO_PHRASES if p in lowered]
    norm = normalize_for_model(text)   # Hindi / Hinglish spelling variants are unified before matching
    found += [p for p in PROMO_PHRASES_MULTI if normalize_for_model(p) in norm and p not in found]
    if found:
        signals.append({
            "name": "Promotional wording",
            "detail": "Contains phrases such as: " + ", ".join(f"'{p}'" for p in found[:3]) + ".",
        })

    if n_words >= 10:
        unique_ratio = len({w.lower() for w in words}) / n_words
        if unique_ratio < 0.55:
            signals.append({
                "name": "Repeated wording",
                "detail": "Many words are repeated, which can indicate templated text.",
            })

    if _URL.search(text):
        signals.append({
            "name": "Contains a link",
            "detail": "Links inside reviews are sometimes promotional.",
        })
    return signals


# ---------------------------------------------------------------- text copied from a shopping site
_FLAG = r"[\U0001F1E6-\U0001F1FF]{2}"
_PASTE_RULES = [
    # (name, regex) - each removes a piece the shopping site adds around the real review
    ("review date line", re.compile(r"Reviewed in [A-Za-z ]+?\s*(?:" + _FLAG + r")?\s*on \d{1,2}[ -][A-Za-z]+[ ,-]+\d{4}", re.I)),
    ("'Verified Purchase' tag", re.compile(r"\bVerified Purchase\b", re.I)),
    ("'Certified Buyer' tag", re.compile(r"\bCertified Buyer\b(?:\s*,\s*[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)?")),
    ("'Verified Buyer' tag", re.compile(r"\bVerified (?:Buyer|Review)\b", re.I)),
    ("'found this helpful' line", re.compile(r"(?:\b\d[\d,]* (?:people|person)|One person) found this helpful\.?", re.I)),
    ("'Helpful' counter", re.compile(r"\bHelpful for \d[\d,]*|^\s*Helpful(?: \(\d+\))?\s*$", re.I | re.M)),
    ("'Was this review helpful?' line", re.compile(r"Was this review helpful\??", re.I)),
    ("time-ago text", re.compile(r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b", re.I)),
    ("button text (Report / Read more / Permalink)", re.compile(r"^\s*(?:Report(?: abuse)?|Read more|Permalink|Share|Like|Comment)\s*$|\bRead more\s*$", re.I | re.M)),
    ("star line", re.compile(r"^\s*[1-5](?:\.0)?\s*(?:\u2605|\u2B50|stars?)\s*$", re.I | re.M)),
]
_RATING_PATTERNS = [re.compile(r"\b([1-5](?:\.\d)?)\s+out of 5 stars\b", re.I), re.compile(r"^\s*([1-5])(?:\.0)?\s*(?:\u2605|\u2B50)\s*$", re.M)]


def clean_pasted_review(text):
    """Remove what a shopping site adds around a review when it is copied (date line, 'Verified Purchase', 'Certified Buyer, Mumbai',
    'Helpful for 12', 'Read more' ...), so the tool judges only the customer's own words.
    Returns (cleaned_text, info). info = {'removed': [names], 'platform': 'Amazon'|'Flipkart'|None, 'rating': float|None}.
    If cleaning would leave nothing, the original text is returned unchanged."""
    original = str(text or "")
    out, removed = original, []
    info = {"removed": removed, "platform": None, "rating": None}
    for pat in _RATING_PATTERNS:
        m = pat.search(out)
        if m:
            info["rating"] = float(m.group(1)); break
    low = original.lower()
    if re.search(r"reviewed in [a-z ]+ on \d|out of 5 stars|verified purchase", low):
        info["platform"] = "Amazon"
    elif "certified buyer" in low or "permalink" in low:
        info["platform"] = "Flipkart"
    out_rating = re.sub(r"\b[1-5](?:\.\d)?\s+out of 5 stars\b", " ", out, flags=re.I)
    if out_rating != out:
        removed.append("star rating text"); out = out_rating
    for name, pat in _PASTE_RULES:
        new = pat.sub(" ", out)
        if new != out:
            removed.append(name); out = new
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\s*\n\s*", "\n", out).strip(" \n,-|")
    if len(out.split()) < 1:
        return original, {"removed": [], "platform": info["platform"], "rating": info["rating"]}
    return out, info
