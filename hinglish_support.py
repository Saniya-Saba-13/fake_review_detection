# -*- coding: utf-8 -*-
"""Extra help for Hinglish / Hindi / mixed reviews.

Why: the training data has very few Hinglish reviews, so the word statistics learned from English reviews were not reaching
Hinglish text ("paisa vasool, must buy" meant nothing to a model that only knew "value for money, must buy").

1. ENGLISH BRIDGE: for Hinglish, Hindi and mixed reviews, English equivalents of known words are APPENDED as extra tokens. The original
   text is always kept; nothing is replaced or discarded. The model then sees both the original words and their English meaning.
2. SPAM-PATTERN CHECK: three transparent, language-independent patterns that need no training data, applied only to non-English
   reviews so the validated English behaviour does not change:
     * the same phrase repeated again and again (stuffing / copy-paste inside the review)
     * four or more different hype phrases stacked together
     * contact details or "message / contact karo" (an advert, not a review)
   Each gives a small capped nudge towards "suspicious". They are hints, not proof, and are shown to the user.

Neither step creates labels or fake training data. Accuracy on Hinglish can only be MEASURED with human-labelled Hinglish reviews
(see docs/REAL_DATA_TRAINING.md); until then the checker keeps saying "not very sure" for Hinglish.
"""
import math
import re

from multilingual import normalize_for_model

NON_ENGLISH = {"Hinglish", "Hindi", "Mixed"}
REPEAT_LOGIT_BOOST = 1.4
STACK_LOGIT_BOOST = 0.6
CONTACT_LOGIT_BOOST = 1.2
# a review that asks readers to contact someone off-platform is advertising, not reviewing (phone number, WhatsApp / Telegram, "message karo")
CONTACT = re.compile(r"(\+?\d[\d\s\-]{8,}\d)|\b(whatsapp|whats app|telegram|insta(gram)?\s*(dm|id)|dm\s*me)\b|\b(message|msg|contact|call|sampark|sandesh)\s+(karo|kare|kijiye|me|us|karein)\b", re.I)

# canonical (post-normalisation) Hinglish word -> English meaning(s)
WORDS = {
    "accha": "good", "badhiya": "good", "mast": "nice", "zabardast": "awesome", "shandar": "excellent", "lajawab": "superb",
    "behtareen": "excellent", "sahi": "right good", "kaafi": "quite", "bahut": "very", "bilkul": "totally", "bekar": "bad useless",
    "bakwas": "bad rubbish", "ghatiya": "bad poor", "kharab": "bad damaged", "faltu": "useless", "dhokha": "fraud cheat",
    "nakli": "fake duplicate", "asli": "original genuine", "original": "original", "pasand": "like", "paisa": "money",
    "vasool": "worth", "wapas": "return back", "return": "return", "refund": "refund", "seller": "seller", "delivery": "delivery",
    "quality": "quality", "product": "product", "size": "size", "colour": "colour", "rang": "colour", "kapda": "fabric cloth",
    "silai": "stitching", "packing": "packing", "damage": "damaged", "toot": "broke", "tuta": "broken", "toota": "broken",
    "chalta": "works", "chal": "works", "nahi": "not", "mat": "do not", "lena": "buy", "kharidna": "buy", "kharido": "buy",
    "sabko": "everyone", "dobara": "again", "behtar": "better", "sasta": "cheap", "mehenga": "expensive", "jaldi": "fast quick",
    "der": "late delay", "late": "late", "time": "time", "pehle": "before", "baad": "after", "din": "day", "mahine": "month",
    "hafte": "week", "garam": "hot", "thanda": "cold", "awaaz": "sound noise", "aawaz": "sound noise", "battery": "battery",
    "camera": "camera", "chhota": "small", "bada": "big", "patla": "thin", "mota": "thick", "halka": "light", "bhaari": "heavy",
    "sundar": "beautiful", "khoobsurat": "beautiful", "ek": "one", "number": "number", "recommend": "recommend",
    # Hindi (Devanagari)
    "अच्छा": "good", "बढ़िया": "good", "बहुत": "very", "खराब": "bad", "बेकार": "bad useless", "घटिया": "bad poor", "शानदार": "excellent",
    "जबरदस्त": "awesome", "लाजवाब": "superb", "पैसा": "money", "वसूल": "worth", "क्वालिटी": "quality", "प्रोडक्ट": "product",
    "डिलीवरी": "delivery", "नकली": "fake duplicate", "असली": "original genuine", "धोखा": "fraud cheat", "वापस": "return back",
    "सुंदर": "beautiful", "पसंद": "like", "कपड़ा": "fabric cloth", "रंग": "colour", "सिलाई": "stitching", "नहीं": "not",
    "खरीदें": "buy", "सबको": "everyone",
}
# multi-word phrases (canonical form) -> English
PHRASES = {
    "paisa vasool": "value for money", "पैसा वसूल": "value for money", "must buy": "must buy", "sabko lena chahiye": "everyone should buy",
    "bahut accha": "very good", "bahut badhiya": "very good", "bahut bekar": "very bad", "bilkul bekar": "totally useless",
    "time pe": "on time", "time par": "on time", "dhokha hai": "it is a fraud", "mat kharido": "do not buy", "mat lena": "do not buy",
    "paise barbad": "waste of money", "paisa barbad": "waste of money", "highly recommended": "highly recommended",
    "wapas kar": "returned", "return kar": "returned", "bahut sundar": "very beautiful", "बहुत अच्छा": "very good",
    "बहुत खराब": "very bad", "बिल्कुल बेकार": "totally useless",
}
# hype phrases (canonical form) used to detect stacking
HYPE = [
    "must buy", "paisa vasool", "पैसा वसूल", "best product", "highly recommended", "zabardast", "lajawab", "shandar", "behtareen", "superb",
    "awesome", "excellent", "amazing", "5 star", "five star", "no 1", "number 1", "buy now", "sabko lena chahiye", "sabko lena",
    "mast product", "ekdum mast", "value for money", "best quality", "perfect", "jabardast", "bahut badhiya", "ever",
    "जबरदस्त", "शानदार", "लाजवाब", "बेस्ट", "सबको लेना चाहिए",
]
_STOP = {"hai", "ho", "he", "ka", "ki", "ke", "ko", "se", "me", "mein", "aur", "yeh", "ye", "to", "bhi", "hi", "tha", "thi", "the",
         "का", "की", "के", "को", "से", "में", "और", "है", "यह", "भी", "ही", "!"}


def _tokens(norm):
    return norm.split()


def english_bridge(clean_text, language):
    """Extra English tokens for a normalised Hinglish/Hindi/mixed review ('' for English or unknown). Original text is untouched."""
    if language not in NON_ENGLISH or not clean_text:
        return "", []
    found, out, padded = [], [], " " + clean_text + " "
    for ph, en in PHRASES.items():
        if " " + ph + " " in padded:
            out.append(en)
            found.append(ph)
    for tok in _tokens(clean_text):
        en = WORDS.get(tok)
        if en and tok not in found:
            out.append(en)
            found.append(tok)
    return " ".join(out), found[:12]


def spam_patterns(raw_text, language):
    """Return dict(repeat, stack, boost_logit, details[]) for non-English reviews; zeros for English."""
    res = {"repeat": False, "stack": False, "contact": False, "boost_logit": 0.0, "details": []}
    if language not in NON_ENGLISH:
        return res
    norm = normalize_for_model(raw_text)
    toks = [t for t in _tokens(norm) if t != "!"]
    if len(toks) >= 6:
        grams = {}
        for n, need in ((3, 2), (2, 3)):
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                if all(w in _STOP for w in g):
                    continue
                grams[(n, g)] = grams.get((n, g), 0) + 1
        rep = [(g, c) for (n, g), c in grams.items() if c >= (2 if n == 3 else 3)]
        content = {}
        for t in toks:
            if t not in _STOP and len(t) > 2:
                content[t] = content.get(t, 0) + 1
        heavy = [(t, c) for t, c in content.items() if c >= 4]
        if rep or heavy:
            ex = " ".join(rep[0][0]) if rep else heavy[0][0]
            res["repeat"] = True
            res["details"].append(f"The same words or phrase are repeated again and again (for example '{ex}').")
    padded = " " + norm + " "
    hits = sorted({h for h in HYPE if " " + normalize_for_model(h) + " " in padded})
    if len(hits) >= 4:
        res["stack"] = True
        res["details"].append("Four or more different hype phrases are stacked together: " + ", ".join(hits[:4]) + ".")
    if CONTACT.search(str(raw_text)):
        res["contact"] = True
        res["details"].append("The review asks readers to contact someone outside the shopping site (phone number, WhatsApp, Telegram or 'message karo').")
    res["boost_logit"] = ((REPEAT_LOGIT_BOOST if res["repeat"] else 0.0) + (STACK_LOGIT_BOOST if res["stack"] else 0.0)
                          + (CONTACT_LOGIT_BOOST if res["contact"] else 0.0))
    return res


def adjust_probability(p_fake, patterns):
    """Capped nudge towards 'suspicious' only; never lowers the score."""
    if not patterns["boost_logit"]:
        return p_fake, False
    p = min(max(p_fake, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) + min(patterns["boost_logit"], 1.6)
    return 1 / (1 + math.exp(-z)), True
