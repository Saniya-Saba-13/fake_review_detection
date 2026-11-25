"""Small lexicon-based sentiment for English, Hindi and Hinglish reviews.

Sentiment is shown NEXT TO the fake/genuine prediction and is never used as evidence of fakeness: a genuine
review can be very negative and a fake one very positive. This is a transparent lexicon heuristic with simple
negation handling, NOT a trained sentiment model; it was not evaluated on labelled sentiment data.
"""
from multilingual import normalize_devanagari, normalize_for_model

_EN_POS = "good great excellent amazing awesome love loved perfect best fantastic superb nice happy satisfied recommend recommended worth comfortable durable sturdy smooth fast wonderful brilliant lovely beautiful useful helpful impressed pleased".split()
_EN_NEG = "bad worst poor terrible awful disappointed disappointing disappointment waste useless broken defective damaged cheap horrible rubbish pathetic faulty torn refund return slow late fake scam overpriced uncomfortable flimsy".split()
_HI_POS = "अच्छा अच्छी अच्छे बढ़िया शानदार बेहतरीन जबरदस्त ज़बरदस्त पसंद सही बेस्ट लाजवाब मस्त कमाल खुश संतुष्ट उम्दा सुंदर".split()
_HI_NEG = "खराब ख़राब बेकार घटिया बकवास बुरा बुरी निराश धोखा नकली टूट टूटा टूटी खराबी गंदा फालतू परेशान".split()
_RO_POS = "accha badhiya mast zabardast shandar lajawab behtareen sahi pasand khush umda kamaal sundar".split()
_RO_NEG = "bura buri kharab bekar bakwas ghatiya dhokha nakli faltu toota tooti kharabi nirash gandi gandaa pareshan".split()
_NEG_EN = {"not", "no", "never", "dont", "don't", "didn't", "doesn't", "isn't", "wasn't", "cannot", "can't", "hardly"}
_NEG_HI = {"नहीं", "न", "मत", "ना", "nahi", "na", "mat"}


def _norm_words(words):
    return {normalize_for_model(w) for w in words}


POS = _norm_words(_EN_POS + _HI_POS + _RO_POS)
NEG = _norm_words(_EN_NEG + _HI_NEG + _RO_NEG)
HINDI_STYLE = _norm_words(_HI_POS + _HI_NEG + _RO_POS + _RO_NEG)
NEGATORS_BEFORE = {normalize_for_model(w) for w in _NEG_EN | _NEG_HI}
NEGATORS_AFTER = {normalize_for_model(w) for w in _NEG_HI}


def analyze_sentiment(text) -> dict:
    """Return {'label': Positive|Negative|Neutral|Mixed, 'score': -1..1, 'positive_terms': [...], 'negative_terms': [...]}."""
    toks = normalize_for_model(text).split()
    pos, neg = [], []
    for i, tok in enumerate(toks):
        polarity = 1 if tok in POS else -1 if tok in NEG else 0
        if not polarity:
            continue
        before = toks[max(0, i - 2):i]
        after = toks[i + 1:i + 3] if tok in HINDI_STYLE else []
        if any(t in NEGATORS_BEFORE for t in before) or any(t in NEGATORS_AFTER for t in after):
            polarity = -polarity
        (pos if polarity > 0 else neg).append(tok)
    n = len(pos) + len(neg)
    score = (len(pos) - len(neg)) / n if n else 0.0
    if n == 0:
        label = "Neutral"
    elif pos and neg and abs(score) < 0.34:
        label = "Mixed"
    elif score > 0:
        label = "Positive"
    else:
        label = "Negative"
    return {"label": label, "score": round(score, 3), "positive_terms": sorted(set(pos))[:6],
            "negative_terms": sorted(set(neg))[:6], "method": "lexicon (not a trained model)"}
