# -*- coding: utf-8 -*-
"""Product context for review checking.

The person picks the product a review belongs to (e.g. "Saree", "Smartphone"). This module answers one question:
does the review text talk about that kind of product?

* A review of an air fryer posted under a saree listing is a recognised warning sign (copy-pasted or re-used reviews).
* A review with no product words at all ("good quality, fast delivery") is NEUTRAL: nothing is learned from it.
* It is a hint, not proof: people also mention accessories, gifts and comparisons. The effect on the score is small and capped.

The word lists are hand-written (English, Hinglish, Hindi). They are a starting point; extend them in LEXICON below.
"""
import math
import re

import config

NONE_LEVEL = "unknown"      # no product chosen / not recognised / too little text
MATCH, WEAK, MISMATCH = "match", "weak_match", "mismatch"
MISMATCH_LOGIT_BOOST = 0.8  # added to the model's log-odds of "fake" on a clear mismatch (about +15-20 points near 50%)
MIN_WORDS_FOR_CHECK = 4

# category -> (strong item words, weaker attribute words). Lower-case; Devanagari written as is.
LEXICON = {
    "Mobiles & Electronics": (
        "phone smartphone mobile iphone watch android earphone earphones earbuds earbud headphone headphones airpods bluetooth speaker smartwatch "
        "laptop notebook charger cable powerbank tablet tv television camera keyboard mouse router fitbit "
        "फोन मोबाइल स्मार्टफोन ईयरबड्स इयरफोन हेडफोन चार्जर लैपटॉप टीवी स्पीकर घड़ी".split(),
        "battery display screen camera ram processor sound bass connectivity pairing charging volume mic app lag hang hanging "
        "बैटरी डिस्प्ले स्क्रीन कैमरा साउंड चार्जिंग हैंग".split()),
    "Fashion - Men": (
        "tshirt t-shirt shirt jeans kurta jacket sneakers shoes shoe hoodie trouser trousers blazer formal sweatshirt watch "
        "शर्ट टीशर्ट जींस कुर्ता जैकेट जूते".split(),
        "fit fabric stitching size colour color cotton comfortable sole sleeve collar wash shrink fade "
        "फिटिंग कपड़ा सिलाई साइज़ साइज रंग कॉटन".split()),
    "Fashion - Women": (
        "saree sari kurti kurta dress top leggings lehenga dupatta salwar heels sandals handbag purse gown earrings jhumka anarkali palazzo "
        "साड़ी साड़ी कुर्ती ड्रेस टॉप लहंगा दुपट्टा सैंडल हैंडबैग झुमके".split(),
        "fit fabric stitching size colour color cotton comfortable border blouse sleeve lining wash shrink fade drape "
        "फिटिंग कपड़ा सिलाई साइज़ रंग ब्लाउज".split()),
    "Kids": (
        "toy toys doll kids baby diaper school schoolbag bag lunchbox stroller crib puzzle cartoon "
        "खिलौना खिलौने बच्चा बच्चे बच्चों डायपर".split(),
        "safe safety child children age plastic colourful battery sound "
        "सुरक्षित उम्र".split()),
    "Home & Kitchen": (
        "mixer grinder airfryer fryer cooker pressure cookware pan tawa kadhai bedsheet bedsheets pillow curtain bottle flask lunchbox "
        "organiser organizer container decor lamp kettle mug utensil "
        "मिक्सर ग्राइंडर कुकर कढ़ाई तवा चादर बेडशीट तकिया पर्दा बोतल कैसरोल".split(),
        "handle lid nonstick non-stick coating leak capacity steel stainless wash scratch cooking jar blade "
        "हैंडल ढक्कन स्टील कोटिंग लीक".split()),
    "Appliances": (
        "refrigerator fridge ac conditioner cooler fan geyser heater iron vacuum washing machine microwave oven purifier chimney inverter "
        "फ्रिज रेफ्रिजरेटर एसी कूलर पंखा गीजर प्रेस वॉशिंग माइक्रोवेव".split(),
        "cooling noise watt power consumption warranty installation service technician compressor motor speed "
        "कूलिंग आवाज वारंटी इंस्टॉलेशन".split()),
    "Beauty & Personal Care": (
        "cream moisturiser moisturizer lipstick makeup foundation shampoo conditioner oil perfume deodorant trimmer shaver serum sunscreen "
        "facewash face-wash kajal mascara lotion fragrance "
        "क्रीम लिपस्टिक शैम्पू तेल परफ्यूम ट्रिमर सीरम सनस्क्रीन फेसवॉश काजल".split(),
        "skin hair smell fragrance shade dry oily pimple glow irritation rash lasting texture scent "
        "त्वचा बाल खुशबू स्किन रिजल्ट".split()),
    "Grocery & Health": (
        "protein supplement vitamin whey oats biscuits snacks tea coffee masala atta rice dal oil ghee honey juice chocolate thermometer "
        "oximeter glucometer bp monitor "
        "प्रोटीन चाय कॉफी मसाला आटा चावल दाल घी शहद".split(),
        "taste flavour flavor expiry sealed fresh packaging quantity pack tasty sweet "
        "स्वाद एक्सपायरी ताज़ा".split()),
    "Books & Stationery": (
        "book books novel notebook pen pencil diary journal stationery marker paperback hardcover ebook "
        "किताब पुस्तक कॉपी पेन पेंसिल डायरी".split(),
        "pages paper print binding story author chapter ink "
        "पन्ने कागज़ कहानी लेखक".split()),
    "Sports & Fitness": (
        "yoga mat dumbbell dumbbells treadmill cycle bicycle bat ball racket cricket football gym resistance skipping "
        "योगा मैट डम्बल ट्रेडमिल साइकिल बैट बॉल".split(),
        "grip workout exercise training durable bounce weight "
        "वर्कआउट एक्सरसाइज".split()),
}

# Groups whose products are often mentioned together; a review from one is not called a mismatch for the other.
RELATED = [
    {"Mobiles & Electronics", "Appliances"},
    {"Home & Kitchen", "Appliances"},
    {"Fashion - Men", "Fashion - Women", "Kids"},
    {"Beauty & Personal Care", "Grocery & Health"},
    {"Kids", "Books & Stationery"},
    {"Kids", "Sports & Fitness"},
]

_LATIN_SPLIT = re.compile(r"[a-z0-9\-']+")
_DEVA = re.compile(r"[\u0900-\u097F]+")


def category_of(product):
    """Group name for a picked product (exact pick-list match), else None."""
    if not product:
        return None
    p = product.strip().lower()
    for cat, items in config.PRODUCTS.items():
        if p == cat.lower() or any(p == i.lower() for i in items):
            return None if cat == "Other" else cat
    return None


def infer_category(product_text):
    """Best-effort group for a free-text product name (e.g. a column in an uploaded file)."""
    cat = category_of(product_text)
    if cat:
        return cat
    words = _words(product_text or "")
    best, best_n = None, 0
    for c, (items, attrs) in LEXICON.items():
        n = len(words & set(items))
        if n > best_n:
            best, best_n = c, n
    return best


def _words(text):
    t = (text or "").lower().replace("’", "'")
    toks = set(_LATIN_SPLIT.findall(t)) | set(_DEVA.findall(t))
    toks |= {w.replace("-", "") for w in toks if "-" in w}
    return toks


def _hits(words, cat):
    items, attrs = LEXICON[cat]
    return sorted(words & set(items)), sorted(words & set(attrs))


def _related(a, b):
    return a == b or any(a in s and b in s for s in RELATED)


def product_fit(text, product):
    """Compare the review text with the chosen product.

    Returns a dict: product, category, level (match | weak_match | mismatch | unknown), matched_terms, other_category,
    other_terms, message. Never raises; unknown means "no information", which has no effect on the score.
    """
    out = {"product": product or None, "category": None, "level": NONE_LEVEL, "matched_terms": [], "other_category": None,
           "other_terms": [], "message": None}
    if not product:
        return out
    cat = category_of(product) or infer_category(product)
    out["category"] = cat
    if not cat:
        out["message"] = "Product not in the list, so the review could not be compared with it."
        return out
    words = _words(text)
    if len(words) < MIN_WORDS_FOR_CHECK:
        out["message"] = "The review is too short to compare with the product."
        return out
    own_items, own_attrs = _hits(words, cat)
    name_hits = sorted(w for w in _words(product) if len(w) > 3 and w in words)      # e.g. "saree" for the pick "Saree"
    if own_items or name_hits:
        out.update(level=MATCH, matched_terms=sorted(set(own_items + name_hits))[:5],
                   message=f"The review mentions things that fit a {product.lower()} ({', '.join(sorted(set(own_items + name_hits))[:3])}).")
        return out
    best, best_items, best_attrs = None, [], []
    for other in LEXICON:
        if _related(cat, other):
            continue
        it, at = _hits(words, other)
        if len(it) > len(best_items) or (len(it) == len(best_items) and len(at) > len(best_attrs)):
            best, best_items, best_attrs = other, it, at
    if best and best_items:
        out.update(level=MISMATCH, other_category=best, other_terms=best_items[:5],
                   message=f"The review talks about {', '.join(best_items[:3])} ({best}), but the product you chose is {product}.")
        return out
    if own_attrs:
        out.update(level=WEAK, matched_terms=own_attrs[:5], message="The review has general wording that could fit this product, but no clear product words.")
        return out
    out["message"] = "The review does not mention anything specific about the product, so this check says nothing either way."
    return out


def adjust_probability(p_fake, fit):
    """Small, capped nudge: only a clear mismatch raises suspicion. A match never proves a review is genuine."""
    if fit["level"] != MISMATCH:
        return p_fake, False
    p = min(max(p_fake, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) + MISMATCH_LOGIT_BOOST
    return 1 / (1 + math.exp(-z)), True


def model_tokens(product, text):
    """Extra tokens appended to the text of a PRODUCT-AWARE model, e.g. 'zzprod_saree zzcat_fashion_women zzfit_match'."""
    if not product:
        return ""
    cat = category_of(product) or infer_category(product)
    slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")  # noqa: E731
    toks = []
    if cat:
        toks.append("zzcat_" + slug(cat))
    toks.append("zzfit_" + product_fit(text, product)["level"].replace("_", ""))
    return " ".join(toks)
