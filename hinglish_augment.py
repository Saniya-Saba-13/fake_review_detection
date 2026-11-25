# -*- coding: utf-8 -*-
"""Hinglish training data made from English labelled reviews (code-mixing augmentation).

WHY: there is no verified fake-vs-genuine Hinglish dataset, so the model has seen only a handful of Hinglish reviews and could not
use Hinglish words such as "zabardast", "paisa vasool" or "bekar" as evidence. This module converts English training reviews into
Hinglish the way people write it (English product words kept, common opinion words, connectors and verbs replaced by romanised
Hindi, spelling varied), keeping the ORIGINAL LABEL of the review it came from.

WHAT IT IS NOT: it is not real Hinglish data and it creates no new knowledge about fakeness. The labels are exactly as good (or as weak)
as the English labels they came from. It only lets what the model learned from English reviews apply to Hinglish words.
Because of that:
  * it is applied to the TRAINING split only (never to validation / test), after the split, so a converted copy of a test review can
    never be in training;
  * the same number of genuine and fake reviews is converted, so "contains Hinglish" is never a clue for either class;
  * Hinglish accuracy is NOT measured on converted rows. It is measured on data/hinglish_eval.csv, hand-written separately.
"""
import random
import re

import hinglish_support
from multilingual import detect_language, normalize_for_model

# (regex on lower-case English, variants). Longest / most specific first.
PHRASES = [
    (r"value for (the )?money", ["paisa vasool", "paisa vasool", "paise ka poora value", "value for money"]),
    (r"worth (the )?(money|price|every penny|every rupee)", ["paisa vasool", "paise vasool", "poora paisa vasool"]),
    (r"waste of (money|time)", ["paisa barbad", "paise barbad", "paisa waste", "paisa barbaad"]),
    (r"highly recommended?", ["sabko recommend karunga", "highly recommended", "sabko lena chahiye", "sabko recommend karta hoon"]),
    (r"(i )?(would|will) (definitely )?recommend( it)?( to (everyone|all|others))?", ["sabko recommend karunga", "main sabko recommend karunga", "recommend karunga"]),
    (r"(do not|don't|dont|never) buy( this)?", ["mat kharido", "mat lena", "kharidna mat", "bilkul mat kharido"]),
    (r"must[- ]buy", ["zaroor kharido", "must buy", "sabko lena chahiye", "lena hi lena"]),
    (r"buy now|order now", ["abhi kharido", "abhi order karo", "buy now", "jaldi order karo"]),
    (r"love (it|this)", ["bahut pasand aaya", "mujhe bahut pasand aaya", "ekdum pasand aaya"]),
    (r"(i am|i'm) (very |really )?(happy|satisfied)( with (it|this|the product))?", ["main bahut khush hoon", "bahut satisfied hoon", "mujhe bahut pasand aaya"]),
    (r"(very |really |so )?disappointed", ["bahut nirash hoon", "bahut disappoint hua", "dil toot gaya"]),
    (r"stopped working", ["kaam karna band kar diya", "chalna band ho gaya", "kaam nahi kar raha"]),
    (r"(does not|doesn't|didn't|did not) work", ["kaam nahi karta", "chalta hi nahi", "kaam nahi kiya"]),
    (r"works? (very |really )?well", ["achhe se kaam karta hai", "badhiya chalta hai", "mast chalta hai"]),
    (r"(is |was )?(very |really )?(fast|quick) delivery|delivered (very )?(fast|quickly|on time)|fast delivery", ["delivery time par aayi", "delivery jaldi hui", "time pe delivery mili", "fast delivery"]),
    (r"(delivery )?(was |is )?(very )?(late|delayed)", ["delivery late aayi", "delivery mein der hui", "bahut der se aaya"]),
    (r"on time", ["time par", "time pe"]),
    (r"(as )?described|as shown( in the (picture|photo))?|exactly like the (picture|photo)", ["jaisa dikhaya waisa hi", "photo jaisa hi hai", "bilkul photo jaisa"]),
    (r"(good|great|nice|decent|fair) price", ["acha price", "sahi daam", "badhiya price"]),
    (r"money back|refund", ["refund", "paise wapas", "refund"]),
    (r"customer (care|service|support)", ["customer care", "customer care", "support"]),
    (r"(of )?(very )?good quality|great quality|excellent quality|superb quality", ["bahut badhiya quality", "zabardast quality", "quality mast hai", "shandar quality"]),
    (r"poor quality|bad quality|low quality|cheap quality", ["quality ghatiya", "quality bekar hai", "quality kharab hai", "ghatiya quality"]),
]

# word -> variants. Applied token by token (probability WORD_P) after the phrase and "X is ADJ" rules.
WORDS = {
    "good": ["accha", "acha", "achha"], "great": ["badhiya", "zabardast", "mast"], "nice": ["mast", "accha", "badhiya"],
    "excellent": ["shandar", "zabardast", "lajawab"], "amazing": ["kamaal ka", "zabardast", "lajawab"], "awesome": ["mast", "zabardast", "jabardast"],
    "superb": ["lajawab", "shandar"], "perfect": ["ekdum perfect", "bilkul sahi", "perfect"], "best": ["sabse accha", "best", "sabse badhiya"],
    "wonderful": ["shandar", "kamaal ka"], "fantastic": ["zabardast", "shandar"], "love": ["pasand", "pasand"], "loved": ["pasand aaya", "bahut pasand aaya"],
    "bad": ["kharab", "bekar", "ghatiya"], "poor": ["ghatiya", "kharab"], "worst": ["sabse bekar", "sabse ghatiya", "worst"], "terrible": ["bahut kharab", "bakwas"],
    "awful": ["bakwas", "bahut bekar"], "useless": ["bekar", "faltu", "bilkul bekar"], "cheap": ["sasta", "ghatiya"], "fake": ["nakli", "duplicate", "nakli"],
    "original": ["asli", "original"], "genuine": ["asli", "original"], "fraud": ["dhokha", "fraud"], "scam": ["dhokha", "scam"],
    "very": ["bahut", "kaafi", "bohot"], "really": ["sach mein", "ekdum", "bahut"], "so": ["bahut", "itna"], "totally": ["bilkul", "ekdum"], "completely": ["bilkul", "ekdum"],
    "quite": ["kaafi"], "too": ["bahut", "zyada"], "much": ["zyada"], "little": ["thoda"], "bit": ["thoda"], "slightly": ["thoda"], "average": ["average", "theek-thaak"],
    "okay": ["theek hai", "ok"], "ok": ["theek", "ok"], "fine": ["theek", "sahi"],
    "but": ["lekin", "par", "magar"], "and": ["aur", "aur"], "or": ["ya"], "because": ["kyunki"], "also": ["bhi"], "again": ["dobara", "phir se"],
    "not": ["nahi", "nahi"], "never": ["kabhi nahi"], "no": ["koi", "nahi"], "this": ["yeh", "ye"], "that": ["woh", "wo"], "it": ["ye", "isko"], "these": ["ye"],
    "i": ["main", "maine", "mai"], "my": ["mera", "meri", "mere"], "we": ["hum"], "you": ["aap"], "everyone": ["sabko", "sab log"], "all": ["sab"], "friends": ["dost"],
    "family": ["ghar walon"], "buy": ["kharid", "lena"], "bought": ["kharida", "liya"], "ordered": ["order kiya", "mangaya"], "received": ["mila", "aaya"],
    "recommend": ["recommend", "recommend"], "money": ["paisa", "paise"], "price": ["price", "daam"], "cost": ["daam"], "expensive": ["mehenga", "mehnga"], "cheaper": ["sasta"],
    "day": ["din"], "days": ["din"], "week": ["hafte"], "weeks": ["hafte"], "month": ["mahine"], "months": ["mahine"], "year": ["saal"], "hours": ["ghante"],
    "broke": ["toot gaya", "tut gaya"], "broken": ["toota hua", "tuta hua"], "damaged": ["damage", "kharab hua"], "returned": ["return kar diya", "wapas kar diya"],
    "return": ["return", "wapas"], "replace": ["replace", "badal"], "replacement": ["replacement"], "wrong": ["galat", "alag"], "different": ["alag"],
    "fast": ["jaldi", "fast"], "quick": ["jaldi", "fast"], "slow": ["slow", "dheere"], "late": ["late", "der se"], "time": ["time", "waqt"],
    "small": ["chhota", "chota"], "big": ["bada", "bada"], "large": ["bada"], "thin": ["patla"], "thick": ["mota"], "light": ["halka", "light"], "heavy": ["bhaari", "bhari"],
    "soft": ["naram", "soft"], "hard": ["sakht", "hard"], "beautiful": ["sundar", "khoobsurat"], "colour": ["rang", "colour"], "color": ["rang", "color"],
    "fabric": ["kapda", "kapda"], "cloth": ["kapda"], "stitching": ["silai", "stitching"], "sound": ["awaaz", "sound"], "noise": ["awaaz", "shor"], "loud": ["tez", "loud"],
    "hot": ["garam"], "warm": ["garam"], "cold": ["thanda"], "easy": ["aasan", "easy"], "difficult": ["mushkil"], "problem": ["dikkat", "problem"], "issue": ["dikkat", "issue"],
    "happy": ["khush", "happy"], "satisfied": ["santusht", "satisfied"], "disappointed": ["nirash", "disappointed"], "sad": ["dukhi"], "waste": ["barbad", "waste"],
    "seller": ["seller", "seller"], "delivery": ["delivery", "delivery"], "packing": ["packing", "packing"], "quality": ["quality", "quality"],
    "worth": ["vasool", "worth"], "only": ["sirf", "bas"], "still": ["abhi bhi"], "now": ["ab", "abhi"], "after": ["baad"], "before": ["pehle"], "with": ["ke saath"],
    "very good": ["bahut accha"], "thank": ["shukriya"], "thanks": ["dhanyawad", "thanks"], "please": ["please", "kripya"], "friend": ["dost", "yaar"],
}
WORD_P = 0.78
ADJECTIVES = ("good|great|nice|excellent|amazing|awesome|superb|perfect|wonderful|fantastic|bad|poor|worst|terrible|awful|useless|cheap|fake|original|genuine|"
              "fast|quick|slow|late|small|big|large|thin|thick|light|heavy|soft|hard|beautiful|loud|average|okay|fine|comfortable|durable|sturdy|smooth|"
              "expensive|cheaper|disappointing|damaged|broken|worth it|loose|tight|dull|bright|clear|noisy")
_IS_ADJ = re.compile(r"\b(\w+(?: \w+)?) (?:is|are|was|were|looks|feels|seems) ((?:very |really |so |quite |too |a bit |slightly )?)(" + ADJECTIVES + r")\b")
_WORD = re.compile(r"[A-Za-z']+|[^A-Za-z']+")
FILLERS_START = ["bhai ", "yaar ", "dosto ", ""]
FILLERS_END = [" hai", " hai bhai", " yaar", "", ""]


def _variant(rng, options):
    return options[rng.randrange(len(options))]


def to_hinglish(text, rng=None):
    """Convert one English review to Hinglish. Deterministic for a given `rng`."""
    rng = rng or random.Random(0)
    t = str(text)
    low = t.lower()
    for pat, variants in PHRASES:
        low = re.sub(pat, lambda m, v=variants: _variant(rng, v) if rng.random() < 0.9 else m.group(0), low)
    # "the product is very good" -> "the product very good hai": Hindi puts the verb last
    low = _IS_ADJ.sub(lambda m: f"{m.group(1)} {m.group(2)}{m.group(3)} hai", low)
    out = []
    for tok in _WORD.findall(low):
        key = tok.lower()
        if key in WORDS and rng.random() < WORD_P:
            out.append(_variant(rng, WORDS[key]))
        else:
            out.append(tok)
    s = "".join(out)
    s = re.sub(r"\b(is|are|was|were)\b", "hai", s) if rng.random() < 0.7 else s
    s = re.sub(r"\s+", " ", s).strip()
    s = _variant(rng, FILLERS_START) + s
    if s and s[-1] not in ".!?" and rng.random() < 0.5:
        s += _variant(rng, FILLERS_END)
    if rng.random() < 0.5:                                   # many people type without capitals
        return s
    return s[:1].upper() + s[1:]


def augment_training(df, per_class=2500, seed=11):
    """df: training frame with `model_text`, `label`, `language`, `platform`. Returns new rows (Hinglish versions of English rows),
    the same number per class, with `clean_text` built exactly as at prediction time (normalised + English bridge)."""
    import pandas as pd
    rng = random.Random(seed)
    src = df[df["language"] == "English"]
    rows = []
    for label in (0, 1):
        pool = src[src["label"] == label]
        pool = pool.sample(n=min(len(pool), per_class * 2), random_state=seed) if len(pool) else pool
        made = 0
        for r in pool.itertuples():
            h = to_hinglish(r.model_text, rng)
            lang = detect_language(h)["language"]
            if lang != "Hinglish":
                continue
            clean = normalize_for_model(h)
            clean = (clean + " " + hinglish_support.english_bridge(clean, "Hinglish")[0]).strip()
            rows.append({"model_text": h, "clean_text": clean, "label": label, "language": "Hinglish", "platform": r.platform, "label_method": "augmented_hinglish"})
            made += 1
            if made >= per_class:
                break
    n = min(sum(r["label"] == 0 for r in rows), sum(r["label"] == 1 for r in rows))     # equal per class
    out, c = [], {0: 0, 1: 0}
    for r in rows:
        if c[r["label"]] < n:
            out.append(r)
            c[r["label"]] += 1
    return pd.DataFrame(out)
