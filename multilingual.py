"""English / Hindi (Devanagari) / Hinglish text handling.

* `normalize_for_model` : cleaning that keeps Devanagari, unifies common Hindi spelling variants (nukta,
  chandrabindu, danda, Devanagari digits) and common Hinglish spelling variants (accha/acha/achha ...).
  Reviews are NOT translated to English; the classifier sees the original-language words plus character n-grams.
* `detect_language`      : script + romanised-Hindi lexicon heuristic (no external service, works offline).
  It is a heuristic: very short or code-mixed texts can be misjudged and the result carries a confidence level.
"""
import re
import unicodedata

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")
_NON_TEXT = re.compile(r"[^a-z0-9'!\u0900-\u097F\s]")
_STRAY_QUOTE = re.compile(r"(?<![a-z])'|'(?![a-z])")
_SPACES = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_LATIN_TOKEN = re.compile(r"[a-z']+")

TOKEN_PATTERN = r"[\u0900-\u097F]+|[a-z0-9']+|!"   # used by the word-level TF-IDF

_DEV_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}


def normalize_devanagari(text: str) -> str:
    """Unicode-normalise Devanagari: NFC, drop nukta / zero-width marks, chandrabindu -> anusvara, danda -> '.'."""
    t = unicodedata.normalize("NFC", text)
    t = t.translate(_DEV_DIGITS)
    t = t.replace("\u093c", "")            # nukta: ज़/ज, फ़/फ, ड़/ड ... treated as the same letter
    t = t.replace("\u200d", "").replace("\u200c", "")
    t = t.replace("\u0901", "\u0902")      # chandrabindu -> anusvara (हैं / हैँ)
    t = t.replace("\u0964", ".").replace("\u0965", ".")   # danda
    return t


# common Hinglish spelling variants -> one canonical form (the char n-grams handle the long tail)
_CANON_GROUPS = {
    "accha": "acha achha achcha aacha accha achi", "bahut": "bohot bahot bhut bohut bahoot bhot bahut",
    "nahi": "nahin nhi nai nahee nhin nahi", "hai": "hei he hain hai", "bekar": "bekaar bekaaar bakar bekar",
    "bakwas": "bakwaas bkwas bakwas", "kharab": "kharaab kharb kharab", "ghatiya": "ghatia ghatiyaa ghatiya",
    "zabardast": "jabardast jabardust zabardast zabrdast", "badhiya": "badiya badhia badhiyaa badhiya",
    "pasand": "pasnd psand pasand", "mast": "mst mast", "paisa": "paise paisa", "yeh": "ye yah yeh",
    "mujhe": "mjhe mujhko mujhe", "bilkul": "bilkl bilkul", "kaafi": "kafi kaafi", "sahi": "sahee sahi",
    "dhokha": "dhoka dhokhaa dhokha", "shandar": "shaandaar shandaar shandar", "lajawab": "lajawaab lajwab lajawab",
    "behtareen": "behtarin behtareen", "product": "prodct produkt prodcut product", "quality": "qualty quallity quality",
    "kya": "kyaa kya", "aur": "or_ aur", "lekin": "lekn lekin", "sabhi": "sab sabhi", "nakli": "nakli nakkli",
    "delivery": "delevery delivry delivery", "faltu": "phaltu faltu", "wapas": "vapas wapas",
}
HINGLISH_CANON = {v: k for k, vs in _CANON_GROUPS.items() for v in vs.split()}

# romanised-Hindi words that are (almost) never English; used only for language detection
HINGLISH_WORDS = set("""hai hain hoon tha thi nahi bahut accha bura buri kharab bekar bakwas ghatiya zabardast mast badhiya shandar
lajawab behtareen sahi galat paisa vasool kaafi bilkul ekdam sach sachmuch yeh woh mujhe mujhko mera meri mere hamara tumhara aapka
aap tum hum mein par pe se ko ka ki ke kya kyun kyu kaise kab kahan kaun koi kuch sabhi aur lekin magar toh bhi abhi phir fir pehle
baad jab jaise waise aisa aisi aise wala wali wale karna karo kiya karke gaya gayi gaye aaya aayi aaye mila mili mile liya diya
milta lena lene dena chahiye chahta pasand dhokha nakli faltu wapas thoda zyada jyada kam bada badi chota choti sasta mehnga
accha acha bohot bahot ek do teen nikla nikli tuta toota tooti kharabi dikhta dikhti lagta lagti lagi laga hua hui hue rahi raha
rahe sakta sakti sakte hota hoti hote wahi isliye kyunki kyuki dukan dukandar samaan saman
sabko sabse sab bhai yaar ekdum jaldi dobara kharid kharido kharidna kharidunga karunga karenge karta karti kiya lijiye zaroor pakka maza mazaa
kamaal khatam aaram dikkat mushkil sundar mazboot dheela dheeli dhila tez halka bhaari garam thanda mahine hafte ghante bheja bhej mana
vasool barbad barbaad pehenna sikudi nikalne nikalna lagne lagta aaya chal chalta chalti chalna raha rahi hoga hogi hogaya ho gaya
khush nirash dost koi kabhi sirf bas theek lagi lagta bilkul zyada thodi thoda kaisa waqt mangaya mila""".split()) - {"do", "par", "bas", "ho", "mat", "raha"}

_TOKEN_NORM = lambda tok: HINGLISH_CANON.get(tok, tok)  # noqa: E731


def normalize_for_model(text) -> str:
    """Lower-case, strip HTML/URLs/emoji, normalise Devanagari, squeeze repeats, canonicalise Hinglish variants."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = normalize_devanagari(str(text)).lower()
    t = _HTML.sub(" ", t)
    t = _URL.sub(" urltoken ", t)
    t = _REPEAT.sub(r"\1\1", t)
    t = t.replace("!", " ! ")
    t = _NON_TEXT.sub(" ", t)
    t = _STRAY_QUOTE.sub(" ", t)
    toks = [_TOKEN_NORM(tok) if tok.isascii() else tok for tok in _SPACES.sub(" ", t).strip().split(" ")]
    return " ".join(x for x in toks if x)


def detect_language(text) -> dict:
    """Return {'language': English|Hindi|Hinglish|Mixed|Unknown, 'confidence': high|medium|low, ...}."""
    s = normalize_devanagari(str(text or ""))
    dev = len(_DEVANAGARI.findall(s))
    lat = len(_LATIN_LETTER.findall(s))
    total = dev + lat
    if total < 2:
        return {"language": "Unknown", "confidence": "low", "devanagari_ratio": 0.0, "hindi_word_ratio": 0.0}
    dev_ratio = dev / total
    toks = _LATIN_TOKEN.findall(normalize_for_model(_DEVANAGARI.sub(" ", s)))
    hits = sum(1 for t in toks if t in HINGLISH_WORDS)
    h_ratio = hits / len(toks) if toks else 0.0
    base = {"devanagari_ratio": round(dev_ratio, 3), "hindi_word_ratio": round(h_ratio, 3)}
    if dev_ratio >= 0.6:
        return {"language": "Hindi", "confidence": "high" if total >= 6 else "medium", **base}
    if dev_ratio >= 0.1:
        return {"language": "Mixed", "confidence": "medium", **base}
    if (hits >= 2 and h_ratio >= 0.14) or (hits >= 4 and h_ratio >= 0.08) or (hits >= 1 and len(toks) <= 3 and h_ratio >= 0.33):
        return {"language": "Hinglish", "confidence": "high" if hits >= 3 or h_ratio >= 0.4 else "medium", **base}
    if len(toks) < 2:
        return {"language": "English", "confidence": "low", **base}
    return {"language": "English", "confidence": "high" if h_ratio == 0 and len(toks) >= 4 else "medium", **base}
