"""Generate SYNTHETIC multilingual demonstration review files (data/demo_sources/<platform>_demo_reviews.csv).

*** NOT REAL DATA. ***  Every review is assembled from hand-written templates in English, Hindi (Devanagari) and Hinglish.
The label ('genuine' / 'fake') only says which template family produced the text (label_method = synthetic_template).
Platform names are only metadata here: they do NOT come from real Meesho / Flipkart / Myntra / Amazon reviews.
Use these files to test the pipeline end-to-end. Metrics measured on them say nothing about real-world accuracy.
Replace them with verified, labelled data (see README) for any real evaluation.

Usage:  python generate_multilingual_demo.py [--genuine 80] [--fake 40] [--seed 11]
"""
import argparse
import random
import re
from datetime import date, timedelta

import pandas as pd

import config
from multilingual import _CANON_GROUPS

ITEMS = {
    "Meesho": {"en": ["kurti", "saree", "bedsheet", "kitchen organiser", "phone cover", "earrings"],
               "hi": ["कुर्ती", "साड़ी", "बेडशीट", "किचन ऑर्गनाइज़र", "फोन कवर", "ईयररिंग्स"],
               "hg": ["kurti", "saree", "bedsheet", "kitchen organiser", "phone cover", "earrings"]},
    "Flipkart": {"en": ["mobile phone", "earbuds", "mixer grinder", "laptop bag", "smart watch", "trimmer"],
                 "hi": ["मोबाइल फोन", "ईयरबड्स", "मिक्सर ग्राइंडर", "लैपटॉप बैग", "स्मार्ट वॉच", "ट्रिमर"],
                 "hg": ["mobile phone", "earbuds", "mixer grinder", "laptop bag", "smart watch", "trimmer"]},
    "Myntra": {"en": ["t-shirt", "jeans", "sneakers", "kurta set", "hoodie", "handbag"],
               "hi": ["टी-शर्ट", "जींस", "स्नीकर्स", "कुर्ता सेट", "हुडी", "हैंडबैग"],
               "hg": ["t-shirt", "jeans", "sneakers", "kurta set", "hoodie", "handbag"]},
    "Amazon": {"en": ["water bottle", "backpack", "power bank", "air fryer", "keyboard", "electric kettle"],
               "hi": ["पानी की बोतल", "बैकपैक", "पावर बैंक", "एयर फ्रायर", "कीबोर्ड", "इलेक्ट्रिक केतली"],
               "hg": ["water bottle", "backpack", "power bank", "air fryer", "keyboard", "electric kettle"]},
}
T = {"en": ["two weeks", "a month", "three weeks", "about two months", "a few days"],
     "hi": ["दो हफ्ते", "एक महीने", "तीन हफ्ते", "करीब दो महीने", "कुछ दिन"],
     "hg": ["do hafte", "ek mahine", "teen hafte", "kareeb do mahine", "kuch din"]}

# ---- genuine building blocks -------------------------------------------------
G_POS = {
    "en": ["The build quality feels solid for the price.", "Delivery came a day early and the packaging was neat.",
           "The size is accurate and matches the photos.", "It is easy to use and I use it every day.",
           "The colour is the same as in the pictures.", "Compared with my old one it is noticeably better.",
           "The seller answered my question within a day.", "The finishing is good."],
    "hi": ["क्वालिटी कीमत के हिसाब से अच्छी है।", "डिलीवरी समय से पहले आ गई और पैकिंग अच्छी थी।", "साइज बिल्कुल सही आया।",
           "इस्तेमाल में आसान है और रोज़ काम आ रहा है।", "रंग फोटो जैसा ही है।", "पुराने वाले से काफी बेहतर है।",
           "सेलर ने मेरे सवाल का जल्दी जवाब दिया।", "फिनिशिंग अच्छी है।"],
    "hg": ["Quality price ke hisaab se accha hai.", "Delivery time se pehle aa gayi aur packing bhi sahi thi.", "Size bilkul sahi aaya.",
           "Use karna aasan hai aur roz kaam aa raha hai.", "Colour photo jaisa hi hai.", "Purane wale se kaafi behtar hai.",
           "Seller ne mere sawal ka jaldi jawab diya.", "Finishing achhi hai."]}
G_CON = {
    "en": ["My only complaint is that the cable is a bit short.", "It is slightly heavier than I expected.",
           "The colour is slightly different from the photos.", "", ""],
    "hi": ["बस इतनी सी शिकायत है कि तार थोड़ा छोटा है।", "थोड़ा भारी लगता है।", "रंग फोटो से हल्का अलग है।", "", ""],
    "hg": ["Bas itni shikayat hai ki wire thoda chhota hai.", "Thoda bhaari lagta hai.", "Colour photo se halka alag hai.", "", ""]}
G_CLOSE = {
    "en": ["Would buy again.", "Good value overall.", "I would recommend it to a friend.", "", ""],
    "hi": ["पैसे के हिसाब से ठीक है।", "दोबारा खरीदूंगा।", "दोस्त को भी बताऊंगा।", "", ""],
    "hg": ["Paise ke hisaab se theek hai.", "Dobara kharidunga.", "Dost ko bhi bataunga.", "", ""]}
G_OPEN = {"en": "I bought this {item} {t} ago and it has held up well.", "hi": "मैंने यह {item} {t} पहले लिया था और अब तक ठीक चल रहा है।",
          "hg": "Maine yeh {item} {t} pehle liya tha aur ab tak achhe se chal raha hai."}
G_NEG_OPEN = {"en": "I wanted to like this {item} but {issue}.", "hi": "मुझे यह {item} पसंद आने की उम्मीद थी लेकिन {issue}।",
              "hg": "Mujhe yeh {item} pasand aane ki umeed thi lekin {issue}."}
G_ISSUES = {
    "en": ["it stopped working after {t}", "the stitching came apart", "the size runs much smaller than listed", "it overheats during longer use",
           "the lid does not close properly", "the finish started peeling"],
    "hi": ["{t} में ही खराब हो गया", "सिलाई खुल गई", "साइज लिखे से बहुत छोटा है", "चलाने पर बहुत गर्म हो जाता है",
           "ढक्कन ठीक से बंद नहीं होता", "रंग उड़ने लगा"],
    "hg": ["{t} mein hi kharab ho gaya", "silai khul gayi", "size likhe se bahut chhota hai", "chalane par bahut garam ho jata hai",
           "dhakkan theek se band nahi hota", "colour udne laga"]}
G_NEG_DET = {
    "en": ["I contacted the seller and they offered a partial refund.", "I tried resetting it twice with no change.",
           "Returning it was easy, at least.", "It looked sturdier in the photos.", "For this price I expected better materials."],
    "hi": ["सेलर से बात की तो आधा पैसा वापस करने को कहा।", "दो बार रीसेट किया पर कोई फर्क नहीं पड़ा।", "वापसी की प्रक्रिया आसान थी।",
           "फोटो में जितना मजबूत दिख रहा था उतना है नहीं।", "इस कीमत में बेहतर मटीरियल की उम्मीद थी।"],
    "hg": ["Seller se baat ki to aadha paisa wapas karne ko bola.", "Do baar reset kiya par koi farak nahi pada.", "Return process aasan tha, at least.",
           "Photo mein jitna mazboot dikh raha tha utna hai nahi.", "Is price mein behtar material ki umeed thi."]}
G_SHORT = {
    "en": ["Decent {item} for the price.", "Works well, does what it says.", "Good {item}, arrived on time.", "Okay {item}. Gets the job done."],
    "hi": ["ठीक-ठाक {item}, कीमत के हिसाब से अच्छा है।", "काम चलाऊ है।", "{item} समय पर आ गया, ठीक है।", "अच्छा {item}, काम करता है।"],
    "hg": ["Theek-thaak {item}, price ke hisaab se accha hai.", "Kaam chalau hai.", "{item} time par aa gaya, theek hai.", "Accha {item}, kaam karta hai."]}
G_SIMPLE_POS = {
    "en": ["The {item} is good and the quality is nice.", "Packing was good and delivery was on time.", "I like this {item}. It works fine.",
           "Good quality for the price, I am happy with it.", "The fabric feels nice and the size is correct.", "This {item} is really good, I am satisfied.", "The quality is excellent and I am pleased with it."],
    "hi": ["इस {item} की क्वालिटी बहुत बढ़िया है, मैं संतुष्ट हूं।", "प्रोडक्ट अच्छा है और क्वालिटी भी ठीक है।", "पैकिंग अच्छी थी और डिलीवरी समय पर हुई।", "यह ठीक लगा, काम कर रहा है।",
           "कीमत के हिसाब से क्वालिटी अच्छी है, मैं खुश हूं।", "कपड़ा अच्छा लगा और साइज सही है।"],
    "hg": ["Is {item} ki quality bahut badhiya hai, main santusht hoon.", "Product accha hai aur quality bhi theek hai.", "Packing achhi thi aur delivery time par hui.", "Yeh theek laga, kaam kar raha hai.",
           "Price ke hisaab se quality achhi hai, main khush hoon.", "Kapda accha laga aur size sahi hai."]}
G_SIMPLE_NEG = {
    "en": ["The {item} is bad and I am disappointed.", "Poor quality, not worth the price.", "Delivery was late and the packing was damaged.",
           "Size is wrong and the material feels cheap.", "It did not work properly, very disappointed.",
           "Very poor {item}, I am not satisfied.", "Quality is really bad, totally disappointed."],
    "hi": ["क्वालिटी खराब है, मैं निराश हूं।", "डिलीवरी देर से हुई और पैकिंग टूटी हुई थी।", "साइज गलत है और कपड़ा सस्ता लगता है।",
           "ठीक से काम नहीं कर रहा, बहुत निराशा हुई।", "कीमत के हिसाब से बहुत खराब प्रोडक्ट है।", "बहुत खराब {item}, मैं संतुष्ट नहीं हूं।", "क्वालिटी सच में खराब है, पूरी तरह निराश।"],
    "hg": ["Quality kharab hai, main nirash hoon.", "Delivery late hui aur packing tuti hui thi.", "Size galat hai aur kapda sasta lagta hai.",
           "Theek se kaam nahi kar raha, bahut nirasha hui.", "Price ke hisaab se bahut kharab product hai."]}
# short, bland, positive fake reviews: in real life many fake reviews look like this, so they overlap with short genuine ones
F_GENERIC = {"en": ["Good product.", "Nice quality, recommended.", "Very good product, nice.", "Value for money.", "Nice product good quality."],
             "hi": ["अच्छा प्रोडक्ट है।", "अच्छी क्वालिटी, लेने लायक।", "बहुत अच्छा प्रोडक्ट।", "पैसे के हिसाब से सही।", "अच्छा प्रोडक्ट अच्छी क्वालिटी।"],
             "hg": ["Accha product hai.", "Achhi quality, lene layak.", "Bahut accha product.", "Paise ke hisaab se sahi.", "Accha product achhi quality."]}
G_TITLES_POS = {"en": ["Good for the price", "Works well", "Solid buy"], "hi": ["कीमत के हिसाब से अच्छा", "ठीक चल रहा है", "अच्छी खरीद"],
                "hg": ["Price ke hisaab se accha", "Theek chal raha hai", "Achhi kharidari"]}
G_TITLES_NEG = {"en": ["Not as expected", "Stopped working", "Disappointed"], "hi": ["उम्मीद से कम", "खराब हो गया", "निराश"],
                "hg": ["Umeed se kam", "Kharab ho gaya", "Nirash"]}

# ---- fake building blocks ------------------------------------------------------
F_OPEN = {"en": ["Best {item} ever", "Amazing {item}", "Perfect {item}", "Excellent product", "Wow just wow", "Superb quality"],
          "hi": ["सबसे बढ़िया {item}", "बहुत ही शानदार {item}", "एकदम परफेक्ट प्रोडक्ट", "वाह क्या बात है", "जबरदस्त क्वालिटी", "लाजवाब {item}"],
          "hg": ["Best {item} ever", "Bahut hi shandar {item}", "Ekdam perfect product", "Wah kya baat hai", "Zabardast quality", "Lajawab {item}"]}
F_MID = {"en": ["highly recommended to everyone", "100% satisfied with this purchase", "totally worth every penny", "this changed my life",
                "must buy for everyone", "five stars, will buy again", "the seller is very trustworthy", "great product great price great seller"],
         "hi": ["सबको खरीदना चाहिए", "100% संतुष्ट हूं", "पैसा वसूल, बिल्कुल पछताएंगे नहीं", "हर पैसे के लायक है", "मेरी ज़िंदगी बदल दी",
                "पांच स्टार, दोबारा खरीदूंगा", "सेलर बहुत भरोसेमंद है", "बढ़िया प्रोडक्ट बढ़िया कीमत बढ़िया सेलर"],
         "hg": ["Sabko kharidna chahiye", "100% satisfied hoon", "Paisa vasool, bilkul pachtayenge nahi", "Har paise ke layak hai", "Meri zindagi badal di",
                "Five star, dobara kharidunga", "Seller bahut bharosemand hai", "Accha product accha price accha seller"]}
F_TAIL = {"en": ["Buy now before the offer ends", "Do not think twice, just order", "Five stars", "Best purchase this year", "Thanks seller", "Everyone should get one"],
          "hi": ["ऑफर खत्म होने से पहले अभी ऑर्डर करें", "ज्यादा मत सोचिए बस खरीद लीजिए", "पांच स्टार", "इस साल की सबसे अच्छी खरीदारी", "सेलर को धन्यवाद", "हर किसी को लेना चाहिए"],
          "hg": ["Offer khatam hone se pehle abhi order karein", "Zyada mat sochiye bas kharid lijiye", "Five star", "Is saal ki sabse achhi kharidari", "Thanks seller", "Har kisi ko lena chahiye"]}
F_NEG = {"en": ["Worst {item} ever, do not buy, total waste of money. Buy the other brand instead, much cheaper and better.",
                "Terrible {item}, scam product and fake seller. Avoid at all costs, go to the competitor."],
         "hi": ["सबसे घटिया {item}, बिल्कुल मत खरीदो, पैसे की बर्बादी। दूसरा ब्रांड खरीदो, सस्ता और बेहतर है।",
                "बेकार {item}, नकली सेलर और धोखा। दूसरी दुकान से लें।"],
         "hg": ["Sabse ghatiya {item}, bilkul mat kharido, paise ki barbaadi. Dusra brand lo, sasta aur behtar hai.",
                "Bekar {item}, nakli seller aur dhokha. Dusri dukaan se lo."]}
F_TITLES = {"en": ["Best product!!!", "Must buy", "Five stars"], "hi": ["सबसे बढ़िया!!!", "जरूर खरीदें", "पांच स्टार"], "hg": ["Best product!!!", "Must buy", "Five star"]}

GENERIC_NOUN = {"en": "product", "hi": "प्रोडक्ट", "hg": "product"}


def pick_item(platform, lang, rng):
    """A platform-typical item, or (40% of the time) just the generic word 'product', so the word itself is not a class cue."""
    return GENERIC_NOUN[lang] if rng.random() < 0.4 else rng.choice(ITEMS[platform][lang])


_VARIANTS = {k: v.split() for k, v in _CANON_GROUPS.items()}


def hinglish_spelling(text, rng):
    """Randomly replace canonical Hinglish words by spelling variants (accha -> acha / achha ...)."""
    def swap(m):
        w = m.group(0)
        group = _VARIANTS.get(w.lower())
        if group and rng.random() < 0.45:
            v = rng.choice(group)
            return v.capitalize() if w[0].isupper() else v
        return w
    return re.sub(r"[A-Za-z]+", swap, text)


def _sep(lang):
    return " "


def genuine(lang, platform, rng):
    item, t = pick_item(platform, lang, rng), rng.choice(T[lang])
    kind = rng.random()
    if kind < 0.10:
        return rng.choice(G_SHORT[lang]).format(item=item), "", rng.choice([3, 4, 4, 5])
    if kind < 0.30:
        return rng.choice(G_SIMPLE_POS[lang]).format(item=item), "", rng.choice([3, 4, 4, 5])
    if kind < 0.42:
        return rng.choice(G_SIMPLE_NEG[lang]).format(item=item), "", rng.choice([1, 1, 2, 2])
    if kind < 0.78:
        parts = [G_OPEN[lang].format(item=item, t=t), rng.choice(G_POS[lang]), rng.choice(G_POS[lang]), rng.choice(G_CON[lang]), rng.choice(G_CLOSE[lang])]
        return " ".join(p for p in parts if p), rng.choice(G_TITLES_POS[lang]) if rng.random() < 0.5 else "", rng.choice([4, 4, 5, 5, 3])
    issue = rng.choice(G_ISSUES[lang]).format(t=t)
    parts = [G_NEG_OPEN[lang].format(item=item, issue=issue), rng.choice(G_NEG_DET[lang]), rng.choice(G_POS[lang]) if rng.random() < 0.4 else ""]
    return " ".join(p for p in parts if p), rng.choice(G_TITLES_NEG[lang]) if rng.random() < 0.5 else "", rng.choice([1, 2, 2, 3])


def fake(lang, platform, rng):
    item = pick_item(platform, lang, rng)
    punct = lambda: rng.choice(["!", "!!", "!!!", "."])  # noqa: E731
    kind = rng.random()
    if kind < 0.10:
        return rng.choice(F_NEG[lang]).format(item=item), "", rng.choice([1, 1, 2])
    if kind < 0.30:
        return rng.choice(F_GENERIC[lang]), "", rng.choice([5, 5, 4])
    if kind < 0.60:  # promotional wording wrapped around a realistic-sounding detail (harder cases)
        text = f"{rng.choice(G_POS[lang])} {rng.choice(F_OPEN[lang]).format(item=item)}{punct()} {rng.choice(F_MID[lang])}{punct()}"
        return text, "", 5
    text = f"{rng.choice(F_OPEN[lang]).format(item=item)}{punct()} {rng.choice(F_MID[lang])}{punct()} {rng.choice(F_TAIL[lang])}{punct()}"
    return text, rng.choice(F_TITLES[lang]) if rng.random() < 0.5 else "", rng.choice([5, 5, 5, 4])


def generate(n_genuine=80, n_fake=40, seed=11, label_noise=0.03):
    rng = random.Random(seed)
    today = date.today()
    frames = {}
    for platform in config.PLATFORMS[:4]:
        rows, seen = [], set()
        for lang, name in (("en", "English"), ("hi", "Hindi"), ("hg", "Hinglish")):
            for label, n, fn in ((0, n_genuine, genuine), (1, n_fake, fake)):
                made, attempts = 0, 0
                while made < n and attempts < n * 60:
                    attempts += 1
                    text, title, rating = fn(lang, platform, rng)
                    if lang == "hg":
                        text, title = hinglish_spelling(text, rng), hinglish_spelling(title, rng)
                    if text in seen:
                        continue
                    seen.add(text)
                    made += 1
                    rows.append({"platform": platform, "review_title": title, "review_text": text, "rating": rating,
                                 "label": "fake" if label else "genuine", "label_method": config.DEMO_LABEL_METHOD,
                                 "collection_date": (today - timedelta(days=rng.randint(0, 90))).isoformat()})
        for r in rows:  # a little label noise: real labels are never perfect
            if rng.random() < label_noise:
                r["label"] = "genuine" if r["label"] == "fake" else "fake"
        rng.shuffle(rows)
        frames[platform] = pd.DataFrame(rows)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genuine", type=int, default=80, help="genuine reviews per platform and language")
    ap.add_argument("--fake", type=int, default=40, help="fake reviews per platform and language")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(config.DEMO_SOURCES_DIR))
    args = ap.parse_args()
    out = config.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for platform, df in generate(args.genuine, args.fake, args.seed).items():
        path = out / f"{platform.lower()}_demo_reviews.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        total += len(df)
        print(f"  wrote {len(df):4d} SYNTHETIC rows -> {path}")
    print(f"Total {total} synthetic rows. This is demonstration data, not real platform reviews.")


if __name__ == "__main__":
    main()
