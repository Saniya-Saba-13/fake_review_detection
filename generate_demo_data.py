"""Generate a SYNTHETIC demonstration dataset (data/sample_reviews.csv).

IMPORTANT: these reviews are produced from hand-written templates. The
'fake'/'genuine' labels describe which template family produced the text. They
are NOT real-world ground truth. Metrics measured on this file only show that
the pipeline runs end to end; they say nothing about accuracy on real reviews.
Replace it with a real labelled dataset (see README) for meaningful results.

Usage:  python generate_demo_data.py [--rows 1000] [--seed 7]
"""
import argparse
import random

import pandas as pd

import config

PRODUCTS = ["headphones", "phone case", "blender", "backpack", "water bottle", "running shoes",
            "laptop stand", "desk lamp", "air fryer", "fitness band", "electric kettle", "wireless mouse",
            "power bank", "office chair", "yoga mat", "bluetooth speaker", "face wash", "coffee maker",
            "monitor", "mechanical keyboard"]
TIMES = ["two weeks", "a month", "three weeks", "about two months", "six weeks", "a few days", "three months"]

GENUINE_POS_DETAILS = [
    "The build quality feels solid for the price.",
    "Setup took about ten minutes and the instructions were clear.",
    "Battery easily lasts a full working day.",
    "It fits well and the material stays comfortable after hours of use.",
    "Packaging was neat and it arrived a day earlier than promised.",
    "Compared with my old one, it is quieter and noticeably faster.",
    "Customer support answered my warranty question within a day.",
    "I use it every morning and it still looks new.",
    "The size is accurate and matches the photos.",
    "It was easy to clean and the parts are dishwasher safe.",
]
GENUINE_CONS = [
    "My only complaint is that the cable is a bit short.",
    "It is slightly heavier than I expected.",
    "The manual is poorly translated, but you do not really need it.",
    "It gets a little noisy at the highest setting.",
    "The strap feels a little cheap.",
    "The color is slightly different from the pictures.",
    "",
    "",
]
GENUINE_CLOSERS = ["Would buy again.", "Good value overall.", "I would recommend it to a friend.",
                   "Happy with the purchase.", "Fine for the price.", "", ""]
GENUINE_NEG_ISSUES = [
    "it stopped working after {t}",
    "the stitching came apart after {t}",
    "the size runs much smaller than listed",
    "it overheats during longer use",
    "the app keeps disconnecting",
    "the lid does not close properly",
    "the finish started peeling after {t}",
]
GENUINE_NEG_DETAILS = [
    "I contacted the seller and they offered a partial refund.",
    "I tried resetting it twice with no change.",
    "Returning it was easy, at least.",
    "The photos made it look much sturdier than it is.",
    "For this price I expected better materials.",
]
GENUINE_SHORT = ["Works well, does what it says.", "Decent {p} for the money.", "Good {p}, arrived on time.",
                 "Great {p}, love it. Works well.", "Nice {p}, nothing special but reliable.",
                 "Okay {p}. Gets the job done."]

FAKE_OPENERS = ["Best {p} ever", "Amazing {p}", "Perfect {p}", "Excellent product", "Wow just wow",
                "Great {p}, love it", "Superb quality", "Awesome {p}"]
FAKE_MIDS = ["highly recommended to everyone", "100% satisfied with this purchase", "value for money, you will not regret",
             "totally worth every penny", "this changed my life", "must buy for everyone",
             "five stars, will buy again", "the seller is very trustworthy", "great product great price great seller",
             "amazing quality amazing price"]
FAKE_TAILS = ["Buy now before the offer ends", "Do not think twice, just order", "Five stars",
              "Best purchase this year", "Thanks seller", "Everyone should get one", "Highly recommend"]
FAKE_NEG = ["Worst {p} ever, do not buy, total waste of money. Buy the other brand instead, much cheaper and better.",
            "Terrible {p}, scam product and fake seller. Avoid at all costs, go to the competitor.",
            "Horrible, do not buy this {p}. Cheap brand X is far better, check my profile for the link."]


def _punct(rng):
    return rng.choice(["!", "!!", "!!!", ".", "."])


def genuine_review(rng):
    p = rng.choice(PRODUCTS)
    kind = rng.random()
    if kind < 0.15:
        return rng.choice(GENUINE_SHORT).format(p=p), rng.choice([3, 4, 4, 5])
    if kind < 0.70:
        parts = [f"I bought this {p} {rng.choice(TIMES)} ago and it has held up well.",
                 rng.choice(GENUINE_POS_DETAILS), rng.choice(GENUINE_POS_DETAILS),
                 rng.choice(GENUINE_CONS), rng.choice(GENUINE_CLOSERS)]
        return " ".join(x for x in parts if x), rng.choice([4, 4, 5, 5, 3])
    issue = rng.choice(GENUINE_NEG_ISSUES).format(t=rng.choice(TIMES))
    parts = [f"I wanted to like this {p} but {issue}.", rng.choice(GENUINE_NEG_DETAILS),
             rng.choice(GENUINE_POS_DETAILS) if rng.random() < 0.4 else ""]
    return " ".join(x for x in parts if x), rng.choice([1, 2, 2, 3])


def fake_review(rng):
    p = rng.choice(PRODUCTS)
    kind = rng.random()
    if kind < 0.12:
        return rng.choice(FAKE_NEG).format(p=p), rng.choice([1, 1, 2])
    if kind < 0.40:  # promotional text wrapped around a realistic-sounding detail (hard cases)
        s = (f"{rng.choice(GENUINE_POS_DETAILS)} {rng.choice(FAKE_OPENERS).format(p=p)}{_punct(rng)} "
             f"{rng.choice(FAKE_MIDS).capitalize()}{_punct(rng)}")
        return s, 5
    s = (f"{rng.choice(FAKE_OPENERS).format(p=p)}{_punct(rng)} {rng.choice(FAKE_MIDS).capitalize()}{_punct(rng)} "
         f"{rng.choice(FAKE_TAILS)}{_punct(rng)}")
    return s, rng.choice([5, 5, 5, 4])


def generate(rows=1000, seed=7, fake_share=0.30, label_noise=0.04):
    rng = random.Random(seed)
    records, seen = [], set()
    n_fake = int(rows * fake_share)
    targets = [1] * n_fake + [0] * (rows - n_fake)
    attempts = 0
    for label in targets:
        while True:
            attempts += 1
            text, rating = fake_review(rng) if label == 1 else genuine_review(rng)
            if text not in seen or attempts > rows * 50:
                break
        seen.add(text)
        records.append({"review_text": text, "rating": rating, "label": label})
    # a little label noise: real-world labels are never perfect
    for r in records:
        if rng.random() < label_noise:
            r["label"] = 1 - r["label"]
    rng.shuffle(records)
    df = pd.DataFrame(records)
    df["label"] = df["label"].map({0: "genuine", 1: "fake"})
    df["source"] = "synthetic_demo"
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(config.SAMPLE_DATA_PATH))
    args = ap.parse_args()
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {args.out}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
