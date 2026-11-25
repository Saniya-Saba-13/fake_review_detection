# ReviewGuard AI: Multilingual Fake Product Review Detection (upgraded)

Upgrade of the original *AI-Powered Fake Product Review Detection System*. The existing routes, single/bulk analysis, explanations, history,
performance page, tests and the `python train_model.py` / `python app.py` workflow are kept; the interface, language support, data pipeline,
database and training were extended.

> **Predictions are estimates, not proof of fraud.** Writing in Hindi or Hinglish, or writing a negative review, is never treated as evidence
> that a review is fake. Language and sentiment are shown next to the verdict but are not model inputs.

> **Read this first: the bundled training data is SYNTHETIC.** I could not download real labelled reviews in the build environment, and I know of
> no verified fake-vs-genuine dataset for Hindi / Hinglish or for Meesho / Flipkart / Myntra reviews. So `build_dataset.py --demo` generates
> *template-based* reviews in English, Hindi and Hinglish, tagged with the four platform names, so the whole pipeline can run end to end.
> **Every score measured on them is a real measurement of a model trained on fake data and says nothing about real-world accuracy.**
> The app shows "Demo model" banners until you train on verified, labelled data that you provide.

## What is new

| Area | Change |
|---|---|
| **UI** | New e-commerce style dashboard (plum / violet / orange, rounded cards, hover effects, responsive). Pages: **Home Dashboard**, **Review Analysis**, **Batch Analysis**, **Prediction Results**, **Model Performance**, **Dataset Information**, **About**. |
| **Languages** | English, Hindi (Devanagari) and Hinglish. No translation: Devanagari normalisation, Hinglish spelling-variant normalisation, TF-IDF on words **and** character n-grams, script/lexicon language detection. |
| **Models** | Compared on a validation split: Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest (word + character TF-IDF). Optional `train_transformer.py` for XLM-RoBERTa / mBERT / IndicBERT (**not run here**, see below). |
| **Evaluation** | Overall, **per language** and **per platform** (precision, recall, F1, false-positive rate, ROC-AUC, confusion matrix) plus a check that negative *genuine* reviews are not flagged more often. |
| **Data** | `ingest.py` (CSV/XLSX, validation, de-duplication, label method required), `build_dataset.py` (unified dataset, per-language balancing); source platform and collection date stored per review. |
| **Database** | SQLite (`data/reviewguard.db`): analysed reviews, batches, labelled training data, ingestions, model versions. |
| **Retraining** | From the Dataset Information page or `python train_model.py`; new versioned model, activated automatically, no server restart. |

## v23: Hinglish now works

| Change | What it does |
|---|---|
| **Hinglish training** | `hinglish_augment.py` converts English *training* reviews into Hinglish (English product words kept; opinion words, connectors and verbs written as romanised Hindi, spelling varied) and keeps the original label. 2,500 genuine + 2,500 fake are added to the **training split only**, after the split, so a converted copy of a validation/test review can never be in training. Equal numbers per class, so "contains Hinglish" is never a clue for either class. |
| **Language detection** | Mixed reviews such as "Best product ever!!! ... paisa vasool, sabko recommend karunga" were classed as English and got no Hinglish handling. The Hinglish word list and threshold were widened (7 of 3,000 English-source reviews changed, all of which really are Hinglish). |
| **Contact / advert pattern** | Non-English reviews that give a phone number, WhatsApp / Telegram or say "message karo / contact karo" get a small capped nudge towards suspicious (shown in the "Hinglish / Hindi check" box). English behaviour is unchanged. |
| **Measured, not assumed** | `data/hinglish_eval.csv`: 80 hand-written Hinglish reviews (40 genuine incl. negative and mixed ones, 40 spam-style), split into `dev` (used while tuning the converter and word lists) and `test` (only for the final numbers). `train_model.py` measures the model on it exactly as the app scores a review, and stores it in `metrics.json` (`hinglish_eval`, and `by_language.Hinglish`). |
| **"Not very sure" removed only if earned** | Hinglish counts as reliable only with at least 30 test reviews of both classes **and** F1 >= 0.75 **and** false-positive rate <= 10%. Otherwise the old "too little data, treat as a guess" note stays. |

**Measured on the 40 hand-written test reviews (v23 model):** F1 0.83, precision 0.94, recall 0.75, ROC-AUC 0.97, 1 of 20 genuine reviews flagged.
Before (v20261031-110704): recall 0.60, F1 0.75. A model with no Hinglish knowledge at all but the same rule layer scores F1 0.55 on the same reviews, so the model does add something.
English did not get worse (test F1 0.771 -> 0.777, same split).

**Be careful with these numbers.** They come from 40 reviews written by one author, in the same spirit as the spam patterns the checker looks for, so they are
*optimistic*: real fake Hinglish reviews are often subtler. Misses on the test set: subtle praise ("ekdum perfect, koi complaint nahi, sabko recommend karunga") and an advert
phrased without hype words. One genuine, slightly negative review was flagged ("heater ... bijli bahut khinchta hai"). The converted training reviews are **not** real Hinglish and the
labels are only as good as the English labels they came from (here: rule-based, see v21). Pure Devanagari Hindi is still "not very sure". To improve this properly,
label real Hinglish reviews (`data/hinglish_labelling_sheet.csv`, `python add_labels.py ...`), grow `hinglish_eval.csv` with them, and run `python train_model.py`.
`python train_model.py --hinglish-per-class 0` turns the conversion off. Tests: 119.

## Honest status of platform integration

**No live marketplace connection exists in this project, and none is faked.** Reviews come from text you paste or CSV/XLSX files you upload.
From a web search (October 2026; please re-check each platform's current developer terms):
* **Amazon:** the Product Advertising API 5.0 does not return customer reviews (removed with the move from 4.0 in 2020), and I found no public review API in the Selling Partner API.
* **Flipkart, Meesho, Myntra:** I found no official public review API. Third-party scraping services exist but may violate the platforms' terms, so none is integrated.

Implemented permitted alternatives: upload exports you are entitled to use (your own seller/brand data, licensed or public datasets) through the
ingestion pipeline, which stores source platform and collection date. Platforms without data show "Upload / export only".

## Folder structure

```
fake_review_detection/
├── app.py                          Flask app: pages + JSON API (database backed)
├── config.py                       paths, platforms, limits
├── multilingual.py                 Devanagari / Hinglish normalisation + language detection
├── sentiment.py                    lexicon sentiment (English / Hindi / Hinglish), shown separately
├── preprocessing.py                original cleaning + text-signal hints (now Hindi/Hinglish aware)
├── data_utils.py                   original dataset helpers (label encoding, column lookup)
├── ingest.py                       CSV/XLSX reading, validation, labelled ingestion, unlabelled batches
├── db.py                           SQLite layer
├── build_dataset.py                unified + balanced training dataset (CLI)
├── hinglish_augment.py             English -> Hinglish training conversion (v23)
├── generate_multilingual_demo.py   SYNTHETIC multilingual demo files
├── train_model.py                  compare, evaluate, version and register the model
├── train_transformer.py            OPTIONAL transformer fine-tuning (needs torch + transformers)
├── model_store.py                  versioned model folders + active pointer
├── predict.py                      prediction, explanations, language, sentiment
├── visualize.py                    charts
├── generate_demo_data.py           original English-only demo generator (kept)
├── requirements.txt  pytest.ini  README.md
├── data/    samples/ (unlabelled upload samples, CSV + XLSX)  demo_sources/ (synthetic, per platform)
│            raw/ (put YOUR labelled files here)  reviewguard.db, unified_training.csv (created)
├── models/  versions/<version>/{model.joblib, metrics.json}, active.json (created)
├── templates/  base, index, single_review, bulk_review, history, performance, dataset, about, error
├── static/     css/style.css   js/{main,dashboard,analyze,bulk,results,dataset}.js   images/
├── tests/      93 automated tests
└── docs/       PROJECT_REPORT.md (original report)   UPGRADE_NOTES.md
```

## Windows 10/11 + VS Code: set-up and run

Python 3.9+ (3.10 to 3.12 recommended). Open the folder that **contains `app.py`** in VS Code, then *Terminal > New Terminal* (PowerShell); check with `dir app.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python build_dataset.py --demo
python train_model.py
python app.py
```
Open <http://127.0.0.1:5000>. (If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.)
* `build_dataset.py --demo` generates the SYNTHETIC demo files, ingests them into the database (flagged *demo*) and writes `data\unified_training.csv`.
* `train_model.py` compares the models (the Random Forest is the slowest; allow a few minutes), prints per-language results and activates the new model.
* Tests: `python -m pytest -v` (or `python -m unittest discover -s tests -t . -v`).
* Fonts (Plus Jakarta Sans, Noto Sans Devanagari) load from Google Fonts; offline, system fonts are used (Nirmala UI renders Hindi on Windows).

## Using the app

* **Review Analysis:** paste a review, pick a platform (Meesho, Flipkart, Myntra, Amazon, Other), optionally a title and star rating, press *Analyse Review*. You get the verdict, confidence, **detected language**, **sentiment**, influential words and surface-pattern hints.
* **Batch Analysis:** paste one review per line, or upload `.csv` / `.xlsx` (column `review`; optional `platform`, `rating`, `title`). Samples: `data\samples\analysis_upload_multilingual.csv` / `.xlsx` (16 hand-written, unlabelled reviews in English, Hindi and Hinglish).
* **Prediction Results:** every stored prediction, filterable by platform, verdict and language.
* **Dataset Information:** what is in the labelled dataset (by platform, language and label method), upload verified labelled files, retrain, platform-integration status.
* **Model Performance:** measured metrics overall, per language, per platform, confusion matrix, all candidates, model versions.

## Bringing in real data (what you need to do)

1. Obtain labelled reviews you are allowed to use. Public fake-review datasets are mostly English and often machine-generated (e.g. the "Fake Reviews Dataset" with CG/OR labels on Kaggle; check its licence). I know of no verified Hindi/Hinglish fake-vs-genuine dataset, so plan to **annotate your own** (several annotators, with agreement checks).
2. Save as CSV (UTF-8, needed for Hindi) or XLSX with: `review` (or `review_text` / `text`), `label` (genuine/fake, OR/CG or 0/1) and optionally `platform`, `review_title`, `rating`, `collection_date`, `label_method`.
3. **State how the labels were obtained** (`label_method`, e.g. `manual_annotation`, `moderator_removed`). Ingestion refuses files without it; a label is never assumed from the platform.
4. Ingest on *Dataset Information*, or:
   ```powershell
   python build_dataset.py --ingest data\raw\meesho_labelled.csv --platform Meesho --label-method manual_annotation
   python train_model.py
   ```
   `train_model.py` uses verified rows only once there are at least 200 (otherwise it falls back to demo rows and says so).

Duplicates (after spelling normalisation) are skipped, and identical text with a conflicting label is reported, not stored.
Training data is balanced **per language** (the larger class is down-sampled) so that no language is mostly "fake".

## Models and the transformer option

`train_model.py` compares four classical models on word + character TF-IDF using a stratified (language x label) 70/15/15 split made **before** fitting; the TF-IDF vocabulary is learned inside the pipeline from the training split only.
Selection: highest validation F1 on the fake class (ties by ROC-AUC); the test split is used once for the report.

`train_transformer.py` fine-tunes XLM-RoBERTa (`xlm-roberta-base`), multilingual BERT or IndicBERT on the **same split**, stores a new version with metrics in the same format, and activates it only with `--activate`.
**I could not run it** (no PyTorch or internet in the build environment); only its dependency check is tested. Use a GPU if possible:
```powershell
pip install torch transformers
python train_transformer.py --model xlm-roberta --epochs 3
python train_transformer.py --model indic-bert --epochs 4
```
Compare versions on *Model Performance*. TF-IDF models are not necessarily worse than transformers, especially with little data.

## JSON API (all return `{"ok": ...}`)

| Method | Route | Notes |
|---|---|---|
| POST | `/api/predict` | `{"review","platform"?,"title"?,"rating"?}`; stores the result |
| POST | `/api/predict-bulk` | JSON `{"reviews":[...],"platform"}` or multipart `file` (.csv/.xlsx) + `platform` |
| GET / DELETE | `/api/history` | filters `platform`, `label`, `language`, `limit`, `offset` |
| GET | `/api/stats`, `/api/dataset-info`, `/api/platforms`, `/api/model-info`, `/api/models`, `/api/health` | |
| POST | `/api/ingest` | multipart `file`, `platform`, `label_method` (localhost only) |
| POST / GET | `/api/retrain`, `/api/retrain/status` | background job (localhost only) |
| POST | `/api/models/activate` | `{"version"}` (localhost only) |

Limits: 5000 characters per review, 1000 reviews per batch, 6 MB uploads, 50,000 rows per ingestion file.

## Testing

`python -m pytest -v` runs 93 tests: the spec sentences in English/Hindi/Hinglish (language, sentiment, prediction structure), normalisation, ingestion (CSV, XLSX, invalid and empty files,
duplicates, conflicts), database, balancing, leakage-free splits, training artefacts, per-language and per-platform metrics, **fairness checks on held-out synthetic reviews** (genuine negative and
positive reviews not flagged fake; Hindi/Hinglish not flagged more than English), suspicious promotional and repetitive reviews, all four platform labels, the API (validation, uploads, limits,
history persistence, retraining job, local-only actions, behaviour without a model) and a cross-check that every element id used by the JavaScript exists in its page.
The fairness tests use **synthetic** data, so they verify the code, not real-world fairness. **Not tested here:** the browser UI itself (no browser was available) and `train_transformer.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Banner "No trained model yet" | `python build_dataset.py --demo` then `python train_model.py`, refresh |
| `can't open file` / `Could not open requirements file` | wrong folder: `cd` into the folder that contains `app.py` |
| Hindi shows as `????` in a CSV | save the CSV as **UTF-8** (Excel: "CSV UTF-8") or use .xlsx |
| `ModuleNotFoundError: openpyxl` | `pip install -r requirements.txt` (needed for .xlsx) |
| Short generic reviews come back "low confidence" | expected: very little evidence; see the explanation text |
| Training says "Not enough labelled data" | needs >= 200 reviews and >= 40 per class after cleaning |
| Hindi words in a chart show as boxes | install a Devanagari font (Windows has Nirmala UI); the Performance page also lists top words as text |
| "Could not load the model" after upgrading libraries | run `python train_model.py` again |

## Product-based checking (v16)

On **Check one review**, pick the product (Add more details → *Product being reviewed*). The checker then also asks: does the review talk about
that kind of product?

* **Clear mismatch** (e.g. an air-fryer review under a Saree): the suspicion score is raised a little (capped; log-odds +0.8) and the result
  card says why. This is a hint, not proof.
* **Match or no product words** ("good quality, fast delivery"): no change. A match never proves a review is genuine.
* Word lists are in `product_context.py` (English, Hinglish, Hindi); extend `LEXICON` for your products.
* **Training:** add a `product` column to your labelled CSV/XLSX (also accepted: `product_name`, `product_category`). If at least 20% of
  the labelled rows have a product, `python train_model.py` trains a *product-aware* model (product group + review-vs-product fit are extra
  features). The model shipped in this zip was trained without product data, so it uses only the product check above.
* Bulk upload also reads an optional `product` column.

## Better Hinglish / Hindi checking (v17)

* **English bridge:** for Hinglish, Hindi and mixed reviews the English meaning of known words is appended to the text the model sees
  ("paisa vasool" -> "value for money"). The original text is kept. This lets what the model learned from English reviews apply to Hinglish.
  Word lists: `hinglish_support.py` (`WORDS`, `PHRASES`, `HYPE`). Add your own spellings and slang there.
* **Spam-pattern check** (Hinglish/Hindi/mixed only): the same phrase repeated again and again, or four or more hype phrases stacked
  together, raises the suspicion score a little (capped). The result card shows a "Hinglish / Hindi check" box.
* **Training:** `python train_model.py` applies the same bridge to non-English training rows.
* **(Superseded by v23, see above.) Accuracy was NOT measured for Hinglish at v17.** The training data has only 12 Hinglish reviews, so the tool still says "Not very sure" for
  Hinglish. To measure and improve it, label `data/hinglish_labelling_sheet.csv` (genuine / fake) and run
  `python add_labels.py "C:\path\hinglish_labelling_sheet_filled.csv"`. Aim for 200+ of each label.
