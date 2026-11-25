# ReviewGuard AI — Multilingual Fake Product Review Detection

ReviewGuard AI is a Flask-based web application that analyses product reviews and estimates whether a review is **likely genuine** or **suspicious / likely fake**. It supports English, Hindi (Devanagari), and Hinglish, and provides single-review analysis, batch analysis, explanations, review history, dataset management, and model-performance reports.

> **Important:** A prediction is a statistical estimate, not proof that a review is fake or genuine. Do not use the result alone to accuse a reviewer or make moderation decisions. Negative opinions and the use of Hindi or Hinglish are not evidence of fraud.

## Features

- **Single review analysis:** paste a review and optionally provide its title, platform, rating, and product.
- **Batch analysis:** analyse multiple reviews by pasting text or uploading CSV/XLSX files.
- **Multilingual text processing:** English, Hindi, Hinglish, and mixed-language detection and normalisation.
- **Prediction explanations:** confidence estimate, influential terms, and writing-pattern signals.
- **Sentiment analysis:** sentiment is displayed separately; it is not treated as proof of fake activity.
- **Product-context check:** highlights a possible mismatch between the review and the selected product category.
- **Dashboard and history:** view stored analyses, filter results, and inspect model information.
- **Dataset management:** import labelled review files, inspect dataset statistics, and retrain the model.
- **Model evaluation:** compare candidate models and view overall, language-wise, and platform-wise metrics.
- **JSON API:** endpoints for predictions, batch predictions, history, dataset information, and model status.

## Technology

- Python and Flask
- scikit-learn, TF-IDF, and classical machine-learning classifiers
- pandas, NumPy, SciPy, and joblib
- SQLite for local data storage
- HTML, CSS, and JavaScript for the web interface
- pytest for automated tests

## Supported languages and platforms

**Languages:** English, Hindi, Hinglish. Mixed or unknown language may also be detected.

**Platform labels:** Amazon, Flipkart, Meesho, Myntra, and Other.

Platform names are metadata supplied by the user. **This project does not connect to marketplace accounts or automatically fetch live reviews.** Reviews must be pasted or uploaded from data you are permitted to use.

## Project structure

```text
fake_review_detection/
├── app.py                   # Flask application and API routes
├── config.py                # Application settings and paths
├── predict.py               # Model loading and review prediction
├── preprocessing.py         # Review cleaning and text signals
├── multilingual.py          # Language detection and normalisation
├── hinglish_support.py      # Hinglish/Hindi language support
├── sentiment.py             # Sentiment analysis
├── product_context.py       # Product/review context checks
├── ingest.py                # CSV/XLSX data ingestion
├── db.py                    # SQLite database operations
├── build_dataset.py         # Dataset preparation
├── train_model.py           # Model comparison and training
├── train_transformer.py     # Optional transformer training
├── model_store.py           # Model version management
├── requirements.txt         # Python dependencies
├── templates/               # Flask HTML templates
├── static/                  # CSS, JavaScript, and images
├── data/                    # Sample, demo, and local application data
├── models/                  # Trained model versions and metrics
└── tests/                   # Automated tests
```

## Setup and run (Windows)

Python 3.10–3.12 is recommended. Python 3.9+ is specified by the project's existing setup notes.

1. Download or clone this repository.
2. Open a terminal in the `fake_review_detection` folder — the folder containing `app.py`.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Build the demonstration dataset:

```powershell
python build_dataset.py --demo
```

6. Train and activate a model:

```powershell
python train_model.py
```

7. Start the web application:

```powershell
python app.py
```

Open the local address printed in the terminal. The application normally uses:

```text
http://127.0.0.1:5000/single
```

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

## First run and demo-data limitation

The bundled demo-data generator creates **synthetic/template-based reviews** so the full application and training workflow can be demonstrated. These are not real customer reviews.

Metrics produced using synthetic data measure performance on that synthetic dataset only. They do **not** establish real-world accuracy. The application identifies demo-trained models as demo models.

For meaningful evaluation, collect or obtain real labelled reviews that you are legally allowed to use, review the labels carefully, and evaluate on a separate representative test set.

## Using the application

### Analyse one review
Open the single-review page, paste the review, select a platform, and optionally enter the title, rating, and product information. Submit it to view the estimate and explanation.

### Analyse multiple reviews
Use the batch-analysis page. You can paste one review per line or upload a `.csv` or `.xlsx` file. The review text column should be named `review`; optional columns include `title`, `platform`, `rating`, and `product`.

### Import labelled data and retrain
Use the Dataset Information page or the command line. Labelled data must include a review-text column, a label column, and an explanation of how the labels were obtained (`label_method`).

Example:

```powershell
python build_dataset.py --ingest data\raw\reviews.csv --platform Flipkart --label-method manual_annotation
python train_model.py
```

Accepted label formats include genuine/fake, OR/CG, or 0/1 as supported by the ingestion pipeline. Use UTF-8 CSV encoding for Hindi text. Review the ingestion report before training.

The training workflow uses verified labelled records when enough are available; otherwise its automatic mode may use demo records. Check the training output and model metadata to confirm which data source was used.

## Model training

The standard training script compares four classical classifiers using word and character TF-IDF features:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine
- Random Forest

The data is split into training, validation, and test subsets. Model selection is based on validation F1 for the fake class, with ROC-AUC used to break ties. The held-out test split is used for reporting.

An optional transformer workflow is provided:

```powershell
pip install torch transformers
python train_transformer.py --model xlm-roberta --epochs 3
```

Transformer training may require substantial memory and a compatible GPU. It is optional; the standard scikit-learn workflow does not require these packages.

## Run tests

From the project folder:

```powershell
python -m pytest -v
```

The tests cover core preprocessing, multilingual support, ingestion, database operations, prediction/training workflows, and API behaviour. Passing automated tests verifies tested software behaviour; it does not prove that predictions are accurate or fair on real-world data.

## API overview

The Flask application exposes JSON endpoints including:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Analyse one review |
| `POST` | `/api/predict-bulk` | Analyse a batch or uploaded file |
| `GET`, `DELETE` | `/api/history` | Read or clear analysis history |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/dataset-info` | Dataset information |
| `GET` | `/api/platforms` | Supported platform labels |
| `GET` | `/api/model-info` | Active model information |
| `GET` | `/api/models` | Available model versions |
| `GET` | `/api/health` | Application health |
| `POST` | `/api/ingest` | Import labelled data (local-only) |
| `POST` | `/api/retrain` | Start model retraining (local-only) |
| `GET` | `/api/retrain/status` | Check retraining status |

For exact request fields and limits, refer to the route definitions in `app.py`.

## Data, privacy, and GitHub notes

- Do not commit private customer data, personal information, API keys, passwords, or other secrets.
- Check the licence and usage terms of every dataset before publishing or using it.
- The local SQLite database and generated model files may contain data or artefacts that you do not want to publish. Review them before pushing the repository.
- Use a `.gitignore` file to exclude virtual environments, caches, local databases, and private raw datasets.
- Do not describe the project as having live Amazon/Flipkart/Meesho/Myntra integration; it currently analyses user-provided text and files.

## Limitations

- Predictions depend on the quality, size, and representativeness of the labelled training data.
- Synthetic demo data is only for demonstrating the pipeline.
- Hindi and Hinglish performance requires more independently labelled real-world examples for reliable evaluation.
- Writing style, language, sentiment, rating, or a product mismatch cannot independently prove that a review is fraudulent.
- No live marketplace review collection is implemented.

## Future improvements

- Expand independently labelled datasets across languages and product categories.
- Improve evaluation with larger, representative, independently annotated test sets.
- Add carefully reviewed integrations only where official APIs and platform terms permit them.
- Monitor false positives and false negatives after deployment.

## License

No licence is specified in the supplied project files. Add a `LICENSE` file before presenting the repository as open source. Until then, all rights remain with the applicable copyright holder.
