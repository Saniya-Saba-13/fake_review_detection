"""Central configuration for ReviewGuard (multilingual fake-review detection)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Paths -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                       # put YOUR real labelled / exported review files here
DEMO_SOURCES_DIR = DATA_DIR / "demo_sources"     # SYNTHETIC per-platform demo files (generate_multilingual_demo.py)
SAMPLES_DIR = DATA_DIR / "samples"               # unlabelled files for testing uploads
UNIFIED_DATASET_PATH = DATA_DIR / "unified_training.csv"   # written by build_dataset.py (what the model trains on)
MODELS_DIR = MODELS_DIR_DEFAULT = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "static" / "images"
DB_PATH = DATA_DIR / "reviewguard.db"

# English-only demo files from the first version of the project (kept)
SAMPLE_DATA_PATH = DATA_DIR / "sample_reviews.csv"
SAMPLE_UPLOAD_PATH = DATA_DIR / "sample_upload.csv"
SAMPLE_UPLOAD_MULTI_CSV = SAMPLES_DIR / "analysis_upload_multilingual.csv"
SAMPLE_UPLOAD_MULTI_XLSX = SAMPLES_DIR / "analysis_upload_multilingual.xlsx"

# ---- Platforms / languages ---------------------------------------------
PLATFORMS = ["Meesho", "Flipkart", "Myntra", "Amazon", "Other"]
# Products people commonly review on Indian shopping sites, grouped for the "Product" dropdown on the Check-one-review page.
# This is only a pick-list of product types; nothing is fetched from any marketplace.
PRODUCTS = {
    "Mobiles & Electronics": ["Smartphone", "Earphones / Earbuds", "Bluetooth Speaker", "Smart Watch", "Laptop", "Power Bank",
                              "Charger / Cable", "Television", "Camera", "Headphones"],
    "Fashion - Men": ["T-shirt", "Shirt", "Jeans", "Kurta", "Shoes / Sneakers", "Jacket", "Watch"],
    "Fashion - Women": ["Saree", "Kurti / Kurta Set", "Dress", "Top", "Leggings", "Lehenga", "Heels / Sandals", "Handbag"],
    "Kids": ["Kids Clothing", "Toys", "School Bag", "Baby Care"],
    "Home & Kitchen": ["Mixer Grinder", "Air Fryer", "Pressure Cooker", "Bedsheet", "Water Bottle", "Cookware", "Home Decor", "Storage / Organiser"],
    "Appliances": ["Refrigerator", "Washing Machine", "Air Conditioner", "Fan", "Iron", "Geyser / Water Heater", "Vacuum Cleaner"],
    "Beauty & Personal Care": ["Face Cream / Moisturiser", "Lipstick / Makeup", "Shampoo / Hair Oil", "Perfume", "Trimmer / Shaver", "Skin Care Serum"],
    "Grocery & Health": ["Grocery / Packaged Food", "Protein / Supplement", "Health Device"],
    "Books & Stationery": ["Book", "Stationery"],
    "Sports & Fitness": ["Sports Equipment", "Yoga Mat", "Gym Equipment"],
    "Other": ["Other (not listed)"],
}
MAX_PRODUCT_CHARS = 80
PRODUCT_AWARE_MIN_SHARE = 0.2   # train with product tokens only if at least this share of labelled rows has a product

LANGUAGES = ["English", "Hindi", "Hinglish"]          # the three supported languages (Mixed/Unknown are detected too)
DEMO_LABEL_METHOD = "synthetic_template"

# ---- Training ----------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_ROWS_FOR_TRAINING = 200
MIN_ROWS_PER_CLASS = 40
MIN_ROWS_PER_LANGUAGE_TO_BALANCE = 40
SELECTION_CRITERION = "validation F1 on the fake class (ties broken by ROC-AUC)"

# ---- Prediction / API limits -------------------------------------------
MIN_REVIEW_CHARS = 2
MAX_REVIEW_CHARS = 5000
MAX_BULK_REVIEWS = 1000
MAX_INGEST_ROWS = 50000
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.60
TOP_TERMS = 5
MAX_RECENT_ANALYSES = 500

# ---- Optional transformer models (train_transformer.py; needs `pip install torch transformers`) ----
TRANSFORMER_CHOICES = {
    "xlm-roberta": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "indic-bert": "ai4bharat/indic-bert",
}

JSON_SORT_KEYS = False
