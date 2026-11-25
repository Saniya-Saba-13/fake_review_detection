"""Run this if the app does not start:   python check_setup.py
It checks the common problems and says what to do. Nothing is changed."""
import importlib
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ok = True


def say(good, msg, fix=""):
    global ok
    ok = ok and good
    print(("[OK]   " if good else "[FAIL] ") + msg + (("\n         -> " + fix) if (fix and not good) else ""))


print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
say(sys.version_info >= (3, 9), "Python version is 3.9 or newer", "Install Python 3.11 or 3.12 from python.org (tick 'Add Python to PATH').")
say((HERE / "app.py").exists() and (HERE / "templates").is_dir(), f"Running inside the project folder ({HERE})",
    "In VS Code use File > Open Folder and pick the folder that contains app.py (not the folder above it).")
say((HERE / "product_context.py").exists() and (HERE / "hinglish_support.py").exists(), "This is the NEW version (v17 files are present)",
    "You are in an old copy. Unzip fake_review_detection_v17.zip and open THAT folder.")

for mod in ("flask", "sklearn", "pandas", "numpy", "scipy", "joblib", "openpyxl"):
    try:
        m = importlib.import_module(mod)
        say(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        say(False, f"{mod} is not installed ({exc.__class__.__name__})", "Run:  pip install -r requirements.txt   (inside the activated venv)")

try:
    sys.path.insert(0, str(HERE))
    from predict import FakeReviewPredictor
    r = FakeReviewPredictor().predict_one("Yeh product bahut accha hai.")
    say(True, f"Model loads and predicts ({r['label']}, version {r['model_version']})")
except Exception as exc:  # noqa: BLE001
    say(False, f"Model could not load: {exc}", "Run:  python train_model.py   then start the app again.")

busy = socket.socket().connect_ex(("127.0.0.1", 5000)) == 0
print(("[NOTE] " if busy else "[OK]   ") + ("Port 5000 is busy (an older copy may be running). The app will pick the next free port and print it."
                                            if busy else "Port 5000 is free"))
print("\nAll good. Start it with:  python app.py" if ok else "\nFix the [FAIL] lines above, then run this check again.")
