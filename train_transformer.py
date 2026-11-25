"""OPTIONAL: fine-tune a multilingual transformer (XLM-RoBERTa, multilingual BERT or IndicBERT) on the same data and splits.

*** This script could NOT be executed in the environment where the project was built (no PyTorch / no internet). ***
It is provided so you can run the comparison on your own machine. Treat the first run as something to verify.

Install (once):   pip install torch transformers
Run examples:     python train_transformer.py --model xlm-roberta --epochs 3
                  python train_transformer.py --model indic-bert --epochs 4 --activate
First run downloads the pre-trained weights from huggingface.co (a few hundred MB). A GPU is strongly recommended;
on CPU use --max-train 2000 --epochs 1 just to check that the script works.

It uses exactly the same unified dataset, balancing and train/validation/test split as train_model.py, so the metrics are
directly comparable (overall, per language, per platform). The result is stored as a new model version; it becomes the active
model only with --activate (compare it on the Model Performance page first).
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np

import config
import db
import model_store
from build_dataset import build_training_frame
from ingest import IngestError
from train_model import choose_rows, group_metrics, overall_metrics, sentiment_check, split_frame

INSTALL_HINT = "This script needs PyTorch and Transformers: pip install torch transformers"


def require_torch():
    try:
        import torch
        import transformers
    except ImportError:
        raise SystemExit(INSTALL_HINT)
    return torch, transformers


def predict_scores(torch, tok, model, texts, device, max_length, batch_size=32):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(config.TRANSFORMER_CHOICES), default="xlm-roberta")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train", type=int, help="use only the first N training reviews (quick smoke test)")
    ap.add_argument("--data-mode", choices=["auto", "real", "all"], default="auto")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--activate", action="store_true", help="make this model the active one")
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args()
    torch, transformers = require_torch()
    hf_name = config.TRANSFORMER_CHOICES[args.model]

    try:
        include_demo, mode_used = choose_rows(args.db, args.data_mode)
        df, report = build_training_frame(args.db, include_demo=include_demo, balance=not args.no_balance)
    except IngestError as exc:
        raise SystemExit(f"ERROR: {exc}")
    if report["rows_final"] < config.MIN_ROWS_FOR_TRAINING:
        raise SystemExit(f"ERROR: only {report['rows_final']} reviews; need at least {config.MIN_ROWS_FOR_TRAINING}.")
    tr, va, te = split_frame(df)
    if args.max_train:
        tr = tr.head(args.max_train)
    print(f"{hf_name}: train={len(tr)} val={len(va)} test={len(te)} ({mode_used})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(hf_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size))
    sched = transformers.get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    texts, labels = tr["model_text"].tolist(), tr["label"].to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    best_f1, best_state, val_rows = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(texts))
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=torch.tensor(labels[idx], dtype=torch.long, device=device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
        vs = predict_scores(torch, tok, model, va["model_text"].tolist(), device, args.max_length)
        vm = overall_metrics(va["label"], (vs >= 0.5).astype(int), vs)
        val_rows.append({"model": hf_name + " (fine-tuned)", "params": {"epoch": epoch, "lr": args.lr}, "val_accuracy": vm["accuracy"],
                         "val_precision": vm["precision"], "val_recall": vm["recall"], "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"]})
        print(f"epoch {epoch}: validation F1(fake)={vm['f1']:.3f} AUC={vm['roc_auc']:.3f}")
        if vm["f1"] > best_f1:
            best_f1, best_state = vm["f1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    score = predict_scores(torch, tok, model, te["model_text"].tolist(), device, args.max_length)
    pred = (score >= 0.5).astype(int)
    overall = overall_metrics(te["label"], pred, score)
    version = model_store.new_version() + "-" + args.model
    vdir = model_store.version_dir(config.MODELS_DIR, version)
    (vdir / "transformer").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(vdir / "transformer")
    tok.save_pretrained(vdir / "transformer")
    counts = report["class_counts"]
    metrics = {
        "version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "is_demo_data": report["is_demo"],
        "data_mode": mode_used, "cleaning_report": report, "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
        "selected_model": {"name": hf_name + " (fine-tuned)", "params": {"epochs": args.epochs, "lr": args.lr, "max_length": args.max_length}, "type": "transformer"},
        "selection_criterion": "best validation F1 epoch", "validation_results": val_rows,
        "test_metrics": {"overall": overall, "by_language": group_metrics(te, pred, score, "language"),
                         "by_platform": group_metrics(te, pred, score, "platform"), "sentiment_check": sentiment_check(te, pred)},
        "majority_class_baseline_accuracy": max(counts.values()) / sum(counts.values()), "top_words": None, "figures": [],
        "notes": ["Fine-tuned with train_transformer.py on the same split as the classical models."] + (
            ["DEMONSTRATION DATA: synthetic training reviews, so these scores are not evidence of real-world accuracy."] if report["is_demo"] else []),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    db.register_model(args.db, version, "transformer: " + hf_name, vdir, report["is_demo"], overall["f1"], overall["accuracy"],
                      f"{report['rows_final']} reviews, {mode_used}", activate=args.activate)
    if args.activate:
        model_store.set_active(config.MODELS_DIR, version)
    print(f"\nTEST: accuracy={overall['accuracy']:.3f} precision={overall['precision']:.3f} recall={overall['recall']:.3f} "
          f"F1={overall['f1']:.3f} ROC-AUC={overall['roc_auc']:.3f}\nSaved {version}" + (" (active)" if args.activate else " (not active; use --activate or the Model Performance page to compare)"))


if __name__ == "__main__":
    main()
