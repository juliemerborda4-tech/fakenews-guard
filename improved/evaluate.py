import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "improved_artifacts"


def eval_baseline(val_csv: Path, model_path: Path):
    d = joblib.load(model_path)
    pipe = d["pipeline"]
    df = pd.read_csv(val_csv)
    X = df[["text", "source_domain", "is_reliable_domain", "reliability_score"]]
    y = df["label"].astype(int).to_numpy()

    probs = pipe.predict_proba(X)
    # class ordering is learned; derive index for fake (=1)
    classes = list(pipe.named_steps["clf"].classes_)
    fake_idx = classes.index(1)
    p_fake = probs[:, fake_idx]
    y_pred = (p_fake >= 0.5).astype(int)

    return y, y_pred, p_fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline"], default="baseline")
    ap.add_argument("--val_csv", default=str(ART / "val.csv"))
    ap.add_argument("--baseline_model", default=str(ART / "baseline_model.joblib"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    y_true, y_pred, p_fake = eval_baseline(Path(args.val_csv), Path(args.baseline_model))
    y_pred = (p_fake >= float(args.threshold)).astype(int)

    print("Label convention: 0=real, 1=fake")
    print("Samples:", len(y_true))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision(fake):", precision_score(y_true, y_pred, zero_division=0))
    print("Recall(fake):", recall_score(y_true, y_pred, zero_division=0))
    print("F1(fake):", f1_score(y_true, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()

