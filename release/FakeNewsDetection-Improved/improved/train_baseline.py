import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "improved_artifacts"


def build_pipeline():
    text_vect = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=60000,
        min_df=2,
    )
    pre = ColumnTransformer(
        transformers=[
            ("text", text_vect, "text"),
            ("domain", OneHotEncoder(handle_unknown="ignore"), ["source_domain"]),
            ("num", "passthrough", ["is_reliable_domain", "reliability_score"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=str(ART / "train.csv"))
    ap.add_argument("--out", default=str(ART / "baseline_model.joblib"))
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    X = train_df[["text", "source_domain", "is_reliable_domain", "reliability_score"]]
    y = train_df["label"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model_type": "baseline", "label_convention": "0=real,1=fake", "pipeline": pipe},
        out,
    )
    print("Saved baseline model to", out)


if __name__ == "__main__":
    main()

