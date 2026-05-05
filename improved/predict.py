import argparse
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "improved_artifacts"


def predict_baseline(text: str, model_path: Path):
    d = joblib.load(model_path)
    pipe = d["pipeline"]
    X = {
        "text": [text],
        "source_domain": [""],
        "is_reliable_domain": [0],
        "reliability_score": [0.0],
    }
    probs = pipe.predict_proba(__import__("pandas").DataFrame(X))[0]
    classes = list(pipe.named_steps["clf"].classes_)
    fake_idx = classes.index(1)
    p_fake = float(probs[fake_idx])
    label = "fake" if p_fake >= 0.5 else "real"
    return label, p_fake, {"real": 1.0 - p_fake, "fake": p_fake}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--baseline_model", default=str(ART / "baseline_model.joblib"))
    args = ap.parse_args()

    label, p_fake, probs = predict_baseline(args.text, Path(args.baseline_model))
    print("label:", label)
    print("p_fake:", round(p_fake, 6))
    print("probs:", probs)


if __name__ == "__main__":
    main()

