import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "improved_artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--baseline_model", default=str(ART / "baseline_model.joblib"))
    args = ap.parse_args()

    bundle = joblib.load(Path(args.baseline_model))
    pipe = bundle["pipeline"]
    X = pd.DataFrame(
        {"text": [args.text], "source_domain": [""], "is_reliable_domain": [0], "reliability_score": [0.0]}
    )
    probs = pipe.predict_proba(X)[0]
    classes = list(pipe.named_steps["clf"].classes_)
    fake_idx = classes.index(1)
    p_fake = float(probs[fake_idx])
    label = "fake" if p_fake >= 0.5 else "real"
    print("label:", label)
    print("p_fake:", round(p_fake, 6))
    print("probs:", {"real": round(1.0 - p_fake, 6), "fake": round(p_fake, 6)})


if __name__ == "__main__":
    main()

