"""
Compatibility layer for older scripts in this repo that expect:

    from predict_model import predict_with_bert

The improved pipeline defaults to a calibrated baseline model stored at:
    improved_artifacts/baseline_model.joblib

Label convention: 0=real, 1=fake
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "improved_artifacts" / "baseline_model.joblib"


_CACHE = {"path": None, "bundle": None}


def _load(path: Path):
    path = Path(path)
    if _CACHE["bundle"] is not None and _CACHE["path"] == str(path):
        return _CACHE["bundle"]
    bundle = joblib.load(path)
    _CACHE["path"] = str(path)
    _CACHE["bundle"] = bundle
    return bundle


def predict_with_bert(text: str, api_feats=None, model_path: str | None = None):
    """
    Returns (label_int, confidence, probs_dict)

    - label_int: 0=real, 1=fake
    - confidence: max(prob_real, prob_fake)
    - probs_dict: {"real": p0, "fake": p1}

    api_feats is accepted for backwards compatibility but not required by the baseline.
    """
    mp = Path(model_path) if model_path else DEFAULT_MODEL
    bundle = _load(mp)
    pipe = bundle["pipeline"]

    X = pd.DataFrame(
        {
            "text": [str(text)],
            "source_domain": [""],
            "is_reliable_domain": [0],
            "reliability_score": [0.0],
        }
    )
    probs = pipe.predict_proba(X)[0]
    classes = list(pipe.named_steps["clf"].classes_)
    fake_idx = classes.index(1)
    p_fake = float(probs[fake_idx])
    p_real = 1.0 - p_fake
    label = 1 if p_fake >= 0.5 else 0
    conf = max(p_real, p_fake)
    return label, conf, {"real": p_real, "fake": p_fake}

