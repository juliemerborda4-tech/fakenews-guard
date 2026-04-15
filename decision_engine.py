# decision_engine.py
# Minimal, safe decision_engine fallback used by the GUI when other backends are not available.
# It prefers to call a local model predict_with_bert if available.

import logging

try:
    from predict_model import predict_with_bert
    _HAS_LOCAL = True
    logging.info("decision_engine: local predict_with_bert loaded")
except Exception as e:
    _HAS_LOCAL = False
    logging.warning("decision_engine: local predictor not available: %s", e)

def decide_label(text):
    """
    Return a dictionary in the shape expected by gui.py decision path:
    {
        "label": "fake" | "real" | "unverified" | "check_needed",
        "prob": float (0-1) optional,
        "reason": str optional,
        "meta": {"sample_articles": [ {title, url, rating}, ... ] }
    }
    This function should be lightweight and deterministic.
    """
    try:
        if not text or not isinstance(text, str):
            return {"label": "unverified", "prob": 0.5, "reason": "empty input"}

        # Use local model when available
        if _HAS_LOCAL:
            try:
                lab, conf, probs = predict_with_bert(text, api_feats=[0,0,0])
                label = "fake" if lab == 1 else "real"
                return {
                    "label": label,
                    "prob": float(conf),
                    "reason": "local model prediction",
                    "meta": {"sample_articles": []}
                }
            except Exception as e:
                logging.exception("decision_engine.local_predict failed: %s", e)
                return {"label": "unverified", "prob": 0.5, "reason": "local model error"}
        else:
            # If no local model, ask GUI to fallback
            return {"label": "unverified", "prob": 0.5, "reason": "no decision engine available"}
    except Exception as e:
        logging.exception("decision_engine.decide_label top-level error: %s", e)
        return {"label": "unverified", "prob": 0.5, "reason": "internal error"}
