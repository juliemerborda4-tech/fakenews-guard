# prediction_logic.py
import logging
from typing import Optional, Dict, Any

TEXTUAL_MAP = {
    "true": "REAL",
    "mostly true": "REAL",
    "false": "FAKE",
    "mostly false": "PARTLY_FAKE",
    "misleading": "MISLEADING",
    "no evidence": "UNVERIFIED",
}

def map_textual(text: Optional[str]) -> str:
    if not text: return "UNVERIFIED"
    return TEXTUAL_MAP.get(text.strip().lower(), "UNVERIFIED")

def extract_factcheck_verdict(fc_json: Optional[Dict[str,Any]]) -> Dict[str,Any]:
    if not fc_json:
        return {"verdict":"UNVERIFIED"}
    items = fc_json.get("claims") or fc_json.get("items") or fc_json.get("claimsSearchResults") or []
    if isinstance(items, dict): items = [items]
    if not items:
        return {"verdict":"UNVERIFIED"}
    for it in items:
        crs = it.get("claimReview") or it.get("claim_reviews") or it.get("claim_review") or []
        if isinstance(crs, dict): crs = [crs]
        for cr in crs:
            txt = cr.get("textualRating") or cr.get("textual_rating") or cr.get("rating")
            if txt:
                return {
                    "verdict": map_textual(txt),
                    "raw_textual": txt,
                    "source": (cr.get("publisher") or {}).get("name") or cr.get("publisher"),
                    "url": cr.get("url"),
                    "matched_claim": cr.get("title") or it.get("claim") or it.get("text")
                }
    return {"verdict":"UNVERIFIED"}

def combine_decision(
    text: str,
    factcheck_json: Optional[Dict[str,Any]],
    gnews_json: Optional[Dict[str,Any]],
    ml_pred: Optional[str],
    ml_prob: Optional[float],
    allow_ml_fallback: bool = False
) -> Dict[str,Any]:
    logging.debug("combine_decision called. text_len=%d ml=%s ml_prob=%s", len(text), ml_pred, ml_prob)

    if len(text.strip()) < 20:
        fc = extract_factcheck_verdict(factcheck_json)
        if fc.get("verdict") != "UNVERIFIED":
            return {"final": fc["verdict"], "source":"FactCheckAPI", "details": fc}
        return {"final":"UNVERIFIED", "source":"heuristic", "reason":"text_too_short"}

    fc = extract_factcheck_verdict(factcheck_json)
    if fc.get("verdict") and fc["verdict"] != "UNVERIFIED":
        return {"final": fc["verdict"], "source":"FactCheckAPI", "details": fc}

    if allow_ml_fallback and ml_pred:
        label = ("REAL" if ml_pred.lower() in ("real","true","not fake") else "FAKE")
        return {"final": label, "source":"ML_SUGGESTION", "confidence": ml_prob, "note":"fallback_used"}

    return {"final":"UNVERIFIED", "source":"none", "gnews_related": bool(gnews_json)}
