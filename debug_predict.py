# debug_predict.py
import json, os, pprint, time
from predict_distilbert import predict_with_distilbert   # your local predictor
from gui import decide_using_sources_and_model, sanitize_backend_out, fetch_factchecks_for_query

def debug_one(text):
    print("="*80)
    print("INPUT:")
    print(text)
    print("-"*80)
    # 1) model raw outputs
    try:
        lab, conf, probs = predict_with_distilbert(text)
    except Exception as e:
        print("predict_with_distilbert ERROR:", e)
        lab, conf, probs = None, None, {}
    print("\nMODEL RAW:")
    pprint.pprint({"label": lab, "conf": conf, "probs": probs})
    # 2) build backend-like dict exactly like GUI expects
    backend_out = {"label": lab or "unverified", "fake_prob": probs.get("fake", 1.0 - probs.get("real", 0.0)), "message": "local model prediction", "related": []}
    backend_out = sanitize_backend_out(backend_out)
    # 3) fetch fact-checks (quick)
    print("\nFACTCHECKS (if API key configured, else empty):")
    try:
        fc = fetch_factchecks_for_query(text, max_results=5)
        pprint.pprint(fc)
    except Exception as e:
        print("factcheck fetch error:", e)
    # 4) run the GUI decision logic so you see final_label + explain
    final = decide_using_sources_and_model(backend_out, text)
    print("\nDECISION ENGINE OUTPUT (after decide_using_sources_and_model):")
    pprint.pprint(final)
    print("="*80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        txt = " ".join(sys.argv[1:])
    else:
        txt = input("Enter test headline/text: ").strip()
    debug_one(txt)
