# debug_predict_arg.py
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_FILE = "model.pkl"
RELATED_VECT = "related_vectorizer.pkl"
TFIDF_MAT = "tfidf_vectors.pkl"
DF_FILE = "news_dataframe.pkl"

model = joblib.load(MODEL_FILE)
related_vec = joblib.load(RELATED_VECT)
tfidf_matrix = joblib.load(TFIDF_MAT)
df = pd.read_pickle(DF_FILE)

def explain_tfidf(input_text, top_n=8):
    try:
        v = related_vec
        x = v.transform([input_text])
        feat_names = None
        try:
            feat_names = v.get_feature_names_out()
        except Exception:
            try:
                feat_names = v.named_steps['tfidf'].get_feature_names_out()
            except Exception:
                feat_names = None
        if feat_names is None:
            print("Feature names not available for this vectorizer.")
            return
        row = x.toarray()[0]
        inds = np.argsort(row)[::-1][:top_n]
        print("Top input TF-IDF tokens:")
        for i in inds:
            if row[i] > 0:
                print(f"  {feat_names[i]}   (value={row[i]:.4f})")
    except Exception as e:
        print("Cannot explain TF-IDF:", e)

def debug(input_text):
    print("=== Debugging prediction for ===")
    print(input_text, "\n")

    try:
        proba = model.predict_proba([input_text])[0]
        classes = list(model.classes_)
        print("Model classes order:", classes)
        print("Predict_proba:", proba)
    except Exception as e:
        print("predict_proba error:", e)

    explain_tfidf(input_text)

    qvec = related_vec.transform([input_text])
    sims = cosine_similarity(qvec, tfidf_matrix)[0]
    cand_idx = sims.argsort()[::-1][:10]
    print("\nTop candidate related articles (raw sims):")
    for idx in cand_idx:
        row = df.iloc[idx]
        text_snip = (row.get('title') or row.get('text') or "")[:80]
        print(f" idx={idx} sim={sims[idx]:.4f} title='{text_snip}' source='{row.get('source')}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_predict_arg.py \"Your news text here\"")
        sys.exit(1)
    q = sys.argv[1]
    debug(q)
