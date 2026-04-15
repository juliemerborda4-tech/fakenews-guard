# convert_dataset.py
import os
import csv
import time
import math
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse, quote_plus

# Config
INPUT = "dataset.csv"      # your old mixed dataset
OUT_TRAIN = "train.csv"
OUT_VAL = "val.csv"
MIN_CHARS = 60             # drop extremely short texts
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Optional API keys (set in environment or leave empty)
GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")
FACTCHECK_KEY = os.getenv("FACTCHECK_API_KEY", "")

# Helper: normalize label to 0/1
def normalize_label(l):
    if pd.isna(l):
        return None
    s = str(l).strip().lower()
    if s in ("1","true","fake","f","false","fraud","hoax","hoaxs"):
        return 1
    if s in ("0","real","r","true_news","true news","true-positive","legit","true"):
        return 0
    # try heuristics
    if "fake" in s or "hoax" in s or "false" in s or "mislead" in s:
        return 1
    if "real" in s or "true" in s or "legit" in s or "verified" in s:
        return 0
    return None

# Optional: FactCheck API search (basic)
def search_factcheck_simple(query):
    if not FACTCHECK_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": FACTCHECK_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        j = r.json()
        return j.get("claims", [])
    except Exception:
        return []

# Optional: GNews API search (basic)
def search_gnews_simple(query, max_results=5):
    if not GNEWS_KEY:
        return {}
    url = "https://gnews.io/api/v4/search"
    params = {"q": query, "token": GNEWS_KEY, "lang": "en", "max": max_results}
    try:
        r = requests.get(url, params=params, timeout=8)
        return r.json()
    except Exception:
        return {}

def build_api_features(text_or_title):
    """
    Returns: (factcheck_bool(int 0/1), num_gnews_hits(int), top_source_score(float 0-1))
    top_source_score: simple heuristic: 1.0 if top source in whitelist else 0.0
    """
    fc = search_factcheck_simple(text_or_title) if FACTCHECK_KEY else []
    factcheck_bool = 1 if len(fc) > 0 else 0

    gnews = search_gnews_simple(text_or_title) if GNEWS_KEY else {}
    num_hits = 0
    top_score = 0.0
    try:
        articles = gnews.get("articles", [])
        num_hits = len(articles)
        # simple whitelist heuristic for credible sources; extend as needed
        whitelist = {"bbc.co.uk","cnn.com","reuters.com","nytimes.com","theguardian.com","inquirer.net","philstar.com"}
        if articles:
            top_url = articles[0].get("url","")
            if top_url:
                parsed = urlparse(top_url).netloc.lower()
                # normalize domain
                p = parsed.replace("www.","")
                if any(w in p for w in whitelist):
                    top_score = 1.0
                else:
                    top_score = 0.0
    except Exception:
        num_hits = 0
        top_score = 0.0

    return factcheck_bool, num_hits, float(top_score)

def try_extract_text_from_row(row):
    """
    Attempt to get article text:
    - prefer 'text' column
    - else 'content'
    - else 'title'
    - else 'url' (leave URL; you may fetch later)
    """
    for c in ("text","content","article_text","body","title"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    # fallback to 'url' column string (not ideal)
    if "url" in row and pd.notna(row["url"]):
        return str(row["url"]).strip()
    # try first column
    for v in row:
        if pd.notna(v) and isinstance(v,str) and len(v) > 20:
            return v
    return ""

def main():
    if not os.path.exists(INPUT):
        print(f"Input file '{INPUT}' not found. Put your old dataset as {INPUT}.")
        return

    print("Loading input CSV...")
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    print(f"Rows loaded: {len(df)}")

    # Prepare columns
    records = []
    skipped = 0
    for idx, row in df.iterrows():
        row = row.to_dict()
        text = try_extract_text_from_row(row)
        if not text or len(text) < MIN_CHARS:
            skipped += 1
            continue
        raw_label = row.get("label") or row.get("tags") or row.get("class") or ""
        label = normalize_label(raw_label)
        if label is None:
            # try infer from columns that contain real/fake words
            found = None
            for k,v in row.items():
                if isinstance(v,str) and ("fake" in v.lower() or "hoax" in v.lower()):
                    found = 1; break
                if isinstance(v,str) and ("real" in v.lower() or "verified" in v.lower()):
                    found = 0; break
            if found is None:
                # skip ambiguous entries (you can manually label later)
                skipped += 1
                continue
            label = found

        # Build API features (if API keys exist, will call; else zeros)
        try:
            fact_bool, num_hits, top_score = build_api_features(text if len(text)<400 else text[:400])
        except Exception:
            fact_bool, num_hits, top_score = 0, 0, 0.0

        records.append({
            "text": text,
            "label": int(label),
            "factcheck_bool": int(fact_bool),
            "num_gnews_hits": int(num_hits),
            "top_source_reliability": float(top_score)
        })

    print(f"Kept records: {len(records)}  Skipped: {skipped}")
    if len(records) < 10:
        print("Warning: too few usable records. Consider adding more labeled real/fake examples.")
    out_df = pd.DataFrame(records)

    # dedupe by text
    out_df.drop_duplicates(subset=["text"], inplace=True)
    out_df = out_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # split
    train_df, val_df = train_test_split(out_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=out_df["label"] if out_df["label"].nunique()>1 else None)

    print(f"Train size: {len(train_df)}  Val size: {len(val_df)}")
    train_df.to_csv(OUT_TRAIN, index=False, quoting=csv.QUOTE_MINIMAL)
    val_df.to_csv(OUT_VAL, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {OUT_TRAIN} and {OUT_VAL}. You're ready to run train_model.py")

if __name__ == "__main__":
    main()
