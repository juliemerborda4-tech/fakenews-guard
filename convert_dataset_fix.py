# convert_dataset_fix.py
import os, csv
import requests
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

# CONFIG
INPUT = "dataset.csv"
OUT_TRAIN = "train.csv"
OUT_VAL = "val.csv"
MIN_CHARS = 30          # lowered to keep more rows for demo
TEST_SIZE = 0.20
RANDOM_STATE = 42

GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")
FACTCHECK_KEY = os.getenv("FACTCHECK_API_KEY", "")

def normalize_label(l):
    if pd.isna(l) or str(l).strip()=="":
        return None
    s = str(l).strip().lower()
    if s in ("1","true","fake","f","false","fraud","hoax"): return 1
    if s in ("0","real","r","true_news","legit","true"): return 0
    if "fake" in s or "hoax" in s or "false" in s or "mislead" in s: return 1
    if "real" in s or "true" in s or "legit" in s or "verified" in s: return 0
    return None

def search_factcheck_simple(query):
    if not FACTCHECK_KEY: return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": FACTCHECK_KEY}
    try:
        r = requests.get(url, params=params, timeout=8).json()
        return r.get("claims", [])
    except Exception:
        return []

def search_gnews_simple(query, max_results=5):
    if not GNEWS_KEY: return {}
    url = "https://gnews.io/api/v4/search"
    params = {"q": query, "token": GNEWS_KEY, "lang": "en", "max": max_results}
    try:
        return requests.get(url, params=params, timeout=8).json()
    except Exception:
        return {}

def build_api_features(text_or_title):
    fc = search_factcheck_simple(text_or_title) if FACTCHECK_KEY else []
    factcheck_bool = 1 if len(fc) > 0 else 0
    gnews = search_gnews_simple(text_or_title) if GNEWS_KEY else {}
    num_hits = 0; top_score = 0.0
    try:
        articles = gnews.get("articles", [])
        num_hits = len(articles)
        whitelist = {"bbc.co.uk","cnn.com","reuters.com","nytimes.com","theguardian.com","inquirer.net","philstar.com"}
        if articles:
            top_url = articles[0].get("url","")
            parsed = urlparse(top_url).netloc.lower() if top_url else ""
            p = parsed.replace("www.","")
            if any(w in p for w in whitelist):
                top_score = 1.0
    except Exception:
        num_hits = 0; top_score = 0.0
    return int(factcheck_bool), int(num_hits), float(top_score)

def try_extract_text_from_row(row):
    for c in ("text","content","article_text","body","title"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    if "url" in row and pd.notna(row["url"]):
        return str(row["url"]).strip()
    # fallback to first long string cell
    for v in row:
        if isinstance(v,str) and len(v) > 20:
            return v
    return ""

def main():
    if not os.path.exists(INPUT):
        print(f"Missing {INPUT}")
        return
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    records = []
    skipped = 0
    for idx, row in df.iterrows():
        rowd = row.to_dict()
        text = try_extract_text_from_row(rowd)
        # remove obvious header-like rows
        if text.strip().lower() in ("text","title","content","article_text"):
            skipped += 1
            continue
        if not text or len(text) < MIN_CHARS:
            skipped += 1
            continue
        raw_label = None
        for c in ("label","tags","class"):
            if c in rowd and rowd[c]:
                raw_label = rowd[c]; break
        label = normalize_label(raw_label)
        if label is None:
            # try to infer from any column that contains fake/real words
            found = None
            for k,v in rowd.items():
                if isinstance(v,str) and ("fake" in v.lower() or "hoax" in v.lower() or "false" in v.lower()):
                    found = 1; break
                if isinstance(v,str) and ("real" in v.lower() or "verified" in v.lower() or "legit" in v.lower()):
                    found = 0; break
            if found is None:
                skipped += 1
                continue
            label = found
        try:
            fc, nh, ts = build_api_features(text[:400])
        except Exception:
            fc, nh, ts = 0, 0, 0.0
        records.append({"text": text, "label": int(label), "factcheck_bool": fc, "num_gnews_hits": nh, "top_source_reliability": ts})
    out_df = pd.DataFrame(records)
    print(f"Rows loaded: {len(df)}  Kept: {len(out_df)}  Skipped: {skipped}")
    if len(out_df) < 10:
        print("Warning: very few usable rows after cleaning. Consider adding more labeled samples.")
    # dedupe
    out_df.drop_duplicates(subset=["text"], inplace=True)
    out_df = out_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ensure each class has at least 2 samples (upsample small classes)
    label_counts = out_df["label"].value_counts().to_dict()
    if label_counts:
        min_count = min(label_counts.values())
    else:
        min_count = 0
    if min_count < 2:
        rows = []
        for label_val, cnt in label_counts.items():
            if cnt < 2:
                cls_rows = out_df[out_df["label"] == label_val]
                need = 2 - cnt
                extra = cls_rows.sample(n=need, replace=True, random_state=RANDOM_STATE)
                rows.append(extra)
        if rows:
            upsampled = pd.concat(rows, ignore_index=True)
            out_df = pd.concat([out_df, upsampled], ignore_index=True)
            out_df = out_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # stratified split if possible
    if out_df["label"].nunique() > 1:
        train_df, val_df = train_test_split(out_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=out_df["label"])
    else:
        train_df, val_df = train_test_split(out_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_df.to_csv(OUT_TRAIN, index=False, quoting=csv.QUOTE_MINIMAL)
    val_df.to_csv(OUT_VAL, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {OUT_TRAIN} ({len(train_df)}) and {OUT_VAL} ({len(val_df)})")

if __name__ == "__main__":
    main()
