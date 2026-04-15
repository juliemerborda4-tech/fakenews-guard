# build_dataset.py
import pandas as pd, random, csv
from sklearn.model_selection import train_test_split

INPUT = "dataset.csv"   # your existing file
OUT_TRAIN = "train.csv"
OUT_VAL = "val.csv"
TARGET_TOTAL_PER_CLASS = 300  # change if you want different size
VAL_RATIO = 0.2
RANDOM_STATE = 42

def normalize_label(s):
    if pd.isna(s): return None
    x = str(s).strip().lower()
    if x in ("fake","1","f","hoax","false","fraud","misleading"):
        return 1
    if x in ("real","0","r","true","legit","verified"):
        return 0
    # heuristics
    if "fake" in x or "hoax" in x or "mislead" in x: return 1
    if "real" in x or "true" in x or "verified" in x: return 0
    return None

df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
# try common columns
text_col = None
for c in ("text","content","headline","title"):
    if c in df.columns:
        text_col = c; break
if not text_col:
    text_col = df.columns[0]

labels = []
rows_by_label = {0:[], 1:[]}
for _, r in df.iterrows():
    lbl = None
    for c in ("label","class","tags"):
        if c in df.columns and r.get(c):
            lbl = normalize_label(r.get(c)); break
    if lbl is None:
        # try to infer if 'fake' appears in row
        for v in r.values:
            if isinstance(v,str) and ("fake" in v.lower() or "hoax" in v.lower()):
                lbl = 1; break
            if isinstance(v,str) and ("real" in v.lower() or "verified" in v.lower()):
                lbl = 0; break
    if lbl is None:
        continue
    text = str(r.get(text_col)).strip()
    if not text: continue
    rows_by_label[lbl].append(text)

print("Found:", {k:len(v) for k,v in rows_by_label.items()})

# Synthetic augmentation if needed (simple paraphrase via shuffling small clauses)
def augment_text(t):
    parts = [p.strip() for p in t.split(",") if p.strip()]
    random.shuffle(parts)
    return ", ".join(parts) if parts else t

balanced = {0:[], 1:[]}
for label in (0,1):
    pool = rows_by_label[label]
    if not pool:
        raise SystemExit(f"No examples for label {label} - you need to add seed examples.")
    # sample with replacement if not enough
    while len(balanced[label]) < TARGET_TOTAL_PER_CLASS:
        s = random.choice(pool)
        if random.random() < 0.25:
            s = augment_text(s)
        balanced[label].append(s)

# combine and split
all_rows = []
for lbl in (0,1):
    for t in balanced[lbl]:
        all_rows.append({"text": t, "label": lbl})

random.shuffle(all_rows)
df2 = pd.DataFrame(all_rows)
train_df, val_df = train_test_split(df2, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=df2["label"])

train_df.to_csv(OUT_TRAIN, index=False, quoting=csv.QUOTE_MINIMAL)
val_df.to_csv(OUT_VAL, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"Saved {OUT_TRAIN} ({len(train_df)}) and {OUT_VAL} ({len(val_df)})")

