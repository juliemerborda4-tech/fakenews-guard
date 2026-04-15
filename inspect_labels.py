# inspect_labels.py
import pandas as pd

def normalize_label(l):
    if pd.isna(l) or str(l).strip()=="":
        return None
    s = str(l).strip().lower()
    if s in ("1","true","fake","f","false","fraud","hoax"): return 1
    if s in ("0","real","r","true_news","legit","true"): return 0
    if "fake" in s or "hoax" in s or "false" in s or "mislead" in s: return 1
    if "real" in s or "true" in s or "legit" in s or "verified" in s: return 0
    return None

df = pd.read_csv("dataset.csv", dtype=str, keep_default_na=False)
labels = []
raws = []
for idx, row in df.iterrows():
    raw = None
    for col in ("label","tags","class"):
        if col in row and row[col]:
            raw = row[col]
            break
    labels.append(normalize_label(raw))
    raws.append((idx, raw, row.to_dict()))

s = pd.Series(labels)
print("Total rows:", len(df))
print("Label counts (after quick normalize):")
print(s.value_counts(dropna=False))
print("\n(Where 0=real, 1=fake, NaN=unknown/ambiguous)\n")

# Print up to 10 ambiguous rows for quick manual review
amb = [r for r,lab in zip(raws, labels) if lab is None]
print(f"Ambiguous / unlabeled rows: {len(amb)} — showing up to 10 examples:\n")
for i, (idx, raw, rowdict) in enumerate(amb[:10]):
    print(f"ROW {idx}: inferred_label=None, raw_label_field={raw}")
    keys = list(rowdict.keys())
    preview_keys = [k for k in ("text","title","content","url") if k in rowdict] + keys[:3]
    for k in preview_keys:
        if k in rowdict:
            v = rowdict[k]
            if isinstance(v,str) and len(v)>200:
                v = v[:200].replace("\n"," ") + "..."
            print(f"  {k}: {v}")
    print("-"*50)
