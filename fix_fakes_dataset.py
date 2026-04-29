import pandas as pd

df = pd.read_csv("FA-KES-Dataset.csv", encoding="latin1")

print("ORIGINAL DATA:", len(df))

# combine title + content
df["text"] = df["article_title"].fillna('') + " " + df["article_content"].fillna('')

# FIX LABEL TYPE
df["label"] = df["labels"].astype(str).str.lower().str.strip()

# keep only needed
df = df[["text", "label"]]

# filter only fake/real
df = df[df["label"].isin(["fake", "real"])]

df.to_csv("dataset.csv", index=False)

print("FINAL CLEAN DATA:", len(df))