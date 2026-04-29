import pandas as pd

# =========================
# LOAD ORIGINAL DATASET
# =========================
df = pd.read_csv("dataset.csv")

print("Original rows:", len(df))

# =========================
# USE TITLE ONLY (VERY IMPORTANT)
# =========================
# convert title to text column
df['text'] = df['title']

# =========================
# CLEAN DATA
# =========================
# remove missing values
df = df.dropna(subset=['text', 'label'])

# convert to string
df['text'] = df['text'].astype(str)

# remove empty text
df = df[df['text'].str.strip() != ""]

# lowercase (optional but recommended)
df['text'] = df['text'].str.lower()

# limit length (para consistent headlines)
df = df[df['text'].str.len() < 150]

print("Clean rows:", len(df))

# =========================
# KEEP ONLY NEEDED COLUMNS
# =========================
df = df[['text', 'label']]

# =========================
# SAVE CLEAN DATASET
# =========================
df.to_csv("clean_dataset.csv", index=False)

print("Cleaned dataset ready!")