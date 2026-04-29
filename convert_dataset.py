import pandas as pd

df = pd.read_csv("dataset.csv")

if 'title' in df.columns and 'text' in df.columns:
    df['text'] = df['title'].fillna('') + " " + df['text'].fillna('')

df = df[['text', 'label']]

df['label'] = df['label'].str.lower()

df.to_csv("dataset_final.csv", index=False)

print("Done!")