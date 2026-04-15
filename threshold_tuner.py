import numpy as np
import pandas as pd
from predict_model import predict_with_bert
from sklearn.metrics import f1_score

df = pd.read_csv("val.csv")
y_true = df["label"].astype(int).tolist()
probs_fake = []

for _, row in df.iterrows():
    text = str(row["text"])
    api_feats = [float(row.get("factcheck_bool",0)), float(row.get("num_gnews_hits",0)), float(row.get("top_source_reliability",0))]
    lab, conf, probs = predict_with_bert(text, api_feats=api_feats)
    probs_fake.append(probs['fake'])

y_true = np.array(y_true)
probs_fake = np.array(probs_fake)

best_t = 0
best_f1 = 0

for t in np.linspace(0.1,0.95,85):
    preds = (probs_fake >= t).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    if f1 > best_f1:
        best_t = t
        best_f1 = f1

print("BEST THRESHOLD:", best_t)
print("BEST F1:", best_f1)
