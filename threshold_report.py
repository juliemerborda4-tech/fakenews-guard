# threshold_report.py
import numpy as np
import pandas as pd
from predict_model import predict_with_bert
from sklearn.metrics import precision_score, recall_score, f1_score

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

print("threshold, precision, recall, f1, tp, fp, tn, fn")
for t in np.linspace(0.1, 0.95, 18):
    preds = (probs_fake >= t).astype(int)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    # confusion
    tp = int(((y_true==1) & (preds==1)).sum())
    fp = int(((y_true==0) & (preds==1)).sum())
    tn = int(((y_true==0) & (preds==0)).sum())
    fn = int(((y_true==1) & (preds==0)).sum())
    print(f"{t:.2f}, {prec:.3f}, {rec:.3f}, {f1:.3f}, {tp}, {fp}, {tn}, {fn}")
