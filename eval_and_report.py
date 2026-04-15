# eval_and_report.py
import pandas as pd
from predict_model import predict_with_bert
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("val.csv")
y_true = df["label"].astype(int).tolist()
y_pred = []
probs_list = []

for _, row in df.iterrows():
    text = str(row["text"])
    api_feats = [float(row.get("factcheck_bool",0)), float(row.get("num_gnews_hits",0)), float(row.get("top_source_reliability",0))]
    lab, conf, probs = predict_with_bert(text, api_feats=api_feats)
    y_pred.append(int(lab))
    probs_list.append((conf, probs))

print("Samples:", len(y_true))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))

# show up to 8 FP and 8 FN
fp = []
fn = []
for i,(t,p) in enumerate(zip(y_true,y_pred)):
    if t==0 and p==1:
        fp.append((i, df.iloc[i]["text"][:400], probs_list[i]))
    if t==1 and p==0:
        fn.append((i, df.iloc[i]["text"][:400], probs_list[i]))

print("\nFalse Positives (0 labeled real but predicted fake):", len(fp))
for item in fp[:8]:
    print(item)
print("\nFalse Negatives (1 labeled fake but predicted real):", len(fn))
for item in fn[:8]:
    print(item)
