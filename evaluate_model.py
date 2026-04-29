import pandas as pd
import pickle
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# LOAD DATASET (adjust file name if needed)
df = pd.read_csv("dataset.csv")

# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# PREPARE DATA
X = df['text'].apply(clean_text)

# FIX LABELS
y = df['label'].str.lower().map({
    "fake": 0,
    "real": 1
})

# REMOVE INVALID DATA
y = y.dropna()
X = X[y.index]

X_vec = vectorizer.transform(X)

# PREDICT
y_pred = model.predict(X_vec)
y_pred = y_pred.astype(int)

# =========================
# RESULTS
# =========================

accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print("\n=== ACCURACY ===")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\n=== CONFUSION MATRIX ===")
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
print(report)