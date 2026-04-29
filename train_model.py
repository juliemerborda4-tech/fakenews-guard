import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

print("=== TRAINING FIXED MODEL ===")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset_final.csv")

print("Total rows before cleaning:", len(df))

# =========================
# CLEAN DATA
# =========================
df = df.dropna(subset=['text', 'label'])

df['label'] = df['label'].str.lower().str.strip()
df = df[df['label'].isin(['real', 'fake'])]

print("Total rows after cleaning:", len(df))

# CHECK BALANCE
print("\nLABEL DISTRIBUTION:")
print(df['label'].value_counts())

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# =========================
# SPLIT (FIXED)
# =========================
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # VERY IMPORTANT
)

# =========================
# VECTORIZER
# =========================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nVECTOR SIZE:", X_train_vec.shape)

# =========================
# MODEL (FIXED)
# =========================
base_model = LinearSVC()

#  ADD PROBABILITY SUPPORT
model = CalibratedClassifierCV(base_model)

model.fit(X_train_vec, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n Model and vectorizer saved successfully!")