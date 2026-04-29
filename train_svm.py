import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# LOAD KAGGLE DATA
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake['label'] = 0
df_true['label'] = 1

df_kaggle = pd.concat([df_fake, df_true], ignore_index=True)
df_kaggle = df_kaggle[['text', 'label']]

# LOAD FA-KES DATA
df_old = pd.read_csv("FA-KES-Dataset.csv", encoding='latin1')

df_old = df_old[['article_content', 'labels']]
df_old.columns = ['text', 'label']

df_old['label'] = df_old['label'].astype(str).str.lower()
df_old['label'] = df_old['label'].map({'fake': 0, 'real': 1})

# LOAD EXTRA DATA (IMPORTANT)
extra = pd.read_csv("extra_data.csv")

# COMBINE ALL DATA
data = pd.concat([df_kaggle, df_old, extra], ignore_index=True)

# CLEAN DATA
data = data.dropna()
data = data.drop_duplicates(subset='text')

# TEXT CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

data['text'] = data['text'].apply(clean_text)

# INPUT / OUTPUT
X = data['text']
y = data['label']

# VECTORIZER (IMPROVED)
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1,2)
)

X_vector = vectorizer.fit_transform(X)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# TRAIN MODEL (CHANGED)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# EVALUATE
accuracy = model.score(X_test, y_test)
print("FINAL ACCURACY:", accuracy)

# SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("TRAINING COMPLETE")