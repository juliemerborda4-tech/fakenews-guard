# build_index.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import os

CSV_PATH = 'dataset.csv'
RELATED_VECT = 'related_vectorizer.pkl'
TFIDF_MAT = 'tfidf_vectors.pkl'
DF_PICKLE = 'news_dataframe.pkl'

def main():
    df = pd.read_csv(CSV_PATH)
    if 'text' not in df.columns:
        raise SystemExit("dataset.csv must contain 'text' column")

    texts = df['text'].astype(str).tolist()

    # Create combined vectorizer: word TF-IDF + char_wb TF-IDF
    word_vect = TfidfVectorizer(ngram_range=(1,2), analyzer='word', max_features=40000)
    char_vect = TfidfVectorizer(ngram_range=(3,5), analyzer='char_wb', max_features=20000)
    vectorizer = FeatureUnion([('word', word_vect), ('char', char_vect)])

    print("Fitting combined TF-IDF vectorizer on dataset...")
    tfidf_matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, RELATED_VECT)
    joblib.dump(tfidf_matrix, TFIDF_MAT)
    df.to_pickle(DF_PICKLE)
    print("Saved related_vectorizer, tfidf matrix and news_dataframe.")

if __name__ == "__main__":
    main()
