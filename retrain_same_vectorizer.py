# retrain_same_vectorizer.py
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

CSV_PATH = 'dataset.csv'
RELATED_VECT = 'related_vectorizer.pkl'  # created by build_index.py
MODEL_FILE = 'model.pkl'
VECT_FILE = 'vectorizer.pkl'  # saved for classification pipeline

def main():
    df = pd.read_csv(CSV_PATH)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise SystemExit("dataset.csv must contain text and label columns")

    X = df['text'].astype(str)
    y = df['label'].astype(str)

    # Load the combined vectorizer (FeatureUnion)
    vectorizer = joblib.load(RELATED_VECT)

    # Instead of calling fit_transform again, we want to use the same vectorizer object.
    # Build a pipeline that uses the already-fit vectorizer (vectorizer is already fitted by build_index.py).
    # But CalibratedClassifierCV expects to call fit on the pipeline; so we will attach the vectorizer
    # (which is a FeatureUnion) and let pipeline refit it (safe) OR fit classifier directly on transformed features.

    # Simpler: transform X using loaded vectorizer, then train classifier on dense/sparse matrix:
    X_vec = vectorizer.transform(X)

    # Train a linear SVM with probability and calibrate
    svc = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    calibrated = CalibratedClassifierCV(svc, cv=2, method='sigmoid')

    print("Training classifier on transformed features...")
    calibrated.fit(X_vec, y)

    # Save the calibrated classifier and also save the vectorizer for later quick transforms
    joblib.dump(calibrated, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)  # vectorizer saved for quick use
    print("Saved model.pkl and vectorizer.pkl")

if __name__ == "__main__":
    main()
