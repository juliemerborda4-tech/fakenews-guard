import os, json, re, requests
from dotenv import load_dotenv
import pickle

load_dotenv()
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"

# Load SVM model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# =========================
# SVM PREDICTION (FIXED)
# =========================
def svm_predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    pred = model.predict(vector)[0]

    # IMPORTANT: adjust if needed
    if pred == 1:
        label = "fake"
    else:
        label = "real"

    confidence = 0.8

    print("PRED:", pred)
    print("LABEL:", label)

    return label, confidence


# =========================
# FACT CHECK
# =========================
def call_factcheck_api(text):
    if not FACTCHECK_API_KEY:
        return None
    try:
        r = requests.get(FACTCHECK_BASE, params={"query": text, "key": FACTCHECK_API_KEY})
        return r.json()
    except:
        return None


# =========================
# NEWS SEARCH
# =========================
def call_gnews(text):
    if not GNEWS_API_KEY:
        return []
    try:
        r = requests.get(GNEWS_BASE, params={"q": text, "token": GNEWS_API_KEY})
        data = r.json()
        return data.get("articles", [])
    except:
        return []


# =========================
# MAIN FUNCTION
# =========================
def predict_and_retrieve(input_text):

    if not input_text.strip():
        return {
            "label": "unverified",
            "fake_prob": 0.5,
            "message": "Empty input",
            "related": []
        }

    ml_label, ml_prob = svm_predict(input_text)

    # FAKE → RETURN DIRECT
    if ml_label == "fake":
        return {
            "label": "fake",
            "fake_prob": ml_prob,
            "message": "Detected as FAKE by SVM model",
            "related": []
        }

    # REAL → FACT CHECK
    fc = call_factcheck_api(input_text)
    if fc and "claims" in fc:
        return {
            "label": "real",
            "fake_prob": 0.1,
            "message": "Verified by Fact Check API",
            "related": fc.get("claims", [])
        }

    # FALLBACK → NEWS
    articles = call_gnews(input_text)
    if articles:
        return {
            "label": "real",
            "fake_prob": 0.2,
            "message": "Related news articles found",
            "related": articles[:5]
        }

    return {
        "label": "real",
        "fake_prob": 0.3,
        "message": "Likely real based on SVM analysis",
        "related": []
    }


# =========================
# TEST
# =========================
if __name__ == "__main__":
    text = input("Enter news: ")
    result = predict_and_retrieve(text)
    print(json.dumps(result, indent=2))