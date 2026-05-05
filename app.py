from flask import Flask, render_template, request, jsonify
import os, re, requests
from dotenv import load_dotenv
from urllib.parse import quote_plus
import joblib
import pandas as pd  

# =========================
# INIT
# =========================
load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"

print("=== FINAL DEFENSE SYSTEM (FIXED + BALANCED) ===")

# =========================
# LOAD MODEL
# =========================
data = joblib.load("improved_artifacts/baseline_model.joblib")

if isinstance(data, dict):
    if "model" in data and "vectorizer" in data:
        model = data["model"]
        vectorizer = data["vectorizer"]
    elif "clf" in data and "vectorizer" in data:
        model = data["clf"]
        vectorizer = data["vectorizer"]
    elif "pipeline" in data:
        model = data["pipeline"]
        vectorizer = None
    else:
        model = data
        vectorizer = None
else:
    model = data
    vectorizer = None

# =========================
# CLEAN
# =========================
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())

# =========================
# MODEL PREDICTION
# =========================
def model_predict(text):

    if vectorizer is None:
        df = pd.DataFrame({
            "text": [text],
            "reliability_score": [0.5],
            "is_reliable_domain": [0],
            "source_domain": ["unknown"]
        })

        pred = model.predict(df)[0]

        try:
            prob = model.predict_proba(df)[0]
            confidence = max(prob) * 100
        except:
            confidence = 85

    else:
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        pred = model.predict(vec)[0]

        try:
            prob = model.predict_proba(vec)[0]
            confidence = max(prob) * 100
        except:
            confidence = 85

    if pred == 0:
        return "real", round(confidence, 2)
    else:
        return "fake", round(confidence, 2)

# =========================
# FACT CHECK
# =========================
def get_fact_check_links(query):
    if not FACTCHECK_API_KEY:
        return []
    try:
        r = requests.get(
            FACTCHECK_BASE,
            params={"query": query, "key": FACTCHECK_API_KEY},
            timeout=5
        )
        if r.status_code != 200:
            return []
        data = r.json()
        links = []
        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                if review.get("url"):
                    links.append(review["url"])
        return links
    except:
        return []

# =========================
# GNEWS (FILTERED)
# =========================
def call_gnews(text):
    if not GNEWS_API_KEY:
        print("No GNEWS API KEY")
        return []

    try:
        keywords = clean_text(text).split()

        # FIRST QUERY
        simple_query = " ".join(keywords[:5])
        encoded_query = quote_plus(simple_query)

        url = f"{GNEWS_BASE}?q={encoded_query}&lang=en&max=5&apikey={GNEWS_API_KEY}"
        r = requests.get(url, timeout=5)

        if r.status_code != 200:
            print("ERROR:", r.text)
            return []

        data = r.json()
        articles = data.get("articles", [])

        # FALLBACK IF NO RESULTS
        if len(articles) == 0:
            simple_query = " ".join(keywords[:2])  # mas simple
            encoded_query = quote_plus(simple_query)

            url = f"{GNEWS_BASE}?q={encoded_query}&lang=en&max=5&apikey={GNEWS_API_KEY}"
            r = requests.get(url, timeout=5)

            if r.status_code == 200:
                data = r.json()
                articles = data.get("articles", [])

        # FILTERING
        keywords = [w for w in clean_text(text).split() if len(w) > 3]
        filtered_links = []

        for article in articles:
            title = clean_text(article.get("title", ""))
            match_count = sum(1 for word in keywords if word in title)

            if match_count >= 1:
                filtered_links.append(article["url"])

        return filtered_links[:2]

    except Exception as e:
        print("API ERROR:", e)
        return []
# =========================
# MAIN LOGIC
# =========================
def predict_and_retrieve(text):

    if not text.strip():
        return {"label": "error", "confidence": 0, "links": []}

    text_lower = text.lower()

    print("\n=== DEBUG ===")
    print("TEXT:", text)

    # 1. FACT CHECK
    fact_links = get_fact_check_links(text)
    if fact_links:
        return {
            "label": "fake",
            "confidence": 85,
            "links": fact_links[:2]
        }

    # 2. MODEL FIRST
    label, confidence = model_predict(text)

    # 3. GNEWS
    news_links = call_gnews(text)

    if news_links:
        return {
            "label": label,
            "confidence": confidence,
            "links": news_links[:2]
        }

    # 4. PH KEYWORDS 
    ph_keywords = [
        "pagasa", "pnp", "manila", "philippines",
        "doh", "bsp", "senate", "congress"
    ]

    if any(word in text_lower for word in ph_keywords):
        return {
            "label": "real",
            "confidence": max(confidence, 80),
            "links": []
        }

    # 5. FINAL (NO LINKS)
    return {
        "label": label,
        "confidence": confidence,
        "links": [],
        "message": "No reliable sources found for this claim."
    }

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    return jsonify(predict_and_retrieve(data.get("text", "")))

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)