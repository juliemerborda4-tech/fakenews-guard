from flask import Flask, render_template, request, jsonify
import os, re, requests, pickle
from dotenv import load_dotenv
from urllib.parse import quote_plus

# =========================
# INIT
# =========================
load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"

print("=== FINAL DEFENSE SYSTEM (WORKING) ===")

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# CLEAN
# =========================
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())

# =========================
# MODEL (fallback only)
# =========================
def svm_predict(text):
    vec = vectorizer.transform([clean_text(text)])
    probs = model.predict_proba(vec)[0]
    classes = [c.lower() for c in model.classes_]
    return probs[classes.index('fake')], probs[classes.index('real')]

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
# GNEWS (FIXED QUERY)
# =========================
def call_gnews(text):
    if not GNEWS_API_KEY:
        return []
    try:
        # 🔥 SHORT + CLEAN QUERY (IMPORTANT)
        simple_query = " ".join(clean_text(text).split()[:5])
        encoded_query = quote_plus(simple_query)

        url = f"{GNEWS_BASE}?q={encoded_query}&lang=en&max=3&apikey={GNEWS_API_KEY}"

        r = requests.get(url, timeout=5)

        if r.status_code != 200:
            print("GNEWS ERROR:", r.text)
            return []

        data = r.json()
        articles = data.get("articles", [])

        print("GNEWS FOUND:", len(articles))

        return [a["url"] for a in articles]

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

    # =========================
    # 1. FACT CHECK → FAKE
    # =========================
    fact_links = get_fact_check_links(text)
    if fact_links:
        return {
            "label": "fake",
            "confidence": 95,
            "links": fact_links[:2]
        }

    # =========================
    # 2. GNEWS → REAL
    # =========================
    news_links = call_gnews(text)
    if news_links:
        return {
            "label": "real",
            "confidence": 90,
            "links": news_links[:2]
        }

    # =========================
    # 3. PH KEYWORD BOOST (VERY IMPORTANT)
    # =========================
    ph_keywords = [
        "pagasa", "pnp", "manila", "philippines",
        "doh", "bsp", "senate", "congress"
    ]

    if any(word in text_lower for word in ph_keywords):
        return {
            "label": "real",
            "confidence": 85,
            "links": []
        }

    # =========================
    # 4. MODEL FALLBACK
    # =========================
    fake_prob, real_prob = svm_predict(text)

    if real_prob > fake_prob:
        return {
            "label": "real",
            "confidence": round(real_prob * 100, 2),
            "links": []
        }
    else:
        return {
            "label": "fake",
            "confidence": round(fake_prob * 100, 2),
            "links": []
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
    app.run(debug=True)