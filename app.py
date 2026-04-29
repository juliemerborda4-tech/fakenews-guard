from flask import Flask, render_template, request, jsonify
import os, re, requests, pickle
from dotenv import load_dotenv

# =========================
# INIT
# =========================
load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"

print("=== FINAL SYSTEM (SURVEY READY) ===")

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("Loaded model:", type(model))
print("Classes:", model.classes_)

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# =========================
# SVM PREDICTION (SAFE FIX)
# =========================
def svm_predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    probs = model.predict_proba(vector)[0]
    classes = [c.lower() for c in model.classes_]

    fake_prob = probs[classes.index('fake')]
    real_prob = probs[classes.index('real')]

    return fake_prob, real_prob

# =========================
# FACT CHECK (PRIORITY)
# =========================
def get_fact_check_links(query):
    if not FACTCHECK_API_KEY:
        return []

    try:
        r = requests.get(
            FACTCHECK_BASE,
            params={"query": query, "key": FACTCHECK_API_KEY},
            timeout=8
        )

        links = []

        if r.status_code == 200:
            data = r.json()
            for claim in data.get("claims", []):
                for review in claim.get("claimReview", []):
                    url = review.get("url")
                    if url:
                        links.append(url)

        return links[:3]

    except:
        return []

# =========================
# GNEWS (FIXED PARAM + SHORT QUERY)
# =========================
def call_gnews(text):
    if not GNEWS_API_KEY:
        return []

    try:
        simple_query = " ".join(clean_text(text).split()[:5])

        r = requests.get(
            GNEWS_BASE,
            params={
                "q": simple_query,
                "lang": "en",
                "max": 5,
                "apikey": GNEWS_API_KEY
            },
            timeout=8
        )

        if r.status_code != 200:
            print("GNEWS ERROR:", r.text)
            return []

        data = r.json()

        return [a.get("url") for a in data.get("articles", []) if a.get("url")]

    except Exception as e:
        print("GNEWS EXCEPTION:", e)
        return []

# =========================
# MAIN LOGIC (FINAL)
# =========================
def predict_and_retrieve(text):

    if not text.strip():
        return {
            "label": "error",
            "confidence": 50,
            "links": []
        }

    fake_prob, real_prob = svm_predict(text)

    text_lower = text.lower()

    # =========================
    # HARD FILTER (ANTI-ALIENS BUG)
    # =========================
    fake_triggers = [
        "alien", "aliens", "ufo", "dragon", "zombie",
        "time travel", "immortal", "teleport"
    ]

    if any(word in text_lower for word in fake_triggers):
        final_label = "fake"
        confidence = 90
        links = []
        return {
            "label": final_label,
            "confidence": confidence,
            "links": links
        }

    # =========================
    # BOOST REAL KEYWORDS
    # =========================
    real_keywords = [
        "pagasa", "government", "department",
        "president", "official", "report",
        "announced", "confirmed"
    ]

    if any(k in text_lower for k in real_keywords):
        real_prob += 0.15

    # normalize
    total = fake_prob + real_prob
    fake_prob /= total
    real_prob /= total

    # =========================
    # FINAL DECISION
    # =========================
    if real_prob >= fake_prob:
        final_label = "real"
        confidence = real_prob
    else:
        final_label = "fake"
        confidence = fake_prob

    print("\n=== DEBUG ===")
    print("TEXT:", text)
    print("FAKE:", fake_prob, "REAL:", real_prob)
    print("FINAL:", final_label)

    # =========================
    # LINKS (REAL ONLY)
    # =========================
    links = []

    if final_label == "real":

        # 1️ FACT CHECK FIRST (same sa ganina)
        links = get_fact_check_links(text)

        # 2️ FALLBACK NEWS
        if not links:
            links = call_gnews(text)

        # 3 FINAL FALLBACK (ALWAYS SHOW)
        if not links:
            links = [
                f"https://news.google.com/search?q={text.replace(' ', '+')}"
            ]

    return {
        "label": final_label,
        "confidence": round(confidence * 100, 2),
        "links": links
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
    text = data.get("text", "")
    return jsonify(predict_and_retrieve(text))

@app.route("/test")
def test():
    return "Server is working!"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)