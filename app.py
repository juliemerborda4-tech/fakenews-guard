from flask import Flask, render_template, request, jsonify
from main_hybrid import predict_and_retrieve
import os
import requests
import urllib.parse

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------------- FACT CHECK FUNCTION ----------------
def get_fact_check_links(query):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "query": query,
        "key": "YOUR_API_KEY"  # 🔥 BUTANGI IMONG GOOGLE API KEY
    }

    try:
        response = requests.get(url, params=params)
        links = []

        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])

            for claim in claims:
                for review in claim.get("claimReview", []):
                    link = review.get("url")
                    if link:
                        links.append(link)

        return links

    except:
        return []


# ---------------- HOME ----------------
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error loading page: {str(e)}"


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({
                "label": "error",
                "message": "No input provided",
                "fake_prob": 0.5,
                "related": [],
                "links": []
            })

        # 🔥 AI RESULT
        result = predict_and_retrieve(text)

        # 🔥 FACT CHECK LINKS
        links = get_fact_check_links(text)

        # 🔥 FALLBACK (kung walay API result)
        if not links:
            query = urllib.parse.quote(text)
            links = [f"https://www.google.com/search?q={query}"]

        # 🔥 ADD LINKS TO RESULT
        result["links"] = links

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "label": "error",
            "message": str(e),
            "fake_prob": 0.5,
            "related": [],
            "links": []
        })


# ---------------- TEST ROUTE ----------------
@app.route("/test")
def test():
    return "Server is working!"


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)