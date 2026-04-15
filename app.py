from flask import Flask, render_template, request, jsonify
from main_hybrid import predict_and_retrieve
import os

# ✅ IMPORTANT: specify folders for Render
app = Flask(__name__, template_folder="templates", static_folder="static")


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
                "related": []
            })

        result = predict_and_retrieve(text)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "label": "error",
            "message": str(e),
            "fake_prob": 0.5,
            "related": []
        })


# ---------------- TEST ROUTE (DEBUG) ----------------
@app.route("/test")
def test():
    return "Server is working!"


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # for Render
    app.run(host="0.0.0.0", port=port)