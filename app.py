from flask import Flask, render_template, request, jsonify
from main_hybrid import predict_and_retrieve
import os

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


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


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # IMPORTANT for Render
    app.run(host="0.0.0.0", port=port)