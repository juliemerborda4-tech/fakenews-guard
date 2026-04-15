import requests
from dotenv import load_dotenv
import os

load_dotenv()

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


# -----------------------
# FACT CHECK API
# -----------------------
def check_fact_check_api(query):
    url = (
        "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        f"?query={query}&key={FACTCHECK_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        data = r.json()

        if "claims" not in data:
            return None

        claim = data["claims"][0]
        review = claim["claimReview"][0]
        rating = review.get("textualRating", "").lower()

        return rating

    except Exception:
        return None


# -----------------------
# GNEWS API
# -----------------------
def check_gnews(query):
    url = (
        "https://gnews.io/api/v4/search"
        f"?q={query}&lang=en&token={GNEWS_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        data = r.json()

        if "articles" not in data or len(data["articles"]) == 0:
            return []

        return data["articles"]

    except Exception:
        return []


# -----------------------
# FINAL PREDICTION LOGIC
# -----------------------
def predict_news(text):

    # 1️⃣ CHECK FACT CHECK API
    fact = check_fact_check_api(text)

    if fact:
        if any(x in fact for x in ["false", "misleading", "fake"]):
            return {
                "result": "FAKE",
                "source": "Google Fact Check API",
                "reason": f"Fact-check rating: {fact}",
                "articles": []
            }

        if any(x in fact for x in ["true", "accurate", "correct"]):
            return {
                "result": "REAL",
                "source": "Google Fact Check API",
                "reason": f"Fact-check rating: {fact}",
                "articles": []
            }

    # 2️⃣ CHECK GNEWS
    articles = check_gnews(text)

    if len(articles) == 0:
        return {
            "result": "FAKE",
            "source": "GNews API",
            "reason": "No similar news found. Highly suspicious.",
            "articles": []
        }

    return {
        "result": "REAL",
        "source": "GNews API",
        "reason": "Similar articles found from credible sources.",
        "articles": articles
    }


if __name__ == "__main__":
    text = input("Enter news: ")
    output = predict_news(text)
    print(output)
