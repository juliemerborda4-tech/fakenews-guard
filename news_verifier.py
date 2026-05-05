import os
import requests

# Use environment variable instead of hardcoding secrets in source.
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")

def check_google_news(query):
    if not NEWS_API_KEY:
        return 0, []
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
    r = requests.get(url)
    data = r.json()

    if "articles" not in data:
        return 0, []

    articles = data["articles"]
    return len(articles), [a["title"] for a in articles[:5]]
