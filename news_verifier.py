import requests

NEWS_API_KEY = "338d534f6f3940eab09966ea05b03401"

def check_google_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
    r = requests.get(url)
    data = r.json()

    if "articles" not in data:
        return 0, []

    articles = data["articles"]
    return len(articles), [a["title"] for a in articles[:5]]
