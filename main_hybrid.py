import os, time, json, requests, feedparser
from dotenv import load_dotenv

load_dotenv()

FACTCHECK_API_KEY = os.getenv("GOOGLE_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"

# ---------------- RSS ----------------
RSS_FEEDS = [
    "https://www.rappler.com/feed/",
    "https://newsinfo.inquirer.net/feed",
    "https://www.gmanetwork.com/news/?format=xml",
    "https://www.philstar.com/rss/headlines",
    "https://news.abs-cbn.com/rss/feeds/news.xml",
]

_RSS_CACHE = {"timestamp": 0, "ttl": 600, "entries": []}

def fetch_rss_feeds():
    entries = []
    for u in RSS_FEEDS:
        try:
            d = feedparser.parse(u)
            for e in d.entries[:50]:
                entries.append({
                    "title": e.get("title", ""),
                    "url": e.get("link", "")
                })
        except:
            continue
    return entries

def ensure_rss_cache():
    if time.time() - _RSS_CACHE["timestamp"] > _RSS_CACHE["ttl"]:
        _RSS_CACHE["entries"] = fetch_rss_feeds()
        _RSS_CACHE["timestamp"] = time.time()

def rss_match(text):
    ensure_rss_cache()
    results = []
    for e in _RSS_CACHE["entries"]:
        if any(word in e["title"].lower() for word in text.lower().split()):
            results.append(e)
    return results[:5]

# ---------------- FACTCHECK ----------------
def call_factcheck_api(text):
    if not FACTCHECK_API_KEY:
        return None
    try:
        r = requests.get(FACTCHECK_BASE, params={
            "query": text,
            "key": FACTCHECK_API_KEY
        }, timeout=8)
        return r.json()
    except:
        return None

def extract_factcheck(resp):
    if not resp or "claims" not in resp:
        return None
    claim = resp["claims"][0]
    review = claim.get("claimReview", [{}])[0]
    return {
        "rating": review.get("textualRating", ""),
        "url": review.get("url", ""),
        "title": review.get("title", "")
    }

# ---------------- GNEWS ----------------
def call_gnews(text):
    if not GNEWS_API_KEY:
        return []
    try:
        r = requests.get(GNEWS_BASE, params={
            "q": text,
            "token": GNEWS_API_KEY,
            "max": 5
        })
        return r.json().get("articles", [])
    except:
        return []

# ---------------- MAIN FUNCTION ----------------
def predict_and_retrieve(text):

    if not text.strip():
        return {"label":"error","fake_prob":0.5,"message":"No input","related":[]}

    score = 0
    related = []

    # 🔥 FACTCHECK
    fc = call_factcheck_api(text)
    fc_data = extract_factcheck(fc)

    if fc_data:
        rating = fc_data["rating"].lower()
        title = fc_data["title"].lower()
        related.append(fc_data)

        words = text.lower().split()
        is_related = any(word in title for word in words)

        if is_related:
            if "false" in rating or "misleading" in rating:
                return {
                    "label":"fake",
                    "fake_prob":0.95,
                    "message":rating,
                    "related":related
                }

            if "true" in rating:
                return {
                    "label":"real",
                    "fake_prob":0.05,
                    "message":rating,
                    "related":related
                }

    # 🔥 RSS
    rss = rss_match(text)
    if rss:
        score += 2
        related.extend(rss)

    # 🔥 GNEWS
    news = call_gnews(text)
    if news:
        score += 1
        related.extend(news)

    # 🔥 FINAL DECISION
    if score >= 2:
        return {
            "label":"real",
            "fake_prob":0.2,
            "message":"Found in multiple sources",
            "related":related[:5]
        }

    elif score == 1:
        return {
            "label":"real",
            "fake_prob":0.35,
            "message":"Some sources found",
            "related":related[:5]
        }

    else:
        return {
            "label":"real",
            "fake_prob":0.55,
            "message":"No strong evidence found",
            "related":[]
        }