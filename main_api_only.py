# main_api_only.py
"""
API-only fake-news helper.
Priority:
  1) Google Fact Check API (authoritative)
  2) GNews (gnews.io)
  3) NewsData.io as fallback (optional key)
Behavior:
  - If fact-check claim found -> use rating and return fact-check as top related
  - Else search news APIs (smart queries) and collect related articles
  - If trusted sources appear -> label REAL
  - If some articles but no trusted -> label UNVERIFIED
  - If none -> label FAKE (conservative)
Return schema:
  {
    "label": "real"|"fake"|"unverified",
    "fake_prob": float(0..1),
    "message": str,
    "related": [ {"title","url","source","excerpt"} ... ]
  }
Requires (in .env):
  FACTCHECK_API_KEY, GNEWS_API_KEY, optional NEWSDATA_API_KEY
"""

import os, json, re, requests, time
from urllib.parse import urlparse
from dotenv import load_dotenv

# Optional translator
try:
    from googletrans import Translator
    _TRANSLATOR = Translator()
except Exception:
    _TRANSLATOR = None

load_dotenv()
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"
NEWSDATA_BASE = "https://newsdata.io/api/1/news"  # NewsData.io API

# Trusted domains - add your local outlets here
TRUSTED_DOMAINS = {
    "rappler.com", "inquirer.net", "gmanetwork.com", "abs-cbn.com", "reuters.com",
    "cnnphilippines.com", "philstar.com", "manilatimes.net", "mb.com.ph", "philnews.com"
}

# ---------- Helpers ----------
def extract_domain(url):
    try:
        p = urlparse(url)
        d = (p.netloc or "").lower()
        return d.replace("www.", "")
    except Exception:
        return (url or "").lower()

def simple_keyword_query(text, max_words=8):
    words = re.findall(r"\w+", (text or "").lower())
    stop = {"ang","sa","ng","na","at","the","a","an","ni","si","mga","now","today"}
    kws = [w for w in words if len(w) > 2 and w not in stop]
    kws = sorted(set(kws), key=lambda w: -len(w))[:max_words]
    return " ".join(kws)

def normalize_article_gnews(a):
    title = a.get("title") or ""
    url = a.get("url") or a.get("link") or ""
    src = ""
    if isinstance(a.get("source"), dict):
        src = a.get("source", {}).get("name") or ""
    else:
        src = a.get("source") or ""
    desc = a.get("description") or a.get("content") or ""
    return {"title": title, "url": url, "source": src, "excerpt": desc}

def normalize_article_newsdata(a):
    title = a.get("title") or ""
    url = a.get("link") or a.get("url") or ""
    src = a.get("source_id") or a.get("source") or ""
    desc = a.get("description") or a.get("content") or ""
    return {"title": title, "url": url, "source": src, "excerpt": desc}

# ---------- FactCheck API ----------
def call_factcheck_api(text, timeout=8):
    if not FACTCHECK_API_KEY:
        return None
    try:
        r = requests.get(FACTCHECK_BASE, params={"query": text, "key": FACTCHECK_API_KEY}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def parse_factcheck_response(j):
    if not j:
        return None
    claims = j.get("claims") or []
    if not claims:
        return None
    top = claims[0]
    reviews = top.get("claimReview") or []
    if not reviews:
        return None
    rev = reviews[0]
    textual = rev.get("textualRating") or rev.get("title") or ""
    return {
        "textualRating": textual,
        "url": rev.get("url") or top.get("url") or "",
        "publisher": (rev.get("publisher") or {}).get("name") if rev.get("publisher") else None,
        "title": rev.get("title") or top.get("text") or "",
        "reviewBody": rev.get("reviewBody") or top.get("text") or ""
    }

def textual_rating_to_fake_prob(textual):
    if not textual:
        return None
    t = textual.lower()
    if any(k in t for k in ["true","mostly true","true -","true."]):
        return 0.02
    if any(k in t for k in ["partly true","partly false","mixture","some truth"]):
        return 0.5
    if any(k in t for k in ["false","mostly false","misleading","fabricated","fake","pants on fire","incorrect","wrong"]):
        return 0.98
    return None

# ---------- GNews ----------
def call_gnews(q, max_results=8, lang=None, timeout=8):
    if not GNEWS_API_KEY or not q:
        return []
    try:
        params = {"q": q, "token": GNEWS_API_KEY, "max": max_results}
        if lang:
            params["lang"] = lang
        r = requests.get(GNEWS_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        articles = j.get("articles") or j.get("results") or []
        return [normalize_article_gnews(a) for a in articles[:max_results]]
    except Exception:
        return []

# ---------- NewsData.io fallback ----------
def call_newsdata(q, max_results=8, lang=None, timeout=8):
    if not NEWSDATA_API_KEY or not q:
        return []
    try:
        params = {"q": q, "apiKey": NEWSDATA_API_KEY, "page": 1}
        if lang:
            params["language"] = lang
        r = requests.get(NEWSDATA_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        results = j.get("results") or []
        return [normalize_article_newsdata(a) for a in results[:max_results]]
    except Exception:
        return []

# ---------- Smart search (try multiple queries) ----------
def call_news_smart(text, max_results=8):
    attempts = []
    if text and text.strip():
        attempts.append({"q": text, "lang": None})
    kw = simple_keyword_query(text, max_words=8)
    if kw and kw != text:
        attempts.append({"q": kw, "lang": None})
    # translate to English fallback if translator available
    if _TRANSLATOR:
        try:
            tr = _TRANSLATOR.translate(text, dest="en").text
            if tr and tr.strip() and tr.lower() != text.lower():
                attempts.append({"q": tr, "lang": "en"})
                kw2 = simple_keyword_query(tr, max_words=8)
                if kw2 and kw2 != tr:
                    attempts.append({"q": kw2, "lang": "en"})
        except Exception:
            pass
    # final attempt using short keywords
    if kw:
        attempts.append({"q": kw, "lang": None})

    seen = {}
    out = []
    for att in attempts:
        q = att["q"]
        lang = att.get("lang")
        # Try GNews first
        arts = call_gnews(q, max_results=max_results, lang=lang)
        if arts:
            for a in arts:
                a["_query_used"] = q
                if a.get("url") and a["url"] not in seen:
                    seen[a["url"]] = True
                    out.append(a)
        # if not enough, try NewsData fallback
        if len(out) < 1:
            arts2 = call_newsdata(q, max_results=max_results, lang=lang)
            if arts2:
                for a in arts2:
                    a["_query_used"] = q
                    if a.get("url") and a["url"] not in seen:
                        seen[a["url"]] = True
                        out.append(a)
        if out:
            # if we found anything for this attempt, return them (prioritize first successful strategy)
            return out[:max_results]
    return out

# ---------- Evidence scoring & decision ----------
def evidence_support_and_trusted(related, top_k=5):
    if not related:
        return 0.0, []
    trusted = []
    domains_seen = set()
    for r in related[:top_k]:
        url = (r.get("url") or "").lower()
        dom = extract_domain(url)
        domains_seen.add(dom)
        for t in TRUSTED_DOMAINS:
            if t in dom:
                trusted.append(r)
                break
    trusted_score = min(1.0, len(trusted) / 2.0)  # 2 trusted -> full
    diversity_score = min(1.0, len(domains_seen) / max(1, top_k))
    score = 0.7 * trusted_score + 0.3 * diversity_score
    return float(score), trusted

# ---------- Main API-only function ----------
def predict_and_retrieve(input_text, top_k=6):
    """
    API-only predictor for GUI to call.
    Returns dict with keys: label, fake_prob, message, related (list)
    """
    if not isinstance(input_text, str) or input_text.strip() == "":
        return {"error":"empty_input","label":"unverified","fake_prob":0.5,"message":"Empty input","related":[]}

    text = input_text.strip()

    # 1) Fact-check API (authoritative)
    fc_json = call_factcheck_api(text)
    fc_claim = parse_factcheck_response(fc_json) if fc_json else None
    related = []
    if fc_claim:
        textual = fc_claim.get("textualRating")
        fc_prob = textual_rating_to_fake_prob(textual)
        if fc_prob is None:
            t = (textual or "").lower()
            fc_prob = 0.95 if "false" in t or "mislead" in t or "incorrect" in t else 0.02
        # Add fact-check article as first related entry
        related.append({
            "title": fc_claim.get("title") or "Fact-check",
            "url": fc_claim.get("url") or "",
            "source": fc_claim.get("publisher") or "FactCheck",
            "excerpt": (fc_claim.get("reviewBody") or "")[:400]
        })
        # Also retrieve news articles for context
        more = call_news_smart(text, max_results=top_k)
        for m in more:
            if m.get("url") not in [r.get("url") for r in related]:
                related.append(m)
        label = "fake" if fc_prob >= 0.5 else "real"
        message = f"Fact-check: {textual or 'rating'}"
        return {"label": label, "fake_prob": float(fc_prob), "message": message, "related": related[:top_k]}

    # 2) No fact-check -> search news
    related = call_news_smart(text, max_results=top_k)
    evidence_score, trusted_list = evidence_support_and_trusted(related, top_k=top_k)

    # Decision rules:
    if trusted_list:
        # trusted sources present -> REAL
        return {
            "label": "real",
            "fake_prob": 0.05,
            "message": "Trusted sources found (news API).",
            "related": related[:top_k]
        }
    if related:
        # some related articles but not trusted -> unverified
        return {
            "label": "unverified",
            "fake_prob": 0.5,
            "message": "Related articles found but not on trusted sources.",
            "related": related[:top_k]
        }
    # No related articles -> return FAKE by conservative default
    return {
        "label": "fake",
        "fake_prob": 0.95,
        "message": "No supporting news sources found.",
        "related": []
    }

# CLI quick test
if __name__ == "__main__":
    s = input("Enter claim or headline: ").strip()
    out = predict_and_retrieve(s, top_k=6)
    print(json.dumps(out, indent=2, ensure_ascii=False))
