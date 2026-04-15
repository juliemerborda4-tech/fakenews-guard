# main_hybrid.py
"""
API-first fake/real predictor (safe defaults).
- Uses Google FactCheck API (if key present)
- Uses local PH RSS feeds (cached)
- Uses GNews API as fallback
- Stricter FactCheck applicability to avoid wrong overrides
- Returns dict with: label ('fake'/'real'), fake_prob (0..1), message, related (list of {title,url,source,excerpt})
"""

import os, re, time, json, requests, feedparser
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GNEWS_BASE = "https://gnews.io/api/v4/search"
NEWSDATA_BASE = "https://newsdata.io/api/1/news"

# ---------------- safe logger ----------------
def _safe_write(file, payload):
    try:
        with open(file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------- RSS (PH news) ----------------
RSS_FEEDS = [
    "https://www.rappler.com/feed/",
    "https://newsinfo.inquirer.net/feed",
    "https://www.gmanetwork.com/news/?format=xml",
    "https://www.philstar.com/rss/headlines",
    "https://news.abs-cbn.com/rss/feeds/news.xml",
    "https://www.officialgazette.gov.ph/feeds/",
    "https://www.instagram.com/abscbnnews?utm_source=ig_web_button_share_sheet&igsh=ZDNlZDc0MzIxNw==",

    # International RSS
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.reuters.com/Reuters/worldNews",
    "https://rss.cnn.com/rss/edition_world.rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://apnews.com/rss"
    "https://www.instagram.com/inquirerdotnet?utm_source=ig_web_button_share_sheet&igsh=ZDNlZDc0MzIxNw==",
    "https://www.philstar.com",

]
_RSS_CACHE = {"timestamp": 0, "ttl": 600, "entries": []}

def fetch_rss_feeds():
    entries = []
    for u in RSS_FEEDS:
        try:
            d = feedparser.parse(u)
            for e in d.entries[:200]:
                entries.append({
                    "title": e.get("title", ""),
                    "url": e.get("link", ""),
                    "summary": e.get("summary", "") or e.get("description", ""),
                    "published": e.get("published", ""),
                    "source": (d.feed.get("title") or "RSS Source")
                })
        except Exception:
            continue
    return entries

def ensure_rss_cache():
    now = time.time()
    if now - _RSS_CACHE["timestamp"] > _RSS_CACHE["ttl"]:
        _RSS_CACHE["entries"] = fetch_rss_feeds()
        _RSS_CACHE["timestamp"] = now

def rss_match(text):
    ensure_rss_cache()
    q = (text or "").lower().strip()
    if not q:
        return []
    kws = set(re.findall(r"\w{4,}", q))
    results = []
    for e in _RSS_CACHE["entries"]:
        title = (e["title"] or "").lower()
        summary = (e["summary"] or "").lower()
        # exact phrase quick match
        if q in title or q in summary:
            results.append(e); continue
        # token overlap
        title_words = set(re.findall(r"\w{4,}", title))
        if len(title_words & kws) >= 2:
            results.append(e)
    return results[:6]

# ---------------- FactCheck API ----------------
def call_factcheck_api(text, timeout=8):
    if not FACTCHECK_API_KEY:
        return None
    try:
        r = requests.get(FACTCHECK_BASE, params={"query": text, "key": FACTCHECK_API_KEY}, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        _safe_write("debug_api_log_raw.json", {"time": time.time(), "api":"factcheck", "q": text, "resp_keys": list(j.keys()) if isinstance(j, dict) else None})
        return j
    except Exception as e:
        _safe_write("debug_api_log_raw.json", {"time": time.time(), "api":"factcheck_error", "error": str(e)})
        return None

def extract_factcheck_verdict(resp):
    if not resp:
        return None
    candidates = resp.get("claims") or resp.get("items") or []
    if isinstance(candidates, dict):
        candidates = [candidates]
    if not candidates:
        return None
    # pick best candidate with textualRating if possible
    best = None
    for c in candidates:
        reviews = c.get("claimReview") or c.get("claim_review") or c.get("claim_reviews") or c.get("reviews") or []
        if isinstance(reviews, dict):
            reviews = [reviews]
        for r in reviews:
            txt = r.get("textualRating") or r.get("textual_rating") or r.get("rating")
            if txt:
                best = (c, r); break
        if best: break
        if not best and reviews:
            best = (c, reviews[0])
        elif not best:
            best = (c, {})
    if not best:
        return None
    top, rev = best
    claim_text = top.get("text") or top.get("claim") or top.get("claimShare") or ""
    textual = (rev.get("textualRating") or rev.get("textual_rating") or rev.get("rating") or "") or ""
    url = (rev.get("url") or rev.get("link") or top.get("url") or "")
    publisher = ""
    if rev.get("publisher"):
        pub = rev.get("publisher")
        if isinstance(pub, dict):
            publisher = pub.get("name") or ""
        else:
            publisher = str(pub)
    title = rev.get("title") or top.get("title") or ""
    reviewBody = rev.get("reviewBody") or rev.get("description") or ""
    return {"claim_text": claim_text, "textualRating": textual, "url": url, "publisher": publisher, "title": title, "reviewBody": reviewBody, "raw": resp}

# ---------------- GNews ----------------
def call_gnews_api(text, max_results=6):
    if not GNEWS_API_KEY:
        return []
    try:
        params = {"q": text, "token": GNEWS_API_KEY, "max": max_results}
        r = requests.get(GNEWS_BASE, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        arts = j.get("articles") or j.get("results") or []
        _safe_write("debug_api_log_raw.json", {"time": time.time(), "api":"gnews", "q": text, "count": len(arts)})
        normalized = []
        for a in arts[:max_results]:
            src = a.get("source")
            src_name = src.get("name") if isinstance(src, dict) else (src or "")
            normalized.append({
                "title": a.get("title") or "",
                "url": a.get("url") or a.get("link") or "",
                "source": src_name,
                "desc": a.get("description") or a.get("content") or ""
            })
        return normalized
    except Exception as e:
        _safe_write("debug_api_log_raw.json", {"time": time.time(), "api":"gnews_error", "error": str(e)})
        return []

# ---------------- Utility: simple predicate and overlap ----------------
_predicate_map = {
    "alive": ["alive","is alive","survived","still alive","buhi","buhay"],
    "dead": ["dead","died","is dead","passed away","killed","deceased","patay","namatay"],
    "arrested": ["arrested","in custody","detained","arestado","dinakpan"],
    "released": ["released","freed","out of custody","pinakawalan","pinalaya"],
}
_pred_token_to_group = {}
for k,toks in _predicate_map.items():
    for t in toks:
        _pred_token_to_group[t] = k

def find_predicate_groups(text):
    s = (text or "").lower()
    found = set()
    for token, group in _pred_token_to_group.items():
        if token in s:
            found.add(group)
    return found

def predicates_conflict(groups_a, groups_b):
    if not groups_a or not groups_b: return False
    conflicts = {("alive","dead"), ("arrested","released")}
    for a in groups_a:
        for b in groups_b:
            if a == b: continue
            if (a,b) in conflicts or (b,a) in conflicts:
                return True
    return False

def claim_matches_input_semantic(fc_claim_text, user_text):
    if not fc_claim_text or not user_text: return False
    g_fc = find_predicate_groups(fc_claim_text)
    g_us = find_predicate_groups(user_text)
    if predicates_conflict(g_fc, g_us): return False
    if g_fc & g_us: return True
    # fallback: token overlap (require 2 long tokens)
    words_fc = set(re.findall(r"\w{4,}", fc_claim_text.lower()))
    words_us = set(re.findall(r"\w{4,}", user_text.lower()))
    if not words_fc or not words_us: return False
    overlap = words_fc & words_us
    if len(overlap) >= 2: return True
    for w in overlap:
        if len(w) > 5: return True
    return False

# ---------------- Main prediction function ----------------
def predict_and_retrieve(input_text, top_k=6, suppress_ui=False):
    text = (input_text or "").strip()
    print(f"[DEBUG_INPUT] {text}")
    if not text:
        return {"label":"real","fake_prob":0.5,"message":"Empty input","related":[]}

    # 1) FactCheck -> strict applicability
    fc_raw = call_factcheck_api(text)
    print("[DEBUG_FACTCHECK_RESP]", bool(fc_raw))
    fc_claim = extract_factcheck_verdict(fc_raw)
    print("[DEBUG_FACTCHECK_PARSED]", fc_claim and {k: fc_claim.get(k) for k in ("textualRating","url","claim_text")} )

    factcheck_applicable = False
    if fc_claim:
        fc_text = ((fc_claim.get("claim_text") or "") + " " + (fc_claim.get("reviewBody") or "")).strip()
        # strict: semantic match AND >=2 token overlap AND claim text reasonably long
        words_fc = set(re.findall(r"\w{4,}", (fc_text or "").lower()))
        words_us = set(re.findall(r"\w{4,}", (text or "").lower()))
        overlap = words_fc & words_us
        if claim_matches_input_semantic(fc_text, text) and len(overlap) >= 2 and len(fc_text) > 30:
            factcheck_applicable = True
        else:
            _safe_write("debug_api_log.json", {
                "note":"factcheck_mismatch_or_too_weak",
                "input":text,
                "fc_title": fc_claim.get("title"),
                "fc_text_len": len(fc_text or ""),
                "overlap_count": len(overlap)
            })

    if factcheck_applicable and fc_claim:
        textual = (fc_claim.get("textualRating") or "") or ""
        tr = textual.lower()
        related = []
        # include factcheck url as first related
        fc_url = fc_claim.get("url") or fc_claim.get("raw", {}).get("url") or ""
        if fc_url:
            related.append({"title": fc_claim.get("title") or "Fact-check", "url": fc_url, "source": fc_claim.get("publisher") or "FactCheck", "desc": fc_claim.get("reviewBody") or ""})
        # get GNews for context
        gnews = call_gnews_api(text, max_results=top_k)
        for a in gnews:
            if not any((a.get("url") or "") == (r.get("url") or "") for r in related):
                related.append(a)
        # decide by textual rating
        if any(k in tr for k in ["false","fabricated","misleading","pants on fire","not true"]):
            return {"label":"fake","fake_prob":0.98,"message":f"Fact-check: {textual}","related": related}
        if any(k in tr for k in ["true","mostly true","accurate","correct"]):
            return {"label":"real","fake_prob":0.02,"message":f"Fact-check: {textual}","related": related}
        # unknown textual -> fallthrough to evidence/GNews below (but include related)
    else:
        related = []

    # 2) RSS (local PH feeds)
    try:
        rss_results = rss_match(text)
    except Exception:
        rss_results = []
    print("[DEBUG_RSS_COUNT]", len(rss_results))
    if rss_results:
        normalized = [{"title": r["title"], "url": r["url"], "source": r.get("source") or "", "excerpt": (r.get("summary") or "")[:400]} for r in rss_results]
        return {"label":"real","fake_prob":0.10,"message":"Matched Philippine RSS news sources.","related": normalized}

    # 3) GNews global search
    gnews = call_gnews_api(text, max_results=top_k)
    print("[DEBUG_GNEWS_COUNT]", len(gnews))
    if not gnews:
        # no related news -> suspicious
        return {"label":"fake","fake_prob":0.85,"message":"No similar news found on trusted sources.","related":[]}
    # found related articles: return them
    return {"label":"real","fake_prob":0.15,"message":"Similar news found from news sources.","related": gnews}

# CLI quick test
if __name__ == "__main__":
    s = input("Enter news text/headline: ").strip()
    import pprint
    pprint.pprint(predict_and_retrieve(s, top_k=6))                