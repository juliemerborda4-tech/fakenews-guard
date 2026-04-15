# api_debug_now.py
import os, json, requests
from dotenv import load_dotenv
load_dotenv()

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

TEXT = "Mindanao has no internet"   # replace if needed

print("FACTCHECK_KEY present:", bool(FACTCHECK_API_KEY))
print("GNEWS_KEY present:", bool(GNEWS_API_KEY))
print("Query text:", TEXT)
print("-" * 60)

# FactCheck
if FACTCHECK_API_KEY:
    try:
        r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                         params={"query": TEXT, "key": FACTCHECK_API_KEY}, timeout=12)
        print("FactCheck HTTP:", r.status_code)
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print("FactCheck error:", e)
else:
    print("No FactCheck key set.")

print("-" * 60)

# GNews (full text)
if GNEWS_API_KEY:
    try:
        r2 = requests.get("https://gnews.io/api/v4/search",
                          params={"q": TEXT, "token": GNEWS_API_KEY, "max": 6}, timeout=12)
        print("GNews HTTP:", r2.status_code)
        print(json.dumps(r2.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print("GNews error:", e)
else:
    print("No GNews key set.")
