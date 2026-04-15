# debug_test.py
from main_hybrid import call_factcheck_api, extract_factcheck_verdict, call_gnews_api, rss_match
t = "Duterte is alive"
print("INPUT:", t)
fc_raw = call_factcheck_api(t)
print("FACTCHECK RAW EXISTS:", bool(fc_raw))
fc_parsed = extract_factcheck_verdict(fc_raw)
import pprint
print("FACTCHECK PARSED:")
pprint.pprint(fc_parsed)
gnews = call_gnews_api(t, max_results=6)
print("GNEWS COUNT:", len(gnews))
for a in gnews[:6]:
    print(" -", a.get("title"), "|", a.get("url"))
rss = rss_match(t)
print("RSS COUNT:", len(rss))
for r in rss[:6]:
    print(" -", r.get("title"), "|", r.get("url"))
