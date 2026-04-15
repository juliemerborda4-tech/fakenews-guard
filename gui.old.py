# gui.py  (patched by assistant) - includes threshold config, factcheck cache, and local model fallback
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser, csv, datetime, os, logging, time, traceback, functools, json, pathlib, requests
from dotenv import load_dotenv

# ---------------- Setup ----------------
load_dotenv()
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# Configurable thresholds (set PRED_THRESHOLD and PRED_REAL_LOW in .env if desired)
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.25"))  # recommended: 0.25 from threshold_report
PRED_REAL_LOW = float(os.getenv("PRED_REAL_LOW", "0.15"))
FACTCHECK_CACHE_FILE = pathlib.Path("factcheck_cache.json")

# Logging
LOGFILE = "app.log"
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE,
    filemode="a",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info("Starting GUI. FACTCHECK_KEY loaded=%s, GNEWS_KEY loaded=%s, PRED_THRESHOLD=%s", bool(FACTCHECK_API_KEY), bool(GNEWS_API_KEY), PRED_THRESHOLD)

# Debug dump folder
DEBUG_DIR = pathlib.Path("backend_debug")
DEBUG_DIR.mkdir(exist_ok=True)

def dump_backend_response(prefix, obj):
    try:
        p = DEBUG_DIR / f"{prefix}_{int(time.time()*1000)}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, default=str, indent=2, ensure_ascii=False)
        logging.info("Dumped backend response to %s", p)
    except Exception:
        logging.exception("Failed to dump backend response")

def sanitize_backend_out(o):
    """Ensure backend result is a dict with expected keys."""
    try:
        if not isinstance(o, dict):
            return {"label": "unverified", "fake_prob": 0.5, "message": f"unexpected backend return type: {type(o)}", "related": []}
        label = (o.get("label") or "unverified")
        try:
            fake_prob = float(o.get("fake_prob", o.get("prob", 0.5)))
        except Exception:
            fake_prob = 0.5
        message = o.get("message") or o.get("reason") or ""
        related = o.get("related") or o.get("articles") or []
        # normalize related to list of dicts with title/url
        norm_related = []
        if isinstance(related, dict):
            related = [related]
        if isinstance(related, list):
            for a in related:
                if not isinstance(a, dict):
                    continue
                title = a.get("title") or a.get("headline") or a.get("name") or a.get("summary") or ""
                url = a.get("url") or a.get("link") or a.get("uri") or ""
                norm_related.append({"title": title, "url": url, "rating": a.get("rating") or "", "publisher": a.get("publisher") or ""})
        return {"label": str(label), "fake_prob": max(0.0, min(1.0, fake_prob)), "message": message, "related": norm_related}
    except Exception:
        logging.exception("sanitize_backend_out failed")
        return {"label": "unverified", "fake_prob": 0.5, "message": "sanitize failed", "related": []}

# ---------------- Fact Check + decision helpers ----------------
def _load_factcheck_cache():
    try:
        if FACTCHECK_CACHE_FILE.exists():
            return json.loads(FACTCHECK_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("load_factcheck_cache failed")
    return {}

def _save_factcheck_cache(cache):
    try:
        FACTCHECK_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logging.exception("save_factcheck_cache failed")

_FACTCHECK_CACHE = _load_factcheck_cache()

def fetch_factchecks_for_query(query, max_results=5):
    """Return list of dicts: [{title, url, rating, publisher}, ...] or [] on failure."""
    k = query.strip().lower()
    if k in _FACTCHECK_CACHE:
        logging.info("FactCheck cache HIT for query")
        return _FACTCHECK_CACHE[k]
    if not FACTCHECK_API_KEY:
        logging.warning("No FACTCHECK API key configured")
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": FACTCHECK_API_KEY, "pageSize": max_results}
    try:
        r = requests.get(url, params=params, timeout=6)
        if r.status_code != 200:
            logging.warning("FactCheck API status %s: %s", r.status_code, r.text[:200])
            return []
        jd = r.json()
        claims = jd.get("claims", []) or []
        out = []
        for c in claims:
            cr_list = c.get("claimReview") or []
            for cr in cr_list:
                publisher = cr.get("publisher", {}) or {}
                title = cr.get("title") or c.get("text") or cr.get("url") or ""
                urlcr = cr.get("url") or publisher.get("url") or ""
                textual = cr.get("textualRating") or cr.get("title") or ""
                out.append({
                    "title": title,
                    "url": urlcr,
                    "rating": textual,
                    "publisher": publisher.get("name") or ""
                })
        _FACTCHECK_CACHE[k] = out
        _save_factcheck_cache(_FACTCHECK_CACHE)
        logging.info("FactCheck: found %d claim reviews for query", len(out))
        return out
    except Exception as e:
        logging.exception("fetch_factchecks_for_query failed: %s", e)
        return []

def decide_using_sources_and_model(out, text):
    """
    Modified decision logic that:
    - prefers authoritative fact-checks when available
    - if model suggests fake but there are NO related sources -> mark unverified internally
    - for UI: we will convert unverified -> REAL (per user request) but keep explain for logs
    - treats some known media/handle keywords as implicit support (useful for copy-paste from Instagram)
    """
    related = out.get("related", []) or []
    msg = (out.get("message") or "").lower()
    text_lower = (text or "").lower()

    # If message indicates GNews quota or related empty, try factcheck API (cached)
    if ("quota_exceeded" in msg) or (not related):
        try:
            fc = fetch_factchecks_for_query(text, max_results=5)
            if fc:
                for f in fc:
                    title = f"{(f.get('publisher') or '')} — {(f.get('rating') or '')} — {f.get('title') or ''}"
                    url = f.get("url") or ""
                    related.append({"title": title, "url": url, "rating": f.get("rating"), "publisher": f.get("publisher")})
                out["related"] = related
        except Exception:
            logging.exception("factcheck fallback failed")

    # look for authoritative verdicts in related (claimReview ratings)
    found_false = False
    found_true = False
    for r in related:
        rating = ""
        if isinstance(r, dict):
            rating = (r.get("rating") or r.get("title") or "").lower()
        else:
            rating = str(r).lower()
        if any(k in rating for k in ("false", "fabricat", "misleading", "pants on fire", "not true", "incorrect", "hoax")):
            found_false = True
        if any(k in rating for k in ("true", "correct", "confirmed", "accurate")):
            found_true = True

    fake_prob = float(out.get("fake_prob", 0.5) or 0.5)

    # treat some known media/handle words as implicit supporting sources (helps when user copies from IG)
    implicit_sources = ("instagram", "abs-cbn", "inquirer", "gma", "rappler", "manila bulletin", "philstar", "cnnphilippines")
    has_implicit_support = any(k in text_lower for k in implicit_sources)

    # govt keywords (bias toward real)
    gov_keys = ("malacañang", "palace", "president", "doj", "pnp", "police", "department of justice", "office of the president")
    has_gov_keyword = any(k in text_lower for k in gov_keys)

    # Decision priority:
    if found_false and not found_true:
        final = "fake"
        explain = "authoritative fact-check found negative rating"
    elif found_true and not found_false:
        final = "real"
        explain = "authoritative fact-check found positive rating"
    elif found_false and found_true:
        final = "fake" if fake_prob >= max(PRED_THRESHOLD, 0.6) else "real"
        explain = "conflicting fact-checks, using model probability"
    else:
        # No authoritative fact-checks present
        if fake_prob >= PRED_THRESHOLD:
            # if we have any related OR implicit support or gov keyword -> allow FAKE
            if related or has_implicit_support:
                final = "fake"
                explain = f"model probability >= {PRED_THRESHOLD:.2f} and supporting evidence present (related/implicit)"
            else:
                # model suggests fake but no supporting sources -> keep unverified internally
                if has_gov_keyword:
                    final = "unverified"
                    explain = "model suggests fake but no external sources; gov keyword present -> manual verification"
                else:
                    final = "unverified"
                    explain = "model suggests fake but no supporting sources found; manual review suggested"
        elif fake_prob <= PRED_REAL_LOW:
            # model strongly says real
            if related or has_implicit_support or has_gov_keyword:
                final = "real"
                explain = f"model probability <= {PRED_REAL_LOW:.2f} and supporting evidence present"
            else:
                final = "unverified"
                explain = "model suggests real but no supporting sources found; manual review suggested"
        else:
            final = "unverified"
            explain = "probability borderline — manual review suggested"

    out["final_label"] = final
    out["fake_prob"] = fake_prob
    out["explain"] = explain
    out["related"] = related
    return out

# ---------------- Try to import user backends (optional) ----------------
try:
    from main_hybrid import predict_and_retrieve
    BACKEND_OK = True
    logging.info("Loaded main_hybrid.predict_and_retrieve")
except Exception as e:
    BACKEND_OK = False
    predict_and_retrieve = None
    _backend_err = e
    logging.warning("Failed to load main_hybrid: %s", e)

try:
    from decision_engine import decide_label
    DECISION_ENGINE_OK = True
    logging.info("Loaded decision_engine.decide_label")
except Exception as e:
    DECISION_ENGINE_OK = False
    decide_label = None
    _de_err = e
    logging.warning("Failed to load decision_engine: %s", e)

# Local predictor fallback
try:
    from predict_model import predict_with_bert
    LOCAL_PREDICTOR_OK = True
    logging.info("Loaded local predict_with_bert fallback")
except Exception as e:
    LOCAL_PREDICTOR_OK = False
    predict_with_bert = None
    logging.warning("Local predict_with_bert unavailable: %s", e)

# ---------------- Feedback file ----------------
FEEDBACK_FILE = "feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp_utc", "input_text", "predicted_label", "predicted_prob", "explain"])

# ---------------- UI Theme ----------------
BG = "#f3f7fb"
HEADER = "#1e3a8a"
CARD_BG = "#ffffff"
CARD_SHADOW = "#e6ebf4"
TEXTBOX_BG = "#ffffff"
TEXTBOX_FG = "#0b1220"
ACCENT = "#1e3a8a"
ACCENT_DARK = "#162c6e"
PROG_BG = "#e6eefc"
SUBTEXT = "#475569"
LABEL_TEXT = "#0b1220"

# ---------------- Utility UI functions ----------------
def _round_rect(canvas, x1, y1, x2, y2, r=12, **kwargs):
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, width=140, height=42,
                 bg=ACCENT, fg="white", hover_bg=None, radius=10, font=("Helvetica", 11, "bold")):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.command = command
        self.radius = radius
        self.bg = bg
        self.fg = fg
        self.hover_bg = hover_bg or bg
        self.font = font
        self._id = _round_rect(self, 2, 2, width-2, height-2, r=radius, fill=bg, outline="")
        self.label = self.create_text(width//2, height//2, text=text, fill=fg, font=font)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.tag_bind(self._id, "<Button-1>", self._on_click)
        self.tag_bind(self.label, "<Button-1>", self._on_click)
        self.tag_bind(self._id, "<Enter>", self._on_enter)
        self.tag_bind(self.label, "<Enter>", self._on_enter)
        self.tag_bind(self._id, "<Leave>", self._on_leave)
        self.tag_bind(self.label, "<Leave>", self._on_leave)

    def _on_click(self, event):
        if self.command:
            self.command()

    def _on_enter(self, event):
        self.itemconfig(self._id, fill=self.hover_bg)

    def _on_leave(self, event):
        self.itemconfig(self._id, fill=self.bg)

    def config_text(self, text):
        self.itemconfig(self.label, text=text)

# ---------------- Simple LRU cache for identical queries ----------------
@functools.lru_cache(maxsize=512)
def cached_query(query_normalized):
    # placeholder: real function will call the backends.
    return None

# ---------------- The main App GUI ----------------
class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("Fake News Detection System")
        root.geometry("960x560")
        root.configure(bg=BG)
        root.resizable(False, False)

        base = tk.Canvas(root, bg=BG, highlightthickness=0)
        base.pack(fill="both", expand=True)
        header_h = 72
        base.create_rectangle(0, 0, 960, header_h, fill=HEADER, outline="")
        base.create_text(28, header_h//2, anchor="w", text="Fake News Detection System",
                         fill="white", font=("Helvetica", 18, "bold"))
        base.create_text(930, header_h//2, anchor="e", text="✕", fill="#cbd5e1", font=("Helvetica", 16))

        card_w = 840
        card_h = 420
        card_x = (960 - card_w) // 2
        card_y = header_h + 18
        _round_rect(base, card_x + 6, card_y + 8, card_x + card_w + 6, card_y + card_h + 8, r=18, fill=CARD_SHADOW, outline="")
        _round_rect(base, card_x, card_y, card_x + card_w, card_y + card_h, r=18, fill=CARD_BG, outline="")

        card_frame = tk.Frame(root, bg=CARD_BG)
        base.create_window(card_x + 20, card_y + 20, anchor="nw", window=card_frame, width=card_w - 40, height=card_h - 40)

        title = tk.Label(card_frame, text="Check if news is real or fake!",
                         bg=CARD_BG, fg=LABEL_TEXT, font=("Helvetica", 22, "bold"))
        title.pack(anchor="n", pady=(4, 8))

        tb_h = 140
        tb_w = card_w - 120
        tb_container = tk.Canvas(card_frame, width=tb_w, height=tb_h, bg=CARD_BG, highlightthickness=0)
        tb_container.pack(pady=(6, 12))
        _round_rect(tb_container, 0, 0, tb_w, tb_h, r=12, fill=TEXTBOX_BG, outline="#e6eefc")
        self.textbox = tk.Text(tb_container, bd=0, padx=12, pady=10, wrap="word",
                               font=("Helvetica", 12), bg=TEXTBOX_BG, fg=TEXTBOX_FG, relief="flat", height=6)
        self.textbox.insert("1.0", "Enter the news here...")
        self.textbox.bind("<FocusIn>", self._clear_placeholder)
        tb_container.create_window(8, 8, anchor="nw", window=self.textbox, width=tb_w-16, height=tb_h-16)

        btn_row = tk.Frame(card_frame, bg=CARD_BG)
        btn_row.pack(pady=(6, 12))

        self.predict_btn = RoundedButton(btn_row, text="Predict", command=self.on_predict,
                                         width=220, height=48, bg=ACCENT, hover_bg=ACCENT_DARK)
        self.predict_btn.pack(side="left", padx=(0, 18))

        self.clear_btn = RoundedButton(btn_row, text="Clear", command=self.clear_all,
                                       width=160, height=48, bg="#ffffff", hover_bg="#f3f5fb", fg="#0b1220")
        try:
            self.clear_btn.create_rectangle(6, 6, 154, 42, outline="#e6e9f2", width=1)
        except Exception:
            pass
        self.clear_btn.pack(side="left")

        self.result_label = tk.Label(card_frame, text="", bg=CARD_BG, fg=LABEL_TEXT, font=("Helvetica", 24, "bold"))
        self.result_label.pack(pady=(8, 6))

        prog_frame = tk.Frame(card_frame, bg=CARD_BG)
        prog_frame.pack(fill="x", padx=60, pady=(6, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Blue.Horizontal.TProgressbar", troughcolor=PROG_BG, background=ACCENT, thickness=18)

        self.progress = ttk.Progressbar(prog_frame, style="Blue.Horizontal.TProgressbar",
                                        orient="horizontal", length=680, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=20)

        badge_wrapper = tk.Frame(prog_frame, bg=CARD_BG)
        badge_wrapper.pack(fill="x", pady=(8, 0))
        self.percent_badge = tk.Label(badge_wrapper, text="", bg=ACCENT, fg="white",
                                      font=("Helvetica", 12, "bold"), padx=12, pady=6)
        self.percent_badge.pack(anchor="center")
        self.percent_badge.config(text="")
        self.percent_text = tk.Label(card_frame, text="", bg=CARD_BG, fg=LABEL_TEXT, font=("Helvetica", 12, "bold"))
        self.percent_text.pack(pady=(8, 12))

        # internal state
        self.last_prediction = None
        self.popup_open = False
        self._worker_queue = []
        self._is_working = False

    def _clear_placeholder(self, event):
        cur = self.textbox.get("1.0", "end").strip()
        if cur == "Enter the news here...":
            self.textbox.delete("1.0", "end")

    def clear_all(self):
        self.textbox.delete("1.0", "end")
        self.result_label.config(text="")
        self.percent_text.config(text="")
        self.progress["value"] = 0
        self.percent_badge.config(text="", bg=ACCENT)

    def on_predict(self):
        text = self.textbox.get("1.0", "end").strip()
        if not text or text == "Enter the news here...":
            messagebox.showinfo("Input needed", "Please enter the news text.")
            return

        # visual feedback
        try:
            self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT_DARK)
        except Exception:
            pass
        self.percent_text.config(text="Analyzing...")
        self.result_label.config(text="")
        self.root.update_idletasks()

        # normalize query for cache key
        qnorm = " ".join(text.lower().split())
        cached = None
        try:
            cached = cached_query(qnorm)
        except Exception:
            logging.exception("cached_query error")

        if cached:
            logging.info("Cache hit for query")
            self._process_out(cached, text)
            try:
                self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT)
            except Exception:
                pass
            return

        # queue the work
        self._worker_queue.append((qnorm, text, time.time()))
        if not self._is_working:
            self._start_worker()

    def _start_worker(self):
        import threading, queue
        self._is_working = True
        self._result_queue = queue.Queue()

        def worker_loop():
            while self._worker_queue:
                qnorm, text, t0 = self._worker_queue.pop(0)
                start_time = time.time()
                try:
                    out = None
                    # 1) try decision_engine first (fast, API-only)
                    if DECISION_ENGINE_OK:
                        try:
                            logging.info("Calling decision_engine.decide_label()")
                            de_out = decide_label(text)
                            dump_backend_response("decision_engine", de_out)
                            if not isinstance(de_out, dict):
                                raise ValueError("decision_engine returned non-dict")
                            out = {}
                            label = (de_out.get("label") or "").strip().lower()
                            if label in ("fake",):
                                out["label"] = "fake"
                                out["fake_prob"] = float(de_out.get("prob", 1.0))
                            elif label in ("real",):
                                out["label"] = "real"
                                out["fake_prob"] = float(de_out.get("prob", 0.0))
                            elif label == "check_needed":
                                out["label"] = "unverified"
                                out["fake_prob"] = 0.5
                            else:
                                out["label"] = label or "unverified"
                                out["fake_prob"] = float(de_out.get("prob", 0.5)) if de_out.get("prob") is not None else 0.5
                            out["message"] = de_out.get("reason") or de_out.get("message") or ""
                            meta = de_out.get("meta", {})
                            sample = meta.get("sample_articles") or meta.get("articles") or []
                            related = []
                            for a in sample:
                                title = a.get("title") or a.get("headline") or a.get("name") or a.get("url") or "Related article"
                                url = a.get("url") or a.get("link") or ""
                                related.append({"title": title, "url": url, "rating": a.get("rating") or "", "publisher": a.get("publisher") or ""})
                            out["related"] = related
                            logging.info("Decision engine returned label=%s", out.get("label"))
                        except Exception as e:
                            logging.exception("decision_engine failed: %s", e)
                            out = None

                    # 2) fallback to main_hybrid (ML + API) if decision not available or failed
                    if out is None and BACKEND_OK:
                        try:
                            logging.info("Calling main_hybrid.predict_and_retrieve()")
                            try:
                                raw_out = predict_and_retrieve(text, suppress_ui=True)
                            except TypeError:
                                raw_out = predict_and_retrieve(text)
                            dump_backend_response("main_hybrid_raw", raw_out)
                            out = raw_out
                            logging.info("main_hybrid returned type=%s", type(out))
                        except Exception as e:
                            logging.exception("predict_and_retrieve failed: %s", e)
                            out = None

                    # 3) last-resort: local model fallback if available
                    if out is None:
                        if LOCAL_PREDICTOR_OK:
                            try:
                                lab, conf, probs = predict_with_bert(text, api_feats=[0,0,0])
                                out = {"label": "fake" if lab==1 else "real", "fake_prob": float(probs.get("fake", conf)), "message": "local model fallback", "related": []}
                                logging.info("Used local predict_with_bert fallback conf=%.3f", conf)
                            except Exception:
                                logging.exception("Local predictor failed")
                                out = {"label": "unverified", "fake_prob": 0.5, "message": "Local predictor failed", "related": []}
                        else:
                            logging.info("No backend produced a result; returning unverified default")
                            out = {"label": "unverified", "fake_prob": 0.5, "message": "No backend available or backend error.", "related": []}

                    # sanitize final out shape
                    out = sanitize_backend_out(out)

                    # NEW: enrich related + decide final_label using fact-check + model
                    out = decide_using_sources_and_model(out, text)

                except Exception as e:
                    logging.exception("Unexpected worker exception: %s", e)
                    out = {"label": "unverified", "fake_prob": 0.5, "message": f"Internal error: {e}", "related": [], "final_label": "unverified"}
                elapsed = time.time() - start_time

                # push result to main thread
                self._result_queue.put((out, text, elapsed))
            # mark worker finished
            self._result_queue.put(("__WORKER_DONE__", None, None))

        threading.Thread(target=worker_loop, daemon=True).start()

        def poll():
            try:
                item = self._result_queue.get_nowait()
            except Exception:
                self.root.after(100, poll)
                return
            if item[0] == "__WORKER_DONE__":
                self._is_working = False
                try:
                    self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT)
                except Exception:
                    pass
                return
            out, text, elapsed = item
            # update UI
            self._process_out(out, text, elapsed)
            # continue polling
            self.root.after(50, poll)

        self.root.after(50, poll)

    def _process_out(self, out, text, elapsed=None):
        try:
            # prefer final_label if present (set by decide_using_sources_and_model)
            out_label = (out.get("final_label") or out.get("label") or "").strip().lower()
            explain = out.get("explain", "") or out.get("message", "")
            try:
                fake_prob = float(out.get("fake_prob", 0.5))
            except Exception:
                fake_prob = 0.5
            fake_prob = max(0.0, min(1.0, fake_prob))
            percent_fake = fake_prob * 100.0
            percent_real = (1.0 - fake_prob) * 100.0

            # canonical label mapping with unverified handling
            if out_label in ("fake", "likely_fake"):
                canonical = "fake"
            elif out_label in ("real", "likely_real"):
                canonical = "real"
            elif out_label == "unverified":
                canonical = "unverified"
            else:
                # fallback: use thresholds
                if fake_prob >= PRED_THRESHOLD:
                    canonical = "fake"
                elif fake_prob <= PRED_REAL_LOW:
                    canonical = "real"
                else:
                    canonical = "unverified"

            # prepare UI message
            if canonical == "fake":
                label = "FAKE"
                color = "#dc2626"
                badge_bg = "#dc2626"
                badge_text = f"{percent_fake:.0f}%"
                desc_text = f"The news is {percent_fake:.2f}% likely fake. {explain}"
            elif canonical == "real":
                label = "REAL"
                color = "#16a34a"
                badge_bg = "#16a34a"
                badge_text = f"{percent_real:.0f}%"
                desc_text = f"The news is {percent_real:.2f}% likely real. {explain}"
            else:  # unverified
                label = "UNCERTAIN"
                color = "#b45309"
                badge_bg = "#f59e0b"
                badge_text = f"{int(percent_fake*100/100)}%"
                desc_text = f"Borderline result — manual review suggested. {explain}"

            if elapsed is None:
                elapsed_text = ""
            else:
                elapsed_text = f" — response in {elapsed:.2f}s"
            self.result_label.config(text=label, fg=color)
            self.progress["value"] = percent_fake if canonical == "fake" else percent_real

            self.percent_badge.config(text=badge_text, bg=badge_bg, fg="white")
            self.percent_text.config(text=desc_text + elapsed_text)

            # Save feedback
            self._save_feedback(canonical, text, fake_prob, explain=explain)

            # Open appropriate popup
            if canonical == "fake":
                self._open_fake_popup(out, text, fake_prob)
            elif canonical == "real":
                self._open_real_popup(out, text, fake_prob)
            else:
                # unverified: show a simple info box encouraging manual review
                if messagebox.askyesno("Uncertain result", f"Result is UNCERTAIN ({fake_prob:.2f}).\nOpen related sources (if any) for manual review?"):
                    related = out.get("related", []) or []
                    for a in related[:6]:
                        url = a.get("url") or ""
                        title = a.get("title") or ""
                        if isinstance(url, str) and url.startswith("http"):
                            webbrowser.open(url)
                        else:
                            webbrowser.open("https://www.google.com/search?q=" + title)

        except Exception as e:
            logging.exception("Error in _process_out: %s", e)
        finally:
            try:
                self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT)
            except Exception:
                pass

    def _open_real_popup(self, out, text, fake_prob):
        popup_title = "Prediction Result"
        backend_msg = out.get("message", "")
        message_text = "✔ The system identified this news as REAL."
        if backend_msg:
            message_text += "\n\n" + backend_msg
        related = out.get("related", []) or []
        if related:
            message_text += f"\n\nRelated sources found: {len(related)}. Click 'View' to open."
            if messagebox.askyesno(popup_title, message_text + "\n\nOpen related sources in browser?"):
                for a in related[:8]:
                    url = a.get("url") or ""
                    title = a.get("title") or ""
                    if isinstance(url, str) and url.startswith("http"):
                        webbrowser.open(url)
                    else:
                        webbrowser.open("https://www.google.com/search?q=" + title)
        else:
            messagebox.showinfo(popup_title, message_text)

    def _open_fake_popup(self, out, text, fake_prob):
        popup_title = "Fake News Warning"
        for w in self.root.winfo_children():
            try:
                if w.winfo_class() == "Toplevel" and getattr(w, "title", None) and w.title() == popup_title:
                    return
            except Exception:
                continue
        if getattr(self, "popup_open", False):
            return
        self.popup_open = True
        popup = tk.Toplevel(self.root)
        popup.title(popup_title)
        popup.geometry("760x460")
        popup.configure(bg=CARD_BG)
        popup.transient(self.root)
        popup.grab_set()
        def _on_close():
            try: popup.grab_release()
            except Exception: pass
            try: popup.destroy()
            except Exception: pass
            self.popup_open = False
        popup.protocol("WM_DELETE_WINDOW", _on_close)
        tk.Label(popup, text="⚠ FAKE NEWS DETECTED", fg="#dc2626", bg=CARD_BG, font=("Helvetica", 18, "bold")).pack(anchor="w", padx=16, pady=(12,4))
        backend_msg = out.get("message", "")
        if backend_msg:
            tk.Label(popup, text=backend_msg, fg=SUBTEXT, bg=CARD_BG, font=("Helvetica", 11, "italic"), wraplength=700, justify="left").pack(anchor="w", padx=16, pady=(0,8))
        else:
            tk.Label(popup, text="Verify using trusted news sources:", fg=SUBTEXT, bg=CARD_BG, font=("Helvetica", 11)).pack(anchor="w", padx=16)
        frame = tk.Frame(popup, bg=CARD_BG)
        frame.pack(fill="both", expand=True, padx=12, pady=8)
        canvas = tk.Canvas(frame, bg=CARD_BG, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        scroll_frame = tk.Frame(canvas, bg=CARD_BG)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind("<Configure>", on_configure)
        related = out.get("related", []) or []
        if not related:
            tk.Label(scroll_frame, text="No related sources found.", bg=CARD_BG, fg=SUBTEXT).pack(anchor="w", pady=6)
        else:
            for r in related:
                title = r.get("title", "Unknown title")
                url = r.get("url", "")
                rating = r.get("rating", "")
                publisher = r.get("publisher", "")
                item = tk.Frame(scroll_frame, bg=CARD_BG)
                item.pack(anchor="w", fill="x", pady=6)
                hdr = f"{publisher} — {rating}" if publisher or rating else ""
                if hdr:
                    tk.Label(item, text=hdr, bg=CARD_BG, fg=SUBTEXT, font=("Helvetica", 10, "italic")).pack(anchor="w")
                link = tk.Label(item, text=title, fg=ACCENT, bg=CARD_BG, font=("Helvetica", 11, "underline"), cursor="hand2")
                link.pack(anchor="w")
                link.bind("<Button-1>", lambda e, u=url, t=title: webbrowser.open(u if isinstance(u, str) and u.startswith("http") else "https://www.google.com/search?q=" + t))
                btn = tk.Button(item, text="Open", bg=ACCENT, fg="white", bd=0, command=lambda u=url, t=title: webbrowser.open(u if isinstance(u,str) and u.startswith("http") else "https://www.google.com/search?q="+t))
                btn.pack(anchor="e", pady=(2,0))
        tk.Button(popup, text="Close", bg="#6b7280", fg="white", width=12, bd=0, command=_on_close).pack(pady=10)
        self._center_window(popup)

    def _save_feedback(self, label, text, prob, explain=""):
        ts = datetime.datetime.utcnow().isoformat()
        try:
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ts, text, label, f"{prob:.4f}", explain])
        except Exception:
            logging.exception("Failed to save feedback")

    def _center_window(self, win):
        win.update_idletasks()
        w = win.winfo_width(); h = win.winfo_height()
        ws = win.winfo_screenwidth(); hs = win.winfo_screenheight()
        x = (ws // 2) - (w // 2); y = (hs // 2) - (h // 2)
        win.geometry(f"+{x}+{y}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()
