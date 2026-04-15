# gui.py  (full updated)
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser, csv, datetime, os, logging, time, traceback, functools, json, pathlib, requests

# ---------------- Setup env + logging ----------------
from dotenv import load_dotenv
load_dotenv()  # optional .env for keys and config

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

LOGFILE = "app.log"
logging.basicConfig(level=logging.INFO, filename=LOGFILE, filemode="a",
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logging.getLogger("").addHandler(console)

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
    try:
        if not isinstance(o, dict):
            return {"label": "unverified", "fake_prob": 0.5,
                    "message": f"unexpected backend return type: {type(o)}", "related": []}
        label = (o.get("label") or "unverified")
        try:
            fake_prob = float(o.get("fake_prob", 0.5))
        except Exception:
            fake_prob = 0.5
        message = o.get("message") or o.get("reason") or ""
        related = o.get("related") or o.get("articles") or []
        norm_related = []
        if isinstance(related, dict):
            related = [related]
        if isinstance(related, list):
            for a in related:
                if not isinstance(a, dict):
                    continue
                title = a.get("title") or a.get("headline") or a.get("name") or a.get("summary") or ""
                url = a.get("url") or a.get("link") or a.get("uri") or ""
                norm_related.append({"title": title, "url": url, "rating": a.get("rating") or ""})
        return {"label": label, "fake_prob": max(0.0, min(1.0, fake_prob)), "message": message, "related": norm_related}
    except Exception:
        logging.exception("sanitize_backend_out failed")
        return {"label": "unverified", "fake_prob": 0.5, "message": "sanitize failed", "related": []}


# ---------------- Fact-check helper ----------------
def fetch_factchecks_for_query(query, max_results=5):
    if not FACTCHECK_API_KEY:
        logging.debug("No FACTCHECK API key configured")
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
                out.append({"title": title, "url": urlcr, "rating": textual, "publisher": publisher.get("name") or ""})
        logging.info("FactCheck: found %d claim reviews for query", len(out))
        return out
    except Exception:
        logging.exception("fetch_factchecks_for_query failed")
        return []


# ---------------- Decision logic (keeps only REAL / FAKE output) ----------------
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.2113"))   # if fake_prob >= this -> consider fake
PRED_REAL_LOW = float(os.getenv("PRED_REAL_LOW", "0.15"))     # if fake_prob <= this -> consider real
WAIT_BEFORE_POPUP = float(os.getenv("WAIT_BEFORE_POPUP", "1"))  # seconds to wait before popup (reduced for dev)

def decide_using_sources_and_model(out, text):
    related = out.get("related", []) or []
    msg = (out.get("message") or "").lower()
    text_lower = (text or "").lower()

    if ("quota_exceeded" in msg) or (not related):
        try:
            fc = fetch_factchecks_for_query(text, max_results=5)
            if fc:
                for f in fc:
                    title = f"{(f.get('publisher') or '')} — {(f.get('rating') or '')} — {f.get('title') or ''}"
                    url = f.get("url") or ""
                    related.append({"title": title, "url": url, "rating": f.get("rating")})
                out["related"] = related
        except Exception:
            logging.exception("factcheck fallback failed")

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

    implicit_sources = ("instagram", "abs-cbn", "inquirer", "gma", "rappler", "manila bulletin", "philstar", "cnnphilippines")
    has_implicit_support = any(k in text_lower for k in implicit_sources)

    gov_keys = ("malacañang", "palace", "president", "doj", "pnp", "police", "department of justice", "office of the president")
    has_gov_keyword = any(k in text_lower for k in gov_keys)

    explain = ""
    if found_false and not found_true:
        final = "fake"
        explain = "authoritative fact-check indicates false"
    elif found_true and not found_false:
        final = "real"
        explain = "authoritative fact-check indicates true"
    elif found_false and found_true:
        final = "fake" if fake_prob >= max(PRED_THRESHOLD, 0.6) else "real"
        explain = "conflicting fact-checks, using model probability"
    else:
        if fake_prob >= PRED_THRESHOLD:
            final = "fake"
            explain = f"model predicts fake (p>={PRED_THRESHOLD})"
        elif fake_prob <= PRED_REAL_LOW:
            final = "real"
            explain = f"model predicts real (p<={PRED_REAL_LOW})"
        else:
            final = "fake" if fake_prob > 0.5 else "real"
            explain = "probability borderline — choosing the closer class"

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
    logging.info("main_hybrid not loaded: %s", e)

# ---------------- Local distil predictor (DISABLED) ----------------
# IMPORTANT: we disable local model to avoid it overriding API-based decisions.
LOCAL_PRED_OK = False
predict_with_distilbert = None
logging.info("Local Distil model disabled to prioritize API/RSS/FactCheck results")

# ---------------- Feedback file ----------------
FEEDBACK_FILE = "feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp_utc", "input_text", "predicted_label", "predicted_prob", "explain"])

# ---------------- UI Theme ----------------
BG = "#f3f7fb"
HEADER = "#1e3a8a"
CARD_BG = "#ffffff"
TEXTBOX_BG = "#ffffff"
TEXTBOX_FG = "#0b1220"
ACCENT = "#1e3a8a"
ACCENT_DARK = "#162c6e"
PROG_BG = "#e6eefc"
SUBTEXT = "#475569"
LABEL_TEXT = "#0b1220"

def _round_rect(canvas, x1, y1, x2, y2, r=12, **kwargs):
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, width=140, height=42, bg=ACCENT, fg="white", hover_bg=None, radius=10, font=("Helvetica", 11, "bold")):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.command = command
        self.radius = radius
        self.bg = bg; self.fg = fg; self.hover_bg = hover_bg or bg; self.font = font
        self._id = _round_rect(self, 2, 2, width-2, height-2, r=radius, fill=bg, outline="")
        self.label = self.create_text(width//2, height//2, text=text, fill=fg, font=font)
        self.bind("<Button-1>", self._on_click); self.bind("<Enter>", self._on_enter); self.bind("<Leave>", self._on_leave)
        self.tag_bind(self._id, "<Button-1>", self._on_click); self.tag_bind(self.label, "<Button-1>", self._on_click)
    def _on_click(self, event):
        if self.command: self.command()
    def _on_enter(self, event): self.itemconfig(self._id, fill=self.hover_bg)
    def _on_leave(self, event): self.itemconfig(self._id, fill=self.bg)
    def config_text(self, text): self.itemconfig(self.label, text=text)

# Circular spinner (Windows-safe: no capstyle)
class CircularSpinner(tk.Canvas):
    def __init__(self, parent, size=36, width=4, speed=8, color=ACCENT, bg=None):
        super().__init__(parent, width=size, height=size, bg=bg if bg is not None else parent["bg"], highlightthickness=0)
        self.size = size
        self.width = width
        self.speed = speed
        self.color = color
        self._angle = 0
        pad = 4
        self._arc = self.create_arc(pad, pad, size-pad, size-pad, start=self._angle, extent=300,
                                    style="arc", outline=self.color, width=self.width)
        self._job = None
        self._running = False

    def _step(self):
        self._angle = (self._angle + self.speed) % 360
        try:
            self.itemconfig(self._arc, start=self._angle)
        except Exception:
            pass
        self._job = self.after(40, self._step)

    def start(self):
        if self._running:
            return
        self._running = True
        self._angle = 0
        self.itemconfig(self._arc, start=self._angle)
        self._step()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._job:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

@functools.lru_cache(maxsize=512)
def cached_query(query_normalized):
    return None

class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("Fake News Detection System")
        root.geometry("960x560"); root.configure(bg=BG); root.resizable(False, False)
        base = tk.Canvas(root, bg=BG, highlightthickness=0); base.pack(fill="both", expand=True)
        header_h = 72
        base.create_rectangle(0, 0, 960, header_h, fill=HEADER, outline="")
        base.create_text(28, header_h//2, anchor="w", text="Fake News Detection System", fill="white", font=("Helvetica", 18, "bold"))
        card_w=840; card_h=420; card_x=(960-card_w)//2; card_y=header_h+18
        _round_rect(base, card_x+6, card_y+8, card_x+card_w+6, card_y+card_h+8, r=18, fill="#e6ebf4", outline="")
        _round_rect(base, card_x, card_y, card_x+card_w, card_y+card_h, r=18, fill=CARD_BG, outline="")
        card_frame = tk.Frame(root, bg=CARD_BG)
        base.create_window(card_x+20, card_y+20, anchor="nw", window=card_frame, width=card_w-40, height=card_h-40)
        title = tk.Label(card_frame, text="Check if news is real or fake!", bg=CARD_BG, fg=LABEL_TEXT, font=("Helvetica", 22, "bold"))
        title.pack(anchor="n", pady=(4,8))
        tb_h=140; tb_w=card_w-120
        tb_container = tk.Canvas(card_frame, width=tb_w, height=tb_h, bg=CARD_BG, highlightthickness=0); tb_container.pack(pady=(6,12))
        _round_rect(tb_container, 0, 0, tb_w, tb_h, r=12, fill=TEXTBOX_BG, outline="#e6eefc")
        self.textbox = tk.Text(tb_container, bd=0, padx=12, pady=10, wrap="word", font=("Helvetica", 12), bg=TEXTBOX_BG, fg=TEXTBOX_FG, relief="flat", height=6)
        self.textbox.insert("1.0", "Enter the news here...")
        self.textbox.bind("<FocusIn>", self._clear_placeholder)
        tb_container.create_window(8, 8, anchor="nw", window=self.textbox, width=tb_w-16, height=tb_h-16)

        # BUTTON ROW (predict + clear)
        btn_row = tk.Frame(card_frame, bg=CARD_BG); btn_row.pack(pady=(6,6))
        self.predict_btn = RoundedButton(btn_row, text="Predict", command=self.on_predict, width=220, height=48, bg=ACCENT, hover_bg=ACCENT_DARK)
        self.predict_btn.pack(side="left", padx=(0,18))
        self.clear_btn = RoundedButton(btn_row, text="Clear", command=self.clear_all, width=160, height=48, bg="#ffffff", hover_bg="#f3f5fb", fg="#0b1220")
        self.clear_btn.pack(side="left")

        # ---------------- progress area (RESULT + progress + percent) ----------------
        self.result_label = tk.Label(card_frame, text="", bg=CARD_BG, fg=LABEL_TEXT, font=("Helvetica", 24, "bold"))
        self.result_label.pack(pady=(8,6))
        prog_frame = tk.Frame(card_frame, bg=CARD_BG); prog_frame.pack(fill="x", padx=60, pady=(6,6))
        style = ttk.Style(); style.theme_use("clam"); style.configure("Blue.Horizontal.TProgressbar", troughcolor=PROG_BG, background=ACCENT, thickness=18)
        self.progress = ttk.Progressbar(prog_frame, style="Blue.Horizontal.TProgressbar", orient="horizontal", length=680, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=20)

        # percent badge: no colored rectangle; text will be colored green/red
        self.percent_badge = tk.Label(prog_frame, text="0%", bg=CARD_BG, fg=ACCENT, font=("Helvetica", 12, "bold"), padx=6, pady=2)
        self.percent_badge.pack(pady=(8,0))

        self.percent_text = tk.Label(card_frame, text="", bg=CARD_BG, fg=LABEL_TEXT, font=("Helvetica", 12, "bold"))
        self.percent_text.pack(pady=(8,12))

        # store original predict command & text for restore later
        self._orig_predict_cmd = self.predict_btn.command
        self._orig_predict_text = "Predict"

        # spinner container placed BELOW the progress area (so progress always visible)
        self._spinner_container = tk.Frame(card_frame, bg=CARD_BG)
        self._spinner_container.pack(pady=(4,0))

        # circular spinner widget (hidden initially)
        self._spinner = CircularSpinner(self._spinner_container, size=36, width=4, speed=12, color=ACCENT, bg=CARD_BG)
        # don't pack yet

        self.last_prediction = None; self.popup_open = False; self._worker_queue = []; self._is_working = False

    def _clear_placeholder(self, event):
        cur = self.textbox.get("1.0", "end").strip()
        if cur == "Enter the news here...":
            self.textbox.delete("1.0", "end")

    def clear_all(self):
        self.textbox.delete("1.0", "end")
        self.result_label.config(text="")
        self.percent_text.config(text="")
        self.progress["value"] = 0
        self.percent_badge.config(text="0%", bg=CARD_BG, fg=ACCENT)

    def on_predict(self):
        text = self.textbox.get("1.0", "end").strip()
        if not text or text == "Enter the news here...":
            messagebox.showinfo("Input needed", "Please enter the news text.")
            return
        try:
            self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT_DARK)
        except Exception:
            pass
        self.percent_text.config(text="Analyzing...")
        self.result_label.config(text="")
        self.root.update_idletasks()
        qnorm = " ".join(text.lower().split())
        cached = None
        try:
            cached = cached_query(qnorm)
        except Exception:
            logging.exception("cached_query error")
        if cached:
            self._process_out(cached, text)
            try:
                self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT)
            except Exception:
                pass
            return
        self._worker_queue.append((qnorm, text, time.time()))
        if not self._is_working:
            self._start_worker()

        # ---------- start UI loading state ----------
        try:
            self.predict_btn.config_text("Analyzing...")
            # temporarily disable the button's command to prevent double-clicks
            self.predict_btn.command = lambda: None
        except Exception:
            pass

        try:
            # ensure spinner is visible and centered under the progress area
            self._spinner.pack(pady=4)
            self._spinner.start()
        except Exception:
            pass
        # --------------------------------------------

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
                    # 1) main_hybrid (API / RSS / FactCheck first)
                    if BACKEND_OK:
                        try:
                            try:
                                raw_out = predict_and_retrieve(text, suppress_ui=True)
                            except TypeError:
                                raw_out = predict_and_retrieve(text)
                            dump_backend_response("main_hybrid_raw", raw_out)
                            out = raw_out
                        except Exception:
                            logging.exception("predict_and_retrieve failed")
                            out = None

                    # NOTE: Local model fallback intentionally DISABLED to avoid overriding API results
                    if out is None:
                        out = {"label": "unverified", "fake_prob": 0.5, "message": "No backend available", "related": []}

                    out = sanitize_backend_out(out)
                    out = decide_using_sources_and_model(out, text)

                except Exception:
                    logging.exception("Unexpected worker exception")
                    out = {"label": "unverified", "fake_prob": 0.5, "message": "Internal error", "related": [], "final_label": "real"}

                # WAIT BEFORE POPUP
                try:
                    wait_sec = float(os.getenv("WAIT_BEFORE_POPUP", str(WAIT_BEFORE_POPUP)))
                except Exception:
                    wait_sec = WAIT_BEFORE_POPUP
                if wait_sec < 0:
                    wait_sec = 0.0
                elif wait_sec > 120:
                    wait_sec = 120.0
                time.sleep(wait_sec)
                elapsed = time.time() - start_time + wait_sec
                self._result_queue.put((out, text, elapsed))
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
            self._process_out(out, text, elapsed)
            self.root.after(50, poll)

        self.root.after(50, poll)

    def _process_out(self, out, text, elapsed=None):
        try:
            out_label = (out.get("final_label") or out.get("label") or "").strip().lower()
            try:
                fake_prob = float(out.get("fake_prob", 0.5))
            except Exception:
                fake_prob = 0.5
            fake_prob = max(0.0, min(1.0, fake_prob))
            percent_fake = fake_prob * 100.0
            percent_real = (1.0 - fake_prob) * 100.0

            # UI mapping: only FAKE or REAL (user requested)
            if out_label in ("fake", "likely_fake"):
                canonical = "fake"
            else:
                canonical = "real"

            backend_msg = out.get("message", "")
            explain = out.get("explain", "")

            if canonical == "fake":
                label_text = "FAKE"
                color = "#dc2626"
                badge_color = "#dc2626"
                badge_text = f"{percent_fake:.0f}%"
                desc_text = f"The news is {percent_fake:.2f}% fake"
            else:
                label_text = "REAL"
                color = "#16a34a"
                badge_color = "#16a34a"
                badge_text = f"{percent_real:.0f}%"
                desc_text = f"The news is {percent_real:.2f}% real"

            elapsed_text = f" — response in {elapsed:.2f}s" if elapsed is not None else ""
            full_desc = desc_text + ((" — " + explain) if explain else "") + elapsed_text

            # update UI elements
            self.result_label.config(text=label_text, fg=color)
            self.progress["value"] = percent_fake if canonical == "fake" else percent_real

            # percent badge is text with colored fg
            self.percent_badge.config(text=badge_text, bg=CARD_BG, fg=badge_color)

            self.percent_text.config(text=full_desc)

            self._save_feedback(canonical, text, fake_prob, explain)

            if canonical == "fake":
                self._open_fake_popup(out, text, fake_prob)
            else:
                self._open_real_popup(out, text, fake_prob)

        except Exception:
            logging.exception("Error in _process_out")
        finally:
            try:
                self.predict_btn.itemconfig(self.predict_btn._id, fill=ACCENT)
            except Exception:
                pass

            # restore Predict UI state (stop spinner and restore button)
            try:
                self._spinner.stop()
                self._spinner.pack_forget()
            except Exception:
                pass

            try:
                self.predict_btn.config_text(self._orig_predict_text)
                self.predict_btn.command = self._orig_predict_cmd
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
            if messagebox.askyesno(popup_title, message_text + f"\n\nRelated sources found: {len(related)}. Open them?"):
                for a in related[:8]:
                    url = a.get("url", "")
                    title = a.get("title", "")
                    if isinstance(url, str) and url.startswith("http"):
                        webbrowser.open(url)
                    else:
                        webbrowser.open("https://www.google.com/search?q=" + title)
        else:
            messagebox.showinfo(popup_title, message_text)

    def _open_fake_popup(self, out, text, fake_prob):
        popup_title = "Fake News Warning"
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
            try:
                popup.grab_release()
            except Exception:
                pass
            try:
                popup.destroy()
            except Exception:
                pass
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
        canvas.create_window((0,0), window=scroll_frame, anchor="nw")
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
                item = tk.Frame(scroll_frame, bg=CARD_BG)
                item.pack(anchor="w", fill="x", pady=6)
                link = tk.Label(item, text=title, fg=ACCENT, bg=CARD_BG, font=("Helvetica", 11, "underline"), cursor="hand2")
                link.pack(anchor="w")
                def _open_url(e, u=url, t=title):
                    if isinstance(u, str) and u.startswith("http"):
                        webbrowser.open(u)
                    else:
                        webbrowser.open("https://www.google.com/search?q=" + t)
                link.bind("<Button-1>", _open_url)

        tk.Button(popup, text="Close", bg="#6b7280", fg="white", width=12, bd=0, command=_on_close).pack(pady=10)
        self._center_window(popup)

    def _save_feedback(self, label, text, prob, explain):
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
