import tkinter as tk
from tkinter import ttk
import random
import time
import threading
import math
from collections import deque
from datetime import datetime, timedelta


# ----------------------------------------------------------------------
# FRAUD DETECTION ENGINE
# ----------------------------------------------------------------------
class FraudDetector:
    """
    Rule-based + statistical fraud detection engine.

    In a real system this would be backed by a trained ML model
    (XGBoost, Isolation Forest, Neural Network, etc.). This engine
    simulates the same decision-making pipeline:
      1. Compute per-rule risk signals
      2. Combine weighted signals into a fraud score (0-1)
      3. Apply decision thresholds
      4. Assign a fraud category
    """

    def __init__(self):
        # Rule weights (must sum to 1.0)
        self.weights = {
            "amount":        0.22,   # unusually large amount
            "velocity":      0.20,   # too many txns in short time
            "location":      0.15,   # unusual geographic location
            "device":        0.12,   # new / untrusted device
            "time_of_day":   0.10,   # odd hour transaction
            "merchant_risk": 0.11,   # high-risk merchant category
            "frequency":     0.10,   # repetitive small txns (card testing)
        }

        # User behavioral profile (simulated baseline)
        self.user_avg_amount = 120.0
        self.user_std_amount = 60.0
        self.user_home_country = "US"
        self.user_trusted_devices = {"iPhone-14", "MacBook-Pro"}
        self.user_typical_hours = range(7, 23)  # 7am - 11pm

        # State tracking
        self.recent_txns = deque(maxlen=200)      # all recent txns
        self.txn_times = deque(maxlen=50)         # timestamps for velocity
        self.small_txns = deque(maxlen=20)        # for card-testing detection

        # Statistics
        self.total_txns = 0
        self.total_fraud = 0
        self.total_amount = 0.0
        self.fraud_amount = 0.0
        self.fraud_rings_detected = 0

        # High-risk merchant categories
        self.high_risk_merchants = {
            "Crypto Exchange", "Gambling", "Wire Transfer",
            "Prepaid Cards", "Adult Content", "Forex Trading"
        }

        # Risky countries (simulated)
        self.high_risk_countries = {"NG", "RU", "CN", "KP", "IR", "BR"}

    # ------------------------------------------------------------------
    def _score_amount(self, amount):
        """Z-score based on user's historical spending."""
        z = (amount - self.user_avg_amount) / max(1, self.user_std_amount)
        if z < 1.5:
            return 0.0
        elif z < 3:
            return 0.4
        elif z < 5:
            return 0.75
        else:
            return 1.0

    def _score_velocity(self):
        """How many transactions in the last 60 seconds."""
        now = time.time()
        recent = [t for t in self.txn_times if now - t < 60]
        n = len(recent)
        if n <= 2:
            return 0.0
        elif n <= 4:
            return 0.3
        elif n <= 7:
            return 0.65
        else:
            return 1.0

    def _score_location(self, country):
        if country == self.user_home_country:
            return 0.0
        elif country in self.high_risk_countries:
            return 1.0
        else:
            return 0.55

    def _score_device(self, device):
        return 0.0 if device in self.user_trusted_devices else 0.7

    def _score_time(self, hour):
        if hour in self.user_typical_hours:
            return 0.0
        elif hour in (5, 6, 23, 0):
            return 0.4
        else:
            return 0.8

    def _score_merchant(self, merchant):
        if merchant in self.high_risk_merchants:
            return 1.0
        elif merchant in ("Electronics", "Travel", "Luxury Goods"):
            return 0.45
        else:
            return 0.1

    def _score_frequency(self, amount):
        """Detect card-testing: many small transactions."""
        if amount < 5.0:
            self.small_txns.append(time.time())
        now = time.time()
        recent_small = [t for t in self.small_txns if now - t < 120]
        n = len(recent_small)
        if n <= 2:
            return 0.0
        elif n <= 5:
            return 0.5
        else:
            return 1.0

    # ------------------------------------------------------------------
    def _categorize(self, scores, txn):
        """Assign a human-readable fraud category."""
        worst = max(scores, key=scores.get)
        if scores[worst] < 0.4:
            return "None"

        mapping = {
            "amount":        "Unusually Large Amount",
            "velocity":      "Rapid Transaction Velocity",
            "location":      "Suspicious Geographic Location",
            "device":        "Untrusted Device",
            "time_of_day":   "Odd-Hour Transaction",
            "merchant_risk": "High-Risk Merchant",
            "frequency":     "Possible Card Testing",
        }
        return mapping[worst]

    # ------------------------------------------------------------------
    def analyze(self, txn):
        """Run all rules and return a comprehensive result dict."""
        scores = {
            "amount":        self._score_amount(txn["amount"]),
            "velocity":      self._score_velocity(),
            "location":      self._score_location(txn["country"]),
            "device":        self._score_device(txn["device"]),
            "time_of_day":   self._score_time(txn["hour"]),
            "merchant_risk": self._score_merchant(txn["merchant"]),
            "frequency":     self._score_frequency(txn["amount"]),
        }

        fraud_score = sum(scores[k] * self.weights[k] for k in scores)
        fraud_score = min(1.0, max(0.0, fraud_score))

        # Decision thresholds
        if fraud_score < 0.25:
            decision = "APPROVED"
        elif fraud_score < 0.45:
            decision = "REVIEW"
        elif fraud_score < 0.70:
            decision = "SUSPICIOUS"
        else:
            decision = "BLOCKED"

        category = self._categorize(scores, txn)

        # Update state
        self.txn_times.append(time.time())
        txn["scores"] = scores
        txn["fraud_score"] = fraud_score
        txn["decision"] = decision
        txn["category"] = category
        self.recent_txns.appendleft(txn)

        self.total_txns += 1
        self.total_amount += txn["amount"]
        if decision in ("SUSPICIOUS", "BLOCKED"):
            self.total_fraud += 1
            self.fraud_amount += txn["amount"]

        # Fraud-ring detection (multiple blocked txns same country/device)
        recent_blocked = [t for t in self.recent_txns
                          if t["decision"] == "BLOCKED"][:5]
        if len(recent_blocked) >= 3:
            countries = [t["country"] for t in recent_blocked]
            if len(set(countries)) <= 2:
                self.fraud_rings_detected += 1

        return {
            "fraud_score": fraud_score,
            "decision": decision,
            "category": category,
            "scores": scores,
        }

    # ------------------------------------------------------------------
    def stats(self):
        detection_rate = (self.total_fraud / max(1, self.total_txns)) * 100
        return {
            "total_txns": self.total_txns,
            "total_fraud": self.total_fraud,
            "total_amount": self.total_amount,
            "fraud_amount": self.fraud_amount,
            "detection_rate": detection_rate,
            "fraud_rings": self.fraud_rings_detected,
        }


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class FraudDetectionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection System")
        self.root.geometry("1300x780")
        self.root.configure(bg="#0a0e1a")
        self.root.minsize(1100, 680)

        # Engine
        self.detector = FraudDetector()
        self.running = True

        # Data
        self.score_history = deque(maxlen=100)
        self.amount_history = deque(maxlen=100)
        self.decision_history = deque(maxlen=100)
        self.alert_log = deque(maxlen=200)
        self.category_counts = {}

        # Simulated data pools
        self.countries = ["US", "US", "US", "US", "UK", "CA", "DE", "FR",
                          "NG", "RU", "CN", "BR", "IN", "AU"]
        self.devices = ["iPhone-14", "MacBook-Pro", "Android-Pixel",
                        "Unknown-Device", "Windows-PC", "Spoofed-ID",
                        "iPhone-14", "MacBook-Pro"]
        self.merchants = ["Grocery", "Restaurant", "Gas Station", "Online Retail",
                          "Electronics", "Travel", "Crypto Exchange", "Gambling",
                          "Wire Transfer", "Prepaid Cards", "Luxury Goods",
                          "Forex Trading", "Adult Content"]
        self.merchant_weights = [15, 12, 10, 12, 6, 5, 4, 3, 3, 3, 4, 2, 2]
        self.customer_names = [f"CUST-{i:04d}" for i in range(1, 500)]

        # Build UI
        self._setup_styles()
        self._build_ui()

        # Start threads
        self.thread = threading.Thread(target=self._simulate, daemon=True)
        self.thread.start()

        self._update_ui()

    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background="#0a0e1a", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#1a1f35", foreground="#94a3b8",
                        padding=[16, 7], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#2d3555")],
                  foreground=[("selected", "#e2e8f0")])
        style.configure("Treeview",
                        background="#131829", foreground="#cbd5e1",
                        fieldbackground="#131829", rowheight=26,
                        font=("Consolas", 9), borderwidth=0)
        style.configure("Treeview.Heading",
                        background="#1a1f35", foreground="#94a3b8",
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Treeview", background=[("selected", "#334155")])

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self.root, bg="#020617", height=62)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="🛡  FRAUD DETECTION SYSTEM",
                 font=("Segoe UI", 17, "bold"),
                 fg="#38bdf8", bg="#020617").pack(side=tk.LEFT, padx=20)

        self.clock = tk.Label(header, text="", font=("Consolas", 11),
                              fg="#94a3b8", bg="#020617")
        self.clock.pack(side=tk.RIGHT, padx=20)

        # ===== KPI STRIP =====
        kpi_frame = tk.Frame(self.root, bg="#0a0e1a", height=90)
        kpi_frame.pack(fill=tk.X, padx=12, pady=(10, 0))
        kpi_frame.pack_propagate(False)

        self.kpi_widgets = {}
        kpis = [
            ("TOTAL TXNS",     "0",    "#38bdf8"),
            ("FRAUD DETECTED", "0",    "#ef4444"),
            ("DETECTION RATE", "0.0%", "#fbbf24"),
            ("AMOUNT AT RISK", "$0",   "#f97316"),
            ("FRAUD RINGS",    "0",    "#a855f7"),
        ]
        for i, (title, init, color) in enumerate(kpis):
            card = tk.Frame(kpi_frame, bg="#131829")
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            kpi_frame.grid_columnconfigure(i, weight=1)

            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"),
                     fg="#64748b", bg="#131829").pack(anchor=tk.W, padx=12, pady=(8, 0))
            val = tk.Label(card, text=init, font=("Segoe UI", 20, "bold"),
                           fg=color, bg="#131829")
            val.pack(anchor=tk.W, padx=12, pady=(0, 8))
            self.kpi_widgets[title] = val

        # ===== MAIN LAYOUT =====
        main = tk.Frame(self.root, bg="#0a0e1a")
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ---- LEFT PANEL: Live Transactions ----
        left = tk.Frame(main, bg="#0a0e1a", width=520)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.pack_propagate(False)

        tk.Label(left, text="LIVE TRANSACTION STREAM",
                 font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#0a0e1a").pack(anchor=tk.W, pady=(0, 6))

        txn_frame = tk.Frame(left, bg="#131829")
        txn_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("time", "customer", "amount", "merchant", "country", "score", "status")
        self.txn_tree = ttk.Treeview(txn_frame, columns=cols, show="headings",
                                     height=22)
        headings = {
            "time": "TIME", "customer": "CUSTOMER", "amount": "AMOUNT",
            "merchant": "MERCHANT", "country": "CTY",
            "score": "SCORE", "status": "STATUS",
        }
        widths = {"time": 65, "customer": 85, "amount": 80, "merchant": 120,
                  "country": 45, "score": 60, "status": 90}
        for c in cols:
            self.txn_tree.heading(c, text=headings[c])
            self.txn_tree.column(c, width=widths[c], anchor=tk.W)

        scroll = ttk.Scrollbar(txn_frame, orient=tk.VERTICAL,
                               command=self.txn_tree.yview)
        self.txn_tree.configure(yscroll=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txn_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Tag colors
        self.txn_tree.tag_configure("approved",   foreground="#22c55e")
        self.txn_tree.tag_configure("review",     foreground="#eab308")
        self.txn_tree.tag_configure("suspicious", foreground="#f97316")
        self.txn_tree.tag_configure("blocked",    foreground="#ef4444",
                                    background="#2a0a0a")

        # ---- RIGHT PANEL: Notebook ----
        right = tk.Frame(main, bg="#0a0e1a")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Fraud Score Timeline
        tab1 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab1, text="  📈 Fraud Score  ")
        self.score_canvas = tk.Canvas(tab1, bg="#020617", highlightthickness=0)
        self.score_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 2: Decision Distribution
        tab2 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab2, text="  📊 Distribution  ")
        self.dist_canvas = tk.Canvas(tab2, bg="#020617", highlightthickness=0)
        self.dist_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 3: Rule Contribution
        tab3 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab3, text="  🧩 Rule Signals  ")
        self.rule_canvas = tk.Canvas(tab3, bg="#020617", highlightthickness=0)
        self.rule_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 4: Alerts
        tab4 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab4, text="  🚨 Alerts  ")
        self.alert_list = tk.Listbox(tab4, bg="#020617", fg="#fca5a5",
                                     font=("Consolas", 9), bd=0,
                                     highlightthickness=0,
                                     selectbackground="#334155")
        self.alert_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ===== FOOTER =====
        footer = tk.Frame(self.root, bg="#020617", height=26)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.footer = tk.Label(footer, text="Monitoring active...",
                               font=("Consolas", 9), fg="#475569", bg="#020617")
        self.footer.pack(side=tk.LEFT, padx=12)
        tk.Label(footer, text="FraudDetector v1.0 | Rule-Based Ensemble",
                 font=("Consolas", 9), fg="#475569", bg="#020617").pack(side=tk.RIGHT, padx=12)

    # ------------------------------------------------------------------
    # SIMULATION THREAD
    # ------------------------------------------------------------------
    def _simulate(self):
        while self.running:
            # 15% chance of a fraudulent-looking transaction burst
            is_fraud_attempt = random.random() < 0.15

            if is_fraud_attempt:
                amount = random.choice([
                    random.uniform(800, 2500),
                    random.uniform(2500, 8000),
                    random.uniform(0.5, 4.0),  # card testing
                ])
                country = random.choice(["NG", "RU", "CN", "BR", "KP", "IR"])
                device = random.choice(["Unknown-Device", "Spoofed-ID", "Windows-PC"])
                merchant = random.choice([
                    "Crypto Exchange", "Gambling", "Wire Transfer",
                    "Prepaid Cards", "Forex Trading", "Adult Content",
                ])
                hour = random.choice([1, 2, 3, 4, 5, 23, 0])
            else:
                amount = max(5, random.gauss(120, 60))
                country = random.choice(["US", "US", "US", "UK", "CA", "DE"])
                device = random.choice(["iPhone-14", "MacBook-Pro", "Android-Pixel"])
                merchant = random.choices(self.merchants,
                                          weights=self.merchant_weights, k=1)[0]
                hour = random.randint(7, 22)

            txn = {
                "time": datetime.now(),
                "customer": random.choice(self.customer_names),
                "amount": round(amount, 2),
                "merchant": merchant,
                "country": country,
                "device": device,
                "hour": hour,
            }

            result = self.detector.analyze(txn)
            self.score_history.append(result["fraud_score"])
            self.amount_history.append(txn["amount"])
            self.decision_history.append(result["decision"])

            # Update category counts
            if result["decision"] in ("SUSPICIOUS", "BLOCKED"):
                cat = result["category"]
                self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
                msg = (f"[{txn['time'].strftime('%H:%M:%S')}] {result['decision']} | "
                       f"{txn['customer']} | ${txn['amount']:.2f} | "
                       f"{txn['country']} | {cat} | score={result['fraud_score']:.2f}")
                self.alert_log.appendleft(msg)

            # Trigger fraud bursts
            if is_fraud_attempt and random.random() < 0.5:
                for _ in range(random.randint(2, 5)):
                    time.sleep(0.15)
                    burst = dict(txn)
                    burst["time"] = datetime.now()
                    burst["amount"] = round(random.uniform(500, 3000), 2)
                    r2 = self.detector.analyze(burst)
                    self.score_history.append(r2["fraud_score"])
                    self.amount_history.append(burst["amount"])
                    self.decision_history.append(r2["decision"])
                    if r2["decision"] in ("SUSPICIOUS", "BLOCKED"):
                        cat = r2["category"]
                        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
                        msg = (f"[{burst['time'].strftime('%H:%M:%S')}] {r2['decision']} | "
                               f"{burst['customer']} | ${burst['amount']:.2f} | "
                               f"{burst['country']} | {cat} | score={r2['fraud_score']:.2f}")
                        self.alert_log.appendleft(msg)

            time.sleep(random.uniform(0.5, 1.2))

    # ------------------------------------------------------------------
    # UI UPDATE
    # ------------------------------------------------------------------
    def _update_ui(self):
        if not self.running:
            return

        self.clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        stats = self.detector.stats()
        self.kpi_widgets["TOTAL TXNS"].config(text=f"{stats['total_txns']}")
        self.kpi_widgets["FRAUD DETECTED"].config(text=f"{stats['total_fraud']}")
        self.kpi_widgets["DETECTION RATE"].config(
            text=f"{stats['detection_rate']:.1f}%")
        self.kpi_widgets["AMOUNT AT RISK"].config(
            text=f"${stats['fraud_amount']:,.0f}")
        self.kpi_widgets["FRAUD RINGS"].config(text=f"{stats['fraud_rings']}")

        # ---- Transaction table ----
        self.txn_tree.delete(*self.txn_tree.get_children())
        for t in list(self.detector.recent_txns)[:50]:
            tag = t["decision"].lower()
            self.txn_tree.insert("", tk.END, values=(
                t["time"].strftime("%H:%M:%S"),
                t["customer"],
                f"${t['amount']:,.2f}",
                t["merchant"][:18],
                t["country"],
                f"{t['fraud_score']:.2f}",
                t["decision"],
            ), tags=(tag,))

        # ---- Alerts ----
        self.alert_list.delete(0, tk.END)
        for msg in list(self.alert_log)[:60]:
            self.alert_list.insert(tk.END, msg)

        # ---- Charts ----
        self._draw_score_chart()
        self._draw_distribution()
        self._draw_rule_signals()

        # Footer
        self.footer.config(
            text=f"Last update: {datetime.now().strftime('%H:%M:%S')}  |  "
                 f"Txns: {stats['total_txns']}  |  "
                 f"Fraud: {stats['total_fraud']}  |  "
                 f"Blocked amount: ${stats['fraud_amount']:,.2f}"
        )

        self.root.after(600, self._update_ui)

    # ------------------------------------------------------------------
    def _draw_score_chart(self):
        c = self.score_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        pad_l, pad_r, pad_t, pad_b = 55, 25, 30, 35
        cw, ch = w - pad_l - pad_r, h - pad_t - pad_b

        c.create_text(w / 2, 15, text="FRAUD SCORE TIMELINE",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        # Grid + Y labels
        for i in range(6):
            y = pad_t + ch * i / 5
            c.create_line(pad_l, y, pad_l + cw, y, fill="#1e293b")
            c.create_text(pad_l - 8, y, text=f"{1.0 - i*0.2:.1f}",
                          anchor=tk.E, fill="#64748b", font=("Consolas", 8))

        # Threshold zones
        c.create_rectangle(pad_l, pad_t, pad_l + cw, pad_t + ch * 0.30,
                           fill="#450a0a", outline="")   # BLOCKED zone
        c.create_rectangle(pad_l, pad_t + ch * 0.30, pad_l + cw, pad_t + ch * 0.55,
                           fill="#431407", outline="")   # SUSPICIOUS zone
        c.create_rectangle(pad_l, pad_t + ch * 0.55, pad_l + cw, pad_t + ch * 0.75,
                           fill="#422006", outline="")   # REVIEW zone

        c.create_text(pad_l + cw - 8, pad_t + 12, text="BLOCKED",
                      anchor=tk.E, fill="#ef4444", font=("Segoe UI", 8, "bold"))
        c.create_text(pad_l + cw - 8, pad_t + ch * 0.42, text="SUSPICIOUS",
                      anchor=tk.E, fill="#f97316", font=("Segoe UI", 8, "bold"))
        c.create_text(pad_l + cw - 8, pad_t + ch * 0.65, text="REVIEW",
                      anchor=tk.E, fill="#eab308", font=("Segoe UI", 8, "bold"))

        n = len(self.score_history)
        if n < 2:
            c.create_text(w / 2, h / 2, text="Waiting for transactions...",
                          fill="#475569", font=("Segoe UI", 12))
            return

        # Plot scores with color per decision
        prev = None
        for i, score in enumerate(self.score_history):
            x = pad_l + cw * i / (n - 1) if n > 1 else pad_l
            y = pad_t + ch - score * ch
            if prev:
                px, py, pscore = prev
                color = "#ef4444" if pscore > 0.70 else \
                        "#f97316" if pscore > 0.45 else \
                        "#eab308" if pscore > 0.25 else "#22c55e"
                c.create_line(px, py, x, y, fill=color, width=2)
            prev = (x, y, score)

        # Current marker
        if prev:
            x, y, s = prev
            color = "#ef4444" if s > 0.70 else \
                    "#f97316" if s > 0.45 else \
                    "#eab308" if s > 0.25 else "#22c55e"
            c.create_oval(x - 5, y - 5, x + 5, y + 5,
                          fill=color, outline="white", width=2)

        c.create_text(pad_l + cw / 2, h - 12, text="Time →",
                      fill="#64748b", font=("Consolas", 8))

    # ------------------------------------------------------------------
    def _draw_distribution(self):
        c = self.dist_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 20, text="DECISION DISTRIBUTION",
                      fill="#94a3b8", font=("Segoe UI", 11, "bold"))

        counts = {"APPROVED": 0, "REVIEW": 0, "SUSPICIOUS": 0, "BLOCKED": 0}
        for d in self.decision_history:
            if d in counts:
                counts[d] += 1

        total = max(1, sum(counts.values()))
        colors = {
            "APPROVED":   "#22c55e",
            "REVIEW":     "#eab308",
            "SUSPICIOUS": "#f97316",
            "BLOCKED":    "#ef4444",
        }

        # Donut chart
        cx, cy = w / 2, h / 2 + 10
        radius = min(w, h) * 0.28
        start = 90

        for name, cnt in counts.items():
            if cnt == 0:
                continue
            extent = (cnt / total) * 360
            c.create_arc(cx - radius, cy - radius, cx + radius, cy + radius,
                         start=start, extent=-extent,
                         fill=colors[name], outline="#020617", width=2)
            start -= extent

        # Inner circle
        inner_r = radius * 0.55
        c.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
                      fill="#020617", outline="")

        c.create_text(cx, cy - 8, text=str(total),
                      fill="#e2e8f0", font=("Segoe UI", 18, "bold"))
        c.create_text(cx, cy + 12, text="TXNS",
                      fill="#64748b", font=("Segoe UI", 9))

        # Legend
        lx = cx + radius + 30
        ly = cy - 60
        for name, cnt in counts.items():
            c.create_rectangle(lx, ly, lx + 12, ly + 12,
                               fill=colors[name], outline="")
            c.create_text(lx + 20, ly + 6,
                          text=f"{name}: {cnt} ({cnt/total*100:.1f}%)",
                          anchor=tk.W, fill="#cbd5e1",
                          font=("Segoe UI", 9))
            ly += 28

    # ------------------------------------------------------------------
    def _draw_rule_signals(self):
        c = self.rule_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 20, text="RULE SIGNAL CONTRIBUTIONS",
                      fill="#94a3b8", font=("Segoe UI", 11, "bold"))

        # Aggregate average scores from recent suspicious txns
        recent = [t for t in list(self.detector.recent_txns)[:30]
                  if t["decision"] in ("SUSPICIOUS", "BLOCKED", "REVIEW")]

        if not recent:
            c.create_text(w / 2, h / 2, text="No suspicious activity yet",
                          fill="#475569", font=("Segoe UI", 12))
            return

        avg = {}
        for t in recent:
            for k, v in t["scores"].items():
                avg[k] = avg.get(k, 0) + v
        for k in avg:
            avg[k] /= len(recent)

        items = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        bar_h = 32
        gap = 14
        total_h = len(items) * (bar_h + gap)
        start_y = (h - total_h) / 2 + 15
        bar_x = 170
        bar_w = w - bar_x - 90

        for i, (label, score) in enumerate(items):
            y = start_y + i * (bar_h + gap)

            c.create_text(bar_x - 12, y + bar_h / 2,
                          text=label.replace("_", " ").title(),
                          anchor=tk.E, fill="#cbd5e1",
                          font=("Segoe UI", 10, "bold"))

            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h,
                               fill="#1e293b", outline="")

            fill_w = bar_w * score
            if score > 0.7:
                color = "#ef4444"
            elif score > 0.45:
                color = "#f97316"
            elif score > 0.25:
                color = "#eab308"
            else:
                color = "#22c55e"

            if fill_w > 0:
                c.create_rectangle(bar_x, y, bar_x + fill_w, y + bar_h,
                                   fill=color, outline="")

            weight = self.detector.weights[label]
            c.create_text(bar_x + bar_w + 12, y + bar_h / 2,
                          text=f"{score*100:.0f}% (w={weight:.2f})",
                          anchor=tk.W, fill=color,
                          font=("Consolas", 9, "bold"))

    # ------------------------------------------------------------------
    def on_close(self):
        self.running = False
        self.root.destroy()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()