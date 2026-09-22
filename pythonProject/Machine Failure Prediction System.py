import tkinter as tk
from tkinter import ttk
import random
import time
import threading
import math
from collections import deque
from datetime import datetime

# ----------------------------------------------------------------------
# PREDICTION ENGINE
# ----------------------------------------------------------------------
class FailurePredictor:
    """
    A lightweight prediction engine using weighted health indicators.
    In a real system, you would load a trained ML model (scikit-learn,
    TensorFlow, etc.). Here we use a rule-based + exponential smoothing
    approach to simulate predictive maintenance logic.
    """

    def __init__(self):
        # Sensor thresholds (min, max, warning_margin)
        self.limits = {
            "temperature": (60, 95, 8),
            "pressure":    (4.0, 8.0, 1.0),
            "vibration":   (0.5, 7.0, 1.5),
            "current":     (5.0, 15.0, 2.0),
            "rpm":         (80, 150, 15),
        }

        # Weights for each sensor's contribution to failure probability
        self.weights = {
            "temperature": 0.25,
            "pressure":    0.20,
            "vibration":   0.30,
            "current":     0.15,
            "rpm":         0.10,
        }

        # Exponential smoothing factor for trend detection
        self.alpha = 0.3
        self.smoothed = {k: None for k in self.limits}
        self.trends = {k: 0.0 for k in self.limits}

        # History for statistical analysis
        self.history = {k: deque(maxlen=30) for k in self.limits}

        # Rolling failure probability
        self.failure_prob = 0.0
        self.health_score = 100.0
        self.risk_level = "LOW"
        self.failure_mode = "None"
        self.time_to_failure = "--"
        self.prediction_history = deque(maxlen=100)

    def _normalize_deviation(self, value, key):
        """Return 0..1 score where 0 = healthy, 1 = extreme deviation."""
        lo, hi, margin = self.limits[key]
        if value < lo:
            dev = (lo - value) / margin
        elif value > hi:
            dev = (value - hi) / margin
        else:
            # Inside range: small deviation from center
            center = (lo + hi) / 2
            half = (hi - lo) / 2
            dev = abs(value - center) / half * 0.3
        return min(1.0, max(0.0, dev))

    def _detect_trend(self, key):
        """Detect increasing/decreasing trend from history."""
        h = list(self.history[key])
        if len(h) < 5:
            return 0.0
        recent = sum(h[-5:]) / 5
        older = sum(h[:5]) / 5
        if older == 0:
            return 0.0
        return (recent - older) / abs(older)

    def _estimate_ttf(self):
        """Estimate time-to-failure in minutes (simulated)."""
        if self.health_score > 85:
            return "Stable"
        elif self.health_score > 70:
            return "> 8 hours"
        elif self.health_score > 50:
            return "2 - 8 hours"
        elif self.health_score > 30:
            return "30 min - 2 hours"
        elif self.health_score > 15:
            return "10 - 30 minutes"
        else:
            return "< 10 minutes"

    def _identify_failure_mode(self, values):
        """Determine the most likely failure mode."""
        deviations = {k: self._normalize_deviation(values[k], k) for k in self.limits}
        worst = max(deviations, key=deviations.get)

        mode_map = {
            "temperature": "Overheating / Bearing Failure",
            "pressure":    "Leak / Valve Malfunction",
            "vibration":   "Misalignment / Bearing Wear",
            "current":     "Motor Overload / Short Circuit",
            "rpm":         "Belt Slip / Load Instability",
        }
        if deviations[worst] < 0.3:
            return "None"
        return mode_map[worst]

    def predict(self, values):
        """Run prediction on a sensor reading dict."""
        scores = {}
        for key in self.limits:
            v = values[key]
            self.history[key].append(v)

            # Exponential smoothing
            if self.smoothed[key] is None:
                self.smoothed[key] = v
            else:
                self.smoothed[key] = self.alpha * v + (1 - self.alpha) * self.smoothed[key]

            # Trend
            self.trends[key] = self._detect_trend(key)

            # Deviation score
            dev = self._normalize_deviation(v, key)

            # Trend amplifies risk if moving away from safe zone
            trend_penalty = 0.0
            lo, hi, _ = self.limits[key]
            if v > hi and self.trends[key] > 0:
                trend_penalty = min(0.3, self.trends[key])
            elif v < lo and self.trends[key] < 0:
                trend_penalty = min(0.3, abs(self.trends[key]))

            scores[key] = min(1.0, dev + trend_penalty)

        # Weighted failure probability
        raw_prob = sum(scores[k] * self.weights[k] for k in self.limits)

        # Smooth the probability (avoid jitter)
        self.failure_prob = 0.7 * self.failure_prob + 0.3 * raw_prob
        self.failure_prob = max(0.0, min(1.0, self.failure_prob))

        # Health score (0 - 100)
        self.health_score = round((1 - self.failure_prob) * 100, 1)

        # Risk level
        if self.health_score >= 85:
            self.risk_level = "LOW"
        elif self.health_score >= 65:
            self.risk_level = "MODERATE"
        elif self.health_score >= 40:
            self.risk_level = "HIGH"
        else:
            self.risk_level = "CRITICAL"

        self.failure_mode = self._identify_failure_mode(values)
        self.time_to_failure = self._estimate_ttf()

        self.prediction_history.append(self.failure_prob)

        return {
            "health": self.health_score,
            "probability": self.failure_prob * 100,
            "risk": self.risk_level,
            "mode": self.failure_mode,
            "ttf": self.time_to_failure,
            "scores": scores,
        }


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class MachineFailurePredictionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Failure Prediction System")
        self.root.geometry("1250x750")
        self.root.configure(bg="#0f172a")
        self.root.minsize(1000, 650)

        # Data
        self.predictor = FailurePredictor()
        self.running = True
        self.max_points = 80
        self.timestamps = deque(maxlen=self.max_points)
        self.history = {
            "temperature": deque(maxlen=self.max_points),
            "pressure":    deque(maxlen=self.max_points),
            "vibration":   deque(maxlen=self.max_points),
            "current":     deque(maxlen=self.max_points),
            "rpm":         deque(maxlen=self.max_points),
        }
        self.prob_history = deque(maxlen=self.max_points)
        self.event_log = deque(maxlen=30)
        self.last_risk = "LOW"
        self.total_readings = 0

        # Style
        self._setup_styles()

        # Build UI
        self._build_ui()

        # Start simulation
        self.thread = threading.Thread(target=self._simulate, daemon=True)
        self.thread.start()

        # Start UI loop
        self._update_ui()

    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "TNotebook", background="#0f172a", borderwidth=0
        )
        style.configure(
            "TNotebook.Tab",
            background="#1e293b",
            foreground="#94a3b8",
            padding=[14, 6],
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#334155")],
            foreground=[("selected", "#e2e8f0")],
        )

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self.root, bg="#020617", height=64)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(
            header, text="🤖  MACHINE FAILURE PREDICTION SYSTEM",
            font=("Segoe UI", 17, "bold"),
            fg="#38bdf8", bg="#020617"
        ).pack(side=tk.LEFT, padx=20)

        self.clock = tk.Label(
            header, text="", font=("Consolas", 11),
            fg="#94a3b8", bg="#020617"
        )
        self.clock.pack(side=tk.RIGHT, padx=20)

        # ===== STATUS BANNER =====
        self.banner = tk.Frame(self.root, bg="#166534", height=52)
        self.banner.pack(fill=tk.X)
        self.banner.pack_propagate(False)

        self.banner_icon = tk.Label(
            self.banner, text="✅", font=("Segoe UI", 20),
            fg="white", bg="#166534"
        )
        self.banner_icon.pack(side=tk.LEFT, padx=(20, 10))

        self.banner_text = tk.Label(
            self.banner, text="SYSTEM HEALTHY — All parameters normal",
            font=("Segoe UI", 13, "bold"),
            fg="white", bg="#166534"
        )
        self.banner_text.pack(side=tk.LEFT)

        self.banner_prob = tk.Label(
            self.banner, text="Failure Probability: 0.0 %",
            font=("Consolas", 11, "bold"),
            fg="white", bg="#166534"
        )
        self.banner_prob.pack(side=tk.RIGHT, padx=20)

        # ===== MAIN LAYOUT =====
        main = tk.Frame(self.root, bg="#0f172a")
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ----- LEFT COLUMN -----
        left = tk.Frame(main, bg="#0f172a", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left.pack_propagate(False)

        # Health Score Card
        health_card = tk.Frame(left, bg="#1e293b", bd=0)
        health_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            health_card, text="MACHINE HEALTH SCORE",
            font=("Segoe UI", 9, "bold"),
            fg="#64748b", bg="#1e293b"
        ).pack(anchor=tk.W, padx=14, pady=(12, 0))

        self.health_value = tk.Label(
            health_card, text="100.0",
            font=("Segoe UI", 40, "bold"),
            fg="#22c55e", bg="#1e293b"
        )
        self.health_value.pack(anchor=tk.W, padx=14, pady=(0, 4))

        self.health_bar_bg = tk.Frame(health_card, bg="#334155", height=10)
        self.health_bar_bg.pack(fill=tk.X, padx=14, pady=(0, 14))
        self.health_bar_bg.pack_propagate(False)
        self.health_bar = tk.Frame(self.health_bar_bg, bg="#22c55e", height=10)
        self.health_bar.place(x=0, y=0, relwidth=1.0, relheight=1)

        # Risk Indicators
        risk_card = tk.Frame(left, bg="#1e293b")
        risk_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            risk_card, text="RISK ASSESSMENT",
            font=("Segoe UI", 9, "bold"),
            fg="#64748b", bg="#1e293b"
        ).pack(anchor=tk.W, padx=14, pady=(12, 6))

        self.risk_value = tk.Label(
            risk_card, text="LOW",
            font=("Segoe UI", 22, "bold"),
            fg="#22c55e", bg="#1e293b"
        )
        self.risk_value.pack(anchor=tk.W, padx=14)

        self.prob_value = tk.Label(
            risk_card, text="Failure Probability: 0.0 %",
            font=("Consolas", 10),
            fg="#94a3b8", bg="#1e293b"
        )
        self.prob_value.pack(anchor=tk.W, padx=14, pady=(0, 12))

        # Failure Mode
        mode_card = tk.Frame(left, bg="#1e293b")
        mode_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            mode_card, text="PREDICTED FAILURE MODE",
            font=("Segoe UI", 9, "bold"),
            fg="#64748b", bg="#1e293b"
        ).pack(anchor=tk.W, padx=14, pady=(12, 4))

        self.mode_value = tk.Label(
            mode_card, text="None",
            font=("Segoe UI", 12, "bold"),
            fg="#e2e8f0", bg="#1e293b", wraplength=280, justify=tk.LEFT
        )
        self.mode_value.pack(anchor=tk.W, padx=14, pady=(0, 12))

        # Time to Failure
        ttf_card = tk.Frame(left, bg="#1e293b")
        ttf_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            ttf_card, text="ESTIMATED TIME TO FAILURE",
            font=("Segoe UI", 9, "bold"),
            fg="#64748b", bg="#1e293b"
        ).pack(anchor=tk.W, padx=14, pady=(12, 4))

        self.ttf_value = tk.Label(
            ttf_card, text="Stable",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8", bg="#1e293b"
        )
        self.ttf_value.pack(anchor=tk.W, padx=14, pady=(0, 12))

        # Stats row
        stats = tk.Frame(left, bg="#0f172a")
        stats.pack(fill=tk.X)
        self.readings_lbl = self._make_stat(stats, "READINGS", "0", "#38bdf8")
        self.readings_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.alerts_lbl = self._make_stat(stats, "ALERTS", "0", "#f43f5e")
        self.alerts_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # ----- RIGHT COLUMN (Notebook with Tabs) -----
        right = tk.Frame(main, bg="#0f172a")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Prediction Chart
        tab1 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab1, text="  📈 Prediction Trend  ")
        self.pred_canvas = tk.Canvas(tab1, bg="#020617", highlightthickness=0)
        self.pred_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 2: Sensor Signals
        tab2 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab2, text="  📊 Sensor Signals  ")
        self.sensor_canvas = tk.Canvas(tab2, bg="#020617", highlightthickness=0)
        self.sensor_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 3: Component Risk Breakdown
        tab3 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab3, text="  🧩 Component Risk  ")
        self.risk_canvas = tk.Canvas(tab3, bg="#020617", highlightthickness=0)
        self.risk_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 4: Event Log
        tab4 = tk.Frame(notebook, bg="#020617")
        notebook.add(tab4, text="  📋 Event Log  ")
        self.log_box = tk.Listbox(
            tab4, bg="#020617", fg="#cbd5e1",
            font=("Consolas", 10), bd=0, highlightthickness=0,
            selectbackground="#334155"
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ===== FOOTER =====
        footer = tk.Frame(self.root, bg="#020617", height=26)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.footer_status = tk.Label(
            footer, text="System initialized. Monitoring active...",
            font=("Consolas", 9), fg="#475569", bg="#020617"
        )
        self.footer_status.pack(side=tk.LEFT, padx=12)
        tk.Label(
            footer, text="FailurePredictor v1.0 | Rule-Based + Trend Analysis",
            font=("Consolas", 9), fg="#475569", bg="#020617"
        ).pack(side=tk.RIGHT, padx=12)

    def _make_stat(self, parent, title, val, color):
        card = tk.Frame(parent, bg="#1e293b")
        tk.Label(
            card, text=title, font=("Segoe UI", 8, "bold"),
            fg="#64748b", bg="#1e293b"
        ).pack(anchor=tk.W, padx=10, pady=(8, 0))
        lbl = tk.Label(
            card, text=val, font=("Segoe UI", 18, "bold"),
            fg=color, bg="#1e293b"
        )
        lbl.pack(anchor=tk.W, padx=10, pady=(0, 8))
        card.value_label = lbl
        return card

    # ------------------------------------------------------------------
    # SIMULATION THREAD
    # ------------------------------------------------------------------
    def _simulate(self):
        # Simulated machine state that degrades over time
        degradation = 0.0
        phase = 0

        while self.running:
            now = datetime.now()
            self.total_readings += 1

            # Introduce gradual degradation cycles
            degradation += 0.002
            if degradation > 1.0:
                degradation = 0.0

            # Occasional random fault injection
            fault_spike = 0.0
            if random.random() < 0.02:
                fault_spike = random.uniform(0.3, 0.8)

            d = min(1.0, degradation + fault_spike)

            # Generate sensor values that drift with degradation
            temperature = 75 + d * 30 + random.uniform(-3, 3)
            pressure    = 6.0 + d * 2.5 + random.uniform(-0.4, 0.4)
            vibration   = 2.5 + d * 6.5 + random.uniform(-0.5, 0.5)
            current     = 9.0 + d * 8.0 + random.uniform(-0.8, 0.8)
            rpm         = 120 - d * 45 + random.uniform(-5, 5)

            values = {
                "temperature": max(40, min(130, temperature)),
                "pressure":    max(2.0, min(12.0, pressure)),
                "vibration":   max(0.3, min(14.0, vibration)),
                "current":     max(3.0, min(22.0, current)),
                "rpm":         max(30, min(180, rpm)),
            }

            # Store
            self.timestamps.append(now)
            for k, v in values.items():
                self.history[k].append(v)

            # Run prediction
            result = self.predictor.predict(values)
            self.prob_history.append(result["probability"] / 100.0)

            # Log events on risk change
            if result["risk"] != self.last_risk:
                msg = (f"[{now.strftime('%H:%M:%S')}] Risk changed: "
                       f"{self.last_risk} → {result['risk']} "
                       f"(Health {result['health']:.1f}%)")
                self.event_log.appendleft(msg)
                self.last_risk = result["risk"]

            if result["risk"] in ("HIGH", "CRITICAL"):
                msg = (f"[{now.strftime('%H:%M:%S')}] ⚠ {result['risk']} risk | "
                       f"{result['mode']} | TTF: {result['ttf']}")
                if not self.event_log or msg != self.event_log[0]:
                    self.event_log.appendleft(msg)

            time.sleep(0.5)

    # ------------------------------------------------------------------
    # UI UPDATE LOOP
    # ------------------------------------------------------------------
    def _update_ui(self):
        if not self.running:
            return

        self.clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        result = {
            "health": self.predictor.health_score,
            "probability": self.predictor.failure_prob * 100,
            "risk": self.predictor.risk_level,
            "mode": self.predictor.failure_mode,
            "ttf": self.predictor.time_to_failure,
            "scores": {
                k: self.predictor._normalize_deviation(
                    self.history[k][-1] if self.history[k] else 0, k
                )
                for k in self.predictor.limits
            },
        }

        # ----- Health & Risk -----
        self.health_value.config(text=f"{result['health']:.1f}")
        ratio = result["health"] / 100.0
        self.health_bar.place(relwidth=ratio)

        if result["health"] >= 85:
            hcolor = "#22c55e"
        elif result["health"] >= 65:
            hcolor = "#eab308"
        elif result["health"] >= 40:
            hcolor = "#f97316"
        else:
            hcolor = "#ef4444"

        self.health_value.config(fg=hcolor)
        self.health_bar.config(bg=hcolor)

        self.risk_value.config(text=result["risk"], fg=hcolor)
        self.prob_value.config(text=f"Failure Probability: {result['probability']:.1f} %")
        self.mode_value.config(text=result["mode"])
        self.ttf_value.config(text=result["ttf"])
        self.readings_lbl.value_label.config(text=f"{self.total_readings}")
        self.alerts_lbl.value_label.config(
            text=str(sum(1 for m in self.event_log if "⚠" in m))
        )

        # ----- Banner -----
        banner_colors = {
            "LOW":      ("#166534", "✅", "SYSTEM HEALTHY — All parameters normal"),
            "MODERATE": ("#854d0e", "⚠️", "MODERATE RISK — Monitor closely"),
            "HIGH":     ("#9a3412", "🔶", "HIGH RISK — Maintenance recommended soon"),
            "CRITICAL": ("#7f1d1d", "🚨", "CRITICAL — Immediate shutdown advised"),
        }
        bg, icon, text = banner_colors[result["risk"]]
        for w in (self.banner, self.banner_icon, self.banner_text, self.banner_prob):
            w.config(bg=bg)
        self.banner_icon.config(text=icon)
        self.banner_text.config(text=text)
        self.banner_prob.config(
            text=f"Failure Probability: {result['probability']:.1f} %  |  "
                 f"Health: {result['health']:.1f}%"
        )

        # ----- Draw charts -----
        self._draw_prediction_chart()
        self._draw_sensor_chart()
        self._draw_risk_chart(result["scores"])

        # ----- Event log -----
        self.log_box.delete(0, tk.END)
        for msg in list(self.event_log)[:25]:
            self.log_box.insert(tk.END, msg)

        # ----- Footer -----
        self.footer_status.config(
            text=f"Last update: {datetime.now().strftime('%H:%M:%S')}  |  "
                 f"Readings: {self.total_readings}  |  "
                 f"Risk: {result['risk']}"
        )

        self.root.after(500, self._update_ui)

    # ------------------------------------------------------------------
    # CHART DRAWING
    # ------------------------------------------------------------------
    def _draw_prediction_chart(self):
        c = self.pred_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        pad_l, pad_r, pad_t, pad_b = 55, 25, 30, 35
        cw, ch = w - pad_l - pad_r, h - pad_t - pad_b

        # Title
        c.create_text(w / 2, 15, text="FAILURE PROBABILITY OVER TIME",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        # Grid + Y labels
        for i in range(6):
            y = pad_t + ch * i / 5
            c.create_line(pad_l, y, pad_l + cw, y, fill="#1e293b")
            c.create_text(pad_l - 8, y, text=f"{100 - i*20}%",
                          anchor=tk.E, fill="#64748b", font=("Consolas", 8))

        # Threshold zones
        c.create_rectangle(pad_l, pad_t, pad_l + cw, pad_t + ch * 0.15,
                           fill="#450a0a", outline="")   # CRITICAL zone
        c.create_rectangle(pad_l, pad_t + ch * 0.15, pad_l + cw, pad_t + ch * 0.35,
                           fill="#431407", outline="")   # HIGH zone
        c.create_rectangle(pad_l, pad_t + ch * 0.35, pad_l + cw, pad_t + ch * 0.60,
                           fill="#422006", outline="")   # MODERATE zone

        # Zone labels
        c.create_text(pad_l + cw - 8, pad_t + 10, text="CRITICAL",
                      anchor=tk.E, fill="#ef4444", font=("Segoe UI", 8, "bold"))
        c.create_text(pad_l + cw - 8, pad_t + ch * 0.25, text="HIGH",
                      anchor=tk.E, fill="#f97316", font=("Segoe UI", 8, "bold"))
        c.create_text(pad_l + cw - 8, pad_t + ch * 0.48, text="MODERATE",
                      anchor=tk.E, fill="#eab308", font=("Segoe UI", 8, "bold"))

        n = len(self.prob_history)
        if n < 2:
            c.create_text(w / 2, h / 2, text="Collecting prediction data...",
                          fill="#475569", font=("Segoe UI", 12))
            return

        # Build line points
        points = []
        for i, p in enumerate(self.prob_history):
            x = pad_l + cw * i / (self.max_points - 1)
            y = pad_t + ch - p * ch
            points.append((x, y))

        # Gradient fill under curve
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            color = "#ef4444" if p > 0.6 else "#f97316" if p > 0.4 else "#eab308" if p > 0.25 else "#22c55e"
            c.create_line(x1, y1, x2, y2, fill=color, width=2)

        # Current marker
        if points:
            x, y = points[-1]
            prob = self.prob_history[-1]
            color = "#ef4444" if prob > 0.6 else "#f97316" if prob > 0.4 else "#eab308" if prob > 0.25 else "#22c55e"
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="white", width=2)
            c.create_text(x, y - 15, text=f"{prob*100:.1f}%",
                          fill=color, font=("Consolas", 9, "bold"))

        # X-axis label
        c.create_text(pad_l + cw / 2, h - 12, text="Time →",
                      fill="#64748b", font=("Consolas", 8))

    def _draw_sensor_chart(self):
        c = self.sensor_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        pad_l, pad_r, pad_t, pad_b = 55, 25, 30, 35
        cw, ch = w - pad_l - pad_r, h - pad_t - pad_b

        c.create_text(w / 2, 15, text="SENSOR SIGNALS (NORMALIZED)",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        # Grid
        for i in range(5):
            y = pad_t + ch * i / 4
            c.create_line(pad_l, y, pad_l + cw, y, fill="#1e293b")

        # Plot each sensor normalized to its range
        configs = [
            ("temperature", "#f87171", 40, 130),
            ("pressure",    "#60a5fa", 2, 12),
            ("vibration",   "#fbbf24", 0, 14),
            ("current",     "#a78bfa", 3, 22),
            ("rpm",         "#34d399", 30, 180),
        ]

        n = len(self.history["temperature"])
        if n < 2:
            c.create_text(w / 2, h / 2, text="Waiting for sensor data...",
                          fill="#475569", font=("Segoe UI", 12))
            return

        for key, color, lo, hi in configs:
            data = list(self.history[key])
            pts = []
            for i, v in enumerate(data):
                x = pad_l + cw * i / (self.max_points - 1)
                norm = (v - lo) / (hi - lo)
                y = pad_t + ch - norm * ch
                pts.append((x, y))
            if len(pts) >= 2:
                c.create_line(pts, fill=color, width=2, smooth=True, splinesteps=8)
            if pts:
                x, y = pts[-1]
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
                c.create_text(x - 8, y, text=f"{data[-1]:.1f}", anchor=tk.E,
                              fill=color, font=("Consolas", 8))

        # Legend
        lx, ly = pad_l + 5, pad_t + 5
        for key, color, _, _ in configs:
            c.create_rectangle(lx, ly, lx + 10, ly + 10, fill=color, outline="")
            c.create_text(lx + 14, ly + 5, text=key.title(), anchor=tk.W,
                          fill="#cbd5e1", font=("Segoe UI", 8))
            lx += 95

    def _draw_risk_chart(self, scores):
        c = self.risk_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 20, text="COMPONENT RISK BREAKDOWN",
                      fill="#94a3b8", font=("Segoe UI", 11, "bold"))

        items = [
            ("Temperature", scores["temperature"], "#f87171"),
            ("Pressure",    scores["pressure"],    "#60a5fa"),
            ("Vibration",   scores["vibration"],   "#fbbf24"),
            ("Current",     scores["current"],     "#a78bfa"),
            ("RPM",         scores["rpm"],         "#34d399"),
        ]

        bar_h = 38
        gap = 18
        total_h = len(items) * (bar_h + gap)
        start_y = (h - total_h) / 2 + 10
        bar_x = 140
        bar_w = w - bar_x - 100

        for i, (label, score, color) in enumerate(items):
            y = start_y + i * (bar_h + gap)

            # Label
            c.create_text(bar_x - 15, y + bar_h / 2, text=label,
                          anchor=tk.E, fill="#cbd5e1", font=("Segoe UI", 11, "bold"))

            # Background bar
            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h,
                               fill="#1e293b", outline="")

            # Filled bar (color by risk)
            fill_w = bar_w * score
            if score > 0.7:
                fill_color = "#ef4444"
            elif score > 0.4:
                fill_color = "#f97316"
            elif score > 0.2:
                fill_color = "#eab308"
            else:
                fill_color = "#22c55e"

            if fill_w > 0:
                c.create_rectangle(bar_x, y, bar_x + fill_w, y + bar_h,
                                   fill=fill_color, outline="")

            # Percentage text
            pct = score * 100
            c.create_text(bar_x + bar_w + 15, y + bar_h / 2, text=f"{pct:.0f}%",
                          anchor=tk.W, fill=fill_color,
                          font=("Consolas", 11, "bold"))

    # ------------------------------------------------------------------
    def on_close(self):
        self.running = False
        self.root.destroy()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MachineFailurePredictionSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()