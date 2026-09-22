import tkinter as tk
from tkinter import ttk
import random
import time
import threading
from collections import deque
from datetime import datetime

class ProductionLineMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Production Line Monitoring System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        # --- Data Storage ---
        self.max_points = 60
        self.timestamps = deque(maxlen=self.max_points)
        self.temperatures = deque(maxlen=self.max_points)
        self.pressures = deque(maxlen=self.max_points)
        self.speeds = deque(maxlen=self.max_points)
        self.vibrations = deque(maxlen=self.max_points)
        self.output_counts = deque(maxlen=self.max_points)
        self.defect_counts = deque(maxlen=self.max_points)

        # --- State ---
        self.running = True
        self.total_output = 0
        self.total_defects = 0
        self.alert_active = False
        self.alert_messages = deque(maxlen=10)
        self.shift_start = datetime.now()

        # --- Thresholds ---
        self.temp_min, self.temp_max = 60, 95
        self.press_min, self.press_max = 4.0, 8.0
        self.speed_min, self.speed_max = 80, 150
        self.vib_max = 7.0

        # --- Build UI ---
        self._build_ui()

        # --- Start Simulation Thread ---
        self.thread = threading.Thread(target=self._simulate, daemon=True)
        self.thread.start()

        # --- Start UI Update Loop ---
        self._update_ui()

    # ----------------------------------------------------------------------
    # UI CONSTRUCTION
    # ----------------------------------------------------------------------
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#181825", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="⚙  PRODUCTION LINE MONITORING SYSTEM",
            font=("Segoe UI", 18, "bold"),
            fg="#cdd6f4",
            bg="#181825",
        )
        title.pack(side=tk.LEFT, padx=20)

        self.clock_label = tk.Label(
            header,
            text="",
            font=("Consolas", 12),
            fg="#a6e3a1",
            bg="#181825",
        )
        self.clock_label.pack(side=tk.RIGHT, padx=20)

        # Main Container
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel - Gauges / Metrics
        left = tk.Frame(main, bg="#1e1e2e", width=340)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        self.metric_frames = {}
        metrics = [
            ("Temperature", "°C", "#f38ba8"),
            ("Pressure", " bar", "#89b4fa"),
            ("Speed", " RPM", "#a6e3a1"),
            ("Vibration", " mm/s", "#fab387"),
        ]
        for name, unit, color in metrics:
            self._create_metric_card(left, name, unit, color)

        # Output & Defect Cards
        stats_frame = tk.Frame(left, bg="#1e1e2e")
        stats_frame.pack(fill=tk.X, pady=(8, 0))

        self.output_card = self._create_stat_card(stats_frame, "Total Output", "0", "#a6e3a1")
        self.output_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.defect_card = self._create_stat_card(stats_frame, "Defects", "0", "#f38ba8")
        self.defect_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # Efficiency
        eff_frame = tk.Frame(left, bg="#1e1e2e")
        eff_frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            eff_frame, text="EFFICIENCY", font=("Segoe UI", 9, "bold"),
            fg="#6c7086", bg="#1e1e2e"
        ).pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.eff_label = tk.Label(
            eff_frame, text="-- %", font=("Segoe UI", 22, "bold"),
            fg="#a6e3a1", bg="#1e1e2e"
        )
        self.eff_label.pack(anchor=tk.W, padx=8, pady=(0, 8))

        # Right Panel - Charts + Alerts
        right = tk.Frame(main, bg="#1e1e2e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Chart Canvas
        chart_frame = tk.Frame(right, bg="#181825", bd=0)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            chart_frame, text="LIVE SENSOR TRENDS",
            font=("Segoe UI", 10, "bold"), fg="#6c7086", bg="#181825"
        ).pack(anchor=tk.W, padx=12, pady=(8, 0))

        self.canvas = tk.Canvas(
            chart_frame, bg="#181825", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Alert Log
        alert_frame = tk.Frame(right, bg="#181825", height=150)
        alert_frame.pack(fill=tk.X, pady=(10, 0))
        alert_frame.pack_propagate(False)

        tk.Label(
            alert_frame, text="⚠  SYSTEM ALERTS",
            font=("Segoe UI", 10, "bold"), fg="#f38ba8", bg="#181825"
        ).pack(anchor=tk.W, padx=12, pady=(6, 0))

        self.alert_list = tk.Listbox(
            alert_frame,
            bg="#181825",
            fg="#f9e2af",
            font=("Consolas", 9),
            bd=0,
            highlightthickness=0,
            selectbackground="#313244",
        )
        self.alert_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 8))

    def _create_metric_card(self, parent, name, unit, color):
        card = tk.Frame(parent, bg="#313244", bd=0)
        card.pack(fill=tk.X, pady=4)

        top = tk.Frame(card, bg="#313244")
        top.pack(fill=tk.X, padx=10, pady=(8, 0))

        tk.Label(
            top, text=name.upper(), font=("Segoe UI", 9, "bold"),
            fg="#6c7086", bg="#313244"
        ).pack(side=tk.LEFT)

        value_label = tk.Label(
            top, text="--", font=("Segoe UI", 14, "bold"),
            fg=color, bg="#313244"
        )
        value_label.pack(side=tk.RIGHT)

        # Progress bar
        bar_bg = tk.Frame(card, bg="#45475a", height=6)
        bar_bg.pack(fill=tk.X, padx=10, pady=(4, 10))
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=color, height=6)
        bar_fill.place(x=0, y=0, relwidth=0, relheight=1)

        self.metric_frames[name] = {
            "value": value_label,
            "bar": bar_fill,
            "unit": unit,
            "color": color,
            "bar_bg": bar_bg,
        }

    def _create_stat_card(self, parent, title, initial, color):
        card = tk.Frame(parent, bg="#313244", bd=0)
        tk.Label(
            card, text=title.upper(), font=("Segoe UI", 8, "bold"),
            fg="#6c7086", bg="#313244"
        ).pack(anchor=tk.W, padx=8, pady=(6, 0))
        val = tk.Label(
            card, text=initial, font=("Segoe UI", 18, "bold"),
            fg=color, bg="#313244"
        )
        val.pack(anchor=tk.W, padx=8, pady=(0, 6))
        card.value_label = val
        return card

    # ----------------------------------------------------------------------
    # SIMULATION THREAD
    # ----------------------------------------------------------------------
    def _simulate(self):
        while self.running:
            now = datetime.now()

            # Generate realistic sensor values with noise
            temp = 78 + random.uniform(-8, 12) + 5 * (random.random() - 0.5)
            press = 6.0 + random.uniform(-1.2, 1.2)
            speed = 115 + random.uniform(-20, 25)
            vib = 3.5 + random.uniform(-1.5, 3.0)

            # Occasional spike
            if random.random() < 0.06:
                temp += random.uniform(8, 18)
            if random.random() < 0.04:
                press += random.uniform(1.5, 3.0)
            if random.random() < 0.05:
                vib += random.uniform(2.5, 4.5)
            if random.random() < 0.03:
                speed -= random.uniform(30, 50)

            temp = max(40, min(120, temp))
            press = max(2.0, min(12.0, press))
            speed = max(20, min(180, speed))
            vib = max(0.5, min(15.0, vib))

            # Output / defect
            output_inc = 1 if random.random() < 0.7 else 0
            defect_inc = 1 if random.random() < 0.04 else 0

            self.timestamps.append(now)
            self.temperatures.append(temp)
            self.pressures.append(press)
            self.speeds.append(speed)
            self.vibrations.append(vib)
            self.output_counts.append(output_inc)
            self.defect_counts.append(defect_inc)
            self.total_output += output_inc
            self.total_defects += defect_inc

            # Check thresholds
            alerts = []
            if temp > self.temp_max:
                alerts.append(f"⚠ HIGH TEMP: {temp:.1f}°C (max {self.temp_max})")
            if temp < self.temp_min:
                alerts.append(f"⚠ LOW TEMP: {temp:.1f}°C (min {self.temp_min})")
            if press > self.press_max:
                alerts.append(f"⚠ HIGH PRESSURE: {press:.2f} bar (max {self.press_max})")
            if press < self.press_min:
                alerts.append(f"⚠ LOW PRESSURE: {press:.2f} bar (min {self.press_min})")
            if speed > self.speed_max:
                alerts.append(f"⚠ OVERSPEED: {speed:.0f} RPM (max {self.speed_max})")
            if speed < self.speed_min:
                alerts.append(f"⚠ LOW SPEED: {speed:.0f} RPM (min {self.speed_min})")
            if vib > self.vib_max:
                alerts.append(f"⚠ HIGH VIBRATION: {vib:.2f} mm/s (max {self.vib_max})")

            if alerts:
                self.alert_active = True
                for a in alerts:
                    self.alert_messages.appendleft(f"[{now.strftime('%H:%M:%S')}] {a}")
            else:
                self.alert_active = False

            time.sleep(0.5)

    # ----------------------------------------------------------------------
    # UI UPDATE LOOP
    # ----------------------------------------------------------------------
    def _update_ui(self):
        if not self.running:
            return

        # Clock
        self.clock_label.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        # Update metric cards
        latest = {
            "Temperature": (self.temperatures[-1] if self.temperatures else 0,
                            self.temp_min, self.temp_max),
            "Pressure": (self.pressures[-1] if self.pressures else 0,
                         self.press_min, self.press_max),
            "Speed": (self.speeds[-1] if self.speeds else 0,
                      self.speed_min, self.speed_max),
            "Vibration": (self.vibrations[-1] if self.vibrations else 0,
                          0, self.vib_max),
        }

        for name, (val, vmin, vmax) in latest.items():
            frame = self.metric_frames[name]
            unit = frame["unit"]

            # Color based on status
            if vmin <= val <= vmax:
                color = frame["color"]
            else:
                color = "#f38ba8"  # red for out-of-range

            frame["value"].config(text=f"{val:.1f}{unit}", fg=color)

            # Update bar
            ratio = min(1.0, max(0.0, (val - vmin) / (vmax - vmin)))
            bar_bg = frame["bar_bg"]
            bar_bg.update_idletasks()
            w = bar_bg.winfo_width()
            frame["bar"].place(x=0, y=0, width=int(w * ratio), relheight=1)
            frame["bar"].config(bg=color)

        # Stats
        self.output_card.value_label.config(text=str(self.total_output))
        self.defect_card.value_label.config(text=str(self.total_defects))

        # Efficiency
        if self.total_output > 0:
            eff = (1 - self.total_defects / max(1, self.total_output)) * 100
            eff = max(0, min(100, eff))
            self.eff_label.config(
                text=f"{eff:.1f} %",
                fg="#a6e3a1" if eff > 90 else "#f9e2af" if eff > 75 else "#f38ba8",
            )

        # Draw chart
        self._draw_chart()

        # Update alerts
        self._update_alerts()

        # Schedule next update
        self.root.after(500, self._update_ui)

    def _draw_chart(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50 or h < 50:
            return

        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
        cw = w - pad_l - pad_r
        ch = h - pad_t - pad_b

        # Grid
        for i in range(5):
            y = pad_t + ch * i / 4
            self.canvas.create_line(pad_l, y, pad_l + cw, y, fill="#313244", width=1)
            val = 100 - i * 25
            self.canvas.create_text(pad_l - 6, y, text=str(val), anchor=tk.E,
                                    fill="#6c7086", font=("Consolas", 8))

        # Axis labels
        self.canvas.create_text(pad_l - 6, pad_t - 8, text="Value",
                                anchor=tk.W, fill="#6c7086", font=("Consolas", 8))
        self.canvas.create_text(pad_l + cw, pad_t + ch + 14, text="Time →",
                                anchor=tk.E, fill="#6c7086", font=("Consolas", 8))

        n = len(self.temperatures)
        if n < 2:
            self.canvas.create_text(w / 2, h / 2, text="Waiting for data...",
                                    fill="#6c7086", font=("Segoe UI", 11))
            return

        def plot(data, color, vmin, vmax, width=2, smooth=True):
            points = []
            for i, v in enumerate(data):
                x = pad_l + cw * i / (self.max_points - 1)
                norm = (v - vmin) / (vmax - vmin)
                y = pad_t + ch - norm * ch
                points.append((x, y))
            if len(points) >= 2:
                self.canvas.create_line(points, fill=color, width=width,
                                        smooth=smooth, splinesteps=12)
            # Current dot
            if points:
                x, y = points[-1]
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                        fill=color, outline="")

        # Plot each sensor (normalized)
        plot(self.temperatures, "#f38ba8", 40, 120)
        plot(self.pressures, "#89b4fa", 2, 12)
        plot(self.speeds, "#a6e3a1", 20, 180)
        plot(self.vibrations, "#fab387", 0, 15)

        # Legend
        legend = [
            ("Temp", "#f38ba8"),
            ("Pressure", "#89b4fa"),
            ("Speed", "#a6e3a1"),
            ("Vibration", "#fab387"),
        ]
        lx = pad_l + 4
        ly = pad_t + 4
        for label, color in legend:
            self.canvas.create_rectangle(lx, ly, lx + 10, ly + 10,
                                         fill=color, outline="")
            self.canvas.create_text(lx + 14, ly + 5, text=label, anchor=tk.W,
                                    fill="#cdd6f4", font=("Segoe UI", 8))
            lx += 80

    def _update_alerts(self):
        # Clear and repopulate
        self.alert_list.delete(0, tk.END)
        for msg in list(self.alert_messages)[:10]:
            self.alert_list.insert(tk.END, msg)
        # Flash border if alert active
        if self.alert_active:
            self.alert_list.config(bg="#3a1e2e")
        else:
            self.alert_list.config(bg="#181825")

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionLineMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()