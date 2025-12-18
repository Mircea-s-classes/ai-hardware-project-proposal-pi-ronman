#!/usr/bin/env python3
"""
yolo_gui.py
Raspberry Pi YOLO GUI with:
- Bird-only filtering (default COCO id 14)
- FPS overlay
- CPU temperature logging to CSV
- Optional CPU temperature plot on exit
- No pop-up alerts
"""

import argparse
import csv
import time
import threading
import subprocess

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# CPU Temperature Logging
# -----------------------------
def get_cpu_temp_c():
    """
    Return CPU temperature in Celsius.
    Raspberry Pi OS: /sys/class/thermal/thermal_zone0/temp
    Fallback: vcgencmd measure_temp
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            millideg = int(f.read().strip())
        return millideg / 1000.0
    except Exception:
        pass

    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        # "temp=48.2'C"
        return float(out.replace("temp=", "").replace("'C", ""))
    except Exception:
        return None


class CPUTemperatureLogger(threading.Thread):
    def __init__(self, interval_s=1.0, filename="cpu_temp_log.csv"):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.filename = filename
        self._stop = threading.Event()
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "cpu_temp_c"])
            while not self._stop.is_set():
                t = time.time() - self.start_time
                temp = get_cpu_temp_c()
                if temp is not None:
                    writer.writerow([f"{t:.3f}", f"{temp:.2f}"])
                    f.flush()
                time.sleep(self.interval_s)

    def stop(self):
        self._stop.set()


def plot_cpu_temperature(csv_file="cpu_temp_log.csv", out_file="cpu_temp_plot.png"):
    """
    Optional: generate a temperature plot after the program exits.
    Only runs if matplotlib is installed.
    """
    if plt is None:
        return

    times = []
    temps = []
    try:
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["time_s"]))
                temps.append(float(row["cpu_temp_c"]))
    except Exception:
        return

    if not times:
        return

    plt.figure()
    plt.plot(times, temps)
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Temperature (°C)")
    plt.title("Raspberry Pi CPU Temperature During Inference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# -----------------------------
# Video Helpers
# -----------------------------
def open_capture(source: str, width: int, height: int, fps: int):
    """
    source can be:
      - "0" or "1" (webcam index)
      - "/dev/video0"
      - file path
      - URL/RTSP
    """
    cap = None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        return cap

    # Try to set properties (may not apply for all sources)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    except Exception:
        pass

    return cap


def draw_boxes_ultralytics(results, frame, show_labels=True, show_conf=True):
    """
    Uses Ultralytics built-in plotting when available.
    """
    try:
        return results[0].plot()
    except Exception:
        return frame


# -----------------------------
# GUI App
# -----------------------------
class YoloGUI(tk.Tk):
    def __init__(
        self,
        model_path: str,
        source: str,
        imgsz: int,
        conf: float,
        width: int,
        height: int,
        fps: int,
        only_class_ids=None,
        temp_log_csv="cpu_temp_log.csv",
        temp_log_interval_s=1.0,
    ):
        super().__init__()
        self.title("YOLO Bird Detection (Pi 5)")
        self.geometry("980x640")

        # Settings
        self.model_path = model_path
        self.source = source
        self.imgsz = imgsz
        self.conf = conf
        self.width = width
        self.height = height
        self.fps_target = fps

        # Filtering: default to bird class in COCO (14)
        self.only_class_ids = only_class_ids if only_class_ids is not None else [14]

        # Runtime state
        self.model = None
        self.cap = None
        self.running = False

        # FPS tracking
        self._last_frame_ts = None
        self._fps_smooth = None
        self._fps_alpha = 0.15  # smoothing factor

        # CPU temp logging
        self.temp_logger = CPUTemperatureLogger(
            interval_s=temp_log_interval_s,
            filename=temp_log_csv
        )

        # UI
        self._build_ui()

        # Clean shutdown
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start automatically
        self.after(50, self.start)

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        ttk.Label(top, text=self.model_path).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(top, text="Source:").pack(side=tk.LEFT)
        ttk.Label(top, text=str(self.source)).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(top, text="Class filter (IDs):").pack(side=tk.LEFT)
        ttk.Label(top, text=",".join(map(str, self.only_class_ids))).pack(side=tk.LEFT, padx=(4, 16))

        self.status_label = ttk.Label(top, text="Initializing...")
        self.status_label.pack(side=tk.RIGHT)

        # Video area
        self.video_label = ttk.Label(self)
        self.video_label.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Bottom buttons
        bot = ttk.Frame(self)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.btn_quit = ttk.Button(bot, text="Quit", command=self.on_close)
        self.btn_quit.pack(side=tk.RIGHT)

    def start(self):
        if YOLO is None:
            self.status_label.config(text="[ERROR] Ultralytics not installed in this environment.")
            return

        # Load model
        try:
            self.model = YOLO(self.model_path)
            self.status_label.config(text="Model loaded. Opening camera...")
        except Exception as e:
            self.status_label.config(text=f"[ERROR] Load model failed: {e}")
            return

        # Open capture
        self.cap = open_capture(self.source, self.width, self.height, self.fps_target)
        if not self.cap or not self.cap.isOpened():
            self.status_label.config(text="[ERROR] Could not open camera/source.")
            return

        # Start temp logging
        try:
            self.temp_logger.start()
        except Exception:
            pass

        self.running = True
        self.status_label.config(text="Running (bird-only). Press Quit to exit.")
        self._update_frame()

    def _update_fps(self):
        now = time.time()
        if self._last_frame_ts is None:
            self._last_frame_ts = now
            return None

        dt = now - self._last_frame_ts
        self._last_frame_ts = now
        if dt <= 1e-6:
            return None

        inst_fps = 1.0 / dt
        if self._fps_smooth is None:
            self._fps_smooth = inst_fps
        else:
            self._fps_smooth = (1 - self._fps_alpha) * self._fps_smooth + self._fps_alpha * inst_fps

        return self._fps_smooth

    def _overlay_text(self, frame, lines, x=10, y=25):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        dy = 26
        for i, line in enumerate(lines):
            yy = y + i * dy
            cv2.putText(frame, line, (x, yy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _update_frame(self):
        if not self.running:
            return

        frame = None
        if self.cap and self.cap.isOpened():
            ok, fr = self.cap.read()
            if ok and fr is not None:
                frame = fr

        if frame is None:
            self.status_label.config(text="[WARN] No frame. Retrying...")
            self.after(10, self._update_frame)
            return

        # YOLO inference
        try:
            # classes filter: only run outputs for chosen IDs
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=self.only_class_ids,
                verbose=False
            )
        except Exception as e:
            self.status_label.config(text=f"[ERROR] Inference failed: {e}")
            self.after(50, self._update_frame)
            return

        # Draw boxes
        vis = draw_boxes_ultralytics(results, frame, show_labels=True, show_conf=True)

        # Update FPS and temp
        fps_val = self._update_fps()
        temp_c = get_cpu_temp_c()

        lines = []
        if fps_val is not None:
            lines.append(f"FPS: {fps_val:.2f}")
        if temp_c is not None:
            lines.append(f"CPU Temp: {temp_c:.1f} C")

        if lines:
            self._overlay_text(vis, lines)

        # Convert to Tk image
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk  # keep reference
        self.video_label.configure(image=imgtk)

        # Schedule next update (fast UI loop)
        self.after(1, self._update_frame)

    def on_close(self):
        self.running = False

        # Stop temp logger
        try:
            if self.temp_logger is not None:
                self.temp_logger.stop()
                self.temp_logger.join(timeout=2.0)
        except Exception:
            pass

        # Release camera
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

        self.destroy()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="YOLO Bird Detection GUI (Pi)")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights (e.g., yolo11n.pt)")
    ap.add_argument("--source", default="0", help="0, /dev/videoX, file path, or URL")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--fps", type=int, default=30, help="Requested capture FPS")
    ap.add_argument(
        "--only_class",
        default="bird",
        help="Filter detections to a class name or comma-separated IDs (default: bird)"
    )
    ap.add_argument("--temp_csv", default="cpu_temp_log.csv", help="CPU temp CSV log filename")
    ap.add_argument("--temp_interval", type=float, default=1.0, help="CPU temp log interval (s)")
    ap.add_argument("--plot_temp", action="store_true", help="Save cpu_temp_plot.png on exit (needs matplotlib)")
    return ap.parse_args()


def parse_only_class_arg(only_class: str):
    """
    Supports:
      --only_class bird
      --only_class 14
      --only_class 14,15
    COCO: bird is class id 14 for Ultralytics YOLO COCO models.
    """
    s = (only_class or "").strip().lower()
    if s == "" or s == "bird":
        return [14]

    # numeric list
    parts = [p.strip() for p in s.split(",")]
    ids = []
    for p in parts:
        if p.isdigit():
            ids.append(int(p))
        else:
            # unknown name; default to bird
            return [14]
    return ids if ids else [14]


def main():
    args = parse_args()

    only_ids = parse_only_class_arg(args.only_class)

    app = YoloGUI(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        width=args.width,
        height=args.height,
        fps=args.fps,
        only_class_ids=only_ids,
        temp_log_csv=args.temp_csv,
        temp_log_interval_s=args.temp_interval
    )
    app.mainloop()

    if args.plot_temp:
        plot_cpu_temperature(args.temp_csv, "cpu_temp_plot.png")


if __name__ == "__main__":
    main()
