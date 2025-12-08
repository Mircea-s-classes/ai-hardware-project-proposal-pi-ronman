#!/usr/bin/env python3
"""
camera_hailo_yolo_birds.py
Hailo YOLOv8s + live MJPEG stream with UINT8 input and ragged YOLO decode.

This version ONLY keeps detections for the COCO "bird" class.

- Captures frames from /dev/video0 with OpenCV
- Runs YOLOv8s HEF on Hailo-8L using InferVStreams
- Decodes yolov8_nms_postprocess output into boxes (ragged per-class results)
- Streams MJPEG with overlays at http://<pi-ip>:8080/
- Draws crosshair, YOLO boxes, COCO labels, and FPS overlay.
"""

from typing import List, Tuple
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import math

import cv2
import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    FormatType,
)

# ---------------------------------------------------------------------
# COCO class names for YOLO (80 classes)
# ---------------------------------------------------------------------

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Bird-only filter: COCO "bird" class
BIRD_CLASS_ID = COCO_CLASS_NAMES.index("bird")  # should be 14

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HEF_PATH = "/home/safri/Downloads/yolov8s_h8l.hef"
CAMERA_INDEX = 0  # /dev/video0

CONF_THRESHOLD = 0.15          # slightly low so we see more boxes
HTTP_PORT = 8080

# ---------------------------------------------------------------------
# Global state for MJPEG streaming
# ---------------------------------------------------------------------

latest_jpeg = None
frame_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle MJPEG requests in separate threads."""


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/video"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with frame_lock:
                    jpg = latest_jpeg
                if jpg is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_http_server(port: int = HTTP_PORT) -> ThreadedHTTPServer:
    server = ThreadedHTTPServer(("0.0.0.0", port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[INFO] MJPEG server started on http://0.0.0.0:{port}/")
    return server


# ---------------------------------------------------------------------
# Pre / post-processing
# ---------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray,
                     input_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Resize BGR frame to HEF input size, keep UINT8.

    Output shape: (1, H, W, 3) uint8.
    """
    if len(input_shape) == 4:
        _, net_h, net_w, _ = input_shape
    elif len(input_shape) == 3:
        net_h, net_w, _ = input_shape
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    inp = resized.astype(np.uint8)
    inp = np.expand_dims(inp, axis=0)  # (1, H, W, 3)
    return inp


# ---------------------------------------------------------------------
# YOLO decode helpers
# ---------------------------------------------------------------------

def _auto_interpret_dense_yolo_tensor(
    out: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Dense tensor path: take raw YOLO NMS tensor and return
    (dets_by_class, coords_are_normalized).
    """
    if out.ndim == 4:
        out = np.squeeze(out, axis=0)

    if out.ndim != 3:
        raise ValueError(f"Unexpected dense YOLO ndim={out.ndim}, shape={out.shape}")

    if out.shape[-1] == 5:
        dets_by_class = out
    elif out.shape[1] == 5:
        dets_by_class = np.transpose(out, (0, 2, 1))
    elif out.shape[0] == 5:
        dets_by_class = np.transpose(out, (1, 2, 0))
    else:
        raise ValueError(f"Could not interpret dense YOLO shape={out.shape}")

    coords = dets_by_class[..., 0:4]
    max_coord = float(np.max(np.abs(coords))) if coords.size > 0 else 0.0
    normalized = max_coord <= 2.0
    print(f"[DEBUG] (dense) YOLO coords max={max_coord:.3f}, normalized={normalized}")

    return dets_by_class, normalized


def _decode_from_dense(
    out: np.ndarray,
    orig_shape: Tuple[int, int],
    conf_thresh: float,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Decode assuming out is a dense numeric tensor.
    """
    h, w = orig_shape
    dets_by_class, normalized = _auto_interpret_dense_yolo_tensor(out)

    confs = dets_by_class[..., 4]
    max_conf_raw = float(np.max(confs)) if confs.size > 0 else 0.0
    min_conf_raw = float(np.min(confs)) if confs.size > 0 else 0.0
    print(f"[DEBUG] (dense) conf raw min={min_conf_raw:.4f}, max={max_conf_raw:.4f}")

    detections: List[Tuple[int, int, int, int, float, int]] = []

    for class_id, dets in enumerate(dets_by_class):
        # >>> BIRD-ONLY FILTER <<<
        if class_id != BIRD_CLASS_ID:
            continue

        for det in dets:
            x_c_raw, y_c_raw, bw_raw, bh_raw, conf_raw = det.tolist()

            if conf_raw == 0 and x_c_raw == 0 and y_c_raw == 0 and bw_raw == 0 and bh_raw == 0:
                continue

            if 0.0 <= conf_raw <= 1.0:
                conf = conf_raw
            else:
                conf = 1.0 / (1.0 + math.exp(-float(conf_raw)))

            if conf < conf_thresh:
                continue

            if normalized:
                x_c = x_c_raw * w
                y_c = y_c_raw * h
                bw = bw_raw * w
                bh = bh_raw * h
            else:
                x_c = x_c_raw
                y_c = y_c_raw
                bw = bw_raw
                bh = bh_raw

            x1 = x_c - bw / 2.0
            y1 = y_c - bh / 2.0
            x2 = x_c + bw / 2.0
            y2 = y_c + bh / 2.0

            detections.append(
                (int(x1), int(y1), int(x2), int(y2), float(conf), int(class_id))
            )

    return detections


def _decode_from_ragged(
    output_tensor,
    orig_shape: Tuple[int, int],
    conf_thresh: float,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Ragged path: handle [batch][class][detections x 5].

    Assume per-det:
        [x1, y1, x2, y2, conf] with coords normalized 0..1.

    This version keeps ONLY the "bird" class.
    """
    h, w = orig_shape
    detections: List[Tuple[int, int, int, int, float, int]] = []

    try:
        batch0 = output_tensor[0]
    except Exception as e:
        print(f"[WARN] Could not index batch 0 in ragged output: {e}")
        return []

    # Sample some coords/confs for debug (all classes)
    sample_coords = []
    sample_confs = []
    for cls_idx, cls_results in enumerate(batch0):
        try:
            arr = np.asarray(cls_results, dtype=np.float32)
        except Exception:
            continue
        if arr.size == 0 or arr.shape[1] < 5:
            continue

        sample_coords.append(arr[:, 0:4].flatten())
        sample_confs.append(arr[:, 4])
        if len(sample_coords) >= 3:
            break

    if sample_coords:
        all_coords = np.concatenate(sample_coords)
        max_coord = float(np.max(np.abs(all_coords)))
        normalized = max_coord <= 2.0
        all_confs = np.concatenate(sample_confs)
        min_conf_raw = float(np.min(all_confs))
        max_conf_raw = float(np.max(all_confs))
        print(f"[DEBUG] (ragged) coords max={max_coord:.3f}, normalized={normalized}")
        print(f"[DEBUG] (ragged) conf raw min={min_conf_raw:.4f}, max={max_conf_raw:.4f}")
    else:
        normalized = True
        print("[DEBUG] (ragged) no coords to inspect; assuming normalized")

    for class_id, cls_results in enumerate(batch0):
        # >>> BIRD-ONLY FILTER <<<
        if class_id != BIRD_CLASS_ID:
            continue

        try:
            arr = np.asarray(cls_results, dtype=np.float32)
        except Exception:
            continue
        if arr.size == 0 or arr.shape[1] < 5:
            continue

        for det in arr:
            x1_raw, y1_raw, x2_raw, y2_raw, conf_raw = det.tolist()

            if conf_raw == 0 and x1_raw == 0 and y1_raw == 0 and x2_raw == 0 and y2_raw == 0:
                continue

            if 0.0 <= conf_raw <= 1.0:
                conf = conf_raw
            else:
                conf = 1.0 / (1.0 + math.exp(-float(conf_raw)))

            if conf < conf_thresh:
                continue

            if normalized:
                x1 = x1_raw * w
                y1 = y1_raw * h
                x2 = x2_raw * w
                y2 = y2_raw * h
            else:
                x1 = x1_raw
                y1 = y1_raw
                x2 = x2_raw
                y2 = y2_raw

            detections.append(
                (int(x1), int(y1), int(x2), int(y2), float(conf), int(class_id))
            )

    return detections


def decode_yolo_hailo(
    output_tensor,
    orig_shape: Tuple[int, int],
    conf_thresh: float = CONF_THRESHOLD,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Top-level YOLO decoder: try dense tensor path, then ragged path.

    This is now BIRD-ONLY thanks to the filters inside the helpers.
    """
    try:
        out_dense = np.asarray(output_tensor, dtype=np.float32)
        if out_dense.ndim >= 3:
            return _decode_from_dense(out_dense, orig_shape, conf_thresh)
    except Exception as e:
        print(f"[DEBUG] Dense decode failed: {e}")

    print("[INFO] Falling back to ragged YOLO decode path")
    return _decode_from_ragged(output_tensor, orig_shape, conf_thresh)


def draw_detections(
    frame: np.ndarray,
    detections: List[Tuple[int, int, int, int, float, int]],
    fps: float = None,
) -> np.ndarray:
    """
    Draw YOLO-style boxes, COCO labels, FPS, and a fixed crosshair.

    detections list already filtered to birds, but we still check class id.
    """
    h, w = frame.shape[:2]

    # YOLO boxes in bright red, thick, with class names
    for (x1, y1, x2, y2, score, class_id) in detections:
        # Safety: only draw if it's the bird class
        if class_id != BIRD_CLASS_ID:
            continue

        x1_clamped = max(0, min(int(x1), w - 1))
        y1_clamped = max(0, min(int(y1), h - 1))
        x2_clamped = max(0, min(int(x2), w - 1))
        y2_clamped = max(0, min(int(y2), h - 1))

        if 0 <= class_id < len(COCO_CLASS_NAMES):
            class_name = COCO_CLASS_NAMES[class_id]
        else:
            class_name = str(class_id)

        label = f"{class_name} {score:.2f}"

        cv2.rectangle(frame, (x1_clamped, y1_clamped),
                      (x2_clamped, y2_clamped), (0, 0, 255), 3)
        cv2.putText(frame, label, (x1_clamped, max(0, y1_clamped - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # FPS overlay (top-left, white text)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Fixed blue crosshair in the center
    cx, cy = w // 2, h // 2
    cross_size = min(w, h) // 10
    cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (255, 0, 0), 2)
    cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (255, 0, 0), 2)

    return frame


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def main():
    server = start_http_server(HTTP_PORT)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {CAMERA_INDEX}")
        server.shutdown()
        return

    print("[INFO] Loading HEF and creating Hailo VDevice...")
    hef = HEF(HEF_PATH)

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef=hef,
            interface=HailoStreamInterface.PCIe,
        )

        ngs = target.configure(hef, configure_params)
        network_group = ngs[0] if isinstance(ngs, (list, tuple)) else ngs
        network_group_params = network_group.create_params()

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        print(f"[INFO] Model input shape : {input_vstream_info.shape}")
        print(f"[INFO] Model output shape: {output_vstream_info.shape}")
        print(f"[INFO] Input vstream name : {input_vstream_info.name}")
        print(f"[INFO] Output vstream name: {output_vstream_info.name}")
        print(f"[INFO] Open your browser at: http://192.168.1.163:{HTTP_PORT}/ (or use hostname -I)")

        input_shape = tuple(input_vstream_info.shape)

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        print("[INFO] Starting live camera + Hailo loop (HTTP MJPEG). Press Ctrl+C to stop.")
        print("[INFO] Bird-only detection: COCO class 'bird'")

        last_print = time.time()
        frames_this_second = 0
        total_frames = 0
        last_dets: List[Tuple[int, int, int, int, float, int]] = []
        current_fps = 0.0

        try:
            with network_group.activate(network_group_params):
                with InferVStreams(
                    network_group, input_vstreams_params, output_vstreams_params
                ) as infer_pipeline:

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("[WARN] Failed to read frame from camera")
                            break

                        orig_h, orig_w = frame.shape[:2]

                        try:
                            input_tensor = preprocess_frame(frame, input_shape)
                        except Exception as e:
                            print(f"[ERROR] Preprocess error: {e}")
                            break

                        input_data = {input_vstream_info.name: input_tensor}

                        try:
                            results = infer_pipeline.infer(input_data)
                            output_tensor = results[output_vstream_info.name]
                        except Exception as e:
                            print(f"[ERROR] Inference error: {e}")
                            break

                        detections = decode_yolo_hailo(
                            output_tensor, (orig_h, orig_w), CONF_THRESHOLD
                        )
                        last_dets = detections

                        # Draw overlays including FPS (bird-only detections)
                        draw_detections(frame, detections, fps=current_fps)

                        ok, encoded = cv2.imencode(".jpg", frame)
                        if ok:
                            jpg_bytes = encoded.tobytes()
                            with frame_lock:
                                global latest_jpeg
                                latest_jpeg = jpg_bytes

                        total_frames += 1
                        frames_this_second += 1
                        now = time.time()
                        if now - last_print >= 1.0:
                            current_fps = frames_this_second  # update FPS display

                            if last_dets:
                                first_det = last_dets[0]
                                print(
                                    f"[INFO] FPS ~ {frames_this_second}  "
                                    f"(total frames: {total_frames}, "
                                    f"bird detections last frame: {len(last_dets)}, "
                                    f"first bird det: {first_det})"
                                )
                            else:
                                print(
                                    f"[INFO] FPS ~ {frames_this_second}  "
                                    f"(total frames: {total_frames}, "
                                    f"bird detections last frame: 0)"
                                )
                            frames_this_second = 0
                            last_print = now

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt – exiting main loop...")

        finally:
            print("[INFO] Cleaning up...")
            cap.release()
            server.shutdown()


if __name__ == "__main__":
    main()
