#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Tuple

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except Exception as e:
    print("Fehlende Abhaengigkeiten. Bitte Umgebung einrichten und Pakete installieren:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install -U pip")
    print("  pip install -r requirements.txt")
    print("")
    print("Danach starten mit:")
    print("  python dsgvo-pixeler.py --input input.mp4 --output output.mp4 --weights models/plates/best.pt")
    raise SystemExit(2) from e


def apply_pad(x1: int, y1: int, x2: int, y2: int, pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def pixelate_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, blocks: int) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh == 0 or rw == 0:
        return
    blocks = max(1, int(blocks))
    small_w = max(1, rw // blocks)
    small_h = max(1, rh // blocks)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixel = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pixel


def blur_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, ksize: int) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh == 0 or rw == 0:
        return
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    max_k = min(rw, rh)
    if max_k % 2 == 0:
        max_k -= 1
    if max_k < 1:
        return
    if k > max_k:
        k = max_k
    if k <= 1:
        return
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred


def boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def clip_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2))


def map_detection_box(
    xyxy,
    tile_x: int,
    tile_y: int,
    scale_x: float,
    scale_y: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    box = (
        math.floor((float(xyxy[0]) + tile_x) / scale_x),
        math.floor((float(xyxy[1]) + tile_y) / scale_y),
        math.ceil((float(xyxy[2]) + tile_x) / scale_x),
        math.ceil((float(xyxy[3]) + tile_y) / scale_y),
    )
    return clip_box(box, w, h)


def box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = box_area((ix1, iy1, ix2, iy2))
    if intersection <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def should_merge_boxes(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    max_gap: int = 2,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_w = min(ax2, bx2) - max(ax1, bx1)
    overlap_h = min(ay2, by2) - max(ay1, by1)
    if overlap_w > 0 and overlap_h > 0:
        # Privacy-first: low-IoU partial detections must not leave a visible seam.
        return True

    a_w, a_h = ax2 - ax1, ay2 - ay1
    b_w, b_h = bx2 - bx1, by2 - by1
    horizontal_gap = max(bx1 - ax2, ax1 - bx2, 0)
    vertical_gap = max(by1 - ay2, ay1 - by2, 0)
    vertical_overlap_ratio = max(0, overlap_h) / max(1, min(a_h, b_h))
    horizontal_overlap_ratio = max(0, overlap_w) / max(1, min(a_w, b_w))
    return (
        horizontal_gap <= max_gap and vertical_overlap_ratio >= 0.5
    ) or (
        vertical_gap <= max_gap and horizontal_overlap_ratio >= 0.5
    )


def merge_overlapping_boxes(
    boxes: list,
    max_gap: int = 2,
) -> list:
    source = sorted(set(tuple(box) for box in boxes if box_area(tuple(box)) > 0))
    merged = []
    visited = set()
    for start_idx in range(len(source)):
        if start_idx in visited:
            continue
        component = []
        stack = [start_idx]
        visited.add(start_idx)
        while stack:
            current_idx = stack.pop()
            component.append(source[current_idx])
            for other_idx in range(len(source)):
                if other_idx in visited:
                    continue
                if should_merge_boxes(source[current_idx], source[other_idx], max_gap):
                    visited.add(other_idx)
                    stack.append(other_idx)
        merged.append(
            (
                min(box[0] for box in component),
                min(box[1] for box in component),
                max(box[2] for box in component),
                max(box[3] for box in component),
            )
        )
    return sorted(merged)


def build_tiles(w: int, h: int, count: int, overlap: float) -> list:
    if count <= 1:
        return [(0, 0, w, h)]

    def axis_segments(length: int) -> list:
        denominator = count - ((count - 1) * overlap)
        tile_size = min(length, max(1, int(math.ceil(length / denominator))))
        max_start = max(0, length - tile_size)
        segments = []
        for index in range(count):
            start = int(round((index * max_start) / (count - 1)))
            segment = (start, min(length, start + tile_size))
            if segment[1] > segment[0] and segment not in segments:
                segments.append(segment)
        return segments

    x_segments = axis_segments(w)
    y_segments = axis_segments(h)
    return [(x0, y0, x1, y1) for y0, y1 in y_segments for x0, x1 in x_segments]


def subtract_box(
    box: Tuple[int, int, int, int],
    cut: Tuple[int, int, int, int],
) -> list:
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = cut
    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)
    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return [box]
    pieces = [
        (x1, y1, x2, iy1),
        (x1, iy2, x2, y2),
        (x1, iy1, ix1, iy2),
        (ix2, iy1, x2, iy2),
    ]
    return [piece for piece in pieces if box_area(piece) > 0]


def subtract_zones(box: Tuple[int, int, int, int], zones: list) -> list:
    pieces = [box]
    for zone in zones:
        next_pieces = []
        for piece in pieces:
            next_pieces.extend(subtract_box(piece, zone))
        pieces = next_pieces
        if not pieces:
            break
    return pieces


def anonymize_box_excluding_zones(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    zones: list,
    mode: str,
    blur_ksize: int,
    blocks: int,
) -> None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    overlapping_zones = [zone for zone in zones if boxes_overlap(box, zone)]
    if not overlapping_zones:
        if mode == "blur":
            blur_roi(frame, x1, y1, x2, y2, blur_ksize)
        else:
            pixelate_roi(frame, x1, y1, x2, y2, blocks)
        return

    visible_pieces = subtract_zones(box, overlapping_zones)
    if not visible_pieces:
        return
    processed = frame[y1:y2, x1:x2].copy()
    if mode == "blur":
        blur_roi(processed, 0, 0, x2 - x1, y2 - y1, blur_ksize)
    else:
        pixelate_roi(processed, 0, 0, x2 - x1, y2 - y1, blocks)
    for px1, py1, px2, py2 in visible_pieces:
        frame[py1:py2, px1:px2] = processed[py1 - y1:py2 - y1, px1 - x1:px2 - x1]


def shift_box(
    box: Tuple[int, int, int, int],
    velocity: Tuple[float, float],
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    dx, dy = velocity
    shifted = (
        int(round(box[0] + dx)),
        int(round(box[1] + dy)),
        int(round(box[2] + dx)),
        int(round(box[3] + dy)),
    )
    return clip_box(shifted, w, h)


def union_boxes(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def temporal_match_score(
    predicted: Tuple[int, int, int, int],
    detection: Tuple[int, int, int, int],
) -> float:
    return box_iou(predicted, detection)


def update_temporal_masks(
    detections: list,
    tracks: list,
    ttl: int,
    w: int,
    h: int,
) -> list:
    current = []
    for kind, box in detections:
        clipped = clip_box(box, w, h)
        if box_area(clipped) > 0:
            current.append((kind, clipped))
    if ttl <= 0:
        tracks.clear()
        return [(kind, box, 0) for kind, box in current]

    for track in tracks:
        track["history"] = [
            (box, age + 1)
            for box, age in track.get("history", [(track["box"], 0)])
            if age + 1 <= ttl
        ]

    predicted = []
    for track in tracks:
        next_box = shift_box(track["box"], track["velocity"], w, h)
        if box_area(next_box) <= 0:
            next_box = track.get("last_observed_box", track["box"])
        predicted.append(next_box)
    candidates = []
    for track_idx, track in enumerate(tracks):
        for detection_idx, (kind, box) in enumerate(current):
            if track["kind"] != kind:
                continue
            score = temporal_match_score(predicted[track_idx], box)
            if score > 0.0:
                candidates.append((score, track_idx, detection_idx))
    candidates.sort(reverse=True)

    matched_tracks = {}
    matched_detections = set()
    for _, track_idx, detection_idx in candidates:
        if track_idx in matched_tracks or detection_idx in matched_detections:
            continue
        matched_tracks[track_idx] = detection_idx
        matched_detections.add(detection_idx)

    next_tracks = []
    for track_idx, track in enumerate(tracks):
        if track_idx in matched_tracks:
            detection_idx = matched_tracks[track_idx]
            kind, box = current[detection_idx]
            old_box = track.get("last_observed_box", track["box"])
            old_center = ((old_box[0] + old_box[2]) / 2.0, (old_box[1] + old_box[3]) / 2.0)
            new_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            elapsed_frames = max(1, int(track["missed"]) + 1)
            history = track["history"]
            history.append((box, 0))
            next_tracks.append(
                {
                    "kind": kind,
                    "box": box,
                    "last_observed_box": box,
                    "velocity": (
                        (new_center[0] - old_center[0]) / elapsed_frames,
                        (new_center[1] - old_center[1]) / elapsed_frames,
                    ),
                    "missed": 0,
                    "history": history,
                }
            )
            continue
        missed = int(track["missed"]) + 1
        if missed <= ttl:
            velocity = tuple(float(value) * 0.75 for value in track["velocity"])
            next_tracks.append(
                {
                    "kind": track["kind"],
                    "box": predicted[track_idx],
                    "last_observed_box": track.get("last_observed_box", track["box"]),
                    "velocity": velocity,
                    "missed": missed,
                    "history": track["history"],
                }
            )

    for detection_idx, (kind, box) in enumerate(current):
        if detection_idx in matched_detections:
            continue
        next_tracks.append(
            {
                "kind": kind,
                "box": box,
                "last_observed_box": box,
                "velocity": (0.0, 0.0),
                "missed": 0,
                "history": [(box, 0)],
            }
        )

    tracks[:] = next_tracks
    masks = []
    for track in tracks:
        for history_box, _ in track["history"]:
            masks.append((track["kind"], history_box, 0))
        if track["missed"] > 0:
            safety_box = union_boxes(track["last_observed_box"], track["box"])
            masks.append((track["kind"], safety_box, track["missed"]))
    return masks


def reset_ultralytics_trackers(models: list) -> Tuple[bool, list]:
    errors = []
    for model, _ in models:
        predictor = getattr(model, "predictor", None)
        trackers = getattr(predictor, "trackers", []) or []
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if not callable(reset):
                errors.append("Tracker bietet keine reset()-Methode")
                continue
            try:
                reset()
            except Exception as e:
                errors.append(str(e))
    return not errors, errors


def capture_raw_detections(predictor) -> None:
    raw_xyxy = []
    for result in getattr(predictor, "results", []) or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            raw_xyxy.append(np.empty((0, 4), dtype=float))
        else:
            raw_xyxy.append(boxes.xyxy.cpu().numpy().copy())
    predictor._dsgvo_raw_xyxy = raw_xyxy


def install_raw_detection_capture(model) -> bool:
    callback_sets = [getattr(model, "callbacks", None)]
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        callback_sets.append(getattr(predictor, "callbacks", None))
    for callbacks in callback_sets:
        if not isinstance(callbacks, dict):
            return False
        event_callbacks = callbacks.get("on_predict_postprocess_end")
        if not isinstance(event_callbacks, list):
            return False
        event_callbacks[:] = [callback for callback in event_callbacks if callback is not capture_raw_detections]
        event_callbacks.insert(0, capture_raw_detections)
    return True


def build_ffmpeg_cmd(
    output: str,
    input_path: str,
    w: int,
    h: int,
    fps: float,
    codec: str,
    bitrate: str,
    use_sw: bool,
    include_audio: bool,
    fade_seconds: float,
    duration_seconds: float,
) -> list:
    if codec == "hevc":
        vcodec = "libx265" if use_sw else "hevc_videotoolbox"
    elif codec == "h264":
        vcodec = "libx264" if use_sw else "h264_videotoolbox"
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    pix_fmt = "yuv420p" if use_sw else "nv12"
    video_filters = []
    if fade_seconds > 0:
        video_filters.append(f"fade=t=in:st=0:d={fade_seconds:.3f}")
        if duration_seconds > 0:
            fade_out_start = max(duration_seconds - fade_seconds, 0.0)
            video_filters.append(f"fade=t=out:st={fade_out_start:.3f}:d={fade_seconds:.3f}")
    video_filters.append(f"format={pix_fmt}")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-vf",
        ",".join(video_filters),
        "-pix_fmt",
        pix_fmt,
        "-c:v",
        vcodec,
        "-b:v",
        bitrate,
        "-movflags",
        "+faststart",
        output,
    ]
    if include_audio:
        cmd[cmd.index("-vf"):cmd.index("-vf")] = ["-map", "1:a?"]
        cmd[cmd.index("-movflags"):cmd.index("-movflags")] = ["-c:a", "copy"]
    else:
        cmd[cmd.index("-movflags"):cmd.index("-movflags")] = ["-an"]
    if not use_sw:
        idx = cmd.index("-b:v")
        cmd[idx:idx] = ["-allow_sw", "1"]
        if codec == "h264":
            cmd[idx:idx] = ["-profile:v", "high", "-level:v", "5.2"]
    return cmd


def open_video(input_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video nicht oeffnbar: {input_path}")
    return cap


def get_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0 or fps > 240.0:
        return 25.0
    return fps


def probe_bitrate(input_path: str) -> str:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=nk=1:nw=1",
            input_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if out.isdigit():
            return f"{int(out) // 1000}k"
    except Exception:
        pass
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=bit_rate",
            "-of",
            "default=nk=1:nw=1",
            input_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if out.isdigit():
            return f"{int(out) // 1000}k"
    except Exception:
        pass
    return "50M"


PRESETS = {
    "fast": {"conf": 0.3, "imgsz": 960, "work_w": 1280, "blocks_plates": 16, "blocks_faces": 24, "pad": 20, "bitrate": "auto"},
    "balanced": {"conf": 0.25, "imgsz": 1280, "work_w": 1920, "blocks_plates": 16, "blocks_faces": 24, "pad": 20, "bitrate": "auto"},
    "quality": {"conf": 0.2, "imgsz": 1600, "work_w": 0, "blocks_plates": 16, "blocks_faces": 24, "pad": 24, "bitrate": "auto"},
}
DEFAULT_BLUR_KSIZE = 80
DEFAULT_FADE_SECONDS = 1.5
DEFAULT_TILE_OVERLAP = 0.2
DEFAULT_MASK_TTL = 3


def parse_args() -> argparse.Namespace:
    default_preset = PRESETS["quality"]
    p = argparse.ArgumentParser(
        description="Kennzeichen und Gesichter in Videos erkennen und anonymisieren (Apple Silicon/MPS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", nargs="+", help="Input-Video, Ordner, Glob oder Komma-Liste (MP4)")
    p.add_argument("--output", help="Output-Video (MP4) oder Zielordner")
    p.add_argument("--weights", help="YOLOv8 plates weights (Liste mit Komma)")
    p.add_argument("--faces_weights", help="YOLOv8 face weights (Liste mit Komma)")
    p.add_argument("--extra_weights", help="Zusatz-Modelle (Liste mit Komma)")
    p.add_argument("--use_extra", action="store_true", help="models/extra/*.pt mitnutzen")
    p.add_argument("--device", default="mps", help="Ultralytics device, z.B. mps oder cpu")
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"Confidence Threshold. Default (preset quality): {default_preset['conf']}. Empfohlen: 0.1-0.6",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help=f"YOLO imgsz. Default (preset quality): {default_preset['imgsz']}. Empfohlen: 640-2048",
    )
    p.add_argument(
        "--work_w",
        type=int,
        default=None,
        help=f"Arbeitsbreite fuer Detektion (0 = Original). Default (preset quality): {default_preset['work_w']}. Empfohlen: 0-3840",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="Pixel-Blockgroesse (deprecated, nur pixelate, groesser = grober). Empfohlen: 4-64",
    )
    p.add_argument(
        "--blocks_plates",
        type=int,
        default=None,
        help=f"Pixel-Blockgroesse fuer Kennzeichen (nur pixelate, groesser = grober). Default (preset quality): {default_preset['blocks_plates']}. Empfohlen: 4-64",
    )
    p.add_argument(
        "--blocks_faces",
        type=int,
        default=None,
        help=f"Pixel-Blockgroesse fuer Gesichter (nur pixelate, groesser = grober). Default (preset quality): {default_preset['blocks_faces']}. Empfohlen: 4-64",
    )
    p.add_argument(
        "--anonymize",
        choices=["pixelate", "blur"],
        default="blur",
        help="Anonymisierung: pixelate|blur (Standard: blur)",
    )
    p.add_argument(
        "--blur_ksize",
        type=int,
        default=DEFAULT_BLUR_KSIZE,
        help=f"Blur-Staerke (Kernel-Size, gerade Werte werden auf ungerade aufgerundet). Standard: {DEFAULT_BLUR_KSIZE}",
    )
    p.add_argument(
        "--pad",
        type=int,
        default=None,
        help=f"Sicherheitsrand in Pixel. Default (preset quality): {default_preset['pad']}. Empfohlen: 0-100",
    )
    p.add_argument("--codec", choices=["hevc", "h264"], default="hevc", help="Video codec")
    p.add_argument(
        "--bitrate",
        default=None,
        help=f"Video bitrate, z.B. 50M oder auto. Default (preset quality): {default_preset['bitrate']}",
    )
    p.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="quality",
        help="Preset fuer Speed/Qualitaet",
    )
    p.add_argument("--force_sw", action="store_true", help="Software-Encoding erzwingen (libx265/libx264)")
    p.add_argument("--debug_pixel", dest="debug_pixel", action="store_true", help="BBox-Overlay fuer Debug einzeichnen")
    p.add_argument("--debug_overlay", dest="debug_pixel", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_audio", action="store_true", help="Audio entfernen")
    p.add_argument(
        "--fade_seconds",
        type=float,
        default=DEFAULT_FADE_SECONDS,
        help=f"Video am Anfang aus Schwarz einblenden und am Ende nach Schwarz ausblenden (0 = aus). Standard: {DEFAULT_FADE_SECONDS}",
    )
    p.add_argument("--no_track", action="store_true", help="Ultralytics-Tracking deaktivieren (Sicherheitsmaske bleibt aktiv)")
    p.add_argument(
        "--snapshot_every",
        type=int,
        default=0,
        help="Snapshot alle N Minuten (0 = aus). Empfohlen: 0-60",
    )
    p.add_argument("--snapshot_dir", default="", help="Snapshot-Ordner (Default: Input-Ordner)")
    p.add_argument("--snapshot_size", default="1920x1080", help="Snapshot-Groesse, z.B. 1920x1080")
    p.add_argument("--debug_no_pixel", dest="debug_no_pixel", action="store_true", help="No-Pixel-Zonen rot einzeichnen")
    p.add_argument("--debug_zones", dest="debug_no_pixel", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_plates", action="store_true", help="Kennzeichen-Erkennung deaktivieren")
    p.add_argument("--no_faces", action="store_true", help="Gesichts-Erkennung deaktivieren")
    p.add_argument(
        "--tiling",
        type=int,
        default=2,
        help="Tiling fuer kleine Objekte (1-10, 1 = aus). Empfohlen: 1-4",
    )
    p.add_argument(
        "--tile_overlap",
        type=float,
        default=DEFAULT_TILE_OVERLAP,
        help=f"Ueberlappung benachbarter Tiles als Anteil (0.0-0.5). Standard: {DEFAULT_TILE_OVERLAP}",
    )
    p.add_argument(
        "--mask_ttl",
        type=int,
        default=DEFAULT_MASK_TTL,
        help=f"Erkannte Boxen bei kurzen Aussetzern N Frames weiter maskieren (0 = aus). Standard: {DEFAULT_MASK_TTL}",
    )
    p.add_argument(
        "--test_minutes",
        type=int,
        default=0,
        help="Nur die ersten N Minuten verarbeiten (0 = alles). Empfohlen: 0-60",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="Log alle n Frames mit aktuellen und durchschnittlichen FPS. Empfohlen: 50-1000",
    )
    p.add_argument(
        "--log_seconds",
        type=int,
        default=5,
        help="Log spaetestens alle N Sekunden (0 = aus). Hilfreich bei langsamem Tiling",
    )
    p.add_argument(
        "--save_preset",
        choices=["off", "json", "txt", "both"],
        default="off",
        help="Speichert verwendete Parameter als Preset im Output-Ordner.",
    )
    p.add_argument(
        "--load_preset",
        default="",
        help="Lade Preset (JSON) per Dateipfad oder Namen.",
    )
    p.add_argument(
        "--no_pixel_zone_px1",
        default="",
        help="Unveraenderte Zone als x1,y1,x2,y2; ueberlappende Boxteile ausserhalb werden weiter anonymisiert. Leer = aus",
    )
    p.add_argument(
        "--no_pixel_zone_px2",
        default="",
        help="Zweite No-Pixel-Zone in Pixeln als x1,y1,x2,y2. Leer = aus",
    )
    p.add_argument(
        "--no_pixel_zone_px3",
        default="",
        help="Dritte No-Pixel-Zone in Pixeln als x1,y1,x2,y2. Leer = aus",
    )
    p.add_argument(
        "--no_pixel_zone_px4",
        default="",
        help="Vierte No-Pixel-Zone in Pixeln als x1,y1,x2,y2. Leer = aus",
    )
    return p.parse_args()


def prompt_value(label: str) -> str:
    return input(f"{label}: ").strip()


def normalize_input_arg(args: argparse.Namespace) -> None:
    if isinstance(args.input, list):
        args.input = ",".join(args.input)


def resolve_paths(args: argparse.Namespace) -> None:
    if args.input:
        return
    if not sys.stdin.isatty():
        return
    print("Interaktiver Modus: Bitte fehlende Werte eingeben.")
    if not args.input:
        args.input = prompt_value("Input-Video (z.B. input.mp4)")


def list_models_in_dir(path: str) -> list:
    if not os.path.isdir(path):
        return []
    candidates = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".pt"):
                candidates.append(os.path.join(root, name))
    candidates.sort()
    return candidates


def parse_model_list(value: str) -> list:
    if not value:
        return []
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts


def apply_preset(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.preset, PRESETS["quality"])
    if args.conf is None:
        args.conf = float(preset["conf"])
    if args.imgsz is None:
        args.imgsz = int(preset["imgsz"])
    if args.work_w is None:
        args.work_w = int(preset["work_w"])
    if args.blocks_plates is None:
        args.blocks_plates = int(args.blocks) if args.blocks is not None else int(preset["blocks_plates"])
    if args.blocks_faces is None:
        args.blocks_faces = int(preset["blocks_faces"])
    if args.pad is None:
        args.pad = int(preset["pad"])
    if args.bitrate is None:
        args.bitrate = str(preset["bitrate"])


def validate_args(args: argparse.Namespace) -> Tuple[bool, list]:
    errors = []
    if args.conf is not None and not (0.0 <= args.conf <= 1.0):
        errors.append("Ungueltiger Wert fuer --conf (erlaubt: 0.0-1.0).")
    if args.imgsz is not None:
        if args.imgsz < 64 or args.imgsz > 4096:
            errors.append("Ungueltiger Wert fuer --imgsz (empfohlen: 640-2048).")
        if args.imgsz % 32 != 0:
            errors.append("Ungueltiger Wert fuer --imgsz (muss ein Vielfaches von 32 sein).")
    if args.work_w is not None and args.work_w < 0:
        errors.append("Ungueltiger Wert fuer --work_w (muss >= 0 sein).")
    for name, value in [("blocks", args.blocks), ("blocks_plates", args.blocks_plates), ("blocks_faces", args.blocks_faces)]:
        if value is not None and value <= 0:
            errors.append(f"Ungueltiger Wert fuer --{name} (muss > 0 sein).")
    if args.blur_ksize is not None and args.blur_ksize <= 0:
        errors.append("Ungueltiger Wert fuer --blur_ksize (muss > 0 sein).")
    if args.pad is not None and args.pad < 0:
        errors.append("Ungueltiger Wert fuer --pad (muss >= 0 sein).")
    if args.snapshot_every is not None and args.snapshot_every < 0:
        errors.append("Ungueltiger Wert fuer --snapshot_every (muss >= 0 sein).")
    if args.fade_seconds is not None and args.fade_seconds < 0:
        errors.append("Ungueltiger Wert fuer --fade_seconds (muss >= 0 sein).")
    if args.tiling is not None and (args.tiling < 1 or args.tiling > 10):
        errors.append("Ungueltiger Wert fuer --tiling (1-10).")
    if args.tile_overlap is not None and not (0.0 <= args.tile_overlap <= 0.5):
        errors.append("Ungueltiger Wert fuer --tile_overlap (0.0-0.5).")
    if args.mask_ttl is not None and (args.mask_ttl < 0 or args.mask_ttl > 120):
        errors.append("Ungueltiger Wert fuer --mask_ttl (0-120).")
    if args.test_minutes is not None and args.test_minutes < 0:
        errors.append("Ungueltiger Wert fuer --test_minutes (muss >= 0 sein).")
    if args.log_every is not None and args.log_every < 0:
        errors.append("Ungueltiger Wert fuer --log_every (muss >= 0 sein).")
    if args.snapshot_size:
        try:
            sw, sh = args.snapshot_size.lower().split("x")
            if int(sw) <= 0 or int(sh) <= 0:
                raise ValueError
        except Exception:
            errors.append("Ungueltiger Wert fuer --snapshot_size (Format: WIDTHxHEIGHT, z.B. 1920x1080).")
    return (len(errors) == 0, errors)


def preset_base_name(args: argparse.Namespace) -> str:
    source = args.input or args.output or "preset"
    base = os.path.splitext(os.path.basename(source))[0]
    return f"{base}_preset"


def arg_provided(flag: str) -> bool:
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def load_preset_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "params" in data and isinstance(data["params"], dict):
        return data["params"]
    if isinstance(data, dict):
        return data
    raise ValueError("Preset JSON hat kein gueltiges Format.")


def apply_loaded_preset(args: argparse.Namespace, params: dict) -> None:
    field_map = {
        "weights": "--weights",
        "faces_weights": "--faces_weights",
        "extra_weights": "--extra_weights",
        "use_extra": "--use_extra",
        "device": "--device",
        "conf": "--conf",
        "imgsz": "--imgsz",
        "work_w": "--work_w",
        "blocks": "--blocks",
        "blocks_plates": "--blocks_plates",
        "blocks_faces": "--blocks_faces",
        "anonymize": "--anonymize",
        "blur_ksize": "--blur_ksize",
        "pad": "--pad",
        "codec": "--codec",
        "bitrate": "--bitrate",
        "preset": "--preset",
        "force_sw": "--force_sw",
        "debug_pixel": "--debug_pixel",
        "debug_no_pixel": "--debug_no_pixel",
        "no_audio": "--no_audio",
        "fade_seconds": "--fade_seconds",
        "no_track": "--no_track",
        "snapshot_every": "--snapshot_every",
        "snapshot_dir": "--snapshot_dir",
        "snapshot_size": "--snapshot_size",
        "no_plates": "--no_plates",
        "no_faces": "--no_faces",
        "tiling": "--tiling",
        "tile_overlap": "--tile_overlap",
        "mask_ttl": "--mask_ttl",
        "test_minutes": "--test_minutes",
        "log_every": "--log_every",
        "no_pixel_zone_px1": "--no_pixel_zone_px1",
        "no_pixel_zone_px2": "--no_pixel_zone_px2",
        "no_pixel_zone_px3": "--no_pixel_zone_px3",
        "no_pixel_zone_px4": "--no_pixel_zone_px4",
    }
    for key, flag in field_map.items():
        if key not in params:
            continue
        if arg_provided(flag):
            continue
        setattr(args, key, params[key])


def save_preset_files(args: argparse.Namespace) -> None:
    if args.save_preset == "off":
        return
    preset_name = preset_base_name(args)
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    params = {
        "weights": args.weights,
        "faces_weights": args.faces_weights,
        "extra_weights": args.extra_weights,
        "use_extra": args.use_extra,
        "device": args.device,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "work_w": args.work_w,
        "blocks": args.blocks,
        "blocks_plates": args.blocks_plates,
        "blocks_faces": args.blocks_faces,
        "anonymize": args.anonymize,
        "blur_ksize": args.blur_ksize,
        "pad": args.pad,
        "codec": args.codec,
        "bitrate": args.bitrate,
        "preset": args.preset,
        "force_sw": args.force_sw,
        "debug_pixel": args.debug_pixel,
        "debug_no_pixel": args.debug_no_pixel,
        "no_audio": args.no_audio,
        "fade_seconds": args.fade_seconds,
        "no_track": args.no_track,
        "snapshot_every": args.snapshot_every,
        "snapshot_dir": args.snapshot_dir,
        "snapshot_size": args.snapshot_size,
        "no_plates": args.no_plates,
        "no_faces": args.no_faces,
        "tiling": args.tiling,
        "tile_overlap": args.tile_overlap,
        "mask_ttl": args.mask_ttl,
        "test_minutes": args.test_minutes,
        "log_every": args.log_every,
        "no_pixel_zone_px1": args.no_pixel_zone_px1,
        "no_pixel_zone_px2": args.no_pixel_zone_px2,
        "no_pixel_zone_px3": args.no_pixel_zone_px3,
        "no_pixel_zone_px4": args.no_pixel_zone_px4,
    }
    if args.save_preset in ("json", "both"):
        json_path = os.path.join(output_dir, f"{preset_name}.json")
        payload = {
            "preset_name": preset_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
    if args.save_preset in ("txt", "both"):
        txt_path = os.path.join(output_dir, f"{preset_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"preset_name: {preset_name}\n")
            f.write(f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in sorted(params.items()):
                f.write(f"{key}: {value}\n")

def extract_ts_from_path(path: str) -> str:
    m = re.search(r"\d{8}-\d{6}", os.path.basename(path))
    return m.group(0) if m else ""



def build_output_path(args: argparse.Namespace, ts: str, output_dir: str = "") -> str:
    in_dir = output_dir or os.path.dirname(args.input)
    in_base = os.path.splitext(os.path.basename(args.input))[0]
    weights_base = "models"
    if args.weights:
        weights_base = "plates"
    if args.faces_weights:
        weights_base = f"{weights_base}-faces"
    if args.extra_weights or args.use_extra:
        weights_base = f"{weights_base}-extra"
    test_tag = f"_test{args.test_minutes}m" if args.test_minutes else ""
    fname = f"{in_base}_plates_{weights_base}_{args.preset}_{args.codec}{test_tag}_{ts}.mp4"
    return os.path.join(in_dir, fname)


def output_is_mp4_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".mp4"


def output_is_directory_target(path: str) -> bool:
    return os.path.isdir(path) or os.path.splitext(path)[1] == ""


def expand_input_item(input_item: str) -> list:
    if os.path.isdir(input_item):
        videos = []
        for name in os.listdir(input_item):
            path = os.path.join(input_item, name)
            if os.path.isfile(path) and name.lower().endswith(".mp4"):
                videos.append(path)
        videos.sort()
        return videos
    if glob.has_magic(input_item):
        matches = [path for path in glob.glob(input_item) if os.path.isfile(path) and path.lower().endswith(".mp4")]
        matches.sort()
        return matches
    return [input_item]


def resolve_input_videos(input_value: str) -> Tuple[list, bool]:
    videos = []
    seen = set()
    input_items = [item.strip() for item in input_value.split(",") if item.strip()]
    for item in input_items:
        for path in expand_input_item(item):
            norm_path = os.path.abspath(path)
            if norm_path in seen:
                continue
            seen.add(norm_path)
            videos.append(path)
    return videos, len(input_items) > 1 or len(videos) > 1 or (len(input_items) == 1 and os.path.isdir(input_items[0]))


def process_video(
    args: argparse.Namespace,
    plate_models: list,
    face_models: list,
    extra_models: list,
    models: list,
    run_ts: str,
) -> int:
    exit_code = 0

    if not args.output:
        args.output = build_output_path(args, run_ts)
    save_preset_files(args)

    try:
        cap = open_video(args.input)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    fps = get_fps(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w <= 0 or h <= 0:
        print("Konnte Videoauflosung nicht ermitteln.", file=sys.stderr)
        cap.release()
        return 2

    use_sw = args.force_sw or os.environ.get("FORCE_SW", "") == "1"
    bitrate = args.bitrate
    if isinstance(bitrate, str) and bitrate.lower() == "auto":
        bitrate = probe_bitrate(args.input)
    bitrate_used = bitrate

    proc = None
    aborted = False
    frame_idx = 0
    max_frames = 0
    if args.test_minutes and args.test_minutes > 0:
        max_frames = int(fps * 60 * args.test_minutes)
    if max_frames and total_frames > 0:
        total_frames = min(total_frames, max_frames)
    duration_frames = total_frames if total_frames > 0 else max_frames
    duration_seconds = (duration_frames / fps) if duration_frames > 0 and fps > 0 else 0.0
    cmd = build_ffmpeg_cmd(
        args.output,
        args.input,
        w,
        h,
        fps,
        args.codec,
        bitrate,
        use_sw,
        not args.no_audio,
        args.fade_seconds,
        duration_seconds,
    )
    start_time = time.time()
    last_log_time = start_time
    last_log_frame = 0
    snapshot_count = 0
    in_base = os.path.splitext(os.path.basename(args.input))[0]
    snapshot_prefix = f"{in_base}_snap_{run_ts}"
    next_snapshot_frame = None
    snap_w = snap_h = 0
    if args.snapshot_every and args.snapshot_every > 0:
        if not args.snapshot_dir:
            args.snapshot_dir = os.path.dirname(args.input) or "."
        os.makedirs(args.snapshot_dir, exist_ok=True)
        try:
            sw, sh = args.snapshot_size.lower().split("x")
            snap_w, snap_h = int(sw), int(sh)
        except Exception:
            snap_w, snap_h = 1920, 1080
        next_snapshot_frame = int(fps * 60 * args.snapshot_every)
    track_notice_printed = False
    zones_px = []
    for zone_arg in [args.no_pixel_zone_px1, args.no_pixel_zone_px2, args.no_pixel_zone_px3, args.no_pixel_zone_px4]:
        if not zone_arg:
            continue
        try:
            zx1, zy1, zx2, zy2 = [int(v) for v in zone_arg.split(",")]
            zx1, zx2 = sorted((zx1, zx2))
            zy1, zy2 = sorted((zy1, zy2))
            zone = clip_box((zx1, zy1, zx2, zy2), w, h)
            if box_area(zone) > 0:
                zones_px.append(zone)
        except Exception:
            continue
    temporal_tracks = []
    ultralytics_tracking_ready, tracker_reset_errors = reset_ultralytics_trackers(models)
    for error in tracker_reset_errors:
        print(
            f"Warnung: Ultralytics-Tracker konnte nicht zurueckgesetzt werden ({error}); "
            "Tracking ist fuer dieses Video deaktiviert.",
            file=sys.stderr,
        )
    for model, _ in models:
        if not install_raw_detection_capture(model):
            print(
                "Rohe YOLO-Detektionen koennen nicht sicher abgegriffen werden; Verarbeitung abgebrochen.",
                file=sys.stderr,
            )
            cap.release()
            return 2
    tracking_requested = (not args.no_track) and args.tiling == 1 and ultralytics_tracking_ready
    tracking_disabled_models = set()
    model_devices = {
        id(model): getattr(model, "_dsgvo_effective_device", args.device)
        for model, _ in models
    }

    def print_progress(processed_frames: int, current_frame: int = 0) -> None:
        nonlocal last_log_time, last_log_frame
        now = time.time()
        elapsed = now - start_time
        log_elapsed = now - last_log_time
        log_frames = processed_frames - last_log_frame
        fps_avg = processed_frames / elapsed if elapsed > 0 and processed_frames > 0 else 0.0
        fps_current = log_frames / log_elapsed if log_elapsed > 0 else fps_avg
        last_log_time = now
        last_log_frame = processed_frames
        prefix = f"Processed: {processed_frames}"
        if current_frame > processed_frames:
            prefix = f"Processing frame: {current_frame} | processed: {processed_frames}"
        if total_frames > 0 and fps_avg > 0:
            remaining = max(total_frames - processed_frames, 0)
            eta_sec = int(remaining / fps_avg)
            eta_min = eta_sec // 60
            eta_rem = eta_sec % 60
            pct = (processed_frames / total_frames) * 100.0
            print(
                f"{prefix} | {pct:.1f}% | ETA {eta_min}m {eta_rem}s | "
                f"{fps_current:.2f} fps current | {fps_avg:.2f} fps avg",
                flush=True,
            )
        else:
            print(f"{prefix} | {fps_current:.2f} fps current | {fps_avg:.2f} fps avg", flush=True)

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if total_frames > 0:
            print(f"Starte Verarbeitung: 0/{total_frames} Frames | Status alle {args.log_seconds}s oder {args.log_every} Frames", flush=True)
        else:
            print(f"Starte Verarbeitung: Status alle {args.log_seconds}s oder {args.log_every} Frames", flush=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            det_frame = frame
            scale_x = 1.0
            scale_y = 1.0
            if args.work_w and args.work_w > 0 and args.work_w < w:
                requested_scale = args.work_w / float(w)
                new_h = max(1, int(h * requested_scale))
                det_frame = cv2.resize(frame, (args.work_w, new_h), interpolation=cv2.INTER_AREA)

            det_h, det_w = det_frame.shape[:2]
            scale_x = det_w / float(w)
            scale_y = det_h / float(h)
            tiles = build_tiles(det_w, det_h, args.tiling, args.tile_overlap)

            if args.tiling > 1 and not args.no_track and not track_notice_printed:
                print(
                    "Hinweis: Ultralytics-Tracking ist bei Tiling deaktiviert; "
                    f"zeitliche Sicherheitsmasken bleiben fuer {args.mask_ttl} Frames aktiv.",
                    file=sys.stderr,
                )
                track_notice_printed = True

            detections_by_kind = {}
            for x0, y0, x1, y1 in tiles:
                tile = det_frame[y0:y1, x0:x1]
                for model, kind in models:
                    model_id = id(model)
                    model_use_tracking = tracking_requested and model_id not in tracking_disabled_models
                    effective_device = model_devices[model_id]
                    predictor = getattr(model, "predictor", None)
                    if predictor is not None:
                        predictor._dsgvo_raw_xyxy = None
                    try:
                        if model_use_tracking:
                            results = model.track(
                                tile,
                                conf=args.conf,
                                imgsz=args.imgsz,
                                device=effective_device,
                                verbose=False,
                                persist=True,
                            )
                        else:
                            results = model.predict(
                                tile,
                                conf=args.conf,
                                imgsz=args.imgsz,
                                device=effective_device,
                                verbose=False,
                            )
                    except Exception as e:
                        if str(effective_device).lower() != "cpu":
                            error_summary = str(e).strip().splitlines()[0] or type(e).__name__
                            print(
                                f"Warnung: Inferenz auf {effective_device} fehlgeschlagen ({error_summary}); "
                                "dieses Modell verwendet ab jetzt CPU.",
                                file=sys.stderr,
                            )
                            effective_device = "cpu"
                            model_devices[model_id] = effective_device
                            model._dsgvo_effective_device = effective_device
                            if model_use_tracking:
                                reset_ready, reset_errors = reset_ultralytics_trackers([(model, kind)])
                                if not reset_ready:
                                    tracking_disabled_models.add(model_id)
                                    model_use_tracking = False
                                    for error in reset_errors:
                                        print(
                                            f"Warnung: Tracker-Neustart fehlgeschlagen ({error}); "
                                            "Tracking ist fuer dieses Modell deaktiviert.",
                                            file=sys.stderr,
                                        )
                            predictor = getattr(model, "predictor", None)
                            if predictor is not None:
                                predictor._dsgvo_raw_xyxy = None
                            # model.track() registered the tracker callbacks before the failed call.
                            # predict() avoids registering them a second time during the device switch.
                            results = model.predict(
                                tile,
                                conf=args.conf,
                                imgsz=args.imgsz,
                                device="cpu",
                                verbose=False,
                            )
                        else:
                            raise e

                    predictor = getattr(model, "predictor", None)
                    raw_results = getattr(predictor, "_dsgvo_raw_xyxy", None)
                    if raw_results is None or len(raw_results) != 1:
                        raise RuntimeError("Rohe YOLO-Detektionen fehlen; unsichere Tracker-Ausgabe wird nicht verwendet")
                    xyxy_batches = [raw_results[0]]
                    if model_use_tracking and results:
                        tracked_boxes = results[0].boxes
                        if tracked_boxes is not None and len(tracked_boxes) > 0:
                            xyxy_batches.append(tracked_boxes.xyxy.cpu().numpy())
                    kind_boxes = detections_by_kind.setdefault(kind, [])
                    for xyxy_list in xyxy_batches:
                        for xyxy in xyxy_list:
                            box = map_detection_box(xyxy, x0, y0, scale_x, scale_y, w, h)
                            if box_area(box) > 0:
                                kind_boxes.append(box)

                    if args.log_seconds > 0 and (time.time() - last_log_time) >= args.log_seconds:
                        print_progress(frame_idx, frame_idx + 1)

            current_detections = []
            for kind in sorted(detections_by_kind):
                for box in merge_overlapping_boxes(detections_by_kind[kind]):
                    current_detections.append((kind, box))

            temporal_masks = update_temporal_masks(
                current_detections,
                temporal_tracks,
                args.mask_ttl,
                w,
                h,
            )
            masks_by_kind = {}
            for kind, box, missed in temporal_masks:
                pad = args.pad
                if missed > 0:
                    max_side = max(box[2] - box[0], box[3] - box[1])
                    pad += max(2 * missed, int(math.ceil(max_side * 0.08 * missed)))
                padded = apply_pad(box[0], box[1], box[2], box[3], pad, w, h)
                if box_area(padded) > 0:
                    masks_by_kind.setdefault(kind, []).append(padded)

            debug_boxes = []
            for kind in sorted(masks_by_kind):
                blocks_val = args.blocks_faces if kind == "faces" else args.blocks_plates
                for box in merge_overlapping_boxes(masks_by_kind[kind]):
                    anonymize_box_excluding_zones(
                        frame,
                        box,
                        zones_px,
                        args.anonymize,
                        args.blur_ksize,
                        blocks_val,
                    )
                    debug_boxes.append(box)

            if args.debug_pixel:
                for bx1, by1, bx2, by2 in debug_boxes:
                    cv2.rectangle(frame, (bx1, by1), (max(bx1, bx2 - 1), max(by1, by2 - 1)), (0, 255, 0), 2)
            if args.debug_no_pixel:
                for zx1, zy1, zx2, zy2 in zones_px:
                    cv2.rectangle(frame, (zx1, zy1), (max(zx1, zx2 - 1), max(zy1, zy2 - 1)), (0, 0, 255), 3)

            if next_snapshot_frame and frame_idx >= next_snapshot_frame:
                snap = frame
                if snap_w > 0 and snap_h > 0:
                    snap = cv2.resize(frame, (snap_w, snap_h), interpolation=cv2.INTER_AREA)
                snap_ts = int(frame_idx / fps) if fps > 0 else frame_idx
                snap_name = f"{snapshot_prefix}_{snapshot_count:04d}_{snap_ts:06d}.jpg"
                cv2.imwrite(os.path.join(args.snapshot_dir, snap_name), snap)
                snapshot_count += 1
                next_snapshot_frame += int(fps * 60 * args.snapshot_every)

            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin nicht verfuegbar")
            proc.stdin.write(frame.tobytes())

            frame_idx += 1
            log_by_frame = args.log_every > 0 and frame_idx % args.log_every == 0
            log_by_time = args.log_seconds > 0 and (time.time() - last_log_time) >= args.log_seconds
            if log_by_frame or log_by_time:
                print_progress(frame_idx)
            if max_frames and frame_idx >= max_frames:
                break

    except KeyboardInterrupt:
        print("Abbruch durch Benutzer (Ctrl+C).", file=sys.stderr)
        aborted = True
        exit_code = 130
    except BrokenPipeError:
        print("ffmpeg Pipe abgebrochen", file=sys.stderr)
        exit_code = 3
    except Exception as e:
        print(f"Fehler: {e}", file=sys.stderr)
        exit_code = 3
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if proc and proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc:
            try:
                if aborted and proc.poll() is None:
                    proc.terminate()
                ret = proc.wait()
                if ret != 0 and not aborted:
                    print(f"ffmpeg exit code: {ret}", file=sys.stderr)
                    exit_code = 3
            except Exception:
                pass

    if exit_code == 0:
        elapsed = time.time() - start_time
        proc_min = int(elapsed // 60)
        proc_sec = int(elapsed % 60)
        dur_sec = int(frame_idx / fps) if fps > 0 else 0
        dur_min = dur_sec // 60
        dur_rem = dur_sec % 60
        print("")
        print("Fertig.")
        print(f"Output: {args.output}")
        print(f"Input: {args.input}")
        print(f"Aufloesung: {w}x{h} @ {fps:.2f} fps")
        print(f"Verarbeitet: {frame_idx} Frames (~{dur_min}m {dur_rem}s)")
        print(f"Bitrate: {bitrate_used}")
        targets = []
        if plate_models:
            targets.append(f"Kennzeichen({len(plate_models)})")
        if face_models:
            targets.append(f"Gesichter({len(face_models)})")
        if extra_models:
            targets.append(f"Extra({len(extra_models)})")
        print(f"Objekte: {', '.join(targets) if targets else 'keine'}")
        print(f"Encoder: {args.codec}{' (SW)' if use_sw else ' (HW)'}")
        print(f"Audio: {'aus' if args.no_audio else 'an'}")
        print(f"Video-Fade: {args.fade_seconds:g}s" if args.fade_seconds > 0 else "Video-Fade: aus")
        tracking_active_models = len(models) - len(tracking_disabled_models) if tracking_requested else 0
        if tracking_active_models == len(models) and tracking_active_models > 0:
            tracking_state = "an"
        elif tracking_active_models > 0:
            tracking_state = "teilweise"
        else:
            tracking_state = "aus"
        print(f"Ultralytics-Tracking: {tracking_state}")
        print(f"Zeitliche Sicherheitsmaske: {args.mask_ttl} Frames" if args.mask_ttl > 0 else "Zeitliche Sicherheitsmaske: aus")
        print(f"Tiling: {args.tiling}x{args.tiling} | Overlap: {args.tile_overlap:g}")
        if snapshot_count:
            print(f"Snapshots: {snapshot_count} ({args.snapshot_dir})")
        print(f"Dauer: {proc_min}m {proc_sec}s")
    return exit_code


def main() -> int:
    if len(sys.argv) == 1:
        print("DSGVO-Pixeler - Kennzeichen & Gesichter anonymisieren (einfacher Start)")
        print("Vorbereitung (einmalig):")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -U pip")
        print("  pip install -r requirements.txt")
        print("Beispiel:")
        print("  python dsgvo-pixeler.py --input input.mp4 --output output.mp4 --weights models/plates/best.pt")
        print("Kurz-Erklaerung:")
        print("  Erkennt Kennzeichen und Gesichter im Video und anonymisiert sie fuer Datenschutz.")
        print("Wichtige Optionen (kurz):")
        print("  --codec hevc|h264     (Standard: hevc)")
        print("  --preset fast|balanced|quality (Standard: quality)")
        print("  --bitrate auto        (passt Bitrate an das Original an)")
        print("  --work_w 0            (Standard quality: Originalbreite; 1920 = schneller)")
        print("  --imgsz 1600          (Standard quality; groesser = genauer, langsamer)")
        print("  --conf 0.2            (Standard quality; niedriger = mehr Treffer)")
        print("  --anonymize blur|pixelate (Standard: blur)")
        print("  --blur_ksize 80       (Blur-Staerke, gerade Werte werden auf ungerade aufgerundet)")
        print("  --blocks_plates 16    (Kennzeichen, groesser = grober)")
        print("  --blocks_faces 24     (Gesichter, groesser = grober)")
        print("  --blocks 16           (deprecated)")
        print("  --pad 24              (Standard quality; Sicherheitsrand)")
        print("  --no_pixel_zone_px1 120,1500,900,2160 (nur der Zonenanteil bleibt unveraendert)")
        print("  --test_minutes 2      (nur erste 2 Minuten verarbeiten)")
        print("  --debug_pixel         (BBox-Overlay fuer Debug)")
        print("  --debug_no_pixel      (No-Pixel-Zonen rot einzeichnen)")
        print("  --no_audio            (Audio entfernen)")
        print("  --fade_seconds 1.5    (Video-Fade aus/nach Schwarz, 0 = aus)")
        print("  --no_track            (Ultralytics-Tracking deaktivieren; Sicherheitsmaske bleibt aktiv)")
        print("  --log_seconds 5       (Status spaetestens alle 5 Sekunden)")
        print("  --snapshot_every 5    (Snapshot alle 5 Minuten)")
        print("  --snapshot_size 1920x1080 (Snapshot-Groesse)")
        print("  --snapshot_dir /pfad/zu/ordner  (Default: Input-Ordner)")
        print("  --input /pfad/zu/ordner  (Batch: alle .mp4-Dateien im Ordner)")
        print("  --input '*.mp4' oder a.mp4,b.mp4  (Glob oder Mehrfachinput)")
        print("  --tiling 2            (2x2 Tiling fuer kleine Kennzeichen, default)")
        print("  --tile_overlap 0.2    (Ueberlappung gegen Trefferluecken an Tile-Grenzen)")
        print("  --mask_ttl 3          (Masken bei kurzen Erkennungsaussetzern weiterfuehren)")
        print("  --no_plates           (nur Gesichter anonymisieren)")
        print("  --no_faces            (nur Kennzeichen anonymisieren)")
        print("  --force_sw            (Software-Encoding erzwingen)")
        print("Weitere Hilfe:")
        print("  python dsgvo-pixeler.py -h")
        return 0
    args = parse_args()
    normalize_input_arg(args)
    resolve_paths(args)
    normalize_input_arg(args)
    if args.load_preset:
        preset_path = args.load_preset
        if not os.path.isfile(preset_path):
            base_dir = os.path.dirname(args.output or args.input or "") or "."
            if not preset_path.endswith(".json"):
                preset_path = os.path.join(base_dir, f"{preset_path}.json")
            else:
                preset_path = os.path.join(base_dir, preset_path)
        if not os.path.isfile(preset_path):
            print(f"Preset-Datei nicht gefunden: {preset_path}", file=sys.stderr)
            return 2
        try:
            preset_params = load_preset_file(preset_path)
        except Exception as e:
            print(f"Preset konnte nicht geladen werden: {e}", file=sys.stderr)
            return 2
        apply_loaded_preset(args, preset_params)
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", "plates"), exist_ok=True)
    os.makedirs(os.path.join("models", "faces"), exist_ok=True)
    os.makedirs(os.path.join("models", "extra"), exist_ok=True)

    plate_models = []
    face_models = []
    extra_models = []

    if not args.no_plates:
        if args.weights:
            plate_models = parse_model_list(args.weights)
        else:
            plate_models = list_models_in_dir(os.path.join("models", "plates"))
    if not args.no_faces:
        if args.faces_weights:
            face_models = parse_model_list(args.faces_weights)
        else:
            face_models = list_models_in_dir(os.path.join("models", "faces"))
    if args.extra_weights:
        extra_models = parse_model_list(args.extra_weights)
    elif args.use_extra:
        extra_models = list_models_in_dir(os.path.join("models", "extra"))

    if not args.no_plates and not plate_models:
        print(
            "Kennzeichen-Modelle nicht gefunden. Lege .pt Dateien in models/plates/ oder nutze --weights.\n"
            "Beispiel: --weights models/plates/best.pt",
            file=sys.stderr,
        )
        return 2
    if not args.no_faces and not face_models:
        print(
            "Gesichts-Modelle nicht gefunden. Lege .pt Dateien in models/faces/ oder nutze --faces_weights.\n"
            "Beispiel: --faces_weights models/faces/face1.pt",
            file=sys.stderr,
        )
        return 2
    apply_preset(args)
    ok, errors = validate_args(args)
    if not ok:
        for msg in errors:
            print(msg, file=sys.stderr)
        return 2

    if not args.input:
        print("Input fehlt. Bitte --input angeben.", file=sys.stderr)
        return 2

    input_paths, is_batch = resolve_input_videos(args.input)
    if not input_paths:
        print(f"Keine .mp4-Dateien gefunden fuer: {args.input}", file=sys.stderr)
        return 2
    if is_batch and args.output and output_is_mp4_file(args.output):
        print("Bei mehreren Inputs muss --output ein Zielordner sein, keine einzelne .mp4-Datei.", file=sys.stderr)
        return 2

    output_dir = ""
    if is_batch and args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    elif args.output and output_is_directory_target(args.output):
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

    run_ts = ""
    if args.output and not is_batch and not output_dir:
        run_ts = extract_ts_from_path(args.output)
    if not run_ts:
        run_ts = time.strftime("%Y%m%d-%H%M%S")
    if output_dir and not is_batch:
        args.output = build_output_path(args, run_ts, output_dir)

    for path in plate_models:
        if not os.path.isfile(path):
            print(f"Kennzeichen-Weights nicht gefunden: {path}", file=sys.stderr)
            return 2
    for path in face_models:
        if not os.path.isfile(path):
            print(f"Gesichts-Weights nicht gefunden: {path}", file=sys.stderr)
            return 2
    for path in extra_models:
        if not os.path.isfile(path):
            print(f"Extra-Weights nicht gefunden: {path}", file=sys.stderr)
            return 2

    if not shutil_which("ffmpeg"):
        print("ffmpeg nicht gefunden. Bitte installieren: brew install ffmpeg", file=sys.stderr)
        return 2
    if (args.bitrate or "").lower() == "auto" and not shutil_which("ffprobe"):
        print("ffprobe nicht gefunden. Bitte ffmpeg komplett installieren: brew install ffmpeg", file=sys.stderr)
        return 2

    models = []
    for path in plate_models:
        try:
            models.append((YOLO(path), "plates"))
        except Exception as e:
            print(f"YOLO Kennzeichen-Modell konnte nicht geladen werden: {path} ({e})", file=sys.stderr)
            return 2
    for path in face_models:
        try:
            models.append((YOLO(path), "faces"))
        except Exception as e:
            print(f"YOLO Gesichts-Modell konnte nicht geladen werden: {path} ({e})", file=sys.stderr)
            return 2
    for path in extra_models:
        try:
            models.append((YOLO(path), "extra"))
        except Exception as e:
            print(f"YOLO Extra-Modell konnte nicht geladen werden: {path} ({e})", file=sys.stderr)
            return 2

    if is_batch:
        print(f"Batch-Modus: {len(input_paths)} .mp4-Datei(en) gefunden.")

    exit_code = 0
    batch_start = time.time()
    batch_results = []
    for index, input_path in enumerate(input_paths, start=1):
        run_args = argparse.Namespace(**vars(args))
        run_args.input = input_path
        if is_batch:
            run_args.output = build_output_path(run_args, run_ts, output_dir)
            run_args.snapshot_dir = args.snapshot_dir
            print("")
            print(f"Batch {index}/{len(input_paths)}: {input_path}")
        ret = process_video(run_args, plate_models, face_models, extra_models, models, run_ts)
        batch_results.append((input_path, run_args.output, ret))
        if ret != 0:
            exit_code = ret
            if ret == 130:
                break
    if is_batch:
        elapsed = time.time() - batch_start
        batch_min = int(elapsed // 60)
        batch_sec = int(elapsed % 60)
        ok_count = sum(1 for _, _, ret in batch_results if ret == 0)
        failed = [(inp, ret) for inp, _, ret in batch_results if ret != 0]
        not_started = len(input_paths) - len(batch_results)
        print("")
        print("Batch-Zusammenfassung")
        print(f"Gefunden: {len(input_paths)}")
        print(f"Erfolgreich: {ok_count}")
        print(f"Fehlgeschlagen: {len(failed)}")
        if not_started:
            print(f"Nicht gestartet: {not_started}")
        print(f"Dauer gesamt: {batch_min}m {batch_sec}s")
        if failed:
            print("Fehler:")
            for inp, ret in failed:
                print(f"  exit {ret}: {inp}")
    return exit_code


def shutil_which(cmd: str) -> str:
    try:
        import shutil
        return shutil.which(cmd)
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
