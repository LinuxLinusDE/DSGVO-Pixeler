#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from typing import Dict, Tuple

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


def boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


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
) -> list:
    if codec == "hevc":
        vcodec = "libx265" if use_sw else "hevc_videotoolbox"
    elif codec == "h264":
        vcodec = "libx264" if use_sw else "h264_videotoolbox"
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    pix_fmt = "yuv420p" if use_sw else "nv12"
    cmd = [
        "ffmpeg",
        "-y",
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
        f"format={pix_fmt}",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kennzeichen in Videos erkennen und verpixeln (Apple Silicon/MPS).")
    p.add_argument("--input", help="Input-Video (MP4)")
    p.add_argument("--output", help="Output-Video (MP4)")
    p.add_argument("--weights", help="YOLOv8 plates weights (Liste mit Komma)")
    p.add_argument("--faces_weights", help="YOLOv8 face weights (Liste mit Komma)")
    p.add_argument("--extra_weights", help="Zusatz-Modelle (Liste mit Komma)")
    p.add_argument("--use_extra", action="store_true", help="models/extra/*.pt mitnutzen")
    p.add_argument("--device", default="mps", help="Ultralytics device, z.B. mps oder cpu")
    p.add_argument("--conf", type=float, default=None, help="Confidence Threshold")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO imgsz")
    p.add_argument("--work_w", type=int, default=None, help="Arbeitsbreite fuer Detektion (0 = Original)")
    p.add_argument("--blocks", type=int, default=None, help="Pixel-Blockgroesse (deprecated, groesser = grober)")
    p.add_argument("--blocks_plates", type=int, default=None, help="Pixel-Blockgroesse fuer Kennzeichen (groesser = grober)")
    p.add_argument("--blocks_faces", type=int, default=None, help="Pixel-Blockgroesse fuer Gesichter (groesser = grober)")
    p.add_argument("--pad", type=int, default=None, help="Sicherheitsrand in Pixel")
    p.add_argument("--codec", choices=["hevc", "h264"], default="hevc", help="Video codec")
    p.add_argument("--bitrate", default=None, help="Video bitrate, z.B. 50M oder auto")
    p.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Preset fuer Speed/Qualitaet",
    )
    p.add_argument("--force_sw", action="store_true", help="Software-Encoding erzwingen (libx265/libx264)")
    p.add_argument("--debug_overlay", action="store_true", help="BBox-Overlay fuer Debug einzeichnen")
    p.add_argument("--no_audio", action="store_true", help="Audio entfernen")
    p.add_argument("--debug_zones", action="store_true", help="No-Pixel-Zonen rot einzeichnen")
    p.add_argument("--no_plates", action="store_true", help="Kennzeichen-Erkennung deaktivieren")
    p.add_argument("--no_faces", action="store_true", help="Gesichts-Erkennung deaktivieren")
    p.add_argument("--test_minutes", type=int, default=0, help="Nur die ersten N Minuten verarbeiten (0 = alles)")
    p.add_argument("--log_every", type=int, default=200, help="Log alle n Frames")
    p.add_argument(
        "--no_pixel_zone",
        default="",
        help="No-Pixel-Zone in Prozent als x1,x2,y1,y2 (z.B. 0,20,63,100). Leer = aus",
    )
    p.add_argument(
        "--no_pixel_zone2",
        default="",
        help="Zweite No-Pixel-Zone in Prozent (z.B. 78,100,59,100). Leer = aus",
    )
    p.add_argument(
        "--no_pixel_zone_px1",
        default="",
        help="No-Pixel-Zone in Pixeln als x1,y1,x2,y2 (z.B. 120,1500,900,2160). Leer = aus",
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
    presets: Dict[str, Dict[str, object]] = {
        "fast": {"conf": 0.3, "imgsz": 960, "work_w": 1280, "blocks_plates": 16, "blocks_faces": 24, "pad": 20, "bitrate": "auto"},
        "balanced": {"conf": 0.25, "imgsz": 1280, "work_w": 1920, "blocks_plates": 16, "blocks_faces": 24, "pad": 20, "bitrate": "auto"},
        "quality": {"conf": 0.2, "imgsz": 1600, "work_w": 0, "blocks_plates": 16, "blocks_faces": 24, "pad": 24, "bitrate": "auto"},
    }
    preset = presets.get(args.preset, presets["balanced"])
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


def build_output_path(args: argparse.Namespace) -> str:
    in_dir = os.path.dirname(args.input)
    in_base = os.path.splitext(os.path.basename(args.input))[0]
    weights_base = "models"
    if args.weights:
        weights_base = "plates"
    if args.faces_weights:
        weights_base = f"{weights_base}-faces"
    if args.extra_weights or args.use_extra:
        weights_base = f"{weights_base}-extra"
    ts = time.strftime("%Y%m%d-%H%M%S")
    test_tag = f"_test{args.test_minutes}m" if args.test_minutes else ""
    fname = f"{in_base}_plates_{weights_base}_{args.preset}_{args.codec}{test_tag}_{ts}.mp4"
    return os.path.join(in_dir, fname)


def main() -> int:
    if len(sys.argv) == 1:
        print("Plater - Kennzeichen verpixeln (einfacher Start)")
        print("Vorbereitung (einmalig):")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -U pip")
        print("  pip install -r requirements.txt")
        print("Beispiel:")
        print("  python dsgvo-pixeler.py --input input.mp4 --output output.mp4 --weights models/plates/best.pt")
        print("Kurz-Erklaerung:")
        print("  Erkennt Kennzeichen im Video und verpixelt sie fuer Datenschutz.")
        print("Wichtige Optionen (kurz):")
        print("  --codec hevc|h264     (Standard: hevc)")
        print("  --preset fast|balanced|quality")
        print("  --bitrate auto        (passt Bitrate an das Original an)")
        print("  --work_w 1920         (schneller, etwas weniger genau)")
        print("  --imgsz 1280          (bessere Erkennung, langsamer)")
        print("  --conf 0.25           (niedriger = mehr Treffer)")
        print("  --blocks_plates 16    (Kennzeichen, groesser = grober)")
        print("  --blocks_faces 24     (Gesichter, groesser = grober)")
        print("  --blocks 16           (deprecated)")
        print("  --pad 20              (Sicherheitsrand)")
        print("  --no_pixel_zone 0,20,63,100  (optional, x1,x2,y1,y2)")
        print("  --no_pixel_zone2 78,100,59,100 (optional, x1,x2,y1,y2)")
        print("  --no_pixel_zone_px1 120,1500,900,2160 (Pixel-Zone, x1,y1,x2,y2)")
        print("  --test_minutes 2      (nur erste 2 Minuten verarbeiten)")
        print("  --debug_overlay       (BBox-Overlay fuer Debug)")
        print("  --no_audio            (Audio entfernen)")
        print("  --debug_zones         (No-Pixel-Zonen rot einzeichnen)")
        print("  --no_audio            (Audio entfernen)")
        print("  --no_plates           (nur Gesichter verpixeln)")
        print("  --no_faces            (nur Kennzeichen verpixeln)")
        print("  --force_sw            (Software-Encoding erzwingen)")
        print("Weitere Hilfe:")
        print("  python dsgvo-pixeler.py -h")
        return 0
    args = parse_args()
    resolve_paths(args)
    os.makedirs("models", exist_ok=True)
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
        print("Kennzeichen-Modelle nicht gefunden. Bitte .pt Dateien nach models/plates legen oder --weights angeben.", file=sys.stderr)
        return 2
    if not args.no_faces and not face_models:
        print("Gesichts-Modelle nicht gefunden. Bitte .pt Dateien nach models/faces legen oder --faces_weights angeben.", file=sys.stderr)
        return 2
    apply_preset(args)
    exit_code = 0

    if not args.input:
        print("Input fehlt. Bitte --input angeben.", file=sys.stderr)
        return 2
    if not args.output:
        args.output = build_output_path(args)

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

    models = []
    for path in plate_models:
        try:
            models.append((YOLO(path), "plates"))
        except Exception as e:
            print(f"YOLO Kennzeichen-Modell konnte nicht geladen werden: {path} ({e})", file=sys.stderr)
            cap.release()
            return 2
    for path in face_models:
        try:
            models.append((YOLO(path), "faces"))
        except Exception as e:
            print(f"YOLO Gesichts-Modell konnte nicht geladen werden: {path} ({e})", file=sys.stderr)
            cap.release()
            return 2
    for path in extra_models:
        try:
            models.append((YOLO(path), "extra"))
        except Exception as e:
            print(f"YOLO Extra-Modell konnte nicht geladen werden: {path} ({e})", file=sys.stderr)
            cap.release()
            return 2

    use_sw = args.force_sw or os.environ.get("FORCE_SW", "") == "1"
    bitrate = args.bitrate
    if isinstance(bitrate, str) and bitrate.lower() == "auto":
        bitrate = probe_bitrate(args.input)
    bitrate_used = bitrate
    cmd = build_ffmpeg_cmd(args.output, args.input, w, h, fps, args.codec, bitrate, use_sw, not args.no_audio)

    proc = None
    frame_idx = 0
    max_frames = 0
    if args.test_minutes and args.test_minutes > 0:
        max_frames = int(fps * 60 * args.test_minutes)
    if max_frames and total_frames > 0:
        total_frames = min(total_frames, max_frames)
    start_time = time.time()
    zones = []
    for zone_arg in [args.no_pixel_zone, args.no_pixel_zone2]:
        try:
            zx1p, zx2p, zy1p, zy2p = [float(v) for v in zone_arg.split(",")]
            zones.append((zx1p, zy1p, zx2p, zy2p))
        except Exception:
            continue
    zones_px = []
    for zone_arg in [args.no_pixel_zone_px1, args.no_pixel_zone_px2, args.no_pixel_zone_px3, args.no_pixel_zone_px4]:
        try:
            zx1, zy1, zx2, zy2 = [int(v) for v in zone_arg.split(",")]
            zones_px.append((zx1, zy1, zx2, zy2))
        except Exception:
            continue
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            det_frame = frame
            scale = 1.0
            if args.work_w and args.work_w > 0 and args.work_w < w:
                scale = args.work_w / float(w)
                new_h = int(h * scale)
                det_frame = cv2.resize(frame, (args.work_w, new_h), interpolation=cv2.INTER_AREA)

            nz_list = []
            for zx1p, zy1p, zx2p, zy2p in zones:
                nz_list.append(
                    (
                        int(w * (zx1p / 100.0)),
                        int(h * (zy1p / 100.0)),
                        int(w * (zx2p / 100.0)),
                        int(h * (zy2p / 100.0)),
                    )
                )
            for zx1, zy1, zx2, zy2 in zones_px:
                nz_list.append((zx1, zy1, zx2, zy2))
            if args.debug_zones and nz_list:
                for zx1, zy1, zx2, zy2 in nz_list:
                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 3)

            for model, kind in models:
                try:
                    results = model.predict(
                        det_frame,
                        conf=args.conf,
                        imgsz=args.imgsz,
                        device=args.device,
                        verbose=False,
                    )
                except Exception as e:
                    if args.device != "cpu":
                        results = model.predict(
                            det_frame,
                            conf=args.conf,
                            imgsz=args.imgsz,
                            device="cpu",
                            verbose=False,
                        )
                    else:
                        raise e

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for b in boxes:
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy.tolist()
                        if scale != 1.0:
                            x1 = int(x1 / scale)
                            y1 = int(y1 / scale)
                            x2 = int(x2 / scale)
                            y2 = int(y2 / scale)
                        x1, y1, x2, y2 = apply_pad(x1, y1, x2, y2, args.pad, w, h)
                        if nz_list and any(boxes_overlap((x1, y1, x2, y2), nz) for nz in nz_list):
                            continue
                        blocks_val = args.blocks_faces if kind == "faces" else args.blocks_plates
                        pixelate_roi(frame, x1, y1, x2, y2, blocks_val)
                        if args.debug_overlay:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin nicht verfuegbar")
            proc.stdin.write(frame.tobytes())

            frame_idx += 1
            if args.log_every > 0 and frame_idx % args.log_every == 0:
                elapsed = time.time() - start_time
                fps_eff = frame_idx / elapsed if elapsed > 0 else 0.0
                if total_frames > 0 and fps_eff > 0:
                    remaining = max(total_frames - frame_idx, 0)
                    eta_sec = int(remaining / fps_eff)
                    eta_min = eta_sec // 60
                    eta_rem = eta_sec % 60
                    pct = (frame_idx / total_frames) * 100.0
                    print(f"Processed frames: {frame_idx} | {pct:.1f}% | ETA {eta_min}m {eta_rem}s")
                else:
                    print(f"Processed frames: {frame_idx}")
            if max_frames and frame_idx >= max_frames:
                break

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
                ret = proc.wait()
                if ret != 0:
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
        print(f"Dauer: {proc_min}m {proc_sec}s")
    return exit_code


def shutil_which(cmd: str) -> str:
    try:
        import shutil
        return shutil.which(cmd)
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
