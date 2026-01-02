# DSGVO-Pixeler
This tool processes 4K videos locally and automatically pixelates vehicle license plates and faces using YOLOv8. Optimized for Apple Silicon (M-series) and action-cam footage, it prioritizes data protection by reliably anonymizing sensitive visual information while preserving video quality and audio.

## Requirements
- Python 3.10+
- ffmpeg via Homebrew: `brew install ffmpeg`
- A YOLOv8 license plate model as a `.pt` file in `models/plates/` (e.g. `models/plates/best.pt`)
- Optional: face models in `models/faces/` (default: all .pt files there).
- Optional: extra models in `models/extra/` (only if enabled via `--use_extra`).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick start (simple)
1) Open a terminal in the project folder.
2) Activate the virtual environment:
```bash
source .venv/bin/activate
```
3) Run (adjust filenames):
```bash
python dsgvo-pixeler.py --input input.mp4 --output output.mp4 --weights models/plates/best.pt
```

If you see errors like `ModuleNotFoundError: No module named 'cv2'`, the environment is missing. Use the setup steps above.

Note: `--output` is optional. If omitted, the file is created in the same folder as the input with useful info (weights, preset, timestamp) in the filename.

If you start without parameters, the program shows a short, easy help message.

## Where do I get `models/plates/best.pt` (plates)?
- Train your own YOLOv8 license plate model and export it as `.pt`.
- Use an existing license plate detection model from a trusted source (check license and privacy).
- Example: A suitable model is available here: https://huggingface.co/Koushim/yolov8-license-plate-detection/tree/main (save as `models/plates/best.pt`)
- Important: The model must detect plates as objects (no OCR required).

## Face models (default)
- Default: all `.pt` files in `models/faces/`.
- Alternatively: `--faces_weights models/faces/a.pt,models/faces/b.pt`

## Examples
HEVC default (4K, MPS, audio is preserved):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output.mp4 \
  --weights /path/to/plate_model.pt
```

H.264 compatible output (plays everywhere, recommend 50M for best quality):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output_h264.mp4 \
  --weights /path/to/plate_model.pt \
  --codec h264 \
  --bitrate 50M
```

Force software encoding (if hardware encoding fails):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output_sw.mp4 \
  --weights /path/to/plate_model.pt \
  --force_sw
```

Quick test with lower detection width (faster, less accurate):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output_fast.mp4 \
  --weights /path/to/plate_model.pt \
  --work_w 1280
```

Quality preset (slower, better detection):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output_quality.mp4 \
  --weights /path/to/plate_model.pt \
  --preset quality
```

More examples (detailed):
Plates only (no faces):
```bash
python dsgvo-pixeler.py --input input.mp4 --weights models/plates/best.pt --no_faces
```

Faces only (no plates):
```bash
python dsgvo-pixeler.py --input input.mp4 --faces_weights models/faces/face1.pt --no_plates
```

Multiple models (plates + faces):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --weights models/plates/a.pt,models/plates/b.pt \
  --faces_weights models/faces/face1.pt,models/faces/face2.pt
```

Use extra models:
```bash
python dsgvo-pixeler.py --input input.mp4 --use_extra
```

Define pixel zones and show them:
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --no_pixel_zone_px1 120,1500,900,2160 \
  --no_pixel_zone_px2 3000,1500,3800,2160 \
  --debug_zones
```

Test run (first 2 minutes, debug overlay):
```bash
python dsgvo-pixeler.py --input input.mp4 --test_minutes 2 --debug_overlay
```

## Key parameters
- `work_w`: detection width (e.g. 1280 or 1920). 0 = original resolution.
- `imgsz`: YOLO inference size (larger = better detection, slower).
- `conf`: confidence threshold (lower = more detections).
- `blocks`: pixel block size (smaller = coarser pixelation).
- `pad`: safety padding around each box (pixels).
- `no_pixel_zone`: no-pixel zone as `x1,x2,y1,y2` in percent (default: off). Example: `0,20,63,100` for bottom-left HUD.
- `no_pixel_zone2`: second no-pixel zone (default: off). Example: `78,100,59,100` for bottom-right HUD.
- `no_pixel_zone_px1..4`: up to four pixel zones as `x1,y1,x2,y2` (top-left -> bottom-right).
- Tip: You can find pixel coordinates here: https://imageonline.io/find-coordinates-of-image/ or https://get-image-coordinates.vercel.app/
- `force_sw`: force software encoding.
- `test_minutes`: process only the first N minutes (0 = full video).
- `preset`: `fast`, `balanced`, `quality` for quick speed/quality choice.
- `debug_overlay`: draws boxes for verification.
- `debug_zones`: draws the no-pixel zones in red.
- `bitrate`: default is `auto` (uses input bitrate), or set e.g. `50M`.
- `faces_weights`: list of face models (default: all in `models/faces/`).
- `weights`: list of plate models (default: all in `models/plates/`).
- `extra_weights`: list of extra models (or `--use_extra` for `models/extra/`).
- `no_faces`: do not pixelate faces.
- `no_plates`: do not pixelate plates.

## Simple usage steps
1) Put your video (e.g. `input.mp4`) into the project folder and weights into `models/plates/` (e.g. `models/plates/best.pt`).
2) Open a terminal in the project folder.
3) Run the command from Quick start.
4) The result will be saved as `output.mp4` in the same folder.
Tip: If `models/plates/` contains models and you forget `--weights`, they will be used automatically.

## FAQ
Are multiple models processed in parallel?
No, they run sequentially (more stable and often faster).

## Insta360 note
Recommendation: reframe/flat export to 16:9 in Insta360 Studio first, then pixelate.

## Troubleshooting
VideoToolbox error -12908 (HW encoding fails): often caused by pixel format negotiation. The script forces `nv12` for VideoToolbox. You can test:
```bash
ffmpeg -y -f lavfi -i testsrc2=size=3840x2160:rate=60 -t 2 -vf format=nv12 -c:v hevc_videotoolbox -b:v 12M vt_test.mp4
```

Rotation looks wrong: some files have rotation metadata. Normalize via ffmpeg:
```bash
ffmpeg -i input.mp4 -vf "transpose=0" -c:a copy normalized.mp4
```

Variable framerate: normalize first:
```bash
ffmpeg -i input.mp4 -vsync cfr -r 25 -c:v libx264 -c:a copy normalized.mp4
```

## Privacy note
Goal is unreadability. Use coarse pixels (`blocks` small) and sufficient `pad` so nothing is missed.
