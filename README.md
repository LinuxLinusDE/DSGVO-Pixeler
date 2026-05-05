# DSGVO-Pixeler
This tool processes 4K videos locally and automatically anonymizes vehicle license plates and faces using YOLOv8 (pixelation or blur). Optimized for Apple Silicon (M-series) and action-cam footage, it prioritizes data protection by reliably anonymizing sensitive visual information while preserving video quality and audio.

## Featured demo (YouTube)
[![DSGVO-Pixeler demo](https://img.youtube.com/vi/VYVoB2Qsij4/hqdefault.jpg)](https://youtu.be/VYVoB2Qsij4)

## Project website
https://linuxlinusde.github.io/DSGVO-Pixeler/

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

You can also use a folder, a glob pattern, or a comma-separated list as the input. The script then processes all matching `.mp4` files. If `--output` is set in this mode, it must be an output folder:
```bash
python dsgvo-pixeler.py --input /videos/source --output /videos/pixelated
python dsgvo-pixeler.py --input "/videos/source/*.mp4" --output /videos/pixelated
python dsgvo-pixeler.py --input a.mp4,b.mp4,c.mp4 --output /videos/pixelated
```

If you start without parameters, the program shows a short, easy help message.

## Screenshots
Example anonymization output:

![DSGVO-Pixeler example 1](misc/dsgvo-pixeler-1.png)
![DSGVO-Pixeler example 2](misc/dsgvo-pixeler-2.png)

Note: the green boxes show detected objects and the red boxes show no-pixel zones; these overlays are optional and only appear when `--debug_pixel` and `--debug_no_pixel` are enabled.

## Where do I get `models/plates/best.pt` (plates)?
- Train your own YOLOv8 license plate model and export it as `.pt`.
- Use an existing license plate detection model from a trusted source (check license and privacy).
- Example: A suitable model is available here: https://huggingface.co/Koushim/yolov8-license-plate-detection/tree/main (save as `models/plates/best.pt`)
- Important: The model must detect plates as objects (no OCR required).

## Face models (default)
- Default: all `.pt` files in `models/faces/`.
- Alternatively: `--faces_weights models/faces/a.pt,models/faces/b.pt`
- Face model link: https://github.com/lindevs/yolov8-face

## Functions
Core features and what the script does in the background:

Detection
- Plates + faces by default (YOLOv8). Can be disabled with `--no_plates` or `--no_faces`.
- Uses all `.pt` models in `models/plates/` and `models/faces/` by default.

Anonymization
- Choose between pixelation (mosaic) and blur: `--anonymize pixelate|blur` (default: blur).
- Pixelation strength: `--blocks_plates`, `--blocks_faces` (pixelate only).
- Blur strength: `--blur_ksize` (odd kernel size).
- Optional padding around boxes: `--pad`.

Tiling
- `--tiling N` splits each frame into an NxN grid to improve detection of small objects.
- Tiling increases processing time and disables tracking. Default is 2x2.

Tracking
- Enabled by default to reduce flicker. Disable with `--no_track`.
- When tiling is active, tracking is turned off.

No‑pixel zones
- Pixel zones: `--no_pixel_zone_px1..4` (x1,y1,x2,y2 in pixels).
- `--debug_no_pixel` draws these zones in red.

Snapshots
- `--snapshot_every` saves JPEG snapshots every N minutes (stored next to the input video by default).
- `--snapshot_size` controls resolution (e.g. 1920x1080). Snapshot filenames include the same timestamp as the output video.
- `--snapshot_dir` sets the output folder (default: same folder as the input video).

Performance
- `--work_w` runs detection on a smaller width for speed.
- `--imgsz` controls YOLO inference size.

Encoding and audio
- Video is encoded via ffmpeg (VideoToolbox on macOS, CPU fallback with `--force_sw`).
- `--no_audio` removes the audio track.
- `--bitrate auto` uses ffprobe to match input bitrate; otherwise use a fixed value.
- `-movflags +faststart` enables fast streaming start.

Logging
- Progress output includes % and ETA, plus effective FPS.
- Summary at the end shows resolution, bitrate, encoder, audio, tracking, tiling, and model counts.

## Tiling
Tiling splits each frame into smaller tiles (e.g. 2x2). This improves detection of very small plates in 4K+ footage, but it increases processing time because YOLO runs on every tile.

Note: Tiling can make processing much slower. Example (MacBook M4, 5760x3240 @ 29.97 fps): 1798 frames (~59s of video) took 9m 36s with 2x2 tiling.

Tip: for quick tests, use `--test_minutes 1`, `--preset fast`, or set `--tiling 1`.

Note: snapshots add a small amount of CPU and disk I/O; for YouTube thumbnails this is usually negligible.

HEVC default (4K, MPS, audio is preserved):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output.mp4 \
  --weights /path/to/plate_model.pt
```

Blur anonymization (compare with pixelation):
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --output output_blur.mp4 \
  --weights /path/to/plate_model.pt \
  --anonymize blur \
  --blur_ksize 80
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

Tiling for small plates (2x2, default):
```bash
python dsgvo-pixeler.py --input input.mp4 --tiling 2
```

Define pixel zones and show them:
```bash
python dsgvo-pixeler.py \
  --input input.mp4 \
  --no_pixel_zone_px1 120,1500,900,2160 \
  --no_pixel_zone_px2 3000,1500,3800,2160 \
  --debug_no_pixel
```

Test run (first 2 minutes, debug overlay):
```bash
python dsgvo-pixeler.py --input input.mp4 --test_minutes 2 --debug_pixel
```

Snapshots every 5 minutes (Full HD):
```bash
python dsgvo-pixeler.py --input input.mp4 --snapshot_every 5 --snapshot_size 1920x1080
```

Examples from real-world workflows

Minimal run with defaults (weights auto-detected from `models/plates/`/`models/faces/`):
```bash
python dsgvo-pixeler.py --input source.mp4
```

Process multiple MP4 files (folder, glob, or list):
```bash
python dsgvo-pixeler.py --input /videos/source --output /videos/pixelated
python dsgvo-pixeler.py --input "/videos/source/*.mp4" --output /videos/pixelated
python dsgvo-pixeler.py --input a.mp4,b.mp4,c.mp4 --output /videos/pixelated
```

Fast test run with two no-pixel zones and visual debugging (1 minute only):
```bash
python dsgvo-pixeler.py \
  --input source.mp4 \
  --preset fast \
  --no_pixel_zone_px1 990,2796,1382,3211 \
  --no_pixel_zone_px2 368,3026,616,3118 \
  --test_minutes 1 \
  --debug_pixel \
  --debug_no_pixel
```

Longer test with snapshots and tiling, plus audio removed (5 minutes):
```bash
python dsgvo-pixeler.py \
  --input source.mp4 \
  --preset fast \
  --no_pixel_zone_px1 990,2796,1382,3211 \
  --no_pixel_zone_px2 368,3026,616,3118 \
  --test_minutes 5 \
  --no_audio \
  --debug_pixel \
  --debug_no_pixel \
  --tiling 1 \
  --snapshot_every 1 \
  --snapshot_size 1920x1080
```

Quick snapshot + debug run without no-pixel zones (1 minute):
```bash
python dsgvo-pixeler.py \
  --input source.mp4 \
  --preset fast \
  --test_minutes 1 \
  --no_audio \
  --debug_pixel \
  --tiling 1 \
  --snapshot_every 1 \
  --snapshot_size 1920x1080
```

Save a preset next to the output, then reuse it for another video:
```bash
python dsgvo-pixeler.py --input source.mp4 --save_preset json
python dsgvo-pixeler.py --input another.mp4 --load_preset source_preset
```
Preset names are derived from the input filename (e.g. `source.mp4` -> `source_preset.json`).

## Key parameters
- `work_w`: detection width (0 = original resolution). Default: 1920. Recommended: 0-3840.
- `imgsz`: YOLO inference size (larger = better detection, slower). Default: 1280. Recommended: 640-2048.
- `conf`: confidence threshold (lower = more detections). Default: 0.25. Recommended: 0.1-0.6.
- `blocks_plates`: pixel block size for plates (larger = coarser). Default: 16. Recommended: 4-64.
- `blocks_faces`: pixel block size for faces (larger = coarser). Default: 24. Recommended: 4-64.
- `blocks`: deprecated alias for `blocks_plates`. Recommended: 4-64.
- `pad`: safety padding around each box (pixels). Default: 20. Recommended: 0-100.
- `no_pixel_zone_px1..4`: up to four pixel zones as `x1,y1,x2,y2` (top-left -> bottom-right).
- Tip: Use the built-in command builder to define no-pixel zones locally: `docs/command-builder.html`
- Hosted version: https://linuxlinusde.github.io/DSGVO-Pixeler/command-builder.html
- `force_sw`: force software encoding.
- `test_minutes`: process only the first N minutes (0 = full video). Default: 0. Recommended: 0-60.
- `preset`: `fast`, `balanced`, `quality` for quick speed/quality choice.
- `anonymize`: `pixelate` or `blur` (default: `blur`).
- `blur_ksize`: blur strength (even values are rounded up to the next odd kernel size).
- `debug_pixel`: draws green boxes for verification.
- `debug_no_pixel`: draws the no-pixel zones in red.
- `no_audio`: remove the audio track in the output.
- `no_track`: disable tracking (tracking is on by default).
- `tiling`: split the frame into tiles for small objects (1-10). Default: 2. Recommended: 1-4.
- `snapshot_every`: save a snapshot every N minutes (0 = off). Default: 0. Recommended: 0-60.
- `snapshot_dir`: output folder for snapshots (default: input folder).
- `snapshot_size`: snapshot size, e.g. 1920x1080.
- `bitrate`: default is `auto` (uses input bitrate), or set e.g. `50M`.
- `log_every`: log output every N frames. Default: 200. Recommended: 50-1000.
- `save_preset`: save used parameters in the output folder as `*_preset.json`/`.txt`.
- `load_preset`: load a preset JSON by file path or name (relative to input/output folder).
- `faces_weights`: list of face models (default: all in `models/faces/`).
- `weights`: list of plate models (default: all in `models/plates/`).
- `extra_weights`: list of extra models (or `--use_extra` for `models/extra/`).
- `no_faces`: do not anonymize faces.
- `no_plates`: do not anonymize plates.

## Simple usage steps
1) Put your video (e.g. `input.mp4`) into the project folder and weights into `models/plates/` (e.g. `models/plates/best.pt`).
2) Open a terminal in the project folder.
3) Run the command from Quick start.
4) The result will be saved as `output.mp4` in the same folder.
Tip: If `models/plates/` contains models and you forget `--weights`, they will be used automatically.

## FAQ
Are multiple models processed in parallel?
No, they run sequentially (more stable and often faster).

Is tracking enabled by default?
Yes. Disable it with `--no_track` if you prefer raw detections.

Why is tracking disabled when tiling is enabled?
Tracking across tiles is unreliable because object IDs cannot be matched consistently between tiles. The tool disables tracking for tiling to avoid unstable results.

What is the default tiling value?
`--tiling` defaults to `2` (2x2 tiling).

## Insta360 note
Recommendation: reframe/flat export to 16:9 in Insta360 Studio first, then anonymize.

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

## License and third-party components
DSGVO-Pixeler's own source code is released under the MIT License. Runtime dependencies, external tools, and model weights are licensed separately.

Important notes:
- `ultralytics` / YOLO may be subject to AGPL-3.0 or an Ultralytics Enterprise License.
- ffmpeg is an external dependency and is not distributed with this project. Please install ffmpeg separately and follow its license terms (LGPL/GPL depending on your build).
- YOLO `.pt` model weights are not covered by DSGVO-Pixeler's MIT license unless their respective rights holders publish them under compatible terms.

See `THIRD_PARTY_NOTICES.md` for the dependency overview.
