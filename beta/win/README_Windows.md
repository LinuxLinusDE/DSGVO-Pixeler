# DSGVO-Pixeler – Windows-Version

`dsgvo-pixeler-win.py` ist die für Windows angepasste Fassung. Sie erkennt
automatisch, ob eine GPU vorhanden ist, und wählt den passenden Video-Encoder.

## Was wurde gegenüber der Mac-Version geändert

| Bereich | Vorher (Mac) | Jetzt |
|---|---|---|
| `--device` | fest `mps` | `auto` → NVIDIA-CUDA, sonst CPU (MPS nur auf Mac) |
| Video-Encoder | fest `*_videotoolbox` | `--hwaccel auto` → nvenc / qsv / amf / videotoolbox, sonst Software |
| VideoToolbox-Flags | immer gesetzt | nur noch bei tatsächlichem VideoToolbox-Encoder |
| Setup-/Fehlertexte | brew / source | plattformneutral (winget, `.venv\Scripts\activate`) |

Die vorhandene automatische CPU-Rückfallebene (bei Inferenz-Fehlern) und das
Software-Encoding (`--force_sw`) bleiben unverändert erhalten.

## Installation (einmalig)

**1. Python 3.10–3.12 installieren** (python.org, Häkchen „Add to PATH").

**2. ffmpeg installieren** (enthält ffprobe) und in den PATH legen:
```
winget install Gyan.FFmpeg
```
Prüfen: `ffmpeg -version`

**3. Virtuelle Umgebung + Pakete:**
```
py -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install ultralytics opencv-python
```

**4. PyTorch für die GPU** (nur bei NVIDIA-Karte – für CUDA-Beschleunigung):
```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
> Passende CUDA-Version je nach Treiber unter https://pytorch.org/get-started/locally/ wählen.
> Ohne diesen Schritt läuft die Erkennung auf der CPU (funktioniert, ist aber langsamer).

## Start

**Automatik (empfohlen)** – wählt GPU und Encoder selbst:
```
python dsgvo-pixeler-win.py --input input.mp4 --output output.mp4 --weights models\plates\best.pt
```

**Encoder / Device explizit erzwingen:**
```
python dsgvo-pixeler-win.py --input input.mp4 --output output.mp4 ^
  --weights models\plates\best.pt --device cuda --hwaccel nvenc
```

## Encoder je nach GPU

| GPU / Fall | Empfehlung |
|---|---|
| NVIDIA (GeForce/RTX/Quadro) | `--hwaccel nvenc` |
| Intel (iGPU / Arc) | `--hwaccel qsv` |
| AMD (Radeon) | `--hwaccel amf` |
| keine GPU / Probleme | `--hwaccel software` oder `--force_sw` |
| unsicher | `--hwaccel auto` (Standard) |

## Hinweise

- `--hwaccel auto` prüft per `ffmpeg -encoders`, welche Encoder deine
  ffmpeg-Version wirklich kann, und fällt zur Not auf Software zurück.
- Schlägt die GPU-Inferenz zur Laufzeit fehl, schaltet das Skript pro Modell
  automatisch auf CPU um (Warnung im Log) – der Lauf bricht nicht ab.
- Pfade unter Windows mit Backslash; in PowerShell Zeilenumbruch mit `` ` ``,
  in cmd.exe mit `^` (siehe Beispiel oben).
