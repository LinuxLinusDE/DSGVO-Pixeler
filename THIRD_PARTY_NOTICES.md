# Third-party notices

DSGVO-Pixeler's own source code is licensed under the MIT License. Runtime
dependencies, external tools, and model weights are licensed separately.

This file is a practical overview, not a replacement for the original license
texts of the projects listed below.

## Python dependencies

- `ultralytics`: Ultralytics YOLO is offered under AGPL-3.0 or an Ultralytics
  Enterprise License. Check the applicable Ultralytics terms before publishing,
  deploying, or using DSGVO-Pixeler in a commercial or proprietary context.
- `opencv-python`: OpenCV 4.5.0 and newer are licensed under Apache-2.0.
- `numpy`: NumPy is licensed under the modified BSD license.

## External tools

- `ffmpeg` / `ffprobe`: These tools are not distributed with DSGVO-Pixeler.
  Install them separately and follow the license terms of the specific build
  you use. Depending on enabled codecs and build options, ffmpeg may be under
  LGPL or GPL terms.

## Model weights

YOLO `.pt` model files are not part of DSGVO-Pixeler's MIT license unless they
are explicitly published with compatible terms by their respective rights
holders. Before distributing model weights, verify the license of each model and
its training source.

