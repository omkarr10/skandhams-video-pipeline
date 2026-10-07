# Smart Video Highlight & Insight Pipeline

A modular Python pipeline for turning a short video into:

- normalized media and 2 FPS analysis frames;
- HSV-based shot boundaries and motion scores;
- per-shot object statistics and annotated keyframes;
- validated VLM captions and excitement verdicts;
- a captioned `highlights.mp4` containing the top three shots;
- machine-readable `summary.json`.

## Run command

```bash
python main.py --input video.mp4 --out results/
```

Optional VLM configuration uses an OpenAI-compatible vision endpoint:

```bash
VLM_ENDPOINT=http://localhost:11434/v1/chat/completions \
VLM_MODEL=llava \
python main.py --input video.mp4 --out results/
```

If no endpoint is configured, the pipeline emits a deterministic fallback
caption and marks the VLM verdict as false. This keeps the output schema
complete while making the missing VLM dependency explicit in `reason`.

## Outputs

```text
results/
├── audio.wav
├── frames/
├── keyframes/
├── normalized.mp4
├── highlights.mp4
├── metadata.json
└── summary.json
```

## Setup and source

Python 3.9+ and FFmpeg (including `ffprobe`) are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The detector and VLM are optional during development. The pipeline reports
missing optional components explicitly and uses a clearly marked fallback for
scene understanding so media and classical-vision stages remain testable on a
CPU-only machine.

The validation run used the open-license Wikimedia Commons clip
[Abdominales.ogv](https://commons.wikimedia.org/wiki/File:Abdominales.ogv).
For a two-minute pipeline test, the 9.9-second source was looped and converted
to MP4; replace it with an original 2–5 minute Pexels, Pixabay, or Commons
video for the final submission.

## Design choices

- Frames are sampled at 2 FPS after resizing to 1280x720 (or a smaller
  letterboxed equivalent when the source is portrait).
- Shot changes use normalized HSV histograms and a configurable Bhattacharyya
  distance threshold.
- Motion is normalized frame-difference energy over consecutive analysis
  frames.
- Highlight score will combine normalized motion, people count, and a VLM
  excitement bonus:
  `0.45 * normalized_motion + 0.25 * normalized_people + 0.30 * exciting`.
- A shot boundary is created when Bhattacharyya HSV histogram distance is at
  least `0.45`.
- The three highest scoring shots are rendered in chronological order.
- Captions are burned into each segment with FFmpeg `drawtext` when that filter
  is available; the implementation falls back to OpenCV text rendering on
  FFmpeg builds without libfreetype. Source audio is retained through each
  segment and concatenation.
- YOLO runs on every extracted frame in each shot. `people_count` is the
  per-shot mean; object counts are aggregated across the same frames, while the
  middle frame is saved as the annotated keyframe.

## Validation run

On an Apple Silicon macOS machine (Python 3.14, CPU, `yolo11n.pt`), the
two-minute validation input completed in **49.73 seconds** and produced 239
analysis frames, one annotated keyframe, `summary.json`, `highlights.mp4`,
`audio.wav`, and `normalized.mp4`. The local FFmpeg build did not expose
`drawtext`, so the tested run used the documented OpenCV caption fallback.

## Input source

The final submission should use a royalty-free sports, street, or event clip
from Pexels or Pixabay and record the exact source URL here.

## Known limitations

- CPU inference is slower and less accurate than GPU inference.
- Histogram boundaries can miss gradual transitions.
- VLM quality depends on the selected local model or compatible endpoint.
- The current implementation expects FFmpeg and the Ultralytics package to be
  installed before running the complete pipeline.
- A repeated short source is only a validation fixture; use a genuine 2–5
  minute source for the hiring submission.
