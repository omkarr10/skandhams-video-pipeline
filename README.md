# Smart Video Highlight & Insight Pipeline

This project is a small end-to-end Python pipeline for a 2–5 minute
royalty-free sports, street, or event video. It prepares the media, detects
shots and motion, runs YOLO object detection, asks a vision-language model for
scene insight, and produces a captioned highlight reel plus JSON output.

The input footage for the walkthrough/submission is sourced from
[Pexels Videos](https://www.pexels.com/videos/). Pexels provides royalty-free
stock video suitable for this task.

## Screen recording

The project walkthrough is available here:
[Watch the screen recording](./Screen%20Recording%202026-10-10%20at%209.08.02%E2%80%AFPM.mov).

GitHub does not support reliable inline playback of repository video files
inside a README. Opening the link above uses GitHub's video/file viewer; the
recording is also available as a Git LFS file in this repository.

## macOS setup

Install Python 3.9+ and FFmpeg:

```bash
brew install python ffmpeg
cd /Users/omkar/Projects/DEV/skandhams-video-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Gemini scene understanding, copy the environment template and add a
Gemini API key. Never commit `.env`.

```bash
cp .env.example .env
nano .env
```

Set:

```env
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.8-flash
```

The VLM is optional. Without `GEMINI_API_KEY` or `VLM_ENDPOINT`, the pipeline
uses a deterministic fallback caption based on detected objects.

## Run

The complete pipeline runs with one command:

```bash
source .venv/bin/activate
python main.py --input video.mp4 --out results/
```

For another Pexels video, change only the input and output paths:

```bash
python main.py --input video4.mp4 --out results_video4/
```

Open the generated highlight on macOS:

```bash
open results/highlights.mp4
```

## Outputs

```text
results/
├── audio.wav          # 16 kHz mono audio, or a silent placeholder
├── frames/            # analysis frames at 2 FPS
├── keyframes/         # one annotated YOLO image per shot
├── normalized.mp4     # letterboxed 1280x720 working video
├── highlights.mp4     # top three shots in chronological order
├── metadata.json      # ffprobe/media metadata
└── summary.json       # per-shot structured results
```

## Implementation and design choices

- **FFmpeg:** `ffprobe` records duration, FPS, resolution, and codec.
  FFmpeg normalizes the input to 1280x720, extracts frames at 2 FPS, and
  extracts 16 kHz mono WAV audio.
- **OpenCV:** consecutive HSV histograms are compared with the Bhattacharyya
  distance. A distance of at least `0.45` starts a new shot. Motion is the
  mean normalized grayscale frame difference within each shot.
- **Object detection:** `yolo11n.pt` runs on every analysis frame on CPU.
  Each shot stores mean people count, aggregated object classes, and a
  middle-frame annotation.
- **VLM:** Gemini receives each annotated keyframe and is prompted to return
  JSON containing a one-sentence caption, an exciting yes/no verdict, and a
  reason. Responses are validated; malformed, unavailable, or unauthorized
  requests fall back to deterministic local captions.
- **Highlight score:**
  `0.45 * normalized_motion + 0.25 * normalized_people + 0.30 * exciting`
  The three highest-scoring shots are selected, restored to chronological
  order, captioned, and concatenated with the original audio.
- If FFmpeg lacks `drawtext`, captions are rendered with OpenCV instead.

## Validation and limitations

Validated on macOS with Python 3.14, CPU YOLO inference, FFmpeg, and Gemini
API scene understanding. CPU processing is slower than GPU inference.
Histogram comparison can miss gradual transitions, YOLO counts detections
rather than unique tracked people, and VLM quality depends on the selected
model and API availability. Videos without audio are supported with a silent
WAV placeholder.

Run the tests with:

```bash
python -m pytest -q
```
