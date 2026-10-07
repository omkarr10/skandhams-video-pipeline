"""FFmpeg/ffprobe media preparation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


def _require_binary(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH. "
            "Install FFmpeg and ensure both ffmpeg and ffprobe are available."
        )
    return executable


def _run(command: List[str]) -> str:
    LOGGER.debug("Running: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def probe_video(input_path: Path) -> Dict[str, Any]:
    """Return duration, FPS, dimensions, and codec from ffprobe."""

    ffprobe = _require_binary("ffprobe")
    output = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height,codec_name,avg_frame_rate",
            "-of",
            "json",
            str(input_path),
        ]
    )
    payload = json.loads(output)
    stream = (payload.get("streams") or [{}])[0]
    frame_rate = stream.get("avg_frame_rate", "0/1")
    numerator, denominator = (int(value) for value in frame_rate.split("/", 1))
    return {
        "duration_seconds": float((payload.get("format") or {}).get("duration", 0.0)),
        "fps": numerator / denominator if denominator else 0.0,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name"),
    }


def prepare_media(input_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Normalize video, extract analysis frames/audio, and persist metadata."""

    ffmpeg = _require_binary("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = out_dir / "normalized.mp4"
    audio_path = out_dir / "audio.wav"

    metadata = probe_video(input_path)
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(normalized_path),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(normalized_path),
            "-vf",
            "fps=2",
            str(frames_dir / "frame_%06d.jpg"),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )

    metadata["normalized_video"] = str(normalized_path)
    metadata["frames_dir"] = str(frames_dir)
    metadata["audio_path"] = str(audio_path)
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
