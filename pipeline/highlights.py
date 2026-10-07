"""Highlight scoring and FFmpeg rendering."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

from pipeline.shots import Shot

LOGGER = logging.getLogger(__name__)


def score_shots(shots: Iterable[Shot]) -> List[Shot]:
    shots = list(shots)
    max_motion = max((shot.motion_score for shot in shots), default=1.0) or 1.0
    max_people = max((shot.people_count for shot in shots), default=1.0) or 1.0
    for shot in shots:
        motion = shot.motion_score / max_motion
        people = shot.people_count / max_people
        verdict = 1.0 if shot.exciting else 0.0
        shot.final_score = round(0.45 * motion + 0.25 * people + 0.30 * verdict, 6)
    return sorted(shots, key=lambda shot: shot.final_score, reverse=True)


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to render highlights.mp4")
    return executable


def render_highlights(shots: Iterable[Shot], video_path: Path, output_path: Path) -> None:
    selected = sorted(score_shots(shots)[:3], key=lambda shot: shot.start_time)
    if not selected:
        raise ValueError("No shots available for highlight rendering")
    ffmpeg = _ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_paths: List[Path] = []
    for shot in selected:
        segment = output_path.parent / f".segment_{shot.index}.mp4"
        caption = shot.caption.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filter_value = f"drawtext=text='{caption}':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.6:x=24:y=h-th-24"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                str(shot.start_time),
                "-t",
                str(max(shot.end_time - shot.start_time, 0.1)),
                "-i",
                str(video_path),
                "-vf",
                filter_value,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(segment),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        segment_paths.append(segment)
    concat_file = output_path.parent / ".concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.name}'\n" for path in segment_paths), encoding="utf-8"
    )
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        concat_file.unlink(missing_ok=True)
        for segment in segment_paths:
            segment.unlink(missing_ok=True)
