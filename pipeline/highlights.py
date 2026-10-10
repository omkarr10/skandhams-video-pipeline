"""Highlight scoring and FFmpeg rendering."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

import cv2

from pipeline.shots import Shot

LOGGER = logging.getLogger(__name__)


def _wrap_caption(
    caption: str,
    font_face: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> List[str]:
    lines: List[str] = []
    current = ""
    for word in caption.split():
        candidate = f"{current} {word}".strip()
        text_width = cv2.getTextSize(
            candidate, font_face, font_scale, thickness
        )[0][0]
        if current and text_width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


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


def _caption_with_opencv(
    video_path: Path, output_path: Path, start: float, duration: float, caption: str
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_count = max(1, int(duration * fps))
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 24
    line_height = 34
    lines = _wrap_caption(
        caption,
        font_face,
        font_scale,
        thickness,
        max_width=width - (padding * 2),
    )
    box_height = (line_height * len(lines)) + padding
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        box_top = max(0, height - box_height)
        cv2.rectangle(frame, (12, box_top), (width - 12, height - 12), (0, 0, 0), -1)
        for line_index, line in enumerate(lines):
            baseline = box_top + padding + (line_index + 1) * line_height - 8
            cv2.putText(
                frame,
                line,
                (padding, baseline),
                font_face,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        writer.write(frame)
    capture.release()
    writer.release()


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
        try:
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
        except subprocess.CalledProcessError as exc:
            if "No such filter" not in exc.stderr:
                raise
            LOGGER.warning("FFmpeg drawtext is unavailable; using OpenCV caption fallback")
            silent_segment = output_path.parent / f".silent_{shot.index}.mp4"
            _caption_with_opencv(
                video_path,
                silent_segment,
                shot.start_time,
                max(shot.end_time - shot.start_time, 0.1),
                shot.caption,
            )
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(silent_segment),
                    "-ss",
                    str(shot.start_time),
                    "-t",
                    str(max(shot.end_time - shot.start_time, 0.1)),
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(segment),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            silent_segment.unlink(missing_ok=True)
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
