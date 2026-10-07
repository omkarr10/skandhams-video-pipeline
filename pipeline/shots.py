"""Shot-boundary and motion analysis for sampled video frames."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class Shot:
    index: int
    start_time: float
    end_time: float
    frame_paths: List[str]
    motion_score: float = 0.0
    people_count: float = 0.0
    objects: Optional[Dict[str, int]] = None
    caption: str = ""
    exciting: bool = False
    excitement_reason: str = ""
    final_score: float = 0.0
    keyframe: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def _motion(previous: np.ndarray, current: np.ndarray) -> float:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(previous_gray, current_gray)
    return float(np.mean(difference) / 255.0)


def detect_shots(
    frame_paths: Sequence[Path],
    sample_fps: float = 2.0,
    histogram_threshold: float = 0.45,
) -> List[Shot]:
    """Group sampled frames into shots using HSV histogram distance."""

    if not frame_paths:
        return []
    frames = [cv2.imread(str(path)) for path in frame_paths]
    if any(frame is None for frame in frames):
        missing = [str(path) for path, frame in zip(frame_paths, frames) if frame is None]
        raise ValueError(f"Could not decode analysis frames: {missing}")

    boundaries = [0]
    histograms = [_histogram(frame) for frame in frames]
    for index in range(1, len(frames)):
        distance = cv2.compareHist(
            histograms[index - 1], histograms[index], cv2.HISTCMP_BHATTACHARYYA
        )
        if distance >= histogram_threshold:
            boundaries.append(index)
    boundaries.append(len(frames))

    shots: List[Shot] = []
    for shot_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        shot_frames = frames[start:end]
        motion_values = [
            _motion(shot_frames[i - 1], shot_frames[i])
            for i in range(1, len(shot_frames))
        ]
        shots.append(
            Shot(
                index=shot_index,
                start_time=start / sample_fps,
                end_time=end / sample_fps,
                frame_paths=[str(path) for path in frame_paths[start:end]],
                motion_score=float(np.mean(motion_values)) if motion_values else 0.0,
            )
        )
    return shots
