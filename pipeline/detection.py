"""YOLO object detection and per-shot aggregation."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2

from pipeline.shots import Shot

LOGGER = logging.getLogger(__name__)


class ObjectDetector:
    """Small adapter around Ultralytics YOLO, loaded lazily."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install requirements.txt to enable detection."
            ) from exc
        self.model = YOLO(model_name)
        self.confidence = confidence

    def analyze_frame(
        self, frame_path: str, annotated_path: Path, save_annotation: bool = True
    ) -> Dict[str, Any]:
        results = self.model.predict(
            source=frame_path, conf=self.confidence, verbose=False, device="cpu"
        )
        result = results[0]
        names = result.names
        labels = [names[int(class_id)] for class_id in result.boxes.cls.tolist()]
        people = labels.count("person")
        annotated = result.plot()
        if save_annotation:
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"Could not write annotated keyframe: {annotated_path}")
        return {"people": people, "objects": dict(Counter(labels))}


def analyze_shots(
    shots: Iterable[Shot],
    keyframes_dir: Path,
    model_name: str = "yolo11n.pt",
) -> None:
    detector = ObjectDetector(model_name=model_name)
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        keyframe_index = len(shot.frame_paths) // 2
        people_counts: List[float] = []
        object_counts: Counter[str] = Counter()
        annotation_path = keyframes_dir / f"shot_{shot.index:03d}.jpg"
        for frame_index, frame_path in enumerate(shot.frame_paths):
            output = detector.analyze_frame(
                frame_path,
                annotation_path,
                save_annotation=frame_index == keyframe_index,
            )
            people_counts.append(float(output["people"]))
            object_counts.update(output["objects"])
        shot.keyframe = str(keyframes_dir / f"shot_{shot.index:03d}.jpg")
        shot.people_count = sum(people_counts) / len(people_counts)
        shot.objects = dict(object_counts)
