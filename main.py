"""Command-line entry point for the smart video highlight pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pipeline.detection import analyze_shots
from pipeline.highlights import render_highlights, score_shots
from pipeline.media import prepare_media
from pipeline.shots import detect_shots
from pipeline.vlm import understand_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create highlights and structured insights from a short video."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "--out", required=True, type=Path, help="Directory for generated artifacts"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity",
    )
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument(
        "--vlm-endpoint",
        default=os.environ.get("VLM_ENDPOINT"),
        help="OpenAI-compatible VLM chat-completions endpoint",
    )
    parser.add_argument("--vlm-model", default=os.environ.get("VLM_MODEL", "llava"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.input.is_file():
        raise SystemExit(f"Input video does not exist: {args.input}")

    metadata = prepare_media(args.input, args.out)
    frame_paths = sorted((args.out / "frames").glob("frame_*.jpg"))
    shots = detect_shots(frame_paths)
    analyze_shots(shots, args.out / "keyframes", model_name=args.detector_model)
    for shot in shots:
        understanding = understand_frame(
            shot.keyframe,
            endpoint=args.vlm_endpoint,
            model=args.vlm_model,
            fallback_objects=shot.objects,
        )
        shot.caption = understanding["caption"]
        shot.exciting = understanding["exciting"]
        shot.excitement_reason = understanding["reason"]
    ranked = score_shots(shots)
    render_highlights(ranked, args.out / "normalized.mp4", args.out / "highlights.mp4")
    summary = {"metadata": metadata, "shots": [shot.to_dict() for shot in ranked]}
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
