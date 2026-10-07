from pathlib import Path

from pipeline.highlights import score_shots
from pipeline.shots import Shot
from pipeline.vlm import fallback_understanding


def test_score_shots_ranks_motion_people_and_verdict():
    shots = [
        Shot(0, 0, 1, ["a"], motion_score=0.1, people_count=1),
        Shot(1, 1, 2, ["b"], motion_score=0.9, people_count=4, exciting=True),
    ]

    ranked = score_shots(shots)

    assert ranked[0].index == 1
    assert ranked[0].final_score > ranked[1].final_score


def test_fallback_understanding_is_valid_schema():
    result = fallback_understanding({"person": 3, "ball": 1})

    assert set(result) == {"caption", "exciting", "reason"}
    assert result["exciting"] is False
    assert "person" in result["caption"]
