import pytest

from pipeline.media import _require_binary


def test_require_binary_reports_missing_tool(monkeypatch):
    monkeypatch.setattr("pipeline.media.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="not found on PATH"):
        _require_binary("ffmpeg")
