from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_web_ui_has_a_separate_ai_pixel_signal_card() -> None:
    html = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    jobs = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "jobs.py").read_text(encoding="utf-8")
    assert "AI 内容可能性评估" in html
    assert 'id="overall-observation"' in html
    assert 'id="metadata-observation"' in html
    assert 'id="watermark-observation"' in html
    assert 'id="p3-observation"' in html
    assert 'id="p3-metrics"' in html
    assert 'id="progress-track"' in html
    assert 'id="progress-bar"' in html
    assert 'id="progress-meta"' in html
    assert "function renderP3" in javascript
    assert "function renderOverall" in javascript
    assert "function renderMetadata" in javascript
    assert "function renderWatermark" in javascript
    assert "const stageLabels" in javascript
    assert "job.progress_percent" in javascript
    assert 'result["origin"] = origin.model_dump' in jobs
    assert '"p3": p3_result.model_dump' in jobs


def test_web_ui_keeps_hidden_panels_out_of_the_layout() -> None:
    stylesheet = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )

    assert "[hidden] { display: none !important; }" in stylesheet
