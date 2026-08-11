from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_web_ui_has_a_separate_ai_pixel_signal_card() -> None:
    html = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    jobs = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "jobs.py").read_text(encoding="utf-8")
    assert "图像来源分析" in html
    assert "检测明细" in html
    assert 'id="overall-observation"' in html
    assert 'id="metadata-observation"' in html
    assert 'id="watermark-observation"' in html
    assert 'id="openai-provenance"' in html
    assert 'id="external-checks" class="external-checks" hidden' in html
    assert 'id="openai-option" class="external-option" for="openai-provenance" hidden' in html
    assert 'id="google-option" class="external-option external-manual" hidden' in html
    assert "https://gemini.google.com/" in html
    assert 'id="p3-observation"' in html
    assert 'id="p3-metrics"' in html
    assert 'class="metric-details"' in html
    assert 'class="evidence-result"' in html
    assert 'class="evidence-label"' in html
    assert 'id="image-preview"' in html
    assert 'id="cancel-button"' in html
    assert 'id="new-analysis"' not in html
    assert 'id="progress-track"' in html
    assert 'id="progress-bar"' in html
    assert 'id="progress-meta"' in html
    assert "function renderP3" in javascript
    assert "function renderOverall" in javascript
    assert "function renderMetadata" in javascript
    assert "function renderWatermark" in javascript
    assert "assessment.status === \"not_configured\"" in javascript
    assert "发现未验证标识" in javascript
    assert "不改变综合判断" in javascript
    assert 'watermark: "正在检查本地水印方案。"' in javascript
    assert 'openai_provenance: "正在请求 OpenAI 来源验证。"' in javascript
    assert 'formData.append("openai_provenance", "1")' in javascript
    assert "externalChecks.hidden = !(openaiConfigured || googleConfigured)" in javascript
    assert "URL.createObjectURL(selected)" in javascript
    assert "cancelButton.addEventListener(\"click\", clearSelected)" in javascript
    assert "function closeMetricDetails" in javascript
    assert "window.addEventListener(\"wheel\", () => closeMetricDetails(), { passive: true })" in javascript
    assert 'return "有一项检测未满足运行条件，未作为判断依据。";' in javascript
    assert 'class="evidence-table-head"' in html
    assert '"watermark": watermark_result.model_dump' in jobs
    assert "const stageLabels" in javascript
    assert "job.progress_percent" in javascript
    assert 'result["origin"] = origin.model_dump' in jobs
    assert '"p3": p3_result.model_dump' in jobs


def test_web_ui_keeps_hidden_panels_out_of_the_layout() -> None:
    stylesheet = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )

    assert "[hidden] { display: none !important; }" in stylesheet
