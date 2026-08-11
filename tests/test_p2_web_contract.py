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
    assert 'id="dda-observation"' in html
    assert 'id="dda-score"' in html
    assert 'id="safe-card"' in html
    assert 'id="forensic-card"' in html
    assert 'id="community-card"' in html
    assert 'id="nonescape-card"' in html
    assert 'id="c2pa-declaration-card"' in html
    assert 'id="c2pa-signature-card"' in html
    assert 'id="c2pa-capture-card"' in html
    assert 'id="overall-summary"' in html
    assert 'id="overall-score-ring"' in html
    assert 'id="overall-score-value"' in html
    assert 'id="dda-metrics"' in html
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
    assert "function renderPixelDetectorCard" in javascript
    assert "function renderC2pa" in javascript
    assert "function renderOverall" in javascript
    assert "function renderMetadata" in javascript
    assert "function renderWatermark" in javascript
    assert "function renderEvidenceOrder" in javascript
    assert "function setOverallScoreRing" in javascript
    assert 'Math.max(-100, Math.min(100, Math.round(score)))' in javascript
    assert 'const magnitude = Math.abs(normalized);' in javascript
    assert 'normalized < 0 ? "var(--success)"' in javascript
    assert 'origin.decision === "possible_photo"' in javascript
    assert 'const conclusion = asText(origin.summary) || "未检出 AI 信号";' in javascript
    assert 'const notRun = ["未运行", "未取得结果", "未取得参数结果", "未完成", "未配置检测"].includes(observation);' in javascript
    assert 'const metadataWithoutData = key === "metadata" && observation === "未发现相机信息";' in javascript
    assert "cameraWithoutData" not in javascript
    assert "const group = points > 0 ? 0 : points < 0 ? 1 : metadataWithoutData ? 3 : notRun ? 4 : 2;" in javascript
    assert '["AI 分数", `${score} / 100`]' not in javascript
    assert "if (payload.result) renderResult(payload.job, payload.result)" in javascript
    assert 'const completedAnalysisStates = new Set(["completed", "partial"])' in javascript
    assert "hasOrigin && hasCompletedAnalysis" in javascript
    assert "这不表示图片一定由相机拍摄" not in javascript
    assert "assessment.status === \"not_configured\"" in javascript
    assert "发现未验证标识" in javascript
    assert "检测到水印或标识。" in javascript
    assert "不改变综合判断" not in javascript
    assert 'watermark: "正在检查本地水印方案。"' in javascript
    assert 'openai_provenance: "正在请求 OpenAI 来源验证。"' in javascript
    assert 'formData.append("openai_provenance", "1")' in javascript
    assert "externalChecks.hidden = !(openaiConfigured || googleConfigured)" in javascript
    assert "URL.createObjectURL(selected)" in javascript
    assert "cancelButton.addEventListener(\"click\", clearSelected)" in javascript
    assert "function closeMetricDetails" in javascript
    assert "cancelSelectedJob" in javascript
    assert 'method: "DELETE"' in javascript
    assert "window.addEventListener(\"scroll\", () => closeMetricDetails(), { passive: true })" in javascript
    assert "window.addEventListener(\"wheel\"" not in javascript
    assert "function keepMetricScrollLocal" in javascript
    assert "function prepareMetricScrollbar" in javascript
    assert "function syncMetricScrollbar" in javascript
    assert 'metrics.addEventListener("wheel", keepMetricScrollLocal, { passive: false })' in javascript
    assert "metric-scrollbar-thumb" in javascript
    assert "scrollbar-width: none" in (
        REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "styles.css"
    ).read_text(encoding="utf-8")
    assert "overscroll-behavior: contain" in (
        REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "styles.css"
    ).read_text(encoding="utf-8")
    assert ".metric-scrollbar-thumb" in (
        REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "styles.css"
    ).read_text(encoding="utf-8")
    assert 'return "有一项检测未满足运行条件，未作为判断依据。";' in javascript
    assert 'class="evidence-table-head"' in html
    assert 'result["watermark"] = watermark_result.model_dump' in jobs
    assert 'web_partial_result.json' in jobs
    assert "const stageLabels" in javascript
    assert "job.progress_percent" in javascript
    assert 'result["origin"] = origin.model_dump' in jobs
    assert 'result["p3"] = p3_result.model_dump' in jobs
    assert "worker_project_root=project_root" in (
        REPOSITORY_ROOT / "src" / "image_trust" / "web" / "server.py"
    ).read_text(encoding="utf-8")


def test_web_ui_keeps_hidden_panels_out_of_the_layout() -> None:
    stylesheet = (REPOSITORY_ROOT / "src" / "image_trust" / "web" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )

    assert "[hidden] { display: none !important; }" in stylesheet
