(() => {
  "use strict";

  const terminalStates = new Set(["completed", "partial", "rejected", "failed"]);
  const stateLabels = {
    queued: ["正在排队", "已创建本地作业，等待开始。"],
    validating: ["正在验证文件", "正在检查图片格式、大小与解码边界。"],
    running: ["正在分析", "正在生成几何证据、相机一致性记录和离线 C2PA 读取结果。"],
    completed: ["分析完成", "全部已配置的分析链路已完成。"],
    partial: ["分析部分完成", "基础几何结果已保留；部分附加分析未形成可用观测。"],
    rejected: ["文件未被接受", "图片未通过输入验证，因此没有继续分析。"],
    failed: ["分析未完成", "保留了可用的错误信息；请检查限制说明后重试。"],
  };
  const observationLabels = {
    positive: "存在待复核候选",
    negative: "未发现候选",
    not_observed: "未形成观测",
    neutral: "中性测量结果",
  };
  const directionLabels = {
    neutral: "中性测量结果",
    supports_ai: "支持 AI 的几何线索",
    supports_camera: "支持相机的几何线索",
    conflicting: "几何线索相互冲突",
  };
  const limitationLabels = {
    no_embedded_c2pa_manifest_found: "未在图片中发现嵌入式 C2PA 清单。",
    ocsp_fetch_disabled: "已禁用 OCSP 联网查询。",
    remote_manifest_fetch_disabled: "已禁用远程 C2PA 清单读取。",
    offline_validation_cannot_confirm_current_revocation_state: "离线验证无法确认当前撤销状态。",
    p1_camera_backend_not_available: "相机一致性后端未形成满足门槛的全图观测。",
    p1_camera_analysis_failed: "相机一致性分析发生错误，其他结果仍已保留。",
    p0_input_rejected: "几何分析拒绝了该输入。",
    p0_analysis_not_available: "几何分析未能完成。",
    web_analysis_failed: "本地分析服务发生未处理错误。",
  };

  const form = document.querySelector("#upload-form");
  const input = document.querySelector("#file-input");
  const dropzone = document.querySelector("#dropzone");
  const selectedFile = document.querySelector("#selected-file");
  const submitButton = document.querySelector("#submit-button");
  const analysisPanel = document.querySelector("#analysis-panel");
  const statusDot = document.querySelector("#status-dot");
  const statusLabel = document.querySelector("#status-label");
  const statusDetail = document.querySelector("#status-detail");
  const jobFilename = document.querySelector("#job-filename");
  const resultPanel = document.querySelector("#result-panel");
  let selected = null;
  let activeJobId = null;
  let pollTimer = null;

  const asText = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : String(value);
  const percentage = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  const degrees = (value) => Number.isFinite(value) ? `${(value * 180 / Math.PI).toFixed(2)}°` : "—";

  function setSelected(file) {
    selected = file || null;
    selectedFile.textContent = selected ? `${selected.name} · ${(selected.size / 1024 / 1024).toFixed(2)} MB` : "尚未选择文件";
    submitButton.disabled = !selected;
  }

  function resetPanels() {
    clearTimeout(pollTimer);
    activeJobId = null;
    analysisPanel.hidden = true;
    resultPanel.hidden = true;
    document.querySelector("#visual-grid").replaceChildren();
    document.querySelector("#limitations-list").replaceChildren();
  }

  function renderStatus(job) {
    const [label, detail] = stateLabels[job.status] || ["状态未知", "服务返回了未知状态。"];
    analysisPanel.hidden = false;
    statusDot.dataset.status = job.status;
    statusLabel.textContent = label;
    statusDetail.textContent = detail;
    jobFilename.textContent = job.original_filename;
    if (job.errors && job.errors.length) {
      statusDetail.textContent = job.errors.map((error) => error.message || error.code).filter(Boolean).join(" ");
    }
  }

  function artifactUrl(jobId, relativePath) {
    const parts = relativePath.split("/").map((part) => encodeURIComponent(part));
    return `/api/jobs/${encodeURIComponent(jobId)}/artifacts/${parts.join("/")}`;
  }

  function addMetric(container, label, value) {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    const definition = document.createElement("dd");
    term.textContent = label;
    definition.textContent = value;
    item.append(term, definition);
    container.append(item);
  }

  function setCard(prefix, value, description, metrics) {
    document.querySelector(`#${prefix}-observation`).textContent = value;
    document.querySelector(`#${prefix}-description`).textContent = description;
    const metricGrid = document.querySelector(`#${prefix}-metrics`);
    metricGrid.replaceChildren();
    metrics.forEach(([label, metricValue]) => addMetric(metricGrid, label, metricValue));
  }

  function renderP0(result) {
    const p0 = result.p0 || {};
    const evidence = p0.evidence || {};
    const features = evidence.features || {};
    const value = directionLabels[evidence.direction] || observationLabels[evidence.observation] || asText(evidence.run_status, "未形成结果");
    const description = evidence.run_status === "not_applicable"
      ? "当前图片不满足稳定的投影几何测量条件。"
      : "叠图显示的是可复核的线段与平行／消失方向族，不是 AI 概率。";
    setCard("p0", value, description, [
      ["运行状态", asText(evidence.run_status)],
      ["适用性", percentage(evidence.applicability)],
      ["覆盖度", percentage(evidence.coverage)],
      ["可靠度", percentage(evidence.reliability)],
      ["稳定线族", asText(features.families?.length, "0")],
      ["候选异常线", asText(features.anomalous_lines?.length, "0")],
    ]);
  }

  function renderCamera(result) {
    const camera = result.camera || {};
    const fullImage = camera.full_image || {};
    const eCam = camera.e_cam || {};
    if (!Object.keys(camera).length) {
      setCard("camera", "未运行", "该作业没有相机一致性结果。", []);
      return;
    }
    if (camera.status === "failed") {
      setCard("camera", "未完成", "相机一致性分析失败，不能据此作任何推断。", []);
      return;
    }
    const value = fullImage.status === "ok" ? observationLabels[eCam.observation] || "已生成相机测量" : "未形成可用相机测量";
    setCard("camera", value, "全图与局部裁剪的相机参数一致性尚未校准为来源结论。", [
      ["后端状态", asText(fullImage.status)],
      ["E_cam", asText(eCam.observation)],
      ["Roll", degrees(fullImage.roll)],
      ["Pitch", degrees(fullImage.pitch)],
      ["垂直视场", fullImage.vfov_or_focal?.kind === "vfov_deg" ? `${Number(fullImage.vfov_or_focal.value).toFixed(2)}°` : "—"],
      ["合格裁剪", `${eCam.qualified_crop_ids?.length || 0}/${eCam.required_qualified_crops || "—"}`],
    ]);
  }

  function renderC2pa(result) {
    const c2pa = result.c2pa || {};
    if (!Object.keys(c2pa).length) {
      setCard("c2pa", "未运行", "该作业没有 C2PA 读取记录。", []);
      return;
    }
    const value = c2pa.manifest_present ? "发现嵌入式清单" : "未发现嵌入式清单";
    setCard("c2pa", value, "只读取图像中已有的嵌入信息；离线模式不会检索远程清单。", [
      ["读取状态", asText(c2pa.status)],
      ["签名状态", asText(c2pa.signature_validation_status)],
      ["网络访问", asText(c2pa.network_access)],
      ["声明动作", asText(c2pa.declared_actions?.length, "0")],
    ]);
  }

  function addVisual(container, title, caption, src) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    const figcaption = document.createElement("figcaption");
    const heading = document.createElement("strong");
    image.src = src;
    image.alt = title;
    image.loading = "lazy";
    heading.textContent = title;
    figcaption.append(heading, document.createTextNode(caption));
    figure.append(image, figcaption);
    container.append(figure);
  }

  function renderVisuals(job, result) {
    const artifacts = result.artifacts || {};
    const visualGrid = document.querySelector("#visual-grid");
    visualGrid.replaceChildren();
    const visuals = [
      ["原始输入", "上传后的本地原图。", artifacts.input_image],
      ["线段与方向族", "同色线段表示同一稳定方向族；请与原图的真实结构对照。", artifacts.lines_overlay],
      ["候选异常线", "只用于复核排序，不能单独解释为 AI 证据。", artifacts.anomalous_lines_overlay],
    ];
    visuals.filter(([, , path]) => typeof path === "string").forEach(([title, caption, path]) => {
      addVisual(visualGrid, title, caption, artifactUrl(job.job_id, path));
    });
    document.querySelector("#evidence-visuals").hidden = visualGrid.childElementCount === 0;
  }

  function humanizeLimitation(value) {
    return limitationLabels[value] || value.replaceAll("_", " ");
  }

  function renderLimitations(job, result) {
    const values = new Set([
      ...(job.limitations || []),
      ...(result.limitations || []),
      ...(result.p0?.evidence?.limitations || []),
      ...(result.camera?.limitations || []),
      ...(result.camera?.e_cam?.limitations || []),
      ...(result.c2pa?.limitations || []),
    ]);
    const section = document.querySelector("#limitations-section");
    const list = document.querySelector("#limitations-list");
    list.replaceChildren();
    values.forEach((value) => {
      const item = document.createElement("li");
      item.textContent = humanizeLimitation(value);
      list.append(item);
    });
    section.hidden = list.childElementCount === 0;
  }

  function renderResult(job, result) {
    resultPanel.hidden = false;
    renderP0(result);
    renderCamera(result);
    renderC2pa(result);
    renderVisuals(job, result);
    renderLimitations(job, result);
  }

  async function poll(jobId) {
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
      if (!response.ok) throw new Error("无法读取本地作业状态。");
      const payload = await response.json();
      renderStatus(payload.job);
      if (terminalStates.has(payload.job.status)) {
        if (payload.result) renderResult(payload.job, payload.result);
        return;
      }
      pollTimer = window.setTimeout(() => poll(jobId), 900);
    } catch (error) {
      statusDot.dataset.status = "failed";
      statusLabel.textContent = "无法读取作业状态";
      statusDetail.textContent = error instanceof Error ? error.message : "请重新上传图片。";
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!selected) return;
    resetPanels();
    submitButton.disabled = true;
    submitButton.textContent = "正在上传…";
    const formData = new FormData();
    formData.append("file", selected, selected.name);
    try {
      const response = await fetch("/api/jobs", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "上传失败");
      activeJobId = payload.job.job_id;
      renderStatus(payload.job);
      await poll(activeJobId);
    } catch (error) {
      analysisPanel.hidden = false;
      statusDot.dataset.status = "failed";
      statusLabel.textContent = "上传未完成";
      statusDetail.textContent = error instanceof Error ? error.message : "请重新选择图片后再试。";
    } finally {
      submitButton.disabled = !selected;
      submitButton.textContent = "开始分析";
    }
  }

  input.addEventListener("change", () => setSelected(input.files?.[0]));
  form.addEventListener("submit", submit);
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) setSelected(file);
  });
  document.querySelector("#new-analysis").addEventListener("click", () => {
    resetPanels();
    input.value = "";
    setSelected(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})();
