(() => {
  "use strict";

  const terminalStates = new Set(["completed", "partial", "rejected", "failed", "cancelled"]);
  const completedAnalysisStates = new Set(["completed", "partial"]);
  const stateLabels = {
    queued: ["等待开始", "文件已接收，正在等待本地分析。"],
    validating: ["检查文件", "正在确认图片格式、大小和可读取性。"],
    running: ["正在分析", "正在依次运行已启用的检测项目。"],
    completed: ["分析完成", "可用的检测项目均已完成。"],
    partial: ["部分完成", "已保留可用结果；部分检测没有取得结果。"],
    rejected: ["文件无法分析", "图片未通过输入检查。"],
    failed: ["分析未完成", "请查看下方说明后重新尝试。"],
    cancelled: ["分析已取消", "这张图片的本地分析已停止。"],
  };
  const stageLabels = {
    queued: "等待本地分析开始。",
    validating: "正在读取图片文件。",
    starting: "正在准备检测环境。",
    geometry: "当前项目：旧几何来源模型。",
    geometry_v2_extraction: "当前项目：几何结构预处理，为 G1–G4 建立局部区域。",
    geometry_v2_g1: "当前项目：G1 局部平行线族。",
    geometry_v2_g2: "当前项目：G2 局部与全局消失方向。",
    geometry_v2_g3: "当前项目：G3 重复结构透视间距。",
    geometry_v2_g4: "当前项目：G4 线段连接与方向突变。",
    geometry_v2_g5: "当前项目：G5 相机与透视场一致性。",
    geometry_perspective: "当前项目：G5 相机与透视场一致性。",
    provenance: "当前项目：来源记录与相机信息。",
    watermark: "当前项目：本地隐式水印。",
    openai_provenance: "当前项目：OpenAI 来源验证。",
    ai_provenance: "已读取可验证的 AI 来源声明。",
    ai_dda: "当前项目：DDA 像素检测。",
    ai_safe: "当前项目：SAFE 纹理检测。",
    ai_forensic_clip: "当前项目：耐压缩像素检测。",
    ai_community_forensics: "当前项目：跨生成器像素检测。",
    ai_nonescape_mini: "当前项目：Nonescape Mini 补充检测。",
    camera: "当前项目：相机参数测量，为 G5 提供数据。",
    synthesis: "正在整理检测结果。",
    complete: "检测结束。",
    cancelled: "已取消这张图片对应的分析任务。",
  };
  const observationLabels = {
    positive: "发现候选",
    negative: "未检出",
    not_observed: "未取得结果",
    neutral: "不作为判断依据",
  };
  const directionLabels = {
    neutral: "不作为判断依据",
    supports_ai: "存在 AI 相关线索",
    supports_camera: "存在相机相关线索",
    conflicting: "线索不一致",
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
    dda_checkpoint_not_available: "DDA 本地检查点不可用，未以启发式分数替代。",
    dda_dinov2_source_not_initialized: "DDA 本地运行时尚未初始化，未以启发式分数替代。",
    dda_worker_failed: "DDA 隔离分析进程未完成，未以启发式分数替代。",
    dda_worker_timed_out: "DDA 隔离分析超时，未以启发式分数替代。",
    dda_input_too_small: "图片短边小于 336 像素，DDA 按登记输入门槛未运行。",
    dda_high_confidence_indicator_is_not_provenance_or_authenticity_proof: "高置信像素信号不是来源、真实性、版权或编辑史的证明。",
    dda_score_below_threshold_is_not_evidence_of_camera_origin: "分数未达阈值并不支持“相机拍摄”结论。",
    dda_no_high_confidence_pixel_signal_is_not_camera_evidence: "未形成高置信 AI 像素信号。",
    dda_jpeg_compression_or_resizing_can_reduce_sensitivity: "JPEG 重编码或缩放会降低此检测器的灵敏度，因此这类结果应视为不确定。",
    dda_scope_is_limited_to_the_registered_generator_holdout_and_unmodified_upload_protocol: "阈值只在登记的 Pixart 校准与原始 SDXL 留出集上审计；未外推为通用 AI 判定。",
    no_high_confidence_pixel_signal_is_not_camera_evidence: "各像素检测均未达到登记的高置信标准。",
    safe_signal_is_scoped_to_lossless_or_unmodified_uploads: "SAFE 补充信号只对无损或未经重编码的上传登记有效。",
    safe_jpeg_reencoding_destroyed_sensitivity_in_local_controls: "本地 JPEG 75 复测中 SAFE 灵敏度消失；传输重编码后不得依赖该信号。",
    safe_failed_the_registered_sdxl_cross_generator_gate: "SAFE 未通过 SDXL 跨生成器门槛，因此只作为范围明确的补充通道。",
    safe_score_below_threshold_is_not_camera_evidence: "SAFE 分数低于阈值不支持相机来源。",
    safe_high_confidence_indicator_is_not_provenance_or_authenticity_proof: "SAFE 高信号不是来源、真实性或编辑史证明。",
    forensic_clip_config_not_available: "耐压缩补充模型的本地配置不可用。",
    forensic_clip_checkpoint_not_available: "耐压缩补充模型的本地检查点不可用。",
    forensic_clip_worker_failed: "耐压缩补充模型的隔离分析进程未完成。",
    forensic_clip_worker_timed_out: "耐压缩补充模型的隔离分析超时。",
    forensic_clip_is_a_low_recall_complement_not_a_general_probability: "耐压缩模型是低召回补充通道，其原始分数不是通用 AI 概率。",
    forensic_clip_registered_holdout_false_positive_rate_is_6_7_percent: "在登记的未见 SDXL 留出集上，该通道的实拍误报率为 6.7%。",
    forensic_clip_limited_review_threshold_has_14_percent_jpeg_holdout_false_positive_rate: "偏向 AI 召回的有限阈值在 JPEG 75 留出集上的实拍误报率为 14%，因此只能给出有限强度结论。",
    forensic_clip_indoor_recall_is_lower_than_outdoor_recall: "该通道对室内场景的检出率明显低于室外场景。",
    forensic_clip_score_below_threshold_is_not_camera_evidence: "耐压缩分数低于阈值不支持相机来源。",
    forensic_clip_high_confidence_indicator_is_not_provenance_or_authenticity_proof: "耐压缩高信号不是来源、真实性或编辑史证明。",
    community_forensics_config_not_available: "跨生成器补充模型的本地配置不可用。",
    community_forensics_checkpoint_not_available: "跨生成器补充模型的本地检查点不可用。",
    community_forensics_worker_failed: "跨生成器补充模型的隔离分析进程未完成。",
    community_forensics_worker_timed_out: "跨生成器补充模型的隔离分析超时。",
    community_forensics_score_is_a_detector_score_not_a_real_world_ai_prevalence_probability: "跨生成器模型分数是检测信号，不是现实世界的 AI 占比。",
    community_forensics_limited_threshold_has_13_5_percent_cross_generator_real_false_positive_rate: "偏召回的有限阈值在登记的跨生成器留出集中有 13.5% 实拍误报，因此只能给出有限强度结论。",
    community_forensics_no_signal_is_not_evidence_of_camera_origin: "跨生成器模型未命中不支持相机来源。",
    community_forensics_high_signal_is_not_provenance_or_authenticity_proof: "跨生成器高信号不是来源、真实性或编辑史证明。",
    nonescape_mini_checkpoint_not_available: "Nonescape Mini 补充模型的本地检查点不可用。",
    nonescape_mini_checkpoint_hash_mismatch: "Nonescape Mini 本地检查点哈希与登记审计不一致，已拒绝运行。",
    nonescape_mini_worker_failed: "Nonescape Mini 隔离分析进程未完成。",
    nonescape_mini_worker_timed_out: "Nonescape Mini 隔离分析超时。",
    nonescape_mini_score_is_a_detector_score_not_a_real_world_ai_prevalence_probability: "Nonescape Mini 分数是检测信号，不是现实世界的 AI 占比。",
    nonescape_mini_high_signal_is_not_provenance_or_authenticity_proof: "Nonescape Mini 高信号不是来源、真实性或编辑史证明。",
    nonescape_mini_score_below_threshold_is_not_camera_evidence: "Nonescape Mini 分数低于阈值不支持相机来源。",
    nonescape_mini_is_a_strict_complement_not_a_standalone_detector: "Nonescape Mini 作为严格的补充通道运行。",
    "nonescape_mini_source_checkpoint_has_no_publisher_cryptographic_checksum; Demirror pins the retrieved file hash locally": "发布方未提供权重哈希；本地仅运行已登记并校验 SHA-256 的文件。",
    nonescape_mini_scope_is_limited_to_registered_Projective_Geometry_cross_generator_and_jpeg85_protocols: "该通道只在登记的跨生成器与 JPEG-85 协议中审计，不能外推为通用保证。",
    exif_metadata_can_be_copied_or_edited: "相机 EXIF 可以被复制或编辑；本次已按规则调整 AI 分数。",
    implicit_watermark_detector_not_configured: "尚未配置兼容的本地隐式水印检测器。",
    sdxl_watermark_pywavelets_not_available: "未安装 SD/SDXL 固定水印所需的本地小波变换依赖。",
    sdxl_watermark_pywavelets_version_not_pinned: "小波变换依赖版本与登记版本不一致，已拒绝运行。",
    sdxl_watermark_worker_timed_out: "SD/SDXL 水印隔离进程超时。",
    sdxl_watermark_worker_failed: "SD/SDXL 水印隔离进程未完成。",
    sdxl_watermark_short_side_below_256: "图片短边小于 256 像素，不适用该固定水印检测。",
    sdxl_watermark_input_pixel_limit_exceeded: "图片像素数超过水印检测的本地资源上限。",
    sdxl_watermark_is_open_and_can_be_copied_to_non_ai_images: "该固定水印和编码器是公开的，标记可以被复制到非 AI 图片。",
    sdxl_watermark_negative_is_not_camera_evidence: "未检出该固定水印不支持相机来源。",
    sdxl_watermark_coverage_depends_on_generator_export_path: "该水印只覆盖保留了固定标记的部分 SD/SDXL 导出路径。",
    sdxl_watermark_resizing_cropping_or_reencoding_can_destroy_signal: "缩放、裁剪或重新编码可能破坏该固定水印。",
    trustmark_q_model_not_available: "尚未安装经过校验的 TrustMark Q 本地模型。",
    trustmark_q_model_not_readable: "无法读取 TrustMark Q 本地模型。",
    trustmark_q_model_sha256_mismatch: "TrustMark Q 模型校验失败，已拒绝运行。",
    trustmark_q_worker_timed_out: "TrustMark Q 隔离进程超时。",
    trustmark_q_worker_failed: "TrustMark Q 隔离进程未完成。",
    trustmark_q_short_side_below_150: "图片短边小于 150 像素，不适用 TrustMark Q 检测。",
    trustmark_q_input_pixel_limit_exceeded: "图片像素数超过 TrustMark Q 的本地资源上限。",
    trustmark_q_variant_only: "当前只检测 TrustMark Q 变体。",
    trustmark_q_bch5_schema_only: "为控制误报，当前只接受 TrustMark 默认的 BCH_5 载荷格式。",
    trustmark_q_correction_budget_reduced_to_3: "为控制误报，只接受最多纠正 3 位错误的 BCH_5 结果；受损更重的水印会保持未检出。",
    trustmark_identifier_is_not_ai_evidence_without_trusted_provenance: "TrustMark 标识本身不证明图片由 AI 生成；需要可信来源记录绑定后才能改变综合判断。",
    trustmark_identifier_can_be_reencoded_or_removed: "TrustMark 标识可能被重新编码、复制或移除。",
    trustmark_payload_withheld_from_result: "为减少标识泄露，结果中仅显示载荷摘要，不返回原始内容。",
    trustmark_remote_resolver_disabled: "离线模式不会连接远程服务解析标识归属。",
    trustmark_negative_is_not_camera_evidence: "未检出 TrustMark 不支持相机来源。",
    openai_api_key_not_configured: "本机未配置 OPENAI_API_KEY，官方 OpenAI 来源验证没有运行。",
    openai_api_key_rejected: "OpenAI 拒绝了本机 API Key；请检查密钥、项目和权限。",
    openai_provenance_access_denied: "当前 OpenAI 项目没有内容来源验证接口的访问权限。",
    openai_provenance_rate_limited: "OpenAI 来源验证暂时达到调用限额。",
    openai_provenance_request_timed_out: "OpenAI 来源验证请求超时；本地检测结果仍已保留。",
    openai_provenance_network_unavailable: "当前网络无法访问 OpenAI 来源验证 API；本地检测结果仍已保留。",
    openai_provenance_http_error: "OpenAI 来源验证 API 返回错误；本地检测结果仍已保留。",
    openai_provenance_response_invalid: "OpenAI 来源验证返回了无法安全解释的响应。",
    openai_provenance_unhandled_failure: "OpenAI 来源验证发生未处理错误；请求细节和密钥没有写入结果。",
    openai_response_too_large: "OpenAI 来源验证响应超过本地安全上限。",
    openai_provenance_input_unavailable: "无法读取要发送给 OpenAI 的图片。",
    openai_provenance_input_too_large: "图片超过 OpenAI 来源验证适配器的本地上传上限。",
    openai_provider_signal_only_covers_supported_openai_content: "OpenAI 官方接口只检测其支持的 OpenAI 来源信号，不是通用 AI 检测器。",
    openai_not_detected_does_not_rule_out_ai: "OpenAI 未检出不排除图片由 OpenAI 旧模型、其他 AI 工具生成，或水印已被变换破坏。",
    openai_c2pa_signal_not_validated: "发现 OpenAI C2PA 相关信号，但没有形成有效或可信的验证状态，因此不参与综合判断。",
    camera_consistency_not_calibrated_for_origin_decision: "拍摄参数一致性尚未按来源类别校准，因此不参与 AI 分数。",
    c2pa_capture_declaration_not_trusted_for_camera_decision: "C2PA 捕获声明虽可读取，但未通过受信任来源链验证，因此不会降低 AI 分数。",
    opencv_lsd_quality_is_backend_relative_not_probability: "线段质量是 OpenCV LSD 内部相对量，不是来源概率。",
    p0_geometry_is_uncalibrated_not_ai_evidence: "几何候选尚未通过来源盲测门槛，不作为 AI 证据。",
    special_imaging_is_only_metadata_gated_in_p0: "特殊成像目前只通过元数据门控，尚未进行视觉识别。",
    special_imaging_not_assessed_without_metadata_or_manual_tag: "没有元数据或人工标签时，系统不判断是否属于特殊成像。",
    p1_e_cam_is_an_uncalibrated_camera_measurement_not_source_evidence: "E_cam 尚未完成来源校准，只是相机参数一致性测量。",
    e_cam_requires_a_qualified_full_image_and_at_least_three_qualified_crops: "E_cam 需要合格的整图估计和至少三个合格裁剪。",
    "full_image_excluded:uncertainty_above_gate": "整图相机估计的不确定性超过测量门槛，E_cam 未形成。",
    trust_list_version_not_configured: "尚未配置离线 C2PA 信任列表版本。",
    web_analysis_failed: "本地分析服务发生未处理错误。",
  };

  const form = document.querySelector("#upload-form");
  const input = document.querySelector("#file-input");
  const dropzone = document.querySelector("#dropzone");
  const uploadPlaceholder = document.querySelector("#upload-placeholder");
  const imagePreview = document.querySelector("#image-preview");
  const selectedFile = document.querySelector("#selected-file");
  const cancelButton = document.querySelector("#cancel-button");
  const submitButton = document.querySelector("#submit-button");
  const externalChecks = document.querySelector("#external-checks");
  const openaiOption = document.querySelector("#openai-option");
  const openaiProvenance = document.querySelector("#openai-provenance");
  const openaiCapability = document.querySelector("#openai-capability");
  const googleOption = document.querySelector("#google-option");
  const privacySummary = document.querySelector("#privacy-summary");
  const analysisPanel = document.querySelector("#analysis-panel");
  const statusDot = document.querySelector("#status-dot");
  const statusLabel = document.querySelector("#status-label");
  const statusDetail = document.querySelector("#status-detail");
  const progressTrack = document.querySelector("#progress-track");
  const progressBar = document.querySelector("#progress-bar");
  const progressMeta = document.querySelector("#progress-meta");
  const jobFilename = document.querySelector("#job-filename");
  const resultPanel = document.querySelector("#result-panel");
  let selected = null;
  let selectedJobId = null;
  let selectionRevision = 0;
  let isSubmitting = false;
  let pollTimer = null;
  let previewUrl = null;
  let visualSignature = "";

  const asText = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : String(value);
  const percentage = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  const degrees = (value) => Number.isFinite(value) ? `${(value * 180 / Math.PI).toFixed(2)}°` : "—";

  function formatDuration(totalSeconds) {
    const seconds = Math.max(0, Math.floor(totalSeconds));
    if (seconds < 60) return `${seconds} 秒`;
    return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
  }

  function detectorMetric(signal, thresholdKey = "high_confidence_threshold", thresholdLabel = "阈值") {
    if (!signal) return "未配置";
    if (signal.status !== "available" || !Number.isFinite(signal.value)) {
      const reason = Array.isArray(signal.limitations) ? signal.limitations[0] : null;
      return reason ? humanizeLimitation(reason) : "本次未形成分数";
    }
    const threshold = signal.details?.[thresholdKey];
    return Number.isFinite(threshold)
      ? `${percentage(signal.value)} / ${thresholdLabel} ${percentage(threshold)}`
      : percentage(signal.value);
  }

  function setSelected(file) {
    selected = file || null;
    selectedFile.textContent = selected ? `${selected.name} · ${(selected.size / 1024 / 1024).toFixed(2)} MB` : "未选择文件";
    updateSelectionControls();
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    if (!selected) {
      imagePreview.removeAttribute("src");
      imagePreview.alt = "";
      imagePreview.hidden = true;
      uploadPlaceholder.hidden = false;
      dropzone.classList.remove("has-preview");
      return;
    }
    previewUrl = URL.createObjectURL(selected);
    imagePreview.src = previewUrl;
    imagePreview.alt = `已选择图片：${selected.name}`;
    imagePreview.hidden = false;
    uploadPlaceholder.hidden = true;
    dropzone.classList.add("has-preview");
  }

  function updateSelectionControls() {
    input.disabled = isSubmitting || selectedJobId !== null;
    submitButton.disabled = !selected || isSubmitting;
    cancelButton.disabled = !selected;
  }

  async function cancelSelectedJob(jobId) {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      // The next submission still remains safe: it is addressed to a different job ID.
    }
    if (!response.ok) throw new Error(payload?.error || "无法取消这张图片的分析任务。");
    return payload;
  }

  function reportCancellationFailure(error) {
    analysisPanel.hidden = false;
    statusDot.dataset.status = "failed";
    statusLabel.textContent = "取消请求未完成";
    statusDetail.textContent = error instanceof Error
      ? `${error.message} 该图片的分析可能仍在继续。`
      : "该图片的分析可能仍在继续。";
  }

  function clearSelected() {
    const jobId = selectedJobId;
    selectionRevision += 1;
    selectedJobId = null;
    isSubmitting = false;
    input.value = "";
    setSelected(null);
    resetPanels();
    submitButton.textContent = "分析图片";
    if (jobId) {
      void cancelSelectedJob(jobId).catch(reportCancellationFailure);
    }
  }

  function updatePrivacySummary() {
    privacySummary.textContent = openaiProvenance.checked
      ? "本次图片会发送至 OpenAI 进行来源验证；其他分析仍在本机完成。"
      : "未启用在线验证。图片和分析结果保存在本机临时目录。";
  }

  async function loadCapabilities() {
    try {
      const response = await fetch("/api/capabilities", { cache: "no-store" });
      if (!response.ok) throw new Error("capability request failed");
      const payload = await response.json();
      const openaiConfigured = payload.external_verification?.openai?.configured === true;
      const googleConfigured = payload.external_verification?.google?.configured === true;
      openaiOption.hidden = !openaiConfigured;
      googleOption.hidden = !googleConfigured;
      externalChecks.hidden = !(openaiConfigured || googleConfigured);
      openaiProvenance.disabled = !openaiConfigured;
      if (!openaiConfigured) openaiProvenance.checked = false;
      openaiCapability.dataset.configured = String(openaiConfigured);
      openaiCapability.textContent = openaiConfigured ? "可用" : "";
      updatePrivacySummary();
    } catch (_) {
      externalChecks.hidden = true;
      openaiOption.hidden = true;
      googleOption.hidden = true;
      openaiProvenance.disabled = true;
      openaiProvenance.checked = false;
      openaiCapability.dataset.configured = "false";
      openaiCapability.textContent = "";
      updatePrivacySummary();
    }
  }

  function resetPanels() {
    clearTimeout(pollTimer);
    analysisPanel.hidden = true;
    resultPanel.hidden = true;
    progressBar.style.width = "0%";
    progressTrack.setAttribute("aria-valuenow", "0");
    progressMeta.textContent = "0% · 已用 0 秒";
    document.querySelector("#visual-grid").replaceChildren();
    visualSignature = "";
    document.querySelector("#limitations-list").replaceChildren();
  }

  function renderStatus(job) {
    const [label, detail] = stateLabels[job.status] || ["状态未知", "服务返回了未知状态。"];
    const progress = Number.isFinite(job.progress_percent)
      ? Math.max(0, Math.min(100, Math.round(job.progress_percent)))
      : (terminalStates.has(job.status) ? 100 : 0);
    const createdAt = Date.parse(job.created_at_utc);
    const elapsedSeconds = Number.isFinite(createdAt) ? (Date.now() - createdAt) / 1000 : 0;
    analysisPanel.hidden = false;
    statusDot.dataset.status = job.status;
    statusLabel.textContent = label;
    statusDetail.textContent = stageLabels[job.stage] || detail;
    progressBar.style.width = `${progress}%`;
    progressTrack.setAttribute("aria-valuenow", String(progress));
    progressMeta.textContent = `${progress}% · 已用 ${formatDuration(elapsedSeconds)}`;
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
    const previousScrollTop = metricGrid.scrollTop;
    metricGrid.replaceChildren();
    metrics.forEach(([label, metricValue]) => addMetric(metricGrid, label, metricValue));
    metricGrid.scrollTop = previousScrollTop;
    const metricDetails = metricGrid.closest("details");
    if (metricDetails) {
      metricDetails.hidden = metrics.length === 0;
      // Polling refreshes completed cards while slower checks are still
      // running. Preserve an explicitly opened data panel across those
      // refreshes; it should close only when the user closes it, opens another
      // panel, clicks outside, scrolls the page, or the card loses its data.
      if (metrics.length === 0) metricDetails.open = false;
    }
    window.requestAnimationFrame(() => syncMetricScrollbar(metricGrid));
  }

  function setOverallScoreRing(score) {
    const ring = document.querySelector("#overall-score-ring");
    const value = document.querySelector("#overall-score-value");
    if (!(ring instanceof HTMLElement) || !(value instanceof HTMLElement)) return;
    if (!Number.isFinite(score)) {
      ring.hidden = true;
      value.textContent = "—";
      return;
    }
    const normalized = Math.max(-100, Math.min(100, Math.round(score)));
    const magnitude = Math.abs(normalized);
    ring.hidden = false;
    ring.style.setProperty("--score", String(magnitude));
    ring.style.setProperty(
      "--score-color",
      normalized < 0 ? "var(--success)" : normalized >= 35 ? "var(--accent)" : "var(--info)",
    );
    ring.dataset.sign = normalized < 0 ? "negative" : normalized > 0 ? "positive" : "neutral";
    ring.setAttribute("aria-label", `AI 分数 ${normalized}，绝对值 ${magnitude} / 100`);
    value.textContent = String(normalized);
  }

  const pixelCardKeys = new Set(["dda", "safe", "forensic", "community", "nonescape"]);
  const c2paCardKeys = new Set(["c2pa-declaration", "c2pa-signature", "c2pa-capture"]);
  const geometryCheckIds = ["G1", "G2", "G3", "G4", "G5"];
  const completedGeometryCheckStates = new Set(["available", "not_applicable", "failed"]);
  const evidenceCardKeys = [
    "dda", "safe", "forensic", "community", "nonescape",
    "c2pa-declaration", "c2pa-signature", "c2pa-capture",
    "metadata", "p0", "geometry-structure",
    "camera", "watermark",
  ];
  const contributionKeys = {
    "c2pa-declaration": "c2pa_declaration",
    "c2pa-signature": "c2pa_signature",
    "c2pa-capture": "c2pa_capture",
  };

  function geometryStructureIsComplete(result) {
    const measurement = result.geometry_v2;
    if (!measurement || typeof measurement !== "object") return false;
    if (measurement.status === "failed") return true;
    const checks = Array.isArray(measurement.checks) ? measurement.checks : [];
    return geometryCheckIds.every((checkId) => {
      const check = checks.find((candidate) => candidate.check_id === checkId);
      return Boolean(check && completedGeometryCheckStates.has(check.status));
    });
  }

  function evidenceIsAvailable(result, key, job = null) {
    if (pixelCardKeys.has(key)) return Boolean(result.p3);
    if (c2paCardKeys.has(key)) return Boolean(result.c2pa);
    if (key === "geometry-structure") {
      const progress = Number(job?.progress_percent);
      const g5StageComplete = job === null || (Number.isFinite(progress) && progress >= 92);
      return g5StageComplete && geometryStructureIsComplete(result);
    }
    if (key === "metadata") return Boolean(result.origin?.camera_metadata);
    return Object.prototype.hasOwnProperty.call(result, key);
  }

  function fallbackContribution(key) {
    if (key === "p0") return { points: 0, state: "neutral", explanation: "等待几何关系模型结果" };
    if (key === "geometry-structure") return { points: 0, state: "neutral", explanation: "本项仅用于几何结构复核" };
    if (key === "camera") return { points: 0, state: "neutral", explanation: "相机参数一致性仅供复核" };
    return { points: 0, state: "neutral", explanation: "等待该项检测完成" };
  }

  function renderEvidenceOrder(result, job) {
    const grid = document.querySelector(".summary-grid");
    const components = result.origin?.score_components || {};
    const cards = evidenceCardKeys.map((key, index) => {
      const card = document.querySelector(`#${key}-card`);
      const score = document.querySelector(`#${key}-score`);
      const available = evidenceIsAvailable(result, key, job);
      if (!(card instanceof HTMLElement) || !(score instanceof HTMLElement)) return null;
      card.hidden = !available;
      const component = components[contributionKeys[key] || key] || fallbackContribution(key);
      const points = Number.isFinite(component.points) ? Math.round(component.points) : 0;
      const sign = points > 0 ? "+" : "";
      score.textContent = `${sign}${points} 分`;
      score.dataset.state = component.state || "neutral";
      score.title = asText(component.explanation, "本项当前不改变 AI 分数。");
      const observation = document.querySelector(`#${key}-observation`)?.textContent?.trim() || "";
      const notRun = ["未运行", "未取得结果", "未取得参数结果", "未完成", "结构检测未完成", "部分结构检查未完成", "未配置检测"].includes(observation);
      const noInformation = (key === "metadata" && observation === "未发现相机信息") || observation === "可测结构不足";
      const group = points > 0 ? 0 : points < 0 ? 1 : noInformation ? 3 : notRun ? 4 : 2;
      return { card, group, points, index };
    }).filter(Boolean);
    cards.sort((left, right) => left.group - right.group || right.points - left.points || left.index - right.index);
    cards.forEach(({ card }) => grid.append(card));
  }

  function closeMetricDetails(except = null) {
    document.querySelectorAll(".metric-details[open]").forEach((details) => {
      if (details !== except) details.open = false;
    });
  }

  function keepMetricScrollLocal(event) {
    const metrics = event.currentTarget;
    if (!(metrics instanceof HTMLElement)) return;
    const canScroll = metrics.scrollHeight > metrics.clientHeight;
    const atTop = metrics.scrollTop <= 0;
    const atBottom = metrics.scrollTop + metrics.clientHeight >= metrics.scrollHeight - 1;
    if (!canScroll || (event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) {
      event.preventDefault();
    }
    event.stopPropagation();
  }

  function syncMetricScrollbar(metrics) {
    const panel = metrics.parentElement;
    if (!(panel instanceof HTMLElement) || !panel.classList.contains("metric-scroll-panel")) return;
    const track = panel.querySelector(".metric-scrollbar");
    const thumb = panel.querySelector(".metric-scrollbar-thumb");
    if (!(track instanceof HTMLElement) || !(thumb instanceof HTMLElement)) return;
    const details = panel.closest(".metric-details");
    if (!(details instanceof HTMLElement) || !details.matches("[open]")) {
      track.hidden = true;
      return;
    }
    const maximumScroll = Math.max(0, metrics.scrollHeight - metrics.clientHeight);
    const scrollable = maximumScroll > 1;
    track.hidden = !scrollable;
    if (!scrollable) {
      thumb.style.height = "";
      thumb.style.transform = "";
      return;
    }
    const trackHeight = track.clientHeight;
    if (!trackHeight) {
      window.requestAnimationFrame(() => syncMetricScrollbar(metrics));
      return;
    }
    const thumbHeight = Math.max(28, Math.round(trackHeight * metrics.clientHeight / metrics.scrollHeight));
    const travel = Math.max(0, trackHeight - thumbHeight);
    const offset = travel * metrics.scrollTop / maximumScroll;
    thumb.style.height = `${thumbHeight}px`;
    thumb.style.transform = `translateY(${offset}px)`;
  }

  function scrollMetricFromTrack(metrics, track, clientY) {
    const thumb = track.querySelector(".metric-scrollbar-thumb");
    if (!(thumb instanceof HTMLElement)) return;
    const maximumScroll = Math.max(0, metrics.scrollHeight - metrics.clientHeight);
    const bounds = track.getBoundingClientRect();
    const thumbHeight = thumb.getBoundingClientRect().height;
    const travel = Math.max(0, bounds.height - thumbHeight);
    if (!maximumScroll || !travel) return;
    const offset = Math.max(0, Math.min(travel, clientY - bounds.top - thumbHeight / 2));
    metrics.scrollTop = offset / travel * maximumScroll;
  }

  function prepareMetricScrollbar(metrics) {
    const details = metrics.closest(".metric-details");
    if (!(details instanceof HTMLElement) || metrics.parentElement?.classList.contains("metric-scroll-panel")) return;
    const panel = document.createElement("div");
    panel.className = "metric-scroll-panel";
    const track = document.createElement("span");
    track.className = "metric-scrollbar";
    track.setAttribute("aria-hidden", "true");
    const thumb = document.createElement("span");
    thumb.className = "metric-scrollbar-thumb";
    track.append(thumb);
    metrics.replaceWith(panel);
    panel.append(metrics, track);

    metrics.addEventListener("scroll", () => syncMetricScrollbar(metrics), { passive: true });
    metrics.addEventListener("wheel", keepMetricScrollLocal, { passive: false });
    track.addEventListener("pointerdown", (event) => {
      if (event.target === thumb) return;
      event.preventDefault();
      event.stopPropagation();
      scrollMetricFromTrack(metrics, track, event.clientY);
    });
    thumb.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      const startY = event.clientY;
      const startScroll = metrics.scrollTop;
      const move = (nextEvent) => {
        const maximumScroll = Math.max(0, metrics.scrollHeight - metrics.clientHeight);
        const travel = Math.max(0, track.clientHeight - thumb.getBoundingClientRect().height);
        if (maximumScroll && travel) metrics.scrollTop = Math.max(0, Math.min(maximumScroll, startScroll + (nextEvent.clientY - startY) / travel * maximumScroll));
      };
      const stop = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", stop);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", stop, { once: true });
    });
    if ("ResizeObserver" in window) {
      const observer = new ResizeObserver(() => syncMetricScrollbar(metrics));
      observer.observe(metrics);
      observer.observe(panel);
    }
  }

  function renderOverall(result) {
    const origin = result.origin || {};
    if (origin.decision) {
      const evidence = Array.isArray(origin.supporting_evidence) ? origin.supporting_evidence : [];
      const score = Number.isFinite(origin.ai_score) ? Math.round(origin.ai_score) : 0;
      setOverallScoreRing(score);
      const sharedMetrics = [
        ["判断强度", origin.evidence_strength === "high" ? "高" : "有限"],
        ["主要依据", evidence.join("、") || "未形成可用依据"],
      ];
      const conclusion = asText(origin.summary) || "未检出 AI 信号";
      if (origin.decision === "possible_ai") {
        setCard("overall", conclusion, asText(origin.explanation), [
          ...sharedMetrics,
          ["综合方式", "按各项已完成线索汇总"],
        ]);
        return;
      }
      if (origin.decision === "possible_photo") {
        setCard("overall", conclusion, asText(origin.explanation), [
          ...sharedMetrics,
          ["综合方式", "拍摄线索已纳入综合判断"],
        ]);
        return;
      }
      setCard("overall", conclusion, asText(origin.explanation), [
        ...sharedMetrics,
        ["综合方式", "已完成线索纳入综合判断"],
      ]);
      return;
    }

    // Compatibility view for local jobs completed before the score policy.
    setOverallScoreRing(null);
    const p3 = result.p3 || {};
    const signals = Array.isArray(p3.signals) ? p3.signals : [];
    const verified = signals.find((signal) => signal.name === "verified_c2pa" && signal.status === "available");
    const dda = signals.find((signal) => signal.name === "dda_pixel_detector");
    const safe = signals.find((signal) => signal.name === "safe_pixel_detector");
    const forensic = signals.find((signal) => signal.name === "forensic_clip_detector");
    const community = signals.find((signal) => signal.name === "community_forensics_detector");
    const nonescape = signals.find((signal) => signal.name === "nonescape_mini_detector");
    const high = p3.status === "available" && p3.risk_band === "high";

    if (verified || high) {
      setCard("overall", "可能为 AI", "AI 像素信号达到预先登记的高阈值。", [
        ["判断强度", "高"],
        ["AI 分数", "旧版结果未提供"],
        ["主要依据", verified ? "已验证的来源记录" : "AI 像素检测"],
      ]);
      return;
    }
    setCard("overall", "未检出 AI 信号", "没有出现足以支持 AI 判断的直接信号。", [
      ["判断状态", "不确定"],
      ["AI 像素检测", p3.status === "available" ? "未达到高置信标准" : "未形成可用结果"],
      ["来源记录", "未发现已验证的 AI 声明"],
    ]);
  }

  function renderP0(result) {
    const p0 = result.p0 || {};
    const evidence = p0.evidence || {};
    const features = evidence.features || {};
    const geometry = result.geometry_ai || {};
    const probability = Number(geometry.probability);
    const geometryAvailable = geometry.status === "available" && Number.isFinite(probability);
    const value = geometry.risk_band === "high"
      ? "检出较强几何 AI 线索"
      : geometry.risk_band === "medium"
        ? "检出辅助几何 AI 线索"
        : evidence.run_status === "ok"
          ? "未发现明显几何线索"
          : "未取得结构结果";
    const description = geometryAvailable
      ? "仅比较画面内线段的位置和方向关系，结果作为受限的辅助线索。"
      : evidence.run_status === "not_applicable"
        ? "图片不满足稳定的几何测量条件。"
        : "已检查线条、平行族和消失方向。";
    const relationshipMetrics = geometryAvailable ? [
      ["几何 AI 值", percentage(probability)],
      ["辅助阈值", percentage(Number(geometry.decision_threshold))],
      ["较强阈值", percentage(Number(geometry.strong_threshold))],
      ["关系模型", asText(geometry.model_version)],
      ["参与线段", asText(geometry.line_count, "0")],
    ] : [];
    setCard("p0", value, description, [
      ...relationshipMetrics,
      ["运行状态", asText(evidence.run_status)],
      ["适用性", percentage(evidence.applicability)],
      ["覆盖度", percentage(evidence.coverage)],
      ["可靠度", percentage(evidence.reliability)],
      ["稳定线族", asText(features.families?.length, "0")],
      ["候选异常线", asText(features.anomalous_lines?.length, "0")],
    ]);
  }

  const geometryCheckDefinitions = {
    G1: {
      label: "G1 局部平行线族",
      measurements: [
        ["comparable_regions", "可比较区域"],
      ],
    },
    G2: {
      label: "G2 消失方向",
      measurements: [
        ["comparison_count", "完成比较"],
      ],
    },
    G3: {
      label: "G3 重复结构间距",
      measurements: [
        ["sequence_count", "可测序列"],
      ],
    },
    G4: {
      label: "G4 线段连接关系",
      measurements: [
        ["tested_pair_count", "测试线对"],
      ],
    },
    G5: {
      label: "G5 相机与透视场",
      measurements: [
        ["attempted_backend_count", "尝试后端"],
        ["measured_backend_count", "可用后端"],
      ],
    },
  };

  function geometryCheckSummary(check) {
    if (check.status === "failed") return "未完成";
    if (check.status === "not_applicable") return "不适用";
    const score = check.anomaly_score === null || check.anomaly_score === undefined
      ? Number.NaN
      : Number(check.anomaly_score);
    const findings = Array.isArray(check.findings) ? check.findings : [];
    if (Number.isFinite(score) && score >= 0.5) {
      return `发现 ${findings.length} 个候选 · ${percentage(score)}`;
    }
    return Number.isFinite(score) ? `未达复核条件 · ${percentage(score)}` : "已完成";
  }

  function renderGeometryStructure(result) {
    const measurement = result.geometry_v2 || {};
    const checks = Array.isArray(result.geometry_v2?.checks) ? result.geometry_v2.checks : [];
    if (measurement.status === "failed") {
      setCard("geometry-structure", "结构检测未完成", "本次没有形成完整的 G1–G5 结构检测结果。", [
        ["检测状态", "未完成"],
      ]);
      return;
    }
    const completedChecks = geometryCheckIds
      .map((checkId) => checks.find((candidate) => candidate.check_id === checkId))
      .filter(Boolean);
    const availableScores = completedChecks
      .filter((check) => check.status === "available")
      .map((check) => Number(check.anomaly_score))
      .filter(Number.isFinite);
    const value = completedChecks.some((check) => check.status === "failed")
      ? "部分结构检查未完成"
      : availableScores.some((score) => score >= 0.5)
        ? "发现结构复核候选"
        : completedChecks.every((check) => check.status === "not_applicable")
          ? "可测结构不足"
          : "未发现达到复核条件的偏差";
    const metrics = [
      ["结构适用性", percentage(Number(measurement.applicability))],
      ["结构区域", asText(measurement.region_count, "0")],
      ["稳定线族", asText(measurement.stable_family_count, "0")],
    ];
    completedChecks.forEach((check) => {
      const definition = geometryCheckDefinitions[check.check_id];
      metrics.push([definition.label, geometryCheckSummary(check)]);
      const measurements = check.measurements && typeof check.measurements === "object"
        ? check.measurements
        : {};
      definition.measurements.forEach(([key, label]) => {
        if (measurements[key] !== null && measurements[key] !== undefined && Number.isFinite(Number(measurements[key]))) {
          metrics.push([`${check.check_id} ${label}`, String(measurements[key])]);
        }
      });
      if (check.check_id === "G5" && measurements.geocalib && typeof measurements.geocalib === "object") {
        const eCam = measurements.geocalib.e_cam === null || measurements.geocalib.e_cam === undefined
          ? Number.NaN
          : Number(measurements.geocalib.e_cam);
        if (Number.isFinite(eCam)) metrics.push(["G5 GeoCalib 一致性残差", percentage(eCam)]);
        if (Number.isFinite(Number(measurements.geocalib.qualified_crop_count))) {
          metrics.push(["G5 有效裁切", String(measurements.geocalib.qualified_crop_count)]);
        }
      }
    });
    setCard(
      "geometry-structure",
      value,
      "汇总局部平行线族、消失方向、重复结构间距、线段连接关系，以及相机与透视场一致性。",
      metrics,
    );
  }

  function renderPixelDetectorCard(prefix, signal, p3, label, pixelSignalCount) {
    if (p3.status !== "available") {
      setCard(prefix, "未取得结果", "本次像素检测没有形成可用结果，不计入 AI 分数。", [
        ["检测状态", asText(p3.status)],
      ]);
      return;
    }
    if (!signal) {
      setCard(prefix, "未运行", "本次已由来源记录处理，未重复运行这一像素检测。", []);
      return;
    }
    const value = Number(signal.value);
    const highThreshold = Number(signal.details?.high_confidence_threshold);
    const limitedThreshold = Number(signal.details?.limited_review_threshold);
    const highEligible = signal.details?.high_confidence_eligible !== false;
    const high = signal.status === "available" && Number.isFinite(value)
      && Number.isFinite(highThreshold) && value >= highThreshold && highEligible;
    const limited = signal.status === "available" && Number.isFinite(value)
      && ((Number.isFinite(limitedThreshold) && value >= limitedThreshold && (!Number.isFinite(highThreshold) || value < highThreshold))
        || (Number.isFinite(highThreshold) && value >= highThreshold && !highEligible));
    const fallbackHigh = signal.status === "available" && !Number.isFinite(highThreshold)
      && pixelSignalCount === 1 && p3.risk_band === "high";
    const fallbackLimited = signal.status === "available" && !Number.isFinite(highThreshold)
      && pixelSignalCount === 1 && p3.risk_band === "medium";
    const observation = high || fallbackHigh
      ? "检出高强度 AI 信号"
      : limited || fallbackLimited
        ? "检出有限 AI 信号"
        : signal.status === "available"
          ? "未检出 AI 信号"
          : "未取得结果";
    const description = high || fallbackHigh
      ? `${label}达到已登记的高强度阈值。原始模型分数和阈值在数据中保留。`
      : limited || fallbackLimited
        ? `${label}达到有限强度复核阈值，需结合原始文件和其他信号复核。`
        : signal.status === "available"
          ? `${label}未达到已登记的提示阈值。`
          : `${label}未形成可用分数，不计入 AI 分数。`;
    setCard(prefix, observation, description, [
      ["模型分数", detectorMetric(signal)],
      ["有限阈值", detectorMetric(signal, "limited_review_threshold", "阈值")],
      ["高强度适用", signal.details?.high_confidence_eligible === false ? "否（当前格式仅复核）" : "是"],
      ["检测说明", asText(signal.interpretation)],
    ]);
  }

  function renderP3(result) {
    const p3 = result.p3 || {};
    const signals = Array.isArray(p3.signals) ? p3.signals : [];
    const definitions = [
      ["dda", "dda_pixel_detector", "DDA 像素检测"],
      ["safe", "safe_pixel_detector", "SAFE 纹理检测"],
      ["forensic", "forensic_clip_detector", "耐压缩像素检测"],
      ["community", "community_forensics_detector", "跨生成器像素检测"],
      ["nonescape", "nonescape_mini_detector", "Nonescape Mini"],
    ];
    const pixelSignalCount = signals.filter((signal) => definitions.some(([, name]) => name === signal.name)).length;
    definitions.forEach(([prefix, name, label]) => {
      renderPixelDetectorCard(prefix, signals.find((signal) => signal.name === name), p3, label, pixelSignalCount);
    });
  }

  function renderLegacyP3Detail(result) {
    const p3 = result.p3 || {};
    if (!Object.keys(p3).length) {
      setCard("p3", "未运行", "本次没有得到可用的像素检测结果。", []);
      return;
    }
    if (p3.status !== "available") {
      setCard("p3", "未取得结果", "本次像素检测没有完成，因此不使用替代分数。", [
        ["检测状态", asText(p3.status)],
      ]);
      return;
    }
    const signals = Array.isArray(p3.signals) ? p3.signals : [];
    const verified = signals.find((signal) => signal.name === "verified_c2pa" && signal.status === "available");
    const dda = signals.find((signal) => signal.name === "dda_pixel_detector");
    const safe = signals.find((signal) => signal.name === "safe_pixel_detector");
    const forensic = signals.find((signal) => signal.name === "forensic_clip_detector");
    const community = signals.find((signal) => signal.name === "community_forensics_detector");
    const nonescape = signals.find((signal) => signal.name === "nonescape_mini_detector");
    const evaluation = p3.evaluation?.dda?.at_high_confidence_threshold || p3.evaluation?.at_high_confidence_threshold || {};
    if (verified) {
      setCard("p3", "来源记录确认 AI", "图片自带且已验证的来源记录说明其由 AI 生成。这是来源信息，不是像素推断。", [
        ["判断依据", "已验证的来源记录"],
        ["可靠程度", percentage(p3.reliability)],
        ["技术记录", "C2PA · trained algorithmic media"],
      ]);
      return;
    }
    const high = p3.risk_band === "high";
    const limited = p3.risk_band === "medium";
    const channelNames = {
      dda_pixel_detector: "DDA 主模型",
      safe_pixel_detector: "SAFE 无损补充",
      forensic_clip_detector: "耐压缩补充",
      community_forensics_detector: "跨生成器补充",
      nonescape_mini_detector: "严格补充模型",
    };
    const hitChannels = signals
      .filter((signal) => signal.status === "available"
        && Number.isFinite(signal.value)
        && Number.isFinite(signal.details?.high_confidence_threshold)
        && signal.value >= signal.details.high_confidence_threshold)
      .map((signal) => channelNames[signal.name])
      .filter(Boolean);
    const limitedChannels = signals
      .filter((signal) => signal.status === "available"
        && Number.isFinite(signal.value)
        && Number.isFinite(signal.details?.limited_review_threshold)
        && signal.value >= signal.details.limited_review_threshold
        && signal.value < signal.details.high_confidence_threshold)
      .map((signal) => channelNames[signal.name])
      .filter(Boolean);
    const compressedEvaluation = p3.evaluation?.forensic_clip?.held_out_sdxl_jpeg75?.at_high_confidence_threshold || {};
    const communityEvaluation = p3.evaluation?.community_forensics?.held_out_sdxl?.at_high_confidence_threshold || {};
    const communityCompressedEvaluation = p3.evaluation?.community_forensics?.held_out_sdxl_jpeg85?.at_high_confidence_threshold || {};
    const nonescapeOriginalEvaluation = p3.evaluation?.nonescape_mini?.held_out_cross_generator_original?.combined_or_of_frozen_high_thresholds || {};
    const nonescapeCompressedEvaluation = p3.evaluation?.nonescape_mini?.held_out_cross_generator_jpeg85?.combined_or_of_frozen_high_thresholds || {};
    setCard(
      "p3",
      high ? "检出明显 AI 信号" : (limited ? "检出有限 AI 信号" : "未检出明显 AI 信号"),
      high
        ? "图片像素特征达到已登记的高阈值，是本次综合判断的主要依据。"
        : (limited
          ? "至少一个补充模型达到有限复核阈值。该档的误报较高，建议结合原始文件和人工检查。"
          : "未达到高阈值。压缩、缩放和生成器差异都可能降低检测灵敏度。"),
      high
        ? [
            ["判断强度", "高"],
            ["命中通道", hitChannels.join("、") || "像素检测"],
            ["DDA 主模型", detectorMetric(dda)],
            ["SAFE 无损补充", detectorMetric(safe)],
            ["耐压缩补充", detectorMetric(forensic)],
            ["跨生成器补充", detectorMetric(community)],
            ["严格补充模型", detectorMetric(nonescape)],
            ["DDA 留出验证", `检出 ${percentage(evaluation.recall)}，误报 ${percentage(evaluation.false_positive_rate)}`],
            ["JPEG 75 留出验证", `补充检出 ${percentage(compressedEvaluation.recall)}，误报 ${percentage(compressedEvaluation.false_positive_rate)}`],
            ["跨生成器留出验证", `检出 ${percentage(communityEvaluation.recall)}，误报 ${percentage(communityEvaluation.false_positive_rate)}`],
            ["JPEG 85 留出验证", `跨生成器检出 ${percentage(communityCompressedEvaluation.recall)}，误报 ${percentage(communityCompressedEvaluation.false_positive_rate)}`],
            ["组合原图验证", `准确 ${percentage(nonescapeOriginalEvaluation.accuracy)}，检出 ${percentage(nonescapeOriginalEvaluation.recall)}，误报 ${percentage(nonescapeOriginalEvaluation.false_positive_rate)}`],
            ["组合 JPEG 85 验证", `准确 ${percentage(nonescapeCompressedEvaluation.accuracy)}，检出 ${percentage(nonescapeCompressedEvaluation.recall)}，误报 ${percentage(nonescapeCompressedEvaluation.false_positive_rate)}`],
            ["技术详情", asText(p3.model_version)],
          ]
        : limited
          ? [
              ["判断强度", "有限"],
              ["命中通道", limitedChannels.join("、") || "耐压缩补充"],
              ["DDA 主模型", detectorMetric(dda)],
              ["SAFE 无损补充", detectorMetric(safe)],
              ["耐压缩补充", detectorMetric(forensic, "limited_review_threshold", "有限阈值")],
              ["跨生成器补充", detectorMetric(community, "limited_review_threshold", "有限阈值")],
              ["严格补充模型", detectorMetric(nonescape)],
              ["JPEG 75 留出验证", "有限阈值：检出 28.0%，误报 14.0%"],
              ["复核要求", "建议取得原始文件，并结合结构、来源记录与人工检查"],
              ["技术详情", asText(p3.model_version)],
            ]
          : [
            ["判断状态", "不确定"],
            ["主要原因", "信号未达到高置信标准"],
            ["DDA 主模型", detectorMetric(dda)],
            ["SAFE 无损补充", detectorMetric(safe)],
            ["耐压缩补充", detectorMetric(forensic)],
            ["跨生成器补充", detectorMetric(community)],
            ["严格补充模型", detectorMetric(nonescape)],
            ["注意", "压缩或缩放会降低灵敏度"],
            ["技术详情", asText(p3.model_version)],
          ],
    );
  }

  function renderCamera(result) {
    const camera = result.camera || {};
    const fullImage = camera.full_image || {};
    const eCam = camera.e_cam || {};
    if (!Object.keys(camera).length) {
      setCard("camera", "未运行", "本次没有相机参数一致性结果。", []);
      return;
    }
    if (camera.status === "failed") {
      setCard("camera", "未完成", "相机参数检查没有完成，不作为判断依据。", []);
      return;
    }
    const value = fullImage.status === "ok" ? "已完成参数检查" : "未取得参数结果";
    setCard("camera", value, "用于比较整图和局部区域的相机参数。", [
      ["后端状态", asText(fullImage.status)],
      ["E_cam", asText(eCam.observation)],
      ["Roll", degrees(fullImage.roll)],
      ["Pitch", degrees(fullImage.pitch)],
      ["垂直视场", fullImage.vfov_or_focal?.kind === "vfov_deg" ? `${Number(fullImage.vfov_or_focal.value).toFixed(2)}°` : "—"],
      ["合格裁剪", `${eCam.qualified_crop_ids?.length || 0}/${eCam.required_qualified_crops || "—"}`],
    ]);
  }

  function sourceTypeSlugs(c2pa) {
    return (c2pa.declared_digital_source_types || []).map((value) => String(value)
      .toLowerCase().replace(/[^a-z0-9]/g, ""));
  }

  function renderC2pa(result) {
    const c2pa = result.c2pa || {};
    const sourceTypes = sourceTypeSlugs(c2pa);
    const signatureValid = c2pa.signature_validation_status === "valid";
    const generated = signatureValid && sourceTypes.some((value) => value.includes("trainedalgorithmicmedia"));
    const trustedCapture = signatureValid && c2pa.trust_status === "trusted"
      && sourceTypes.some((value) => value.includes("digitalcapture") || value.includes("computationalcapture"));
    const commonMetrics = [
      ["读取状态", asText(c2pa.status)],
      ["来源类型", (c2pa.declared_digital_source_types || []).join("、") || "—"],
      ["声明动作", asText(c2pa.declared_actions?.length, "0")],
    ];
    setCard(
      "c2pa-declaration",
      generated ? "声明生成式内容" : "未发现生成式声明",
      generated
        ? "来源记录明确声明生成式内容；签名验证状态会单独列出。"
        : "未检出已验证的生成式来源声明。",
      commonMetrics,
    );
    setCard(
      "c2pa-signature",
      signatureValid ? "签名通过验证" : "签名未通过验证",
      generated && signatureValid
        ? "该签名验证使生成式来源声明可以计入 AI 分数。"
        : "签名状态已单独记录。",
      [
        ["签名状态", asText(c2pa.signature_validation_status)],
        ["信任状态", asText(c2pa.trust_status)],
        ["信任列表版本", asText(c2pa.trust_list_version)],
        ["验证状态", asText(c2pa.validation_state)],
      ],
    );
    setCard(
      "c2pa-capture",
      trustedCapture ? "可信数字拍摄来源" : "未形成可信拍摄来源",
      trustedCapture
        ? "可信拍摄来源已使 AI 分数降低。"
        : "未形成可信数字拍摄来源链。",
      [
        ["来源类型", (c2pa.declared_digital_source_types || []).join("、") || "—"],
        ["签名状态", asText(c2pa.signature_validation_status)],
        ["信任状态", asText(c2pa.trust_status)],
      ],
    );
  }

  function renderLegacyC2paDetail(result) {
    const c2pa = result.c2pa || {};
    if (!Object.keys(c2pa).length) {
      setCard("c2pa", "未运行", "本次没有读取到 C2PA 记录。", []);
      return;
    }
    const value = c2pa.manifest_present ? "发现嵌入式清单" : "未发现嵌入式清单";
    setCard("c2pa", value, "只读取图片中已有的嵌入记录。可信的数字拍摄来源链可作为实拍线索；离线模式不检索远程清单。", [
      ["读取状态", asText(c2pa.status)],
      ["签名状态", asText(c2pa.signature_validation_status)],
      ["信任状态", asText(c2pa.trust_status)],
      ["信任列表版本", asText(c2pa.trust_list_version)],
      ["来源类型", (c2pa.declared_digital_source_types || []).join("、") || "—"],
      ["网络访问", asText(c2pa.network_access)],
      ["声明动作", asText(c2pa.declared_actions?.length, "0")],
    ]);
  }

  function renderMetadata(result) {
    const metadata = result.origin?.camera_metadata || {};
    if (!Object.keys(metadata).length) {
      setCard("metadata", "未运行", "本次没有相机信息审阅结果。", []);
      return;
    }
    if (metadata.status === "coherent") {
      setCard("metadata", "相机信息较完整", "相机品牌、型号、拍摄时间和多项拍摄参数均存在。", [
        ["相机", `${asText(metadata.camera_make)} · ${asText(metadata.camera_model)}`],
        ["拍摄时间", asText(metadata.captured_at_local)],
        ["物理参数", (metadata.physical_capture_fields || []).join("、") || "—"],
        ["编辑软件", asText(metadata.software)],
      ]);
      return;
    }
    if (metadata.status === "partial") {
      setCard("metadata", "相机信息不完整", "发现了部分相机信息。", [
        ["相机", `${asText(metadata.camera_make)} · ${asText(metadata.camera_model)}`],
        ["拍摄时间", asText(metadata.captured_at_local)],
        ["物理参数", (metadata.physical_capture_fields || []).join("、") || "—"],
      ]);
      return;
    }
    setCard("metadata", "未发现相机信息", "没有可用的 EXIF 信息。", []);
  }

  function renderWatermark(result) {
    const assessment = result.watermark || result.origin?.implicit_watermark;
    if (!assessment || assessment === "not_configured" || assessment.status === "not_configured") {
      setCard("watermark", "未配置检测", "当前没有可用的本地水印检测器，因此不作为判断依据。", [
        ["检测状态", "未配置"],
      ]);
      return;
    }
    const adapters = Array.isArray(assessment.adapters) ? assessment.adapters : [];
    const positives = adapters.filter((item) => item.observation === "positive");
    const completed = adapters.filter((item) => item.run_status === "ok");
    const metrics = [
      ["整体状态", assessment.status === "completed" ? "已完成" : assessment.status === "partial" ? "部分完成" : "不可用"],
      ["已完成方案", `${completed.length} / ${adapters.length}`],
    ];
    adapters.forEach((item) => {
      const label = item.observation === "positive"
        ? "检出匹配"
        : item.observation === "negative"
          ? "未检出"
          : item.run_status === "not_applicable"
            ? "不适用"
            : item.run_status === "failed"
              ? "运行失败"
              : "不可用";
      const name = item.provider === "openai"
        ? "OpenAI 官方来源"
        : item.adapter_id || item.scheme || "未命名方案";
      metrics.push([name, label]);
      (item.provider_signals || []).forEach((signal) => {
        const signalName = signal.signal_type === "synthid" ? "OpenAI SynthID" : "OpenAI C2PA";
        const outcome = signal.outcome === "detected" ? "已检出" : "未检出";
        const validation = signal.validation_state ? ` · ${signal.validation_state}` : "";
        metrics.push([signalName, `${outcome}${validation}`]);
      });
    });
    if (positives.some((item) => item.evidence_class === "verified_provider_ai")) {
      setCard("watermark", "检出可信 AI 来源信号", "官方验证检出了受支持的来源信号，已计入 AI 分数。", metrics);
      return;
    }
    if (positives.some((item) => item.evidence_class === "known_open_ai_watermark")) {
      setCard("watermark", "检出开放 AI 水印", "像素标记与已知开放水印匹配。", metrics);
      return;
    }
    if (positives.length) {
      setCard("watermark", "发现未验证标识", "检测到水印或标识。", metrics);
      return;
    }
    if (completed.length) {
      const usedOpenAI = adapters.some((item) => item.provider === "openai");
      setCard(
        "watermark",
        usedOpenAI ? "已完成来源信号检查" : "已完成本地水印检查",
        usedOpenAI
          ? "本地方案和已选择的 OpenAI 检查均未检出匹配。"
          : "已启用的本地方案未检出匹配。",
        metrics,
      );
      return;
    }
    setCard("watermark", "未取得水印结果", "已配置或已选择的方案本次没有取得结果，不支持 AI 或实拍判断。", metrics);
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
    const visuals = [
      ["线段与方向组", "同色线段表示同一稳定方向组，请与原图结构对照。", artifacts.lines_overlay],
      ["待复核线段", "用于定位需要复核的位置。", artifacts.anomalous_lines_overlay],
      ["局部结构区域", "不同边框表示分别分析的连续结构区域。", artifacts.geometry_v2_regions_overlay],
      ["局部线族", "颜色区分各结构区域内拟合出的稳定线族。", artifacts.geometry_v2_families_overlay],
      ["几何一致性候选", "红色线段和橙色区域用于定位本次复核候选。", artifacts.geometry_v2_consistency_overlay],
      ["重复结构间距", "标出参与重复间距检查的局部线族。", artifacts.geometry_v2_repeat_spacing_overlay],
    ];
    const available = visuals.filter(([, , path]) => typeof path === "string");
    const nextSignature = JSON.stringify([job.job_id, ...available.map(([, , path]) => path)]);
    if (nextSignature === visualSignature) return;
    visualSignature = nextSignature;
    visualGrid.replaceChildren();
    available.forEach(([title, caption, path]) => {
      addVisual(visualGrid, title, caption, artifactUrl(job.job_id, path));
    });
    document.querySelector("#evidence-visuals").hidden = visualGrid.childElementCount === 0;
  }

  function humanizeLimitation(value) {
    if (limitationLabels[value]) return limitationLabels[value];
    if (typeof value !== "string") return "有一项检测未满足运行条件，未作为判断依据。";
    if (value.startsWith("camera_metadata_unavailable:")) return "无法读取图片中的相机信息。";
    if (value.startsWith("dependency_not_installed:")) return "相机参数检测所需组件未安装，本次未运行。";
    if (value.startsWith("perspective_fields_inference_failed:") || value.startsWith("geocalib_inference_failed:")) {
      return "相机参数检测没有完成，本次不作为判断依据。";
    }
    if (value.startsWith("dda_audit_record_unavailable:") || value.startsWith("safe_audit_record_unavailable:")) {
      return "部分像素检测配置无法读取，本次未使用替代分数。";
    }
    if (value.startsWith("forensic_clip_audit_record_unavailable:") || value.startsWith("community_forensics_audit_record_unavailable:") || value.startsWith("nonescape_mini_audit_record_unavailable:")) {
      return "部分补充像素检测无法运行，本次不作为判断依据。";
    }
    return "有一项检测未满足运行条件，未作为判断依据。";
  }

  function renderLimitations(job, result) {
    const values = new Set([
      ...(job.limitations || []),
      ...(result.limitations || []),
      ...(result.p0?.evidence?.limitations || []),
      ...(result.p3?.limitations || []),
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
    const overall = document.querySelector("#overall-summary");
    const hasOrigin = Boolean(result.origin?.decision);
    const hasCompletedAnalysis = completedAnalysisStates.has(job.status);
    if (overall instanceof HTMLElement) overall.hidden = !(hasOrigin && hasCompletedAnalysis);
    if (hasOrigin && hasCompletedAnalysis) renderOverall(result);
    if (evidenceIsAvailable(result, "p0", job)) renderP0(result);
    if (evidenceIsAvailable(result, "geometry-structure", job)) renderGeometryStructure(result);
    if (evidenceIsAvailable(result, "p3", job)) renderP3(result);
    if (evidenceIsAvailable(result, "camera", job)) renderCamera(result);
    if (evidenceIsAvailable(result, "c2pa", job)) renderC2pa(result);
    if (evidenceIsAvailable(result, "metadata", job)) renderMetadata(result);
    if (evidenceIsAvailable(result, "watermark", job)) renderWatermark(result);
    renderEvidenceOrder(result, job);
    renderVisuals(job, result);
    renderLimitations(job, result);
  }

  async function poll(jobId, revision) {
    if (revision !== selectionRevision || jobId !== selectedJobId) return;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
      if (!response.ok) throw new Error("无法读取本地作业状态。");
      const payload = await response.json();
      if (revision !== selectionRevision || jobId !== selectedJobId) return;
      renderStatus(payload.job);
      if (payload.result) renderResult(payload.job, payload.result);
      if (terminalStates.has(payload.job.status)) {
        selectedJobId = null;
        updateSelectionControls();
        return;
      }
      pollTimer = window.setTimeout(() => poll(jobId, revision), 900);
    } catch (error) {
      if (revision !== selectionRevision || jobId !== selectedJobId) return;
      statusDot.dataset.status = "failed";
      statusLabel.textContent = "无法读取分析状态";
      statusDetail.textContent = error instanceof Error ? error.message : "请重新选择图片。";
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!selected) return;
    const revision = selectionRevision;
    resetPanels();
    isSubmitting = true;
    updateSelectionControls();
    submitButton.textContent = "正在提交…";
    const formData = new FormData();
    formData.append("file", selected, selected.name);
    if (openaiProvenance.checked && !openaiProvenance.disabled) {
      formData.append("openai_provenance", "1");
    }
    try {
      const response = await fetch("/api/jobs", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "提交失败");
      if (revision !== selectionRevision) {
        void cancelSelectedJob(payload.job.job_id).catch(reportCancellationFailure);
        return;
      }
      selectedJobId = payload.job.job_id;
      isSubmitting = false;
      updateSelectionControls();
      renderStatus(payload.job);
      await poll(selectedJobId, revision);
    } catch (error) {
      if (revision !== selectionRevision) return;
      analysisPanel.hidden = false;
      statusDot.dataset.status = "failed";
      statusLabel.textContent = "提交未完成";
      statusDetail.textContent = error instanceof Error ? error.message : "请重新选择图片后再试。";
    } finally {
      if (revision === selectionRevision) {
        isSubmitting = false;
        updateSelectionControls();
        submitButton.textContent = "分析图片";
      }
    }
  }

  input.addEventListener("change", () => {
    if (isSubmitting || selectedJobId !== null) {
      input.value = "";
      return;
    }
    setSelected(input.files?.[0]);
  });
  cancelButton.addEventListener("click", clearSelected);
  openaiProvenance.addEventListener("change", updatePrivacySummary);
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
    if (isSubmitting || selectedJobId !== null) return;
    const file = event.dataTransfer?.files?.[0];
    if (file) setSelected(file);
  });
  document.querySelectorAll(".metric-details").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (details.open) {
        closeMetricDetails(details);
        const metrics = details.querySelector(".metric-grid");
        if (metrics instanceof HTMLElement) window.requestAnimationFrame(() => syncMetricScrollbar(metrics));
      }
    });
  });
  document.querySelectorAll(".metric-details .metric-grid").forEach((metrics) => {
    prepareMetricScrollbar(metrics);
  });
  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element) || !event.target.closest(".metric-details")) closeMetricDetails();
  });
  window.addEventListener("scroll", () => closeMetricDetails(), { passive: true });
  updatePrivacySummary();
  loadCapabilities();
})();
