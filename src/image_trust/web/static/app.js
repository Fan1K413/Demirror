(() => {
  "use strict";

  const terminalStates = new Set(["completed", "partial", "rejected", "failed"]);
  const stateLabels = {
    queued: ["正在排队", "已创建本地作业，等待开始。"],
    validating: ["正在验证文件", "正在检查图片格式、大小与解码边界。"],
    running: ["正在分析", "正在生成 AI 像素信号、相机元数据、几何证据、相机一致性记录和离线来源记录。"],
    completed: ["分析完成", "全部已配置的分析链路已完成。"],
    partial: ["分析部分完成", "基础几何结果已保留；部分附加分析未形成可用观测。"],
    rejected: ["文件未被接受", "图片未通过输入验证，因此没有继续分析。"],
    failed: ["分析未完成", "保留了可用的错误信息；请检查限制说明后重试。"],
  };
  const stageLabels = {
    queued: "等待本地分析线程。",
    validating: "正在验证图片格式、大小与解码边界。",
    starting: "正在准备本地检测环境。",
    geometry: "正在提取线段并检查画面几何结构。",
    provenance: "正在读取元数据与离线来源记录。",
    watermark: "正在检查已配置的本地隐式水印方案。",
    ai_provenance: "已从可验证来源记录取得 AI 声明。",
    ai_dda: "正在运行 DDA 像素检测。",
    ai_safe: "正在运行 SAFE 频域检测。",
    ai_forensic_clip: "正在运行耐压缩像素检测。",
    ai_community_forensics: "正在运行跨生成器像素检测。",
    ai_nonescape_mini: "正在运行 Nonescape Mini 补充检测。",
    camera: "正在检查相机参数与画面一致性。",
    synthesis: "正在汇总各项证据并形成结论。",
    complete: "全部已配置的分析链路已结束。",
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
    dda_checkpoint_not_available: "DDA 本地检查点不可用，未以启发式分数替代。",
    dda_dinov2_source_not_initialized: "DDA 本地运行时尚未初始化，未以启发式分数替代。",
    dda_worker_failed: "DDA 隔离分析进程未完成，未以启发式分数替代。",
    dda_worker_timed_out: "DDA 隔离分析超时，未以启发式分数替代。",
    dda_input_too_small: "图片短边小于 336 像素，DDA 按登记输入门槛未运行。",
    dda_high_confidence_indicator_is_not_provenance_or_authenticity_proof: "高置信像素信号不是来源、真实性、版权或编辑史的证明。",
    dda_score_below_threshold_is_not_evidence_of_camera_origin: "分数未达阈值并不支持“相机拍摄”结论。",
    dda_no_high_confidence_pixel_signal_is_not_camera_evidence: "未形成高置信 AI 像素信号，不等同于相机照片。",
    dda_jpeg_compression_or_resizing_can_reduce_sensitivity: "JPEG 重编码或缩放会降低此检测器的灵敏度，因此这类结果应视为不确定。",
    dda_scope_is_limited_to_the_registered_generator_holdout_and_unmodified_upload_protocol: "阈值只在登记的 Pixart 校准与原始 SDXL 留出集上审计；未外推为通用 AI 判定。",
    no_high_confidence_pixel_signal_is_not_camera_evidence: "三路像素检测均未达到各自的高置信标准，这仍不等同于相机照片。",
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
    nonescape_mini_is_a_strict_complement_not_a_standalone_detector: "Nonescape Mini 仅作为严格的补充通道，不能单独判断来源。",
    "nonescape_mini_source_checkpoint_has_no_publisher_cryptographic_checksum; Demirror pins the retrieved file hash locally": "发布方未提供权重哈希；本地仅运行已登记并校验 SHA-256 的文件。",
    nonescape_mini_scope_is_limited_to_registered_Projective_Geometry_cross_generator_and_jpeg85_protocols: "该通道只在登记的跨生成器与 JPEG-85 协议中审计，不能外推为通用保证。",
    exif_metadata_can_be_copied_or_edited: "相机 EXIF 可以被复制或编辑，因此只能在 AI 检测已完成且未命中时支持有限强度的“可能为实拍”。",
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
    camera_consistency_not_calibrated_for_origin_decision: "拍摄参数一致性尚未按来源类别校准，因此不参与三档判断。",
    c2pa_capture_declaration_not_trusted_for_camera_decision: "C2PA 捕获声明虽可读取，但未通过受信任来源链验证，因此不作为“可能为实拍”的依据。",
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
  const selectedFile = document.querySelector("#selected-file");
  const submitButton = document.querySelector("#submit-button");
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
  let activeJobId = null;
  let pollTimer = null;

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
    selectedFile.textContent = selected ? `${selected.name} · ${(selected.size / 1024 / 1024).toFixed(2)} MB` : "尚未选择文件";
    submitButton.disabled = !selected;
  }

  function resetPanels() {
    clearTimeout(pollTimer);
    activeJobId = null;
    analysisPanel.hidden = true;
    resultPanel.hidden = true;
    progressBar.style.width = "0%";
    progressTrack.setAttribute("aria-valuenow", "0");
    progressMeta.textContent = "0% · 已用 0 秒";
    document.querySelector("#visual-grid").replaceChildren();
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
    metricGrid.replaceChildren();
    metrics.forEach(([label, metricValue]) => addMetric(metricGrid, label, metricValue));
  }

  function renderOverall(result) {
    const origin = result.origin || {};
    if (origin.decision) {
      const metadata = origin.camera_metadata || {};
      const evidence = Array.isArray(origin.supporting_evidence) ? origin.supporting_evidence : [];
      const sharedMetrics = [
        ["判断强度", origin.evidence_strength === "high" ? "高" : "有限"],
        ["主要依据", evidence.join("、") || "未形成可用依据"],
      ];
      if (origin.decision === "possible_ai") {
        setCard("overall", "可能为 AI", `原因：${asText(origin.explanation)}`, [
          ...sharedMetrics,
          ["相机元数据", metadata.status === "coherent" ? "存在，但不能推翻 AI 信号" : "未形成强相机证据"],
        ]);
        return;
      }
      if (origin.decision === "possible_camera") {
        setCard("overall", "可能为实拍", `原因：${asText(origin.explanation)}`, [
          ...sharedMetrics,
          ["AI 像素检测", "未检出高置信 AI 信号"],
        ]);
        return;
      }
      setCard("overall", "未检出 AI 信号", `原因：${asText(origin.explanation)}`, [
        ...sharedMetrics,
        ["相机元数据", metadata.status === "partial" ? "信息不完整" : "未形成正向相机证据"],
      ]);
      return;
    }

    // Compatibility view for local jobs completed before the three-band policy.
    const p3 = result.p3 || {};
    const signals = Array.isArray(p3.signals) ? p3.signals : [];
    const verified = signals.find((signal) => signal.name === "verified_c2pa" && signal.status === "available");
    const dda = signals.find((signal) => signal.name === "dda_pixel_detector");
    const safe = signals.find((signal) => signal.name === "safe_pixel_detector");
    const forensic = signals.find((signal) => signal.name === "forensic_clip_detector");
    const community = signals.find((signal) => signal.name === "community_forensics_detector");
    const nonescape = signals.find((signal) => signal.name === "nonescape_mini_detector");
    const high = p3.status === "available" && p3.risk_band === "high";

    if (verified) {
      setCard("overall", "已确认包含 AI 生成内容", "原因：已验证的图片来源记录明确声明其由 AI 生成。这个结论优先于像素检测；画面几何和拍摄参数不参与这一判断。", [
        ["判断强度", "已确认"],
        ["主要依据", "已验证的来源记录"],
        ["像素检测", dda?.status === "available" ? "已完成" : "未参与"],
      ]);
      return;
    }
    if (high) {
      setCard("overall", "高度疑似 AI 生成", "原因：AI 像素检测命中严格的高置信标准。画面几何、拍摄参数和一般元数据只用于解释或复核，不会被换算成 AI 概率。", [
        ["判断强度", "高"],
        ["主要依据", "AI 像素检测"],
        ["检测分数", percentage(dda?.value ?? safe?.value ?? forensic?.value ?? community?.value ?? nonescape?.value)],
      ]);
      return;
    }
    setCard("overall", "目前无法可靠判断", "原因：没有出现足以支持高置信 AI 判断的直接证据。这不表示图片一定由相机拍摄；几何或拍摄参数也不会被当作反向证明。", [
      ["判断状态", "不确定"],
      ["AI 像素检测", p3.status === "available" ? "未达到高置信标准" : "未形成可用结果"],
      ["来源记录", "未发现已验证的 AI 声明"],
    ]);
  }

  function renderP0(result) {
    const p0 = result.p0 || {};
    const evidence = p0.evidence || {};
    const features = evidence.features || {};
    const value = evidence.run_status === "ok" ? "结构线索已提取" : "未形成结构线索";
    const description = evidence.run_status === "not_applicable"
      ? "当前图片不满足稳定的几何测量条件；不会强行把缺失测量换算成来源结论。"
      : "已检查线条、平行族和消失方向。三种几何判别实验在未见 SDXL 上均未通过门槛，因此当前作为可审阅的一致性证据，不会伪装成可靠 AI 概率。";
    setCard("p0", value, description, [
      ["运行状态", asText(evidence.run_status)],
      ["适用性", percentage(evidence.applicability)],
      ["覆盖度", percentage(evidence.coverage)],
      ["可靠度", percentage(evidence.reliability)],
      ["稳定线族", asText(features.families?.length, "0")],
      ["候选异常线", asText(features.anomalous_lines?.length, "0")],
    ]);
  }

  function renderP3(result) {
    const p3 = result.p3 || {};
    if (!Object.keys(p3).length) {
      setCard("p3", "未运行", "本次没有得到 AI 内容判断。", []);
      return;
    }
    if (p3.status !== "available") {
      setCard("p3", "目前无法可靠判断", "原因：本次检测未能完成，因此不会用不可靠的替代分数下结论。", [
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
      setCard("p3", "已确认包含 AI 生成内容", "原因：图片自带且已验证的来源记录，明确说明它由 AI 生成。这是来源记录，不是对像素的猜测。", [
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
      high ? "检测信号达到高标准" : (limited ? "检测到有限 AI 信号" : "未达到高标准"),
      high
        ? "该检测是总判断的主要依据：图片的像素特征与已知生成图模式高度相符。它支持进一步复核，但不能证明来源或真实性。"
        : (limited
          ? "耐压缩模型达到偏向 AI 检出率的复核阈值，因此总判断为“可能为 AI（有限）”。该档在 JPEG 75 留出集的实拍误报率为 14%，必须结合原图和人工复核。"
          : "该检测没有提供高置信 AI 依据。这不表示它一定是相机照片；压缩、缩放等处理也可能减弱检测信号。"),
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
      setCard("camera", "未运行", "该作业没有相机一致性结果。", []);
      return;
    }
    if (camera.status === "failed") {
      setCard("camera", "未完成", "相机一致性分析失败，不能据此作任何推断。", []);
      return;
    }
    const value = fullImage.status === "ok" ? "拍摄参数已测量" : "未形成可用测量";
    setCard("camera", value, "用于检查全图和局部裁剪之间的参数关系。本地真实/AI 对照未形成稳定区分，因此不独立改变来源结论。", [
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
    setCard("c2pa", value, "只读取图像中已有的嵌入信息；只有已验证且受信任的数字拍摄来源链才可支持“可能为实拍”。离线模式不会检索远程清单。", [
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
      setCard("metadata", "未运行", "该作业没有相机元数据审阅结果。", []);
      return;
    }
    if (metadata.status === "coherent") {
      setCard("metadata", "发现完整相机元数据", "相机品牌、型号、拍摄时间以及至少三项拍摄参数彼此完整。在 AI 像素检测已完成且未命中时，它支持有限强度的“可能为实拍”；EXIF 可被复制或编辑，不能作为真实性证明。", [
        ["相机", `${asText(metadata.camera_make)} · ${asText(metadata.camera_model)}`],
        ["拍摄时间", asText(metadata.captured_at_local)],
        ["物理参数", (metadata.physical_capture_fields || []).join("、") || "—"],
        ["编辑软件", asText(metadata.software)],
      ]);
      return;
    }
    if (metadata.status === "partial") {
      setCard("metadata", "相机元数据不完整", "发现了部分相机信息，但不足以作为“可能为实拍”的正向依据。", [
        ["相机", `${asText(metadata.camera_make)} · ${asText(metadata.camera_model)}`],
        ["拍摄时间", asText(metadata.captured_at_local)],
        ["物理参数", (metadata.physical_capture_fields || []).join("、") || "—"],
      ]);
      return;
    }
    setCard("metadata", "未发现相机元数据", "没有可用的相机 EXIF 信息；这很常见，也不表示图片一定是 AI 生成。", []);
  }

  function renderWatermark(result) {
    const assessment = result.watermark || result.origin?.implicit_watermark;
    if (!assessment || assessment === "not_configured" || assessment.status === "not_configured") {
      setCard("watermark", "尚未接入检测", "当前没有兼容的本地隐式水印检测器，因此它不参与本次 AI 或实拍判断。", [
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
      metrics.push([item.adapter_id || item.scheme || "未命名方案", label]);
    });
    if (positives.some((item) => item.evidence_class === "verified_provider_ai")) {
      setCard("watermark", "检测到可信 AI 水印", "本地适配器形成了可进入综合判断的供应商来源证据。", metrics);
      return;
    }
    if (positives.some((item) => item.evidence_class === "known_open_ai_watermark")) {
      setCard("watermark", "检测到开放 AI 水印", "像素标记与已知开放 AI 生态水印匹配；该标记可被复制，不能验证创建者。", metrics);
      return;
    }
    if (positives.length) {
      setCard("watermark", "发现未验证的隐式标识", "检测到水印或标识，但尚未验证其归属和当前图像绑定，因此不改变综合判断。", metrics);
      return;
    }
    if (completed.length) {
      setCard("watermark", "已完成本地检查", "已启用的方案未检出匹配；其他工具可能不加水印，水印也可能已被移除。", metrics);
      return;
    }
    setCard("watermark", "本地检测不可用", "已配置的水印方案本次未形成观测；这不支持 AI 或实拍来源。", metrics);
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
    renderOverall(result);
    renderP0(result);
    renderP3(result);
    renderCamera(result);
    renderC2pa(result);
    renderMetadata(result);
    renderWatermark(result);
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
