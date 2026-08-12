"use strict";

const $ = (id) => document.getElementById(id);

const SURFACE_LABELS = {
  facade: "立面",
  roof: "屋面",
  road: "道路",
  floor: "地面／地板",
  ceiling: "天花",
  window_or_door_array: "门窗阵列",
  fence_or_railing: "围栏／栏杆",
  object_surface: "物体表面",
  other: "其它",
  uncertain: "不确定",
};

const VERDICT_LABELS = {
  pending: "待复核",
  coherent_within_surface: "同一表面内自洽",
  split_across_surfaces: "跨表面误并",
  geometry_inconsistent_within_surface: "同一表面内存在冲突",
  unassessable: "无法判断",
};

const STATUS_LABELS = {
  pending: "待审",
  completed: "已完成",
  unassessable: "无法判断",
};

const ASSET_LABELS = {
  image: "匿名原图",
  line_ids_overlay: "线号叠图",
  regions_overlay: "区域候选",
  local_families_overlay: "局部线族",
  global_families_overlay: "全局线族",
  consistency_overlay: "一致性复核图",
  repeat_spacing_overlay: "重复间距复核图",
};

const SURFACE_COLORS = [
  "#9b4056",
  "#2f6d94",
  "#3e7d60",
  "#9a6b22",
  "#6e58a1",
  "#a34f2d",
  "#44777b",
  "#765342",
];

const state = {
  overview: null,
  currentIndex: -1,
  packet: null,
  annotation: null,
  dirty: false,
  selectedSurfaceId: null,
  selectedFamilyId: null,
  drawing: false,
  lineMode: false,
  draftPoints: [],
  activeAsset: null,
  saving: false,
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    // The response status remains the useful part for a local service failure.
  }
  if (!response.ok) {
    const code = payload && payload.error ? payload.error : `HTTP ${response.status}`;
    throw new Error(code);
  }
  return payload;
}

function taskAssetUrl(relativePath) {
  const reviewerId = state.packet.reviewer_id;
  const encodedPath = relativePath.split("/").map(encodeURIComponent).join("/");
  return `api/tasks/${encodeURIComponent(reviewerId)}/assets/${encodedPath}`;
}

function currentTaskSummary() {
  return state.overview && state.currentIndex >= 0
    ? state.overview.tasks[state.currentIndex]
    : null;
}

function selectedSurface() {
  return state.annotation
    ? state.annotation.surfaces.find((surface) => surface.surface_id === state.selectedSurfaceId) || null
    : null;
}

function selectedFamily() {
  return state.packet
    ? state.packet.family_proposals.find((family) => family.family_id === state.selectedFamilyId) || null
    : null;
}

function familyReview(familyId) {
  return state.annotation.proposed_family_reviews.find(
    (review) => review.proposed_family_id === familyId,
  );
}

function setDirty(value = true) {
  state.dirty = value;
  updateTaskHeading();
}

function showMessage(message, kind = "") {
  const target = $("save-message");
  target.textContent = message;
  target.className = `save-message${kind ? ` ${kind}` : ""}`;
}

function statusClass(status) {
  return `status-pill status-${status}`;
}

async function boot() {
  bindEvents();
  try {
    state.overview = await requestJson("api/state");
    renderTaskOptions();
    const firstPending = state.overview.tasks.findIndex((task) => task.status === "pending");
    await loadTask(firstPending >= 0 ? firstPending : 0, true);
    $("app").setAttribute("aria-busy", "false");
  } catch (error) {
    showFatal(error);
  }
}

function showFatal(error) {
  $("app").hidden = true;
  $("fatal-error").hidden = false;
  $("fatal-error-copy").textContent = `请确认盲审服务仍在运行，然后刷新页面。错误：${error.message}`;
}

function renderTaskOptions() {
  const select = $("task-select");
  select.replaceChildren();
  state.overview.tasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = task.reviewer_id;
    option.textContent = `${String(task.position).padStart(2, "0")} · ${STATUS_LABELS[task.status]}`;
    select.append(option);
  });
  renderOverallProgress();
}

function renderOverallProgress() {
  const completed = state.overview.counts.completed + state.overview.counts.unassessable;
  const total = state.overview.task_count;
  $("progress-copy").textContent = `已处理 ${completed} / ${total}`;
  $("progress-bar").style.width = `${total ? (completed / total) * 100 : 0}%`;
}

async function loadTask(index, force = false) {
  if (!force && state.dirty && !window.confirm("当前修改尚未保存，仍要切换任务吗？")) {
    $("task-select").value = currentTaskSummary().reviewer_id;
    return;
  }
  if (!state.overview.tasks.length) {
    throw new Error("盲审清单中没有任务");
  }
  const boundedIndex = Math.max(0, Math.min(index, state.overview.tasks.length - 1));
  const summary = state.overview.tasks[boundedIndex];
  showMessage("正在载入任务…");
  const payload = await requestJson(`api/tasks/${encodeURIComponent(summary.reviewer_id)}`);
  state.currentIndex = boundedIndex;
  state.packet = payload.packet;
  state.annotation = payload.annotation;
  state.dirty = false;
  state.selectedSurfaceId = state.annotation.surfaces[0]?.surface_id || null;
  state.selectedFamilyId = state.packet.family_proposals[0]?.family_id || null;
  state.drawing = false;
  state.lineMode = false;
  state.draftPoints = [];
  state.activeAsset = state.packet.assets.image;
  renderTask();
  showMessage("");
}

function renderTask() {
  renderTaskOptions();
  $("task-select").value = state.packet.reviewer_id;
  $("previous-task").disabled = state.currentIndex === 0;
  $("next-task").disabled = state.currentIndex === state.overview.tasks.length - 1;
  $("instructions").replaceChildren(
    ...state.packet.instructions.map((instruction) => {
      const item = document.createElement("li");
      item.textContent = instruction;
      return item;
    }),
  );
  $("assessability-reason").value = state.annotation.assessability_reason;
  $("review-note").value = state.annotation.review_note;
  buildAssetOptions();
  renderAllEditors();
  setImageAsset(state.packet.assets.image);
}

function updateTaskHeading() {
  if (!state.packet) return;
  const summary = currentTaskSummary();
  $("task-heading").textContent = `匿名任务 ${String(summary.position).padStart(2, "0")}${state.dirty ? " · 未保存" : ""}`;
  $("task-status").textContent = STATUS_LABELS[state.annotation.status];
  $("task-status").className = statusClass(state.annotation.status);
}

function buildAssetOptions() {
  const select = $("asset-select");
  select.replaceChildren();
  Object.entries(ASSET_LABELS).forEach(([key, label]) => {
    const path = state.packet.assets[key];
    if (!path) return;
    const option = document.createElement("option");
    option.value = path;
    option.textContent = label;
    select.append(option);
  });
  if (state.packet.family_proposals.length) {
    const group = document.createElement("optgroup");
    group.label = "单族细图";
    state.packet.family_proposals.forEach((family) => {
      const option = document.createElement("option");
      option.value = family.detail_overlay;
      option.textContent = family.family_id;
      group.append(option);
    });
    select.append(group);
  }
  select.value = state.activeAsset;
}

function setImageAsset(path) {
  state.activeAsset = path;
  $("asset-select").value = path;
  $("image-loading").textContent = "正在载入图像…";
  $("image-loading").hidden = false;
  const image = $("background-image");
  image.alt = path === state.packet.assets.image ? "匿名待审图片" : "匿名几何复核叠图";
  image.src = taskAssetUrl(path);
}

function renderAllEditors() {
  updateTaskHeading();
  renderMetrics();
  renderSurfaceControls();
  renderSurfaces();
  renderFamilies();
  redrawCanvas();
}

function renderMetrics() {
  const reviewed = state.annotation.proposed_family_reviews.filter(
    (review) => review.verdict !== "pending",
  ).length;
  $("surface-count").textContent = String(state.annotation.surfaces.length);
  $("family-progress").textContent = `${reviewed} / ${state.packet.family_proposals.length}`;
  $("family-count").textContent = `${state.packet.family_proposals.length} 项`;
}

function renderSurfaceControls() {
  const surface = selectedSurface();
  $("surface-kind").disabled = !surface;
  $("surface-visibility").disabled = !surface;
  $("surface-note").disabled = !surface;
  $("delete-surface").disabled = !surface;
  $("line-mode").disabled = !surface || state.drawing;
  $("new-surface").disabled = state.drawing;
  $("close-surface").disabled = !state.drawing || state.draftPoints.length < 3;
  $("cancel-drawing").disabled = !state.drawing;
  if (surface) {
    $("surface-kind").value = surface.surface_kind;
    $("surface-visibility").value = surface.visibility;
    $("surface-note").value = surface.note;
  } else {
    $("surface-note").value = "";
  }
  $("line-mode").textContent = state.lineMode ? "结束点选" : "点选线段";
  $("interaction-mode").textContent = state.drawing
    ? `正在绘制表面 · 已放置 ${state.draftPoints.length} 个顶点`
    : state.lineMode
      ? `正在为 ${state.selectedSurfaceId} 点选线段`
      : surface
        ? `已选择 ${surface.surface_id}`
        : "选择一个表面后可分配线段";
  $("canvas-help").textContent = state.drawing
    ? "依次点击表面边界，至少三个顶点；按 Enter 闭合，Esc 取消。"
    : state.lineMode
      ? "点击最接近的线段切换它与当前表面的归属；同一边界线可以属于多个表面。"
      : "先画出同一屋面、立面、道路或物体表面；之后再逐族复核。";
}

function renderSurfaces() {
  const list = $("surface-list");
  list.replaceChildren();
  if (!state.annotation.surfaces.length) {
    list.className = "surface-list empty-copy";
    list.textContent = "尚未标出表面";
  } else {
    list.className = "surface-list";
    state.annotation.surfaces.forEach((surface, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `surface-item${surface.surface_id === state.selectedSurfaceId ? " selected" : ""}`;
      button.dataset.surfaceId = surface.surface_id;
      const color = document.createElement("span");
      color.className = "surface-color";
      color.style.backgroundColor = SURFACE_COLORS[index % SURFACE_COLORS.length];
      const title = document.createElement("span");
      title.textContent = `${surface.surface_id} · ${SURFACE_LABELS[surface.surface_kind]}`;
      const count = document.createElement("small");
      count.textContent = `${surface.line_ids.length} 条线`;
      button.append(color, title, count);
      list.append(button);
    });
  }

  const selectedLines = $("selected-lines");
  selectedLines.replaceChildren();
  const surface = selectedSurface();
  if (surface && surface.line_ids.length) {
    surface.line_ids.forEach((lineId) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "line-chip";
      chip.dataset.removeLineId = lineId;
      chip.title = "从当前表面移除";
      chip.textContent = `${lineId} ×`;
      selectedLines.append(chip);
    });
  }
}

function surfaceIndex(surfaceId) {
  return state.annotation.surfaces.findIndex((surface) => surface.surface_id === surfaceId);
}

function redrawCanvas() {
  const canvas = $("annotation-canvas");
  if (!state.packet || !canvas.width || !canvas.height) return;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);

  state.annotation.surfaces.forEach((surface, index) => {
    drawPolygon(
      context,
      surface.polygon_normalized,
      SURFACE_COLORS[index % SURFACE_COLORS.length],
      surface.surface_id === state.selectedSurfaceId,
    );
  });

  if ($("show-lines").checked) {
    state.packet.lines.forEach((line) => drawLine(context, line, "rgba(44, 88, 120, 0.42)", 1));
    const family = selectedFamily();
    if (family) {
      const ids = new Set(family.member_line_ids);
      state.packet.lines
        .filter((line) => ids.has(line.line_id))
        .forEach((line) => drawLine(context, line, "#e0802d", 2.2));
    }
    const surface = selectedSurface();
    if (surface) {
      const ids = new Set(surface.line_ids);
      state.packet.lines
        .filter((line) => ids.has(line.line_id))
        .forEach((line) => drawLine(context, line, "#2b9a68", 3));
    }
  }

  if (state.draftPoints.length) {
    drawDraft(context, state.draftPoints);
  }
}

function fitImageToStage() {
  if (!state.packet) return;
  const stage = $("image-stage");
  const image = $("background-image");
  const canvas = $("annotation-canvas");
  const [width, height] = state.packet.canonical_size;
  const availableWidth = Math.max(160, stage.clientWidth - 24);
  const availableHeight = Math.max(160, Math.min(stage.clientHeight - 24, window.innerHeight - 290));
  const scale = Math.max(0.25, Math.min(availableWidth / width, availableHeight / height));
  const cssWidth = Math.round(width * scale);
  const cssHeight = Math.round(height * scale);
  [image, canvas].forEach((element) => {
    element.style.width = `${cssWidth}px`;
    element.style.height = `${cssHeight}px`;
  });
}

function drawPolygon(context, points, color, selected) {
  if (!points.length) return;
  context.save();
  context.beginPath();
  points.forEach((point, index) => {
    const x = point.x * context.canvas.width;
    const y = point.y * context.canvas.height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.closePath();
  context.fillStyle = `${color}22`;
  context.fill();
  context.strokeStyle = color;
  context.lineWidth = selected ? 3 : 1.5;
  context.stroke();
  context.restore();
}

function drawLine(context, line, color, width) {
  context.save();
  context.beginPath();
  context.moveTo(line.start.x * context.canvas.width, line.start.y * context.canvas.height);
  context.lineTo(line.end.x * context.canvas.width, line.end.y * context.canvas.height);
  context.strokeStyle = color;
  context.lineWidth = width;
  context.stroke();
  context.restore();
}

function drawDraft(context, points) {
  context.save();
  context.beginPath();
  points.forEach((point, index) => {
    const x = point.x * context.canvas.width;
    const y = point.y * context.canvas.height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = "#9b4056";
  context.lineWidth = 2;
  context.setLineDash([5, 4]);
  context.stroke();
  context.setLineDash([]);
  points.forEach((point) => {
    context.beginPath();
    context.arc(point.x * context.canvas.width, point.y * context.canvas.height, 3.5, 0, Math.PI * 2);
    context.fillStyle = "#9b4056";
    context.fill();
  });
  context.restore();
}

function renderFamilies() {
  const list = $("family-list");
  list.replaceChildren();
  state.packet.family_proposals.forEach((family) => {
    const review = familyReview(family.family_id);
    const card = document.createElement("article");
    card.className = `family-card${family.family_id === state.selectedFamilyId ? " selected" : ""}`;
    card.dataset.familyId = family.family_id;

    const header = document.createElement("div");
    header.className = "family-card-header";
    const title = document.createElement("div");
    title.className = "family-title";
    const identifier = document.createElement("strong");
    identifier.textContent = family.family_id;
    const scope = document.createElement("span");
    scope.className = "scope-tag";
    scope.textContent = family.region_id === "global" ? "全图" : "局部";
    title.append(identifier, scope);
    if (family.priority_reason === "check_finding") {
      const priority = document.createElement("span");
      priority.className = "priority-tag";
      priority.textContent = "优先复核";
      title.append(priority);
    }
    const inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "text-button";
    inspect.dataset.inspectFamily = family.family_id;
    inspect.textContent = family.family_id === state.selectedFamilyId ? "正在查看" : "查看此族";
    header.append(title, inspect);

    const body = document.createElement("div");
    body.className = "family-card-body";
    const summary = document.createElement("div");
    summary.className = "family-summary";
    summary.textContent = `${family.member_line_ids.length} 条成员线 · ${family.kind === "parallel" ? "平行候选" : "消失方向候选"}`;

    const verdict = document.createElement("select");
    verdict.dataset.verdictFamily = family.family_id;
    verdict.setAttribute("aria-label", `${family.family_id} 复核结论`);
    Object.entries(VERDICT_LABELS).forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      verdict.append(option);
    });
    verdict.value = review.verdict;

    body.append(summary, verdict);
    const surfacePicker = buildSurfacePicker(family, review);
    if (surfacePicker) body.append(surfacePicker);

    const memberDetails = document.createElement("details");
    const memberSummary = document.createElement("summary");
    memberSummary.textContent = `成员线（${family.member_line_ids.length}）`;
    const memberLines = document.createElement("div");
    memberLines.className = "family-lines";
    const assigned = new Set(
      state.annotation.surfaces.flatMap((surface) =>
        surface.line_ids.filter((lineId) => family.member_line_ids.includes(lineId)),
      ),
    );
    family.member_line_ids.forEach((lineId) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `member-line-button${assigned.has(lineId) ? " assigned" : ""}`;
      button.dataset.familyLineId = lineId;
      button.dataset.familyId = family.family_id;
      button.title = selectedSurface() ? `切换与 ${state.selectedSurfaceId} 的归属` : "请先选择一个表面";
      button.textContent = lineId;
      memberLines.append(button);
    });
    memberDetails.append(memberSummary, memberLines);
    body.append(memberDetails);

    if (review.verdict === "geometry_inconsistent_within_surface") {
      body.append(buildOutlierPicker(family, review));
    }

    const note = document.createElement("textarea");
    note.rows = 2;
    note.maxLength = 500;
    note.placeholder = "本族备注（可选）";
    note.value = review.note;
    note.dataset.familyNote = family.family_id;
    body.append(note);
    card.append(header, body);
    list.append(card);
  });
}

function buildSurfacePicker(family, review) {
  if (!["coherent_within_surface", "split_across_surfaces", "geometry_inconsistent_within_surface"].includes(review.verdict)) {
    return null;
  }
  const wrapper = document.createElement("div");
  wrapper.className = "surface-options";
  const inputType = review.verdict === "split_across_surfaces" ? "checkbox" : "radio";
  state.annotation.surfaces.forEach((surface) => {
    const label = document.createElement("label");
    label.className = "surface-option";
    const input = document.createElement("input");
    input.type = inputType;
    input.name = inputType === "radio" ? `surface-${family.family_id}` : "";
    input.value = surface.surface_id;
    input.checked = review.surface_ids.includes(surface.surface_id);
    input.dataset.reviewSurface = family.family_id;
    const copy = document.createElement("span");
    copy.textContent = `${surface.surface_id} · ${SURFACE_LABELS[surface.surface_kind]}`;
    label.append(input, copy);
    wrapper.append(label);
  });
  if (!state.annotation.surfaces.length) {
    wrapper.classList.add("family-summary");
    wrapper.textContent = "请先画出表面，再选择此结论。";
  }
  return wrapper;
}

function buildOutlierPicker(family, review) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `异常成员线（${review.outlier_line_ids.length}）`;
  const options = document.createElement("div");
  options.className = "family-lines";
  family.member_line_ids.forEach((lineId) => {
    const label = document.createElement("label");
    label.className = "surface-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = review.outlier_line_ids.includes(lineId);
    input.value = lineId;
    input.dataset.outlierFamily = family.family_id;
    const copy = document.createElement("span");
    copy.textContent = lineId;
    label.append(input, copy);
    options.append(label);
  });
  details.append(summary, options);
  return details;
}

function beginSurface() {
  state.drawing = true;
  state.lineMode = false;
  state.draftPoints = [];
  renderSurfaceControls();
  redrawCanvas();
  $("annotation-canvas").focus();
}

function cancelSurface() {
  state.drawing = false;
  state.draftPoints = [];
  renderSurfaceControls();
  redrawCanvas();
}

function nextSurfaceId() {
  const used = new Set(state.annotation.surfaces.map((surface) => surface.surface_id));
  let index = 1;
  while (used.has(`surface-${String(index).padStart(3, "0")}`)) index += 1;
  return `surface-${String(index).padStart(3, "0")}`;
}

function closeSurface() {
  if (!state.drawing || state.draftPoints.length < 3) {
    showMessage("表面至少需要三个顶点。", "error");
    return;
  }
  const surface = {
    surface_id: nextSurfaceId(),
    surface_kind: "facade",
    polygon_normalized: state.draftPoints.map((point) => ({ ...point })),
    line_ids: [],
    visibility: "clear",
    note: "",
  };
  state.annotation.surfaces.push(surface);
  state.selectedSurfaceId = surface.surface_id;
  state.drawing = false;
  state.draftPoints = [];
  setDirty();
  renderAllEditors();
  showMessage(`已建立 ${surface.surface_id}，现在可点选属于它的线段。`);
}

function deleteSelectedSurface() {
  const surface = selectedSurface();
  if (!surface || !window.confirm(`删除 ${surface.surface_id} 及其线段归属吗？`)) return;
  state.annotation.surfaces = state.annotation.surfaces.filter(
    (candidate) => candidate.surface_id !== surface.surface_id,
  );
  state.annotation.proposed_family_reviews.forEach((review) => {
    if (review.surface_ids.includes(surface.surface_id)) {
      review.verdict = "pending";
      review.surface_ids = [];
      review.outlier_line_ids = [];
    }
  });
  state.selectedSurfaceId = state.annotation.surfaces[0]?.surface_id || null;
  state.lineMode = false;
  setDirty();
  renderAllEditors();
  showMessage("已删除表面；受影响的线族结论已恢复为待复核。", "success");
}

function toggleLineMembership(lineId) {
  const surface = selectedSurface();
  if (!surface) {
    showMessage("请先选择一个表面。", "error");
    return;
  }
  const index = surface.line_ids.indexOf(lineId);
  if (index >= 0) surface.line_ids.splice(index, 1);
  else surface.line_ids.push(lineId);
  surface.line_ids.sort();
  setDirty();
  renderSurfaces();
  renderFamilies();
  redrawCanvas();
}

function canvasPoint(event) {
  const canvas = $("annotation-canvas");
  const bounds = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
    y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
  };
}

function handleCanvasClick(event) {
  if (!state.packet) return;
  const point = canvasPoint(event);
  if (state.drawing) {
    state.draftPoints.push(point);
    renderSurfaceControls();
    redrawCanvas();
    return;
  }
  if (state.lineMode && selectedSurface()) {
    const nearest = nearestLine(point);
    if (!nearest || nearest.distance > 14) {
      showMessage("附近没有可点选的线段，请放大或使用成员线按钮。", "error");
      return;
    }
    toggleLineMembership(nearest.line.line_id);
    return;
  }
  for (let index = state.annotation.surfaces.length - 1; index >= 0; index -= 1) {
    const surface = state.annotation.surfaces[index];
    if (pointInsidePolygon(point, surface.polygon_normalized)) {
      state.selectedSurfaceId = surface.surface_id;
      renderAllEditors();
      return;
    }
  }
}

function nearestLine(point) {
  let best = null;
  const bounds = $("annotation-canvas").getBoundingClientRect();
  state.packet.lines.forEach((line) => {
    const distance = pointSegmentDistance(point, line.start, line.end, bounds.width, bounds.height);
    if (!best || distance < best.distance) best = { line, distance };
  });
  return best;
}

function pointSegmentDistance(point, start, end, width, height) {
  const px = point.x * width;
  const py = point.y * height;
  const sx = start.x * width;
  const sy = start.y * height;
  const ex = end.x * width;
  const ey = end.y * height;
  const dx = ex - sx;
  const dy = ey - sy;
  const lengthSquared = dx * dx + dy * dy;
  if (!lengthSquared) return Math.hypot(px - sx, py - sy);
  const projection = Math.max(
    0,
    Math.min(1, ((px - sx) * dx + (py - sy) * dy) / lengthSquared),
  );
  return Math.hypot(
    px - (sx + projection * dx),
    py - (sy + projection * dy),
  );
}

function pointInsidePolygon(point, polygon) {
  let inside = false;
  for (let first = 0, second = polygon.length - 1; first < polygon.length; second = first++) {
    const a = polygon[first];
    const b = polygon[second];
    const crosses = (a.y > point.y) !== (b.y > point.y)
      && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

function changeVerdict(familyId, verdict) {
  const review = familyReview(familyId);
  if (["coherent_within_surface", "geometry_inconsistent_within_surface"].includes(verdict)) {
    const candidate = selectedSurface() || state.annotation.surfaces[0];
    if (!candidate) {
      showMessage("请先画出至少一个表面，再选择这个结论。", "error");
      renderFamilies();
      return;
    }
    review.surface_ids = [candidate.surface_id];
  } else if (verdict === "split_across_surfaces") {
    if (state.annotation.surfaces.length < 2) {
      showMessage("“跨表面误并”需要至少两个已标表面。", "error");
      renderFamilies();
      return;
    }
    review.surface_ids = state.annotation.surfaces.slice(0, 2).map((surface) => surface.surface_id);
  } else {
    review.surface_ids = [];
  }
  review.verdict = verdict;
  if (verdict !== "geometry_inconsistent_within_surface") review.outlier_line_ids = [];
  setDirty();
  renderMetrics();
  renderFamilies();
  showMessage("");
}

function changeReviewSurface(familyId, surfaceId, checked) {
  const review = familyReview(familyId);
  if (review.verdict === "split_across_surfaces") {
    const next = new Set(review.surface_ids);
    if (checked) next.add(surfaceId);
    else next.delete(surfaceId);
    if (next.size < 2) {
      showMessage("跨表面结论至少保留两个表面。", "error");
      renderFamilies();
      return;
    }
    review.surface_ids = [...next];
  } else {
    review.surface_ids = [surfaceId];
  }
  setDirty();
  renderFamilies();
}

function toggleOutlier(familyId, lineId, checked) {
  const review = familyReview(familyId);
  const next = new Set(review.outlier_line_ids);
  if (checked) next.add(lineId);
  else next.delete(lineId);
  review.outlier_line_ids = [...next].sort();
  setDirty();
  renderFamilies();
}

function selectFamily(familyId) {
  state.selectedFamilyId = familyId;
  const family = selectedFamily();
  renderFamilies();
  redrawCanvas();
  if (family) setImageAsset(family.detail_overlay);
}

function selectSurface(surfaceId) {
  state.selectedSurfaceId = surfaceId;
  renderAllEditors();
}

function validateClient(status) {
  const annotation = state.annotation;
  if (status === "unassessable" && !annotation.assessability_reason.trim()) {
    return "请填写整张图片无法判断的原因。";
  }
  if (status === "completed") {
    if (!annotation.surfaces.length) return "完成前至少需要标出一个可见表面。";
    if (annotation.proposed_family_reviews.some((review) => review.verdict === "pending")) {
      return "还有线族处于待复核状态。";
    }
  }
  const surfaces = new Map(annotation.surfaces.map((surface) => [surface.surface_id, surface]));
  const families = new Map(state.packet.family_proposals.map((family) => [family.family_id, family]));
  for (const review of annotation.proposed_family_reviews) {
    if (["pending", "unassessable"].includes(review.verdict)) continue;
    const memberIds = new Set(families.get(review.proposed_family_id).member_line_ids);
    const sharesMember = review.surface_ids.some((surfaceId) =>
      (surfaces.get(surfaceId)?.line_ids || []).some((lineId) => memberIds.has(lineId)),
    );
    if (!sharesMember) {
      return `${review.proposed_family_id} 的所选表面尚未包含该族成员线。请先用画布或成员线按钮分配至少一条。`;
    }
  }
  return null;
}

async function saveAnnotation(status) {
  if (state.saving) return;
  const previousStatus = state.annotation.status;
  state.annotation.status = status;
  const validationError = validateClient(status);
  if (validationError) {
    state.annotation.status = previousStatus;
    showMessage(validationError, "error");
    updateTaskHeading();
    return;
  }
  state.saving = true;
  ["save-draft", "mark-unassessable", "complete-review"].forEach((id) => {
    $(id).disabled = true;
  });
  showMessage("正在保存…");
  try {
    const reviewerId = state.packet.reviewer_id;
    const payload = await requestJson(
      `api/tasks/${encodeURIComponent(reviewerId)}/annotation`,
      { method: "PUT", body: JSON.stringify(state.annotation) },
    );
    state.annotation = payload.annotation;
    state.dirty = false;
    state.overview = await requestJson("api/state");
    const nextIndex = state.overview.tasks.findIndex((task) => task.reviewer_id === reviewerId);
    state.currentIndex = nextIndex;
    renderTaskOptions();
    renderAllEditors();
    showMessage(
      status === "pending" ? "草稿已保存。" : status === "completed" ? "本项已完成。" : "已标记为无法判断。",
      "success",
    );
  } catch (error) {
    state.annotation.status = previousStatus;
    updateTaskHeading();
    showMessage(`保存失败：${error.message}`, "error");
  } finally {
    state.saving = false;
    ["save-draft", "mark-unassessable", "complete-review"].forEach((id) => {
      $(id).disabled = false;
    });
  }
}

function syncTextFields() {
  state.annotation.assessability_reason = $("assessability-reason").value;
  state.annotation.review_note = $("review-note").value;
  setDirty();
}

function bindEvents() {
  const requestTask = (index) => {
    loadTask(index).catch((error) => showMessage(`任务载入失败：${error.message}`, "error"));
  };
  $("previous-task").addEventListener("click", () => requestTask(state.currentIndex - 1));
  $("next-task").addEventListener("click", () => requestTask(state.currentIndex + 1));
  $("task-select").addEventListener("change", (event) => {
    const index = state.overview.tasks.findIndex((task) => task.reviewer_id === event.target.value);
    requestTask(index);
  });
  $("next-pending").addEventListener("click", () => {
    const tasks = state.overview.tasks;
    let target = -1;
    for (let offset = 1; offset <= tasks.length; offset += 1) {
      const index = (state.currentIndex + offset) % tasks.length;
      if (tasks[index].status === "pending") {
        target = index;
        break;
      }
    }
    if (target >= 0) requestTask(target);
    else showMessage("没有待审任务。", "success");
  });
  $("asset-select").addEventListener("change", (event) => setImageAsset(event.target.value));
  $("show-lines").addEventListener("change", redrawCanvas);
  $("background-image").addEventListener("load", () => {
    const [width, height] = state.packet.canonical_size;
    const canvas = $("annotation-canvas");
    canvas.width = width;
    canvas.height = height;
    fitImageToStage();
    $("image-loading").hidden = true;
    redrawCanvas();
  });
  $("background-image").addEventListener("error", () => {
    $("image-loading").hidden = false;
    $("image-loading").textContent = "图像载入失败";
  });
  $("new-surface").addEventListener("click", beginSurface);
  $("close-surface").addEventListener("click", closeSurface);
  $("cancel-drawing").addEventListener("click", cancelSurface);
  $("delete-surface").addEventListener("click", deleteSelectedSurface);
  $("line-mode").addEventListener("click", () => {
    if (!selectedSurface()) return;
    state.lineMode = !state.lineMode;
    renderSurfaceControls();
  });
  $("annotation-canvas").addEventListener("click", handleCanvasClick);
  $("annotation-canvas").addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.drawing) cancelSurface();
    if (event.key === "Enter" && state.drawing) closeSurface();
  });
  $("surface-kind").addEventListener("change", (event) => {
    const surface = selectedSurface();
    if (!surface) return;
    surface.surface_kind = event.target.value;
    setDirty();
    renderSurfaces();
  });
  $("surface-visibility").addEventListener("change", (event) => {
    const surface = selectedSurface();
    if (!surface) return;
    surface.visibility = event.target.value;
    setDirty();
  });
  $("surface-note").addEventListener("input", (event) => {
    const surface = selectedSurface();
    if (!surface) return;
    surface.note = event.target.value;
    setDirty();
  });
  $("surface-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-surface-id]");
    if (button) selectSurface(button.dataset.surfaceId);
  });
  $("selected-lines").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-line-id]");
    if (button) toggleLineMembership(button.dataset.removeLineId);
  });
  $("family-list").addEventListener("click", (event) => {
    const inspect = event.target.closest("[data-inspect-family]");
    if (inspect) {
      selectFamily(inspect.dataset.inspectFamily);
      return;
    }
    const member = event.target.closest("[data-family-line-id]");
    if (member) {
      state.selectedFamilyId = member.dataset.familyId;
      toggleLineMembership(member.dataset.familyLineId);
    }
  });
  $("family-list").addEventListener("change", (event) => {
    if (event.target.dataset.verdictFamily) {
      changeVerdict(event.target.dataset.verdictFamily, event.target.value);
    } else if (event.target.dataset.reviewSurface) {
      changeReviewSurface(event.target.dataset.reviewSurface, event.target.value, event.target.checked);
    } else if (event.target.dataset.outlierFamily) {
      toggleOutlier(event.target.dataset.outlierFamily, event.target.value, event.target.checked);
    }
  });
  $("family-list").addEventListener("input", (event) => {
    if (!event.target.dataset.familyNote) return;
    familyReview(event.target.dataset.familyNote).note = event.target.value;
    setDirty();
  });
  $("assessability-reason").addEventListener("input", syncTextFields);
  $("review-note").addEventListener("input", syncTextFields);
  $("save-draft").addEventListener("click", () => saveAnnotation("pending"));
  $("mark-unassessable").addEventListener("click", () => saveAnnotation("unassessable"));
  $("complete-review").addEventListener("click", () => saveAnnotation("completed"));
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotation("pending");
    }
  });
  window.addEventListener("resize", fitImageToStage);
}

boot();
