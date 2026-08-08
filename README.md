# Demirror P0

Demirror 的第一阶段是投影几何测量基线，而不是通用 AI 图像分类器。它对 PNG、JPEG 和静态 WebP 执行安全门控、坐标规范化、线段检测、消失方向拟合和异常候选可视化。

P0 的输出不能用于断言“图片由 AI 生成”。它只报告几何测量的适用性、覆盖范围、线族稳定性与待复核异常候选。真实性可信度 T 与证据充分度 Q 在独立校准完成前不输出。

## 当前能力

- 文件魔数、解码、像素上限和 SHA-256 门控；
- EXIF 方向规范化与 canonical / analysis 坐标映射；
- 默认 OpenCV LSD 线段后端；
- 明确不可用的 DeepLSD 适配器占位；
- 确定性的多消失方向 RANSAC 基线；
- 线段覆盖图，以及仅显示线族和候选的异常复核图；
- 版本化 evidence-result-v1 JSON。

## 本地安装

建议在项目根目录创建隔离环境：

    python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.lock
    .venv\Scripts\python -m pip install -e ".[dev]"

## 运行

    .venv\Scripts\python -m image_trust.cli analyze path\to\image.jpg --config configs\p0.yaml --output outputs\sample

成功输出：

    outputs\sample\result.json
    outputs\sample\lines.json
    outputs\sample\lines_overlay.png
    outputs\sample\anomalous_lines_overlay.png

DeepLSD 不是 P0 默认依赖。若在配置中明确指定 deeplsd 但本地没有该依赖和合法权重，命令会失败并给出安装提示；它不会悄悄改用 OpenCV。

## 测试

    .venv\Scripts\python -m pytest

## P0 验证流程

P0 的“稳定”只表示几何测量链和拒判边界经过验证，绝不表示具备通用 AI
图像判别能力。F1–F5 是完全离线、可再生成的测试夹具：

    .venv\Scripts\python scripts\generate_p0_fixtures.py --output data\p0_fixtures
    .venv\Scripts\python scripts\evaluate_p0.py data\p0_fixtures --config configs\p0.yaml --output outputs\p0_f1_f5 --manifest data\p0_fixtures\manifest.json

F6 用于人工审核真实强透视照片的叠图。下载只在显式运行脚本时发生；图片、
来源清单、审核日志和输出都保留在已忽略的 `data/`、`outputs/` 中：

    .venv\Scripts\python scripts\download_f6_validation_set.py --output data\p0_f6_real_v2
    .venv\Scripts\python scripts\evaluate_p0.py data\p0_f6_real_v2\images --config configs\p0.yaml --output outputs\p0_f6_v2
    .venv\Scripts\python scripts\make_overlay_contact_sheet.py outputs\p0_f6_v2\artifacts --artifact lines_overlay.png --output outputs\p0_f6_v2\lines_contact_sheet.png
    .venv\Scripts\python scripts\make_overlay_contact_sheet.py outputs\p0_f6_v2\artifacts --output outputs\p0_f6_v2\anomalous_contact_sheet.png
    .venv\Scripts\python scripts\audit_p0_validation.py --f1-f5-summary outputs\p0_f1_f5\summary.json --f6-source-manifest data\p0_f6_real_v2\source_manifest.json --f6-evaluation outputs\p0_f6_v2 --output outputs\p0_completion_audit.json --require-human-review

在将 P0 视为验收完成前，项目负责人必须在 `data/p0_f6_real_v2/validation_log.json`
为每张 F6 图片填写审核人、日期、叠图截图、线段对齐、线族混入和处置结论。
异常候选只用于复核排序，不能被写作 AI 证据或 AI 概率。

## 约束

- 只分析本地静态图片；不下载 URL、不调用外部 API。
- 不在导入时下载模型权重。
- 叠加图使用 canonical（已应用 EXIF 方向）的原图坐标。
- OpenCV LSD 的 quality 是后端相对质量，不是概率，不参与 AI 来源判断。
- outputs、data 和 weights 被 gitignore 排除。
