# Demirror

## 本地网页演示

在已安装项目依赖的环境中运行：

    .venv\Scripts\python -m image_trust.cli serve

然后打开 `http://127.0.0.1:8765`。服务只绑定本机地址；上传文件、持久化的
作业状态和证据叠图会写入已忽略的 `.demirror_web_jobs/`，不会调用外部 API 或
自动下载模型。P0 几何链路始终会被尝试；C2PA 或 P1 相机链路不可用时，页面会
明确显示“部分完成”和相应限制，而不会伪造结果。
如果本地 Python 进程意外停止，下一次启动服务会把未完成的网页作业标成失败，
避免浏览器无限等待。

## P1 audit commands

`camera-calibration-summary` accepts only result files made with the same P1
configuration, requested backend, actual backend/model/weights provenance, and
unique input hashes. It produces descriptive gate and `E_cam` coverage data;
it does not fit or emit a decision threshold:

    .venv\Scripts\python -m image_trust.cli camera-calibration-summary outputs\cohort\*\camera_result.json --config configs\p1_geocalib.yaml --cohort independent-camera-v1 --output outputs\p1_calibration_summary.json

`c2pa-analyze` reads embedded C2PA data only from a local file. It disables
remote-manifest and OCSP requests, so the record cannot claim current
revocation coverage. Install the dedicated overlay first:

    .venv\Scripts\python -m pip install -r requirements-p1-c2pa.lock
    .venv\Scripts\python -m image_trust.cli c2pa-analyze path\to\image.jpg --config configs\p1_c2pa.yaml --output outputs\p1_c2pa_sample

### Registered camera cohorts

A P1 registry stores each local file's SHA-256, canonical resolution, license,
provenance reference, transformations, and capture-or-generator family. The
audit rejects missing files, hash or size mismatches, unsafe paths, duplicate
hashes, and a family that appears in more than one split. The files remain in
the ignored `data/` directory.

The existing F6 set can be registered only as a real-camera control smoke
cohort. It is intentionally blocked from threshold calibration:

    .venv\Scripts\python scripts\import_p0_f6_as_p1_control.py --dataset-root data\p0_f6_real_v2 --source-manifest data\p0_f6_real_v2\source_manifest.json --output data\p0_f6_real_v2\p1_camera_control_registry.json
    .venv\Scripts\python -m image_trust.cli camera-calibration-registry-audit --registry data\p0_f6_real_v2\p1_camera_control_registry.json --dataset-root data\p0_f6_real_v2
    .venv\Scripts\python -m image_trust.cli camera-calibration-run --registry data\p0_f6_real_v2\p1_camera_control_registry.json --dataset-root data\p0_f6_real_v2 --config configs\p1_geocalib.yaml --split control --allow-control-smoke --output outputs\p1_geocalib_f6_control_v2

An independent calibration registry uses `intended_use:
independent_calibration`, places its new images in the `calibration` split, and
keeps every capture or generator family out of the `holdout` split. Run it with
the same command but omit `--split control --allow-control-smoke`.

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

## P1：相机参数一致性（当前为可审计骨架）

P1 与 P0 的 JSON 契约和运行链完全独立。它为每个后端统一记录：相机模型、
roll、pitch、垂直视场角或焦距、主点、地平线、不确定性、适用性、覆盖度、
局限，以及模型提交、权重 SHA-256、推理设备和耗时。

当前已提供 Perspective Fields 与 GeoCalib 的独立适配器边界。GeoCalib 的 CPU
推理代码与依赖已固定在独立锁文件中；适配器只接受显式提供的本地权重，也没有安装模型权重。
Perspective Fields 采用仅限非商业用途的 Adobe Research License，因此没有被放入
默认环境。在这两个后端真正可运行前，命令会写出明确的 `unavailable` 结果，而不会
下载模型、静默换用另一模型，或伪造相机参数：

    .venv\Scripts\python -m image_trust.cli camera-analyze path\to\image.jpg --config configs\p1_camera.yaml --output outputs\p1_sample

如需复现已验证的 GeoCalib CPU 依赖环境，先安装 P0 锁文件，再安装 P1 叠加锁文件：

    .venv\Scripts\python -m pip install -r requirements.lock
    .venv\Scripts\python -m pip install -r requirements-p1-geocalib.lock

GeoCalib 已在本地安装且权重文件存在时，可使用 `configs\p1_geocalib.yaml`。该配置将
权重路径、官方代码提交、CC-BY-4.0 权重许可、推理设备和相机模型写入每次结果；若缺少
任一必需项，结果仍是 `unavailable`。GeoCalib 默认假定主点在图像中心且不优化它，此限制
会写入输出，不能将它与能估计主点的后端直接混合为同一阈值。
为避免高像素照片在其全分辨率后处理阶段耗尽内存，`geocalib_max_input_edge` 默认限制
模型副本的最长边为 1280 像素；相机几何结果会映射回原图坐标，并在结果中留下限制记录。

已确认符合非商业用途时，Perspective Fields 使用独立的
`configs\p1_perspective_fields.yaml`。当前固定的是能预测主点偏移的
`Paramnet-360Cities-edina-uncentered`；其预期输入是裁剪图像，且官方输出没有可校准的
不确定度或直接地平线。因此参数仍可审计地写入结果，但在不确定度缺失时不会进入 `E_cam`。
其运行时需在 PowerShell 的 UTF-8 模式安装，避免上游 `setup.py` 的 Windows 编码问题：

    $env:PYTHONUTF8 = '1'; .venv\Scripts\python -m pip install -r requirements-p1-perspective-fields.lock

输出为 `outputs\p1_sample\camera_result.json`。P1 会对全图及 4、6 或 8 个候选
重叠方形裁剪分别请求估计；每个裁剪都保存 `crop_to_canonical` 仿射映射，主点和
地平线会先映射回 canonical 坐标再进行比较。只有全图合格、至少 3 个裁剪同时
通过低纹理／特殊成像／不确定性门控时，才会产生 `E_cam`；否则结果固定为
`observation: not_observed`，不是高异常。

`E_cam` 目前没有校准后的来源判定阈值，不能解释为 AI 分数、真实性分数或来源结论。
Perspective Fields 和 GeoCalib 必须使用各自独立的配置、模型提交与权重哈希运行，
不得在同一次结果中混合或互相回退。

## 约束

- 只分析本地静态图片；不下载 URL、不调用外部 API。
- 不在导入时下载模型权重。
- 叠加图使用 canonical（已应用 EXIF 方向）的原图坐标。
- OpenCV LSD 的 quality 是后端相对质量，不是概率，不参与 AI 来源判断。
- outputs、data 和 weights 被 gitignore 排除。
- P1 不自动安装依赖或下载相机模型权重；后端不可用会显式写入结果。
- GeoCalib 代码为 Apache-2.0；其训练权重须在取得时单独记录 CC-BY-4.0 来源与 SHA-256。
- Perspective Fields 的 Adobe Research License 仅允许非商业使用，接入前必须确认项目用途符合该许可。
