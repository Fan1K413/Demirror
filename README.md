# Demirror

Demirror 是一个本地静态图片审阅工具，将来源记录、AI 像素信号、画面几何和相机参数
拆成可审计证据，再给出有边界的综合判断。运行时配置与模型审计保存在 `models/`；不会
参与运行的实验协议、对照结果和阶段报告统一保存在 [research](research/README.md)。

## 本地网页演示

在已安装项目依赖的环境中运行：

    .venv\Scripts\python -m image_trust.cli serve

然后打开 `http://127.0.0.1:8765`。服务只绑定本机地址；上传文件、持久化的
作业状态和证据叠图会写入已忽略的 `.demirror_web_jobs/`。默认不会调用外部 API，
也不会自动下载模型。P0 几何链路始终会被尝试；C2PA 或 P1 相机链路不可用时，页面会
明确显示“部分完成”和相应限制，而不会伪造结果。
如果本地 Python 进程意外停止，下一次启动服务会把未完成的网页作业标成失败，
避免浏览器无限等待。

### Docker Compose

仓库提供 `Dockerfile`、`compose.yaml` 和环境变量示例。首次使用可复制配置后启动：

    Copy-Item .env.example .env
    docker compose up --build -d

然后打开 `http://127.0.0.1:8765`。默认发布的端口只绑定宿主机回环地址；容器以非 root
用户、只读根文件系统和移除全部 Linux capabilities 的方式运行。作业与模型缓存保存在
具名卷中，`weights/` 和 `data/` 以只读方式挂载，均不会被写进镜像。停止并保留数据使用
`docker compose down`；同时移除作业和缓存卷需显式执行 `docker compose down --volumes`。

默认镜像包含 C2PA、CPU 像素通道和离线水印适配器的运行库，但不包含受许可约束的模型
权重、数据集或 Perspective Fields/GeoCalib 上游代码。缺少挂载资产的通道会按现有规则
明确显示不可用。若只需核心几何、元数据和网页能力，可在 `.env` 中设置
`DEMIRROR_INSTALL_OPTIONAL_DETECTORS=0` 后本地重新构建。`OPENAI_API_KEY` 只从本地
`.env` 注入运行容器，不能作为 Docker build 参数或提交到仓库。

GitHub Actions 会在拉取请求中构建并调用 `/api/health` 做容器冒烟测试，但不发布镜像；
推送到 `main`、推送 `v*` 版本标签或手动运行工作流时，AMD64 与 ARM64 的构建、启动和
健康检查会分成并行任务执行。两种架构的健康检查都通过后，发布任务复用这两份架构缓存，
将多架构镜像及 SBOM、构建来源证明发布
到 `ghcr.io/fan1k413/demirror`。服务器拉取同一标签时由 Docker 自动选择本机架构，无需
修改 `compose.yaml`。私有包拉取时使用具备
`read:packages` 权限的 GitHub 凭据登录 `ghcr.io`；发布工作流本身仅使用仓库内置的
`GITHUB_TOKEN`，不需要另建 Registry Key。Docker 构建摘要仍会生成，但关闭可下载的
`.dockerbuild` 构建记录上传，避免该诊断文件继续占用 Actions Artifact 配额。

分析过程中，页面会显示当前检测阶段、阶段完成度和已用时间。进度按几何、来源记录、
各像素模型、相机检查与综合推断的实际边界更新；它表示流水线已走到的位置，不是对剩余
时间的估算。修改后端代码后需要重启本地服务，浏览器再强制刷新一次。

### 几何关系盲审工具

已生成语义表面关系试点包后，正常启动原来的本地服务：

    .venv\Scripts\python -m image_trust.cli serve

然后打开 `http://127.0.0.1:8765/geometry-review/`。页面按盲审清单依次展示匿名原图、几何叠图和
逐线族细图，可画出可见表面多边形、把检测线归入表面并填写固定线族结论。草稿直接
原子写回每项的 `annotation.json`；刷新后可以继续。服务只接受本机地址，只提供盲包
明确声明的文件，不读取或暴露相邻 `posthoc/` 中的事后来源密钥。

该工具服务于几何标注一致性研究；它与图片分析入口共用本机服务，但使用独立子路径、
资源白名单和数据 API，也不改变 AI 分数。完成全部
任务后仍须运行预注册审计器，通过隐藏重复一致性门禁后才能开启来源密钥。

### 可选 OpenAI 官方来源验证

OpenAI 已提供 `POST /v1/content_provenance_checks`，用于检查其支持的图片 C2PA 与
SynthID 来源信号。它不是通用 AI 检测器。Demirror 默认关闭该项；只有网页中对当前
图片勾选后才会把文件发送给 OpenAI。网络、鉴权或限额错误只让附加检查标为不可用，
不会丢失本地结果。

在项目根目录新建已被忽略的 `.env`，然后重启服务：

    OPENAI_API_KEY=你的项目API密钥

也可以在启动服务的同一个 PowerShell 会话设置 `$env:OPENAI_API_KEY`。密钥只在服务
进程内读取，不写入作业 JSON、日志或 Git。不要把密钥粘贴到聊天、截图或提交中。

Google 当前只在登录后的 Gemini 应用和受限测试门户提供图片 SynthID 验证，没有公开的
图片检测 API；页面只提供官方手动验证入口，不模拟网页上传，也不需要 Google API Key。
在线验证区域默认完全隐藏：配置 `OPENAI_API_KEY` 后只显示 OpenAI 项；如需显示 Google
手动入口，再配置 `GOOGLE_SYNTHID_MANUAL_ENABLED=1`。两项均未配置时页面不显示该区域。

## P3 多路 AI 像素信号

P3 不是把几个原始分数平均成“AI 概率”，而是顺序执行五条已登记、作用范围不同的
正向检测通道：

- DDA 是高阈值补充通道；固定阈值 0.94 在原始 SDXL 留出集上召回 85%、误报 0%，
  但后续跨生成器审计表明它不能单独担当主判。
- SAFE 只补充无损或未经重编码的生成纹理；本地 15 张去来源 ChatGPT 图召回 100%，
  但 JPEG 75 后召回降为 0%，因此不得外推到压缩图。
- forensic CLIP 是低召回、耐 JPEG 的补充通道。高阈值 0.9925177 在未见 SDXL JPEG 75
  上召回 17.3%、误报 6.7%；偏向 AI 检出的有限阈值 0.9919478 召回 28.0%、误报 14.0%，
  只能触发“可能为 AI（有限）”。
- Community Forensics 是跨生成器补充通道；固定高阈值在四生成器原图留出集上召回
  80.8%、误报 4.2%。
- Nonescape Mini 是严格补充通道；它与 Community Forensics 的冻结高阈值取并集后，
  同一留出集召回为 86.9%、误报仍为 4.2%。

任一高阈值命中可触发“可能为 AI（高）”；有限阈值只触发有限强度复核。全部低于阈值
仍只表示“未检出 AI 信号”，不会反推为相机照片。完整相机 EXIF 仅在 AI 检测已成功完成
且未命中时支持“可能为实拍（有限）”，因为 EXIF 可以复制或编辑。已登记的旧 P0 全图
几何来源模型按冻结档位提供 `+10/+20` 的受限辅助分；新的局部几何 v2 与 P1 相机一致性
仍只作为可审阅解释，不参与当前来源判定。

审计记录位于 `models/ai_likelihood_dda_v1.json`、`models/ai_likelihood_safe_v1.json`、
`models/ai_likelihood_forensic_clip_v1.json`、`models/ai_likelihood_community_forensics_v1.json`
和 `models/ai_likelihood_nonescape_mini_v1.json`。P3 运行时不联网、不自动下载权重；五个模型
逐个在短生命周期 CPU 子进程中运行，结束即释放模型内存，避免在网页服务中常驻叠加。
可选依赖固定在 `requirements-p3-pixel.lock`。

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

未安装研究叠加依赖时，SIDBench 专用测试会明确跳过，不影响基础产品测试环境。

## SIDBench 离线候选复现环境

PatchCraft/RPTC 与 NPR 的 SIDBench 筛选只用于离线研究，不属于网页或基础安装。
冻结评估器已经被既有报告按 SHA-256 绑定，因此不在原文件中追加环境逻辑；其 CPU
复现环境由独立叠加锁文件和环境记录约束。参考环境固定为 CPython 3.10.6、
`torch==2.13.0+cpu` 与 `torchvision==0.28.0+cpu`，CPU wheel 来自 PyTorch 官方索引：

    py -3.10 -m venv .venv-sidbench
    .venv-sidbench\Scripts\python -m pip install --upgrade pip
    .venv-sidbench\Scripts\python -m pip install -r requirements.lock
    .venv-sidbench\Scripts\python -m pip install -r requirements-research-sidbench.lock
    .venv-sidbench\Scripts\python -m pip install -e ".[dev]"
    .venv-sidbench\Scripts\python scripts\verify_sidbench_research_environment.py

校验器会同时检查 Python、PyTorch、TorchVision、两份依赖锁，以及冻结评估器、协议和
最终审计的 SHA-256；任一项不一致都会在模型加载前失败。它不会下载上游代码、权重或数据。
在已按协议准备 SIDBench、NPR 源码、checkpoint 和登记数据后，完整评分与汇总命令为：

    $python = ".venv-sidbench\Scripts\python"
    $protocol = "research\records\2026-08-12\pixel\sidbench_patchcraft_npr_screen_protocol_v1.json"
    $sidbench = "outputs\research\sidbench_candidate_v1"
    $npr = "outputs\research\npr_upstream_candidate_v1"
    $result = "outputs\research\sidbench_patchcraft_npr_replay_v1"
    & $python scripts\screen_sidbench_candidates.py score --candidate patchcraft_rptc --repo-root . --protocol $protocol --sidbench-root $sidbench --npr-upstream-root $npr --output "$result\patchcraft_rptc.json"
    & $python scripts\screen_sidbench_candidates.py score --candidate npr --repo-root . --protocol $protocol --sidbench-root $sidbench --npr-upstream-root $npr --output "$result\npr.json"
    & $python scripts\screen_sidbench_candidates.py summarize --repo-root . --protocol $protocol --patchcraft-report "$result\patchcraft_rptc.json" --npr-report "$result\npr.json" --output "$result\audit.json"

环境补充记录位于
`research/records/2026-08-12/pixel/sidbench_patchcraft_npr_environment_v1.json`。原始审计
没有保存包版本，所以该记录是后续重放的固定环境，不追溯声称历史进程已记录这些版本。

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

- 只分析本地静态图片且不下载 URL；OpenAI 官方来源验证是默认关闭、逐次明确同意的唯一外部调用。
- 不在导入时下载模型权重。
- 叠加图使用 canonical（已应用 EXIF 方向）的原图坐标。
- OpenCV LSD 的 quality 是后端相对质量，不是概率，不参与 AI 来源判断。
- outputs、data 和 weights 被 gitignore 排除。
- P1 不自动安装依赖或下载相机模型权重；后端不可用会显式写入结果。
- GeoCalib 代码为 Apache-2.0；其训练权重须在取得时单独记录 CC-BY-4.0 来源与 SHA-256。
- Perspective Fields 的 Adobe Research License 仅允许非商业使用，接入前必须确认项目用途符合该许可。
