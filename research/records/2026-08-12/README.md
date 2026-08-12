# 2026-08-12 研究记录清单

本目录只保留能够支撑阶段结论的 28 份机器可读记录。筛选标准是：最终审计优先；保留
复现最终审计所必需的预注册协议；保留会影响资源边界理解的关键失败收据；不保留已被
最终结果取代的重复候选屏、连续采集协议版本和中间选择表。

## Pixel

- `ai_likelihood_openfake_available_pixel_baseline_audit_v1.json`：五通道可用行修正基线；
- `ai_likelihood_openfake_confirmation_audit_v1.json`：两项候选的独立确认结果；
- `ai_likelihood_forensic_clip_two_view_candidate_v1.json`：forensic 双视图候选定义；
- `ai_likelihood_community_two_view_protocol_v1.json`：Community 第四窗召回协议；
- `ai_likelihood_community_two_view_recall_audit_v1.json`：第四窗召回安全结果；
- `ai_likelihood_dda_low_memory_audit_v1.json`：DDA 等价低内存加载审计；
- `pixel_fusion_recall_audit_v1.json`：历史像素融合的唯一实拍去重审计。

## Geometry

- `geometry_v2_fivek_extension_protocol_v1.json` 与对应 audit：FiveK G1–G4 复核负担；
- `geometry_v2_openfake_comparison_protocol_v1.json` 与对应 audit：生成／实拍对照；
- `fivek_cross_signal_diagnostic_v1.json`：像素与几何集合交叉诊断；
- `geometry_flip_equivariance_protocol_v1.json` 与对应 audit：已被 SDXL 留出门禁拒绝的水平翻转等变性路线；
- `geometry_semantic_relation_pilot_protocol_v1.json`：32 个唯一源图加 4 个隐藏重复的语义表面关系盲审协议，当前不含人工结果。
- `geometry_semantic_relation_agent_assisted_protocol_v1.json`：三组来源盲化 AI 辅助预标注的角色、独立性、输入哈希与合并门禁；结果不是人工真值，也不授权来源计分。
- `geometry_semantic_relation_agent_assisted_audit_v1.json`：36 项 AI 辅助盲审的来源中立最终闭包、人工优先复核队列与不计分决策。
- `geometry_deterministic_surface_baseline_protocol_v1.json`：在实现前冻结的粗表面候选、双侧采样、线段邻接和线族拆分参数；不读取预标注答案，不授权来源计分或网页接入。
- `geometry_deterministic_surface_baseline_audit_v1.json`：36 个来源盲化 packet 的双次确定性、产物闭包、实现哈希和资源统计；只授权进入独立人工关系对比。
- `geometry_surface_continuation_protocol_v1.json`：在读取独立人工答案前冻结基线—人工比较指标、门禁和通过后才可运行的表面条件 G1–G4 离线重放规则。
- `geometry_surface_continuation_readiness_v1.json`：不读取人工标注、AI 辅助标注或来源密钥的非人工工程就绪审计；全部门禁通过，当前只等待独立人工标注与来源中立质量回执。
- `geometry_surface_agent_diagnostic_protocol_v1.json`：按用户要求预先冻结 AI 辅助盲标的描述性表面对比和 G1–G4 诊断重放；明确不能代替人工门禁、来源计分或网页接入。
- `geometry_surface_agent_diagnostic_audit_v1.json`：30 项完成态 AI 盲标的确定性表面对比与表面条件 G1–G4 诊断；反事实人审阈值失败，仍须以后独立人工确认。

## Real controls

- `real_control_fivek_pilot_v1.json`：FiveK 首批 8 个源图簇；
- `real_control_fivek_extension_audit_v1.json`：FiveK 后续 32 个源图簇；
- `real_control_hdrplus_pixel_protocol_v1_failure.json`：错误 DDA 资源上限的失败收据；
- `real_control_hdrplus_pixel_audit_v1.json`：HDR+ 40 个手机 RAW 源图簇结果；
- `real_control_source_screen_v1.json`：真实对照来源与许可筛选。

这些文件全部不被网页运行时读取，也不授权阈值、分值或判断文案变更。被筛除的 16 份
中间记录保存在本地 `outputs/rollback_backups/night_research_records_2026-08-12/`。
