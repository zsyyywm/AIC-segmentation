# AIC 航拍图像语义分割

以ConvNeXt-B + UPerHead为基线的单模型语义分割项目。**v0数据增强与损失实验阶段已结束，当前进入v1网络结构改进阶段。** 原v02—v05统一归档到`code/v0/`，保留四组配置及独立成绩；`code/v1/`提供v04等价控制（U0）。v1系列已有M0（Mask2Former）、UP（UPerHead+多原型模块）和MP（Mask2Former+多原型模块）的独立实现，尚无新结构的正式训练、完整验证或平台成绩。

## 快速开始

1. 拉取项目并初始化 `code/mmsegmentation` 子模块，将另行收到的官方数据放入根目录 `data/`。
2. 按 [全局复现与训练后操作](docs/setup.md) 检查数据、复现环境，并对完成的run进行分析、测试推理和提交打包。
3. 阅读所用版本的 `version_notes.md`，获取配置、训练命令、预算与验证状态。本轮先比较 [M0：Mask2Former](code/v1_m0_mask2former/version_notes.md) 和 [UP：UPerHead加多原型](code/v1_up_uper_proto/version_notes.md)；[MP：Mask2Former加同一多原型](code/v1_mp_mask2former_proto/version_notes.md)为条件候选。第一轮见 [v0说明](code/v0/version_notes.md)，结构控制见 [v1说明](code/v1/version_notes.md)，原始基线见 [baseline说明](code/baseline_v1/version_notes.md)。
4. 按 [成员开发指南](docs/development.md) 在code中创建自己的版本、修改并提交分支。

修改前遵守 [协作与赛事底线](AGENTS.md)。理解模型可选读 [模型说明](docs/model.md)，核对标签和划分可选读 [数据规范](docs/data_spec.md)。

## 目录与职责

| 位置 | 内容 |
|---|---|
| `code/baseline_v1/` | 原始基线及其 `version_notes.md`，不直接覆盖 |
| `code/v0/` | v02—v05 的四组数据增强与损失实验配置、源码和历史映射 |
| `code/v1/` | v04等价的网络结构改进控制配置 |
| `code/v1_m0_mask2former/` | M0：完整Mask2Former区域查询对照，不含多原型模块 |
| `code/v1_up_uper_proto/` | UP：UPerHead加多原型区域语义模块 |
| `code/v1_mp_mask2former_proto/` | MP：Mask2Former加同一多原型模块 |
| `code/<其他改进版本>/` | 成员独立修改的源码、配置及版本说明 |
| `code/mmsegmentation/` | 各版本共享的官方框架，保持原样 |
| `tools/` | 各版本共用的数据检查、绘图、单模型分析、对比和提交工具 |
| `data/` | 官方图像、mask和固定划分，图片只读 |
| `docs/` | 操作、开发、模型与数据说明，各自独立 |
| `result.md` | 按日期追加的实验分析、对比和平台结果 |
| `AGENTS.md` | 需求澄清、协作与赛事约束 |

各版本训练时自动生成 `runs/<实验名>/<时间戳>_train/`，保存配置、日志、权重和趋势图。同一代码版本可训练多次，结果不覆盖。
v0旧版的本地导出位于被Git忽略的根 `runs/archives/v0/`；v03诊断仍在 `runs/diagnostics/20260930_barren/`。这些路径不属于GitHub交付物。
本次GitHub目录更新包含v0归档、v1控制及M0、UP、MP三个独立版本；每个版本保留自己的配置、组件和入口，不跨版本导入或继承。原成员分支和历史实验身份保留。

## 当前进度

| 版本/组件 | 当前状态 | 已有证据 | 下一步 |
|---|---|---|---|
| `baseline_v1` | 已完成一次ConvNeXt-B 40k正式训练 | 本地已归档服务器run、日志、配置快照和最佳权重；最佳验证mIoU 74.41。平台64.4665来源于成员反馈 | 作为后续版本比较基准 |
| `v0`：v02—v05 | 第一阶段已结束；四个独立40k正式run已完成并归档；整理后未在服务器重跑 | 固定验证mIoU依次75.53、75.59、76.32、76.32；本地保留原run导出，无新平台成绩 | 作为v1历史对照；保留四组配置、成绩及原始路径 |
| `v1` 控制配置（U0） | 已按v04实际解析配置建立，尚无v1正式run | 本地配置解析仅在实验名与run目录上区别于v04快照；历史成绩不算v1成绩 | 补齐同口径控制run和完整验证预测 |
| M0 | 独立代码完成；版本说明记录已通过本地检查 | 分类预训练、Ignore监督、768 AMP、1024滑窗及4micro官方数据检查的历史记录；冻结清单和统一发布复核摘要仍在，原验证run当前未找到 | 服务器环境和smoke待验证；恢复或重新取得原始检查证据；无正式精度或平台成绩 |
| UP | 独立代码完成；本地模块、完整模型及官方数据短程Runner检查通过 | 24项核心/头检查、预训练核验、16micro开发检查、4micro官方Runner及1024滑窗；短程Runner的更新点lr为0，详见版本说明 | 服务器smoke待验证；正式训练待授权，无完整验证或平台成绩 |
| MP | 独立代码完成；版本说明记录已通过本地检查 | P旁路、768 AMP、1024滑窗及8micro官方数据检查的历史记录；冻结清单和统一发布复核摘要仍在，原验证run当前未找到 | 服务器smoke待验证；恢复或重新取得原始检查证据；正式训练为条件候选，无精度或平台成绩 |
| `postprocess_rgb_crf` | 独立RGB DenseCRF流程完成 | 9项合成单元测试通过；尚无真实模型概率导出和全量验证结果 | 等训练模型胜出后再验证是否接入 |

v0四版使用同一增强流程，主要差异是CE+Lovasz、CE+Lovasz+ABL、CE+RMI、CE+RMI+ABL。v03的真实混淆比例不能替代v04结果。M0与U0比较完整分割头及其必要训练配套，不能把未来收益完全归因于结构；UP/MP分别与对应无P对照比较。CRF只在最终候选checkpoint上决定是否采用。详细实验记录见 `result.md`，当前入口与证据边界见各阶段 `version_notes.md`。

2026-10-02发布核查：M0、MP源码目录中的本地产物已不存在，之前迁移计划中的对应诊断目标也不存在；不能据计划清单宣称原始证据已完整归档。UP诊断仍在根`runs/diagnostics/20261001_up_feasibility/`。本次发布只做源码、配置与文档检查，不新增训练结果。

正式run、开发检查证据、权重、数据、预测图、环境、缓存和提交ZIP仅本地或服务器留存，不上传GitHub。成员拿到版本源码后，需要标准AIC目录、同级MMSeg、另行提供的官方数据和该版本声明的环境；算法入口不依赖其他实验版本。使用GitHub交付源码，无需额外生成源码ZIP。

项目自编README只有本文件。第三方框架自带README和LICENSE保留，不属于项目自编教程。
