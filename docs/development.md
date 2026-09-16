# 成员开发指南

本文只说明拿到项目后如何开发独立版本和提交分支。环境、数据检查、训练和提交产物的命令见 `setup.md`；模型原理见 `model.md`。

## 1. 协作流程

维护者提供基线代码与说明，官方数据另行交给有权限的参赛成员。成员先用 `tools/check_uploaded_dataset.py` 核验数据，再学习基线、创建版本、提交分支。维护者选择审核与合并，并下载选定代码执行训练。

成员不必自行完成长训练，但必须如实标记哪些检查已执行、哪些未运行；不能把预计效果写成实验结果。

## 2. code中的目录关系

```text
code/
  mmsegmentation/                 共享框架，不复制到每个版本
  baseline_v1/                    原始参照，不在这里做新改进
  v02_convnextb_loss_dice/         独立改进版本示例
  v03_convnextb_aug_color/         另一个独立改进版本示例
```

版本目录必须处于code同层，不嵌套到baseline内。每个版本保留自己的configs、aicseg、tools、requirements和version_notes；共享项目根的data与tools以及同级框架。

根tools负责通用检查、曲线、单模型报告和比较；版本tools负责该版本的训练与底层推理。不要把公共脚本复制到各版本。

## 3. 创建自己的版本

1. 从最新基线或维护者指定的起点创建工作分支，例如 `feature/v02-convnextb-loss-dice`。
2. 在code下新建 `v02_convnextb_loss_dice`，命名体现版本、模型与主要改动，避免final2/new等名称。
3. 从起点复制 `configs/`、`aicseg/`、`tools/`、`requirements.txt`、`version_notes.md`。不要复制runs、预测目录、权重、pycache、图片；已有tests不属于启动训练必需文件。
4. 更新该版本配置的 `experiment_name`，例如 `v02_convnextb_loss_dice_768`，防止结果归属混淆。
5. 在自己的版本中改代码，一次只改变一个主要因素。保留配置继承所依赖的文件；当前B继承同目录L配置，不能只复制B文件。

不要只改文件夹名而仍把配置指向baseline目录。保持同层结构后，现有 `../../data` 与相邻框架路径通常不用改；具体以实际配置为准。

## 4. 改动放在哪里

| 内容 | 位置，相对于自己的版本目录 |
|---|---|
| 主干、分割头、loss、增强、优化器、预算 | `configs/` |
| 新的模型或loss组件 | `aicseg/` 内新增模块，并注册/配置导入 |
| 数据元信息与标签映射 | `aicseg/dataset.py`，非必要不改 |
| 每类IoU记录 | `aicseg/metrics.py`，保留口径和键名 |
| 训练摘要与自动绘图 | `aicseg/hooks.py`，保留输出契约 |
| 训练入口与配置快照 | `tools/train.py`，算法改进通常无需改 |
| tmux保活 | `tools/start_train.sh`，算法改进通常无需改 |
| 底层预测 | `tools/predict_and_pack.py`，保留根提交工具调用接口 |

初始版本中的 `aicseg` 使用相同包名，但在各自训练进程中导入本版本代码，不要在同一进程混用两个版本包。确需调整共享框架或根tools，单独向维护者说明，不混入算法版本的大量重复补丁。

## 5. 写好版本说明

每个版本用 `version_notes.md`，不要新增README。只记录这个版本的情况：

- 名称、负责人、日期、来源版本。
- 改进假设和唯一主要因素，修改文件及配置名称。
- 模型、实验名、种子、batch/累积、iter、约等效epoch与评测方式。
- 检查状态、已有run/日志/权重路径、指标和结论；未运行明确写“待维护者验证”。

不要复制环境安装、目录创建或Git教程到版本说明。详细逐次分析/对比记录在result，版本说明只保留该版本摘要。

## 6. 提交分支供维护者选择

示例在项目根、已有Git仓库中执行；分支和版本名换成自己的实际值：

```bash
git switch -c feature/v02-convnextb-loss-dice
# 完成上面的独立版本修改之后
git add code/v02_convnextb_loss_dice
git diff --cached --stat
git commit -m "Add ConvNeXt-B Dice loss experiment"
git push -u origin feature/v02-convnextb-loss-dice
```

创建分支应在开始改动前做；已在该分支时不要再次执行switch -c。暂存后检查只包含本版本必要文件，不包含图片、权重、缓存、密钥或运行产物。

在GitHub发起PR或将分支名、提交号交给维护者，并说明：改动目的、版本目录、配置、已做检查、未验证事项和新增依赖。不要自行合并或直接覆盖main。

维护者在独立、干净的代码副本中审核选定分支，决定是否合并和执行训练；训练时记录实际提交号与run。root result由维护者集中追加，减少多人写同一台账的冲突。

## 7. 交付前检查

- baseline未被修改，自己的目录有清楚名称与版本说明。
- 所有依赖配置和组件都提交，没有指向自己的机器私有路径。
- 增量依赖已写requirements，保留日志字段、配置快照和标签映射。
- 相对基线的改动明确；没有修改图片、官方测试标签或使用额外数据。
- 分支、提交号、配置和验证状态已告知维护者。
