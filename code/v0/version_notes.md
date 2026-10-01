# v0：数据增强与损失基线实验

整理日期：2026-10-01。v0 归档 v02—v05 四次已完成的正式实验；四者使用同一套随机缩放、768 裁剪、D4 与 PhotoMetricDistortion，主要变量是主头损失。它们是四个独立 run，不合并成绩。旧版说明原文保存在 `history/`；其中的旧训练路径仅用于追溯。

## 当前可运行入口与历史映射

| 历史版本目录 | 当前 B 配置（相对 v0） | 主头损失 | 40k 最终/最佳验证 mIoU | 本地归档 run |
|---|---|---|---:|---|
| `code/v02_convnextb_d4_lovasz_gn` | `configs/v02_convnextb_d4_lovasz_gn.py` | CE1.0 + Lovasz0.5 | 75.53 | `runs/archives/v0/v02_convnextb_d4_lovasz_gn/20260929_192049_train/` |
| `code/v03_convnextb_abl` | `configs/v03_convnextb_abl.py` | CE1.0 + Lovasz0.5 + ABL0.1 | 75.59 | `runs/archives/v0/v03_convnextb_abl/20260929_193345_train/` |
| `code/v04_convnextb_rmi` | `configs/v04_convnextb_rmi.py` | CE1.0 + RMI0.5 | 76.32 | `runs/archives/v0/v04_convnextb_rmi/20260930_200850_train/` |
| `code/v05_convnextb_rmi_abl` | `configs/v05_convnextb_rmi_abl.py` | CE1.0 + RMI0.5 + ABL0.1 | 76.32 | `runs/archives/v0/v05_convnextb_rmi_abl/20260930_203814_train/` |

表中归档路径相对 AIC 根目录。每个 B 配置在本目录继承对应的 L 配置，八个配置文件均随 v0 交付。L 文件是这些已执行 B 配置的继承依赖，不代表本轮执行过 L 训练；直接运行旧 L 配置仍引用原 `code/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth`，不属于本轮验证入口。四个 B 配置均覆盖为公开 ConvNeXt-B 分类预训练地址，不依赖该本地权重。四个 B 配置继续使用各自原有 `experiment_name`；新训练 run 位于 `code/v0/runs/<experiment_name>/`，不会覆盖旧服务器 run。

历史服务器 run 的实际路径依次是：

- v02：`/root/autodl-tmp/AIC/code/v02_convnextb_d4_lovasz_gn/runs/v02_convnextb_d4_lovasz_gn_768/20260929_192049_train`
- v03：`/root/autodl-tmp/AIC/code/v03_convnextb_abl/runs/v03_convnextb_abl_768/20260929_193345_train`
- v04：`/root/autodl-tmp/AIC/code/v04_convnextb_rmi/runs/v04_convnextb_rmi_768/20260930_200850_train`
- v05：`/root/autodl-tmp/AIC/code/v05_convnextb_rmi_abl/runs/v05_convnextb_rmi_abl_768/20260930_203814_train`

本地归档原样保留解析配置、manifest、summary、日志与 checkpoint 记录。只有 v03 归档有实际最佳权重；v03—v05 有测试 ZIP，v02、v04、v05 的本地归档没有可用 checkpoint。v03 完整验证诊断及真实混淆矩阵仍在 `runs/diagnostics/20260930_barren/`，未搬动。v04 尚无完整验证预测/真实混淆矩阵；其精确荒地混淆比例不能引用 v03 的结果。逐次证据和平台关联边界见根 `result.md`。

## 版本单元与配置口径

把整个 `code/v0/` 放入标准 AIC 项目的 `code/` 后，可依赖同级 `mmsegmentation/`、根 `data/`、根公共 `tools/` 和 `docs/setup.md` 声明的 Python 环境。`aicseg/` 汇集与旧版逐文件相同的公共组件，以及与 v03/v04 对应实现哈希一致的 ABL/RMI；`requirements.txt` 含四组配置所需依赖。一个 Python 进程只加载 v0 的 `aicseg`，不能混用其他版本同名包。

四个当前 B 配置经 MMEngine 解析后与对应旧版配置逐项相等。对应实际 run 的解析快照与旧版配置仅在训练时写入的绝对 `work_dir` 上不同。归档快照中的旧路径不改写；历史 run 不是新 `code/v0/runs/` 下已经执行的 run。当前源码目录未经过服务器 smoke 或正式训练，不能把历史成绩称为重构后代码的新成绩。

## 后续复跑入口（本次未执行）

训练、恢复和监控均需显式选择上表某一 `CONFIG`；v0 工具不设默认实验。下例选 v04 损失组，其余三组只替换配置路径。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v0"
export CONFIG=configs/v04_convnextb_rmi.py
export OMP_NUM_THREADS=1
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --smoke
# smoke通过并取得长训练授权后，另起独立正式run：
bash tools/start_train.sh --config "$CONFIG"
# 仅实际正式run中断时，用其绝对路径恢复：
bash tools/start_train.sh --config "$CONFIG" --resume /absolute/path/to/run
python tools/monitor_training.py --config "$CONFIG" --run /absolute/path/to/run --once
```

共同预算为 ConvNeXt-B + UPerHead、固定 Fold0、seed2026、物理 batch2/累积4、40k micro iterations、8万样本曝光、1万优化器更新，slide768/stride512 验证。数据与环境检查、完成 run 的通用分析及提交操作见根 `docs/setup.md`。

本地历史 run 可传绝对归档路径给根分析工具。历史推理需同时提供原 `resolved_config_*.py`、实际 checkpoint 和显式 `--model-dir code/v0`；目前只有 v03 本地归档具备该 checkpoint。根 `tools/test_and_pack.py` 支持这些显式参数；这一步未在本次归档任务中重新执行推理。
