# v1：结构改进阶段控制配置

建立日期：2026-10-01；阶段状态更新：2026-10-02。本目录是v1结构改进阶段的v04等价控制入口（U0），尚无本控制的正式run或新成绩。v0阶段已结束，v1系列候选M0、UP、MP已分别在 `code/v1_m0_mask2former/`、`code/v1_up_uper_proto/`、`code/v1_mp_mask2former_proto/` 实现，具体变量和检查状态见各自 `version_notes.md`；新候选不覆盖本控制目录。

## 来源与等价边界

控制入口：`configs/control_convnextb_rmi_768.py`；原始依据是 v04 实际正式 run 的 `resolved_config_20260930_200850.py`，SHA256 为 `09646cce458c49c1d72c434b6c0cb44e116da3ab343a05198cf6ebf7ab051e5f`。该快照保存在根 `runs/archives/v0/v04_convnextb_rmi/20260930_200850_train/`，其记录的服务器 run 路径保持原样。

MMEngine 解析后，新控制配置与该快照仅有两处差异：

- `experiment_name`：从 `v04_convnextb_rmi_768` 改为 `v1_convnextb_rmi_control_768`。
- `work_dir`：从原服务器绝对 run 路径改为 `runs/v1_convnextb_rmi_control_768`；实际启动时训练入口再创建独立时间戳 run。

其余配置保持 v04 实际运行口径：ConvNeXt-B + UPerHead、主头 CE1.0 + RMI0.5、辅助头 CE0.4，无 ABL；随机缩放、768 裁剪、D4、PhotoMetricDistortion；GroupNorm、AdamW、stage-wise 层衰减与原调度；公开分类主干预训练、固定数据划分、Ignore255；物理 batch2/累积4、40k micro iterations、8万样本曝光、1万优化器更新；slide768/stride512 验证。v1 的 `aicseg/`、`tools/` 和依赖从 v04 对应实现建立，不导入 `code/v0/`。

v04 历史正式验证 mIoU 为 76.32；这是旧 run 结果，**不是 v1 成绩**。目前没有 v04 完整验证预测/真实混淆矩阵，应在结构对照前补齐；v03 的真实混淆比例不能代替 v04。

## 可摘取前提与后续入口（本次未执行）

整个 `code/v1/` 可放入标准 AIC 项目 `code/` 下，依赖同级 `mmsegmentation/`、根 `data/`、根公共 `tools/` 与 `docs/setup.md` 声明的环境。一个 Python 进程只加载 v1 的 `aicseg`。控制配置为独立文件，不跨目录继承 v0；新训练 run 在 `code/v1/runs/v1_convnextb_rmi_control_768/` 下。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v1"
export CONFIG=configs/control_convnextb_rmi_768.py
export OMP_NUM_THREADS=1
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --smoke
# smoke通过并取得长训练授权后，另起正式run：
bash tools/start_train.sh --config "$CONFIG"
# 仅正式run确实中断时恢复：
bash tools/start_train.sh --config "$CONFIG" --resume /absolute/path/to/run
python tools/monitor_training.py --config "$CONFIG" --run /absolute/path/to/run --once
```

本次只完成本地配置解析、模块注册、路径及历史等价性检查；未执行 GPU 前向、服务器 smoke、正式训练或推理。若后续新结构需要修改专用损失或优化器参数分组，需在对应设计任务中单独记录与验证，不能把这类调整视为当前控制配置的一部分。
