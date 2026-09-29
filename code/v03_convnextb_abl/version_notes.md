# V03 ConvNeXt-B D4 + Lovász + GN + ABL

更新：2026-09-28。负责人：待用户指定。来源版本：`code/v02_convnextb_d4_lovasz_gn`。

## 改进目标

本版本是v02的边界监督候选，只在主解码头增加Active Boundary Loss（ABL）。D4、GN、ConvNeXt-B、UPerNet、数据划分、训练预算和推理方式保持不变，不新增推理网络或后处理。

## 配置与主要改动

- 正式候选配置：`configs/aic_convnext_base_upernet_768.py`。
- 实验名：`v03_convnextb_abl_768`。
- 主头损失：`CrossEntropyLoss(1.0) + LovaszLoss(0.5) + ActiveBoundaryLoss(0.1)`。
- 辅助头：保持 `CrossEntropyLoss(0.4)`。
- ABL默认参数：预测边界采样上限1%、最大GT边界距离20、标签平滑0.2、Ignore255。
- `aicseg/abl_core.py` 为独立数值实现，`aicseg/abl.py` 只负责MMSeg注册；增加SciPy距离变换依赖。
- L配置只作为B配置的继承依赖保留，不是本轮正式训练入口。

ABL会在训练时增加CPU距离变换和CPU/GPU数据转换，实际吞吐与显存必须以服务器smoke为准。目前没有精度提升证据。

## 服务器训练命令

公共数据检查、环境安装以及训练完成后的分析、预测和打包见根 `docs/setup.md`。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v03_convnextb_abl"
export CONFIG=configs/aic_convnext_base_upernet_768.py
export OMP_NUM_THREADS=1
cd "$MODEL_DIR"
```

先执行最多300 iter的独立smoke：

```bash
bash tools/start_train.sh --config "$CONFIG" --smoke
```

smoke通过并取得长训练授权后，从分类预训练权重开始40k正式训练，不从smoke续训：

```bash
bash tools/start_train.sh --config "$CONFIG"
```

只有正式run确实中断时，使用其实际绝对路径恢复：

```bash
bash tools/start_train.sh --config "$CONFIG" --resume /absolute/path/to/run
```

训练期间查看和重连：

```bash
tmux ls
tmux attach -t 实际会话名
export RUN_DIR=/absolute/path/to/run
python "$MODEL_DIR/tools/monitor_training.py" --run "$RUN_DIR"
tail -f "$RUN_DIR/console.log"
nvidia-smi
```

`tail` 会持续占用当前终端。连接断开后先检查已有tmux会话，不要重复启动训练。未经明确授权不得启动超过300 iter或预计超过2小时的训练。

## 验证状态

- 本地CPU PyTorch单元测试18项：17项通过，CUDA AMP测试因本机无CUDA跳过。
- 已覆盖全Ignore、Ignore邻域、无真实边界、无预测边界、小目标、半精度输入和有限反向传播。
- 尚未完成MMSeg真实注册表构建、GPU前向、300 iter服务器smoke或正式训练。
- 暂无run、权重或精度结果；当前只能称为“代码已实现且CPU合成测试通过”。

论文依据：Wang等，*Active Boundary Loss for Semantic Segmentation*，AAAI 2022，https://arxiv.org/abs/2102.02696 。本项目实现是对论文算法的独立适配，不使用外部分割训练数据或checkpoint。
