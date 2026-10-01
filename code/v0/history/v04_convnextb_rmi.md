# V04 ConvNeXt-B D4 + GN + CE/RMI

更新：2026-09-28。负责人：待用户指定。来源版本：`code/v02_convnextb_d4_lovasz_gn`。

## 改进目标

本版本是v02的区域关系损失候选。保留D4与GN，将主头的 `CE + Lovász` 替换为 `CE + RMI`；它与v03竞争，不在ABL之上继续叠加RMI。

## 配置与主要改动

- 正式候选配置：`configs/aic_convnext_base_upernet_768.py`。
- 实验名：`v04_convnextb_rmi_768`。
- 主头损失：`CrossEntropyLoss(1.0) + RegionMutualInformationLoss(0.5)`。
- RMI默认参数：3×3邻域、4倍平均池化、协方差稳定项 `1e-6`、Ignore255。
- 辅助头：保持 `CrossEntropyLoss(0.4)`。
- `aicseg/rmi_core.py` 为纯PyTorch数值实现，`aicseg/rmi_loss.py` 负责MMSeg注册。
- ConvNeXt-B、UPerNet、ImageNet-21K分类预训练、Fold0、物理batch2/累积4、40k iter及768/512滑窗与v02一致。
- L配置只作为B配置的继承依赖保留，不是本轮正式训练入口。

RMI在训练中使用局部协方差、float64 Cholesky及严格Ignore邻域屏蔽，实际速度、显存和AMP行为必须以服务器smoke为准。目前没有精度提升证据。

## 服务器训练命令

公共数据检查、环境安装以及训练完成后的分析、预测和打包见根 `docs/setup.md`。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v04_convnextb_rmi"
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

- 本地CPU PyTorch单元测试15项：14项通过，CUDA AMP测试因本机无CUDA跳过。
- 已覆盖全Ignore、Ignore邻域、非整除尺寸、小目标、半精度输入、公式对照和有限反向传播。
- 尚未完成MMSeg真实注册表构建、GPU前向、300 iter服务器smoke或正式训练。
- 暂无run、权重或精度结果；当前只能称为“代码已实现且CPU合成测试通过”。

论文依据：Zhao等，*Region Mutual Information Loss for Semantic Segmentation*，NeurIPS 2019，https://proceedings.neurips.cc/paper/2019/hash/a67c8c9a961b4182688768dd9ba015fe-Abstract.html 。本项目实现是对论文损失的独立适配，不使用外部分割训练数据或checkpoint。
