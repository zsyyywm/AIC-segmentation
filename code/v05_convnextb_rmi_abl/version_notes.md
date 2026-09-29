# V05 ConvNeXt-B D4 + GN + CE/RMI + ABL

更新：2026-09-29。负责人：待用户指定。来源版本：`code/v04_convnextb_rmi`。

## 改进目标

本版本用于验证RMI区域关系监督与ABL边界监督能否互补。它在v04的 `CE + RMI` 主头损失上增加ABL，不重新加入Lovasz。网络结构和推理过程保持不变。

四个训练候选形成以下对照：

- v02：`CE + Lovasz`。
- v03：`CE + Lovasz + ABL`。
- v04：`CE + RMI`。
- v05：`CE + RMI + ABL`。

其中 `v05-v04` 检查ABL在RMI方案上的增量，`v05-v03` 比较有ABL时RMI与Lovasz两种区域损失方案。

## 配置与主要改动

- 正式候选配置：`configs/aic_convnext_base_upernet_768.py`。
- 实验名：`v05_convnextb_rmi_abl_768`。
- 主头损失：`CrossEntropyLoss(1.0) + RegionMutualInformationLoss(0.5) + ActiveBoundaryLoss(0.1)`。
- RMI参数：3x3邻域、4倍平均池化、协方差稳定项 `1e-6`、Ignore255。
- ABL参数：预测边界采样上限1%、最大GT边界距离20、标签平滑0.2、Ignore255。
- 辅助头：保持 `CrossEntropyLoss(0.4)`。
- `aicseg/rmi_core.py` 与 `aicseg/rmi_loss.py` 提供RMI实现和注册。
- `aicseg/abl_core.py` 与 `aicseg/abl.py` 提供ABL实现和注册；SciPy用于距离变换。
- ConvNeXt-B、UPerNet、D4、GN、ImageNet-21K分类预训练、Fold0、物理batch2/累积4、40k iter及768/512滑窗与v04一致。
- L配置作为B配置的同目录继承依赖保留，不是本轮正式训练入口。

RMI包含float64 Cholesky运算，ABL包含CPU距离变换及CPU/GPU数据转换。两项同时启用后的吞吐、显存、AMP行为和梯度稳定性必须以服务器smoke为准。目前没有精度提升证据。

## 服务器训练命令

公共数据检查、环境安装以及训练完成后的分析、预测和打包见根 `docs/setup.md`。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v05_convnextb_rmi_abl"
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

连接断开后先检查已有tmux会话，不要重复启动训练。未经明确授权不得启动超过300 iter或预计超过2小时的训练。

## 验证状态

- 已复用v03的ABL数值实现与测试，并复用v04的RMI数值实现与测试。
- 本地CPU PyTorch单元测试33项：31项通过，2项CUDA AMP测试因本机无CUDA跳过。
- 已覆盖ABL与RMI的Ignore处理、半精度输入、数值有限性和反向传播；额外完成 `CE + 0.5*RMI + 0.1*ABL` 合成输入的有限反向传播检查。
- 已静态核对解析前配置契约：主头损失顺序为CE、RMI、ABL，B配置具有独立实验名和run目录。
- 尚未完成MMSeg真实注册表构建、GPU前向、300 iter服务器smoke或正式训练。
- 暂无run、权重或精度结果；当前只能称为“代码已实现且本地合成测试通过”。

论文依据：

- Wang等，*Active Boundary Loss for Semantic Segmentation*，AAAI 2022，https://arxiv.org/abs/2102.02696 。
- Zhao等，*Region Mutual Information Loss for Semantic Segmentation*，NeurIPS 2019，https://proceedings.neurips.cc/paper/2019/hash/a67c8c9a961b4182688768dd9ba015fe-Abstract.html 。

本项目实现是对两种训练损失的独立适配，不使用外部分割训练数据或checkpoint。
