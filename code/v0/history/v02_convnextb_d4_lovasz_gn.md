# V02 ConvNeXt-B D4 + Lovász + GN

更新：2026-09-28。负责人：待用户指定。来源版本：`code/baseline_v1`。

## 改进目标

这是优先提分的第一版组合模型，不承担单因素消融。它同时加入航拍方向不变性、与mIoU更接近的损失以及小物理batch下不依赖批统计的解码头归一化。

## 组合改动

- D4增强：用均匀 `RandomD4` 替换基线水平翻转。0°、90°、180°、270°旋转及对应镜像各占1/8；image和mask同步，使用数组旋转/翻转，不进行插值。
- 主头损失：`CrossEntropyLoss(1.0) + LovaszLoss(0.5)`；Lovász采用多类、当前批次出现类别、跨批次像素计算及 `reduction='none'`。
- 辅助头损失：保持原 `CrossEntropyLoss(0.4)`。
- 解码头归一化：UPerHead和FCNHead由BN改为32组GN；ConvNeXt主干不变。

## 配置与不变量

- 开发配置：`configs/aic_convnext_base_upernet_768.py`。
- 实验名：`v02_convnextb_d4_lovasz_gn_768`。
- 新增：`aicseg/transforms.py`；修改：`aicseg/__init__.py`和两份配置。
- 保持：ConvNeXt-B + UPerNet、ImageNet-21K分类预训练、Fold0验证、8类、Ignore255、768裁剪、现有尺度/颜色增强、物理batch2/累积4、40k iter、单尺度滑窗768/stride512。
- L配置只作为B配置的继承依赖保留；本轮正式候选是ConvNeXt-B配置。

## 服务器训练命令

命令格式与baseline完全一致，只将 `MODEL_DIR` 指向本版本；`CONFIG` 文件名保持不变。公共环境安装、数据检查、监控、分析和提交见根目录 `docs/setup.md`。

每个新终端先设置：

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v02_convnextb_d4_lovasz_gn"
export CONFIG=configs/aic_convnext_base_upernet_768.py
export OMP_NUM_THREADS=1
```

先执行不超过300 iter的smoke：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --smoke
```

smoke通过后，正式训练仍为40k iter：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG"
```

确认训练中断后，使用该实验的实际绝对run路径恢复：

```bash
cd "$MODEL_DIR"
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

`tail` 会持续占用当前终端；只在训练确实中断时执行恢复命令。连接断开后先检查已有tmux会话，不要重复启动训练。

smoke只验证链路，不作为正式模型成绩，也不从smoke续成正式训练。未经明确授权不得启动超过300 iter或预计超过2小时的训练。

## 验证状态

- 已完成本地语法、配置结构、D4八变换唯一性、image/mask同步性、标签集合保持及公共工具单元测试。
- 本机未安装MMCV/MMEngine/MMSeg/MMPReTrain，尚未完成真实注册表构建、GPU前向或300 iter smoke。
- 服务器先完成导入检查和不超过300 iter的smoke；通过后才可安排40k正式训练。
- 暂无正式run、权重或提升结论。
