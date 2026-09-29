# Baseline V1

更新：2026-09-16。本文件只介绍baseline配置与结果，不包含开发或操作教程。

## 基线设置

- 框架：MMSegmentation 1.2.2。
- 模型：EncoderDecoder、ConvNeXt主干、UPerHead主分割头、FCNHead辅助头。
- 预训练：公开ImageNet-21K分类主干权重，不加载外部分割checkpoint。
- 数据：Fold0验证，训练5596张、验证1400张；种子2026。
- 输入与评测：训练裁剪768，单尺度滑窗768/stride512。
- 优化：AMP、AdamW，有效batch8。
- 类别：8个有效类，内部编号0..7，Ignore255，导出恢复官方1..8。

| 设置 | 默认B配置 | 保留L配置 |
|---|---|---|
| 文件 | `configs/aic_convnext_base_upernet_768.py` | `configs/aic_convnext_large_upernet_768.py` |
| experiment_name | `b0_convnextb_upernet_768` | `b0_convnextl_upernet_768` |
| 主干 | ConvNeXt-Base | ConvNeXt-Large |
| 物理batch / 累积 | 2 / 4 | 1 / 8 |
| 默认训练 | 40k iter | 80k iter |
| 验证及保存间隔 | 4k iter | 8k iter |
| 约等效epoch | 14.3 | 14.3 |

B配置继承L配置再覆盖B参数。表中是当前代码默认设置，不代表服务器该次实际运行的配置。

## 服务器训练命令

以下命令均在AutoDL Linux终端执行。公共环境安装、数据检查、监控、分析和提交见根目录 `docs/setup.md`。

每个新终端先设置：

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/baseline_v1"
export CONFIG=configs/aic_convnext_base_upernet_768.py
export OMP_NUM_THREADS=1
```

先执行不超过300 iter的smoke：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --smoke
```

smoke通过后，默认ConvNeXt-B正式训练为40k iter：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG"
```

如需先停在20k进行开发筛选：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --cfg-options train_cfg.max_iters=20000
```

确认训练中断后，使用该实验的实际绝对run路径恢复；20k实验若仍计划停20k，需继续附带相同的 `--cfg-options`：

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

## 已有结果

- 首轮平台成绩：**64.4665**，来源为成员反馈。
- 首个提交包已通过500张同名预测、尺寸、8位灰度、标签范围和CRC检查。
- 已将服务器正式run归档到本地 `runs/20260916_084306/`：ConvNeXt-B、seed2026、40k iter、8万样本曝光、1万优化器更新，RTX 4090 D训练2.5037小时。
- 该run最佳及最终验证mIoU均为74.41（iter40000）；最佳权重SHA256为 `1E4B89044B3698CDDDC993043E771472CBD5E2522918A87D37BABA6FA9A374C3`。
- 平台成绩与该run的对应关系来自现有成员记录，尚未重新执行“该checkpoint生成ZIP并上传”的完整链路验证；详细证据见根 `result.md`。
- 共用工具已有语法与合成日志功能测试；这些证据不代替服务器实际训练及推理验证。
- 暂无已确认的改进结论，L是否优于B待验证。
