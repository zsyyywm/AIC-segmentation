# 服务器操作流程

适用：baseline_v1及保持相同接口的改进版本。以下命令均在AutoDL Linux终端执行。文件职责见根README，模型设置见所用版本的version_notes，数据规范见 `data_spec.md`。

## 1. 放置项目

推荐单卡4090/4090D 24GB、CPU至少8核、内存至少32GB、数据盘150GB；镜像Python3.10/PyTorch2.1.0/CUDA12.1。

项目放在 `/root/autodl-tmp/AIC`，保持code、data、tools同级，MMSegmentation与版本目录同级。不要只上传版本目录。

每个新终端先设置：

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/baseline_v1"
export CONFIG=configs/aic_convnext_base_upernet_768.py
export OMP_NUM_THREADS=1
```

新版本只把MODEL_DIR和CONFIG改为该版本实际值；以下命令保持不变。

## 2. 检查图片

```bash
cd "$AIC_ROOT"
python -m pip install Pillow
python tools/check_uploaded_dataset.py
echo $?
```

末尾PASS且退出码0才继续；FAIL先补传对应文件再重查。默认检查train5596对、val1400对和test500张，完整解码、尺寸、模式、标签范围及配对，不修改图片。换数据位置时先确认训练配置也已匹配，检查脚本的`--data-root`不会修改训练配置。

## 3. 配置环境

使用同一个激活的PyTorch环境。先检查镜像：

```bash
which python
python --version
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

应为Python3.10、torch2.1.0、torchvision0.16.0、torch运行时CUDA12.1、GPU可用True。只有PyTorch不匹配时执行：

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

安装依赖：

```bash
apt-get update
apt-get install -y tmux
cd "$AIC_ROOT"
python -m pip install "numpy==1.26.4" "opencv-python==4.11.0.86"
python -m pip install "mmcv==2.1.0" --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
python -m pip install "mmengine==0.10.7" "mmpretrain==1.2.0" "ftfy>=6.1,<7" regex
python -m pip install -r "$MODEL_DIR/requirements.txt"
python -m pip install -v -e code/mmsegmentation
python -m pip check
python -c "import torch, mmcv, mmengine, mmseg, mmpretrain, ftfy, regex; from mmcv.ops import nms; print(mmcv.__version__, mmengine.__version__, mmseg.__version__); assert torch.cuda.is_available(); print('IMPORT PASS:', torch.cuda.get_device_name(0))"
cd "$MODEL_DIR"
python -c "from tools._bootstrap import bootstrap; bootstrap(); import mmpretrain.models; import aicseg; print('CUSTOM IMPORT PASS')"
```

通过条件：无依赖冲突，MMCV2.1.0/MMEngine0.10.7/MMSeg1.2.2，出现IMPORT PASS和CUSTOM IMPORT PASS。不要换成mmcv-lite、MMCV2.2或NumPy2；wheel失败先查环境与网络，不直接改为源码编译。

## 4. 冒烟测试

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --smoke
```

记录打印的tmux会话名和RUN_DIR。默认300iter后验证全量1400张并保存；首次B预训练下载约440MB，验证也需等待。通过条件：有限loss、无OOM/NaN/数据错误、验证指标和checkpoint正常、训练结束生成summary。

每次验证及训练结束后自动刷新run中的`training_curves.png`。绘图失败会告警，不中止训练。smoke仅验链路，不作为正式模型成绩或正式续训起点。

## 5. 正式训练与20k续训

确认没有其他训练占用GPU，下面两种方式二选一。

B默认40k：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG"
```

B开发筛选先停20k：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --cfg-options train_cfg.max_iters=20000
```

20k保持原40k调度终点，适合随后续训。有持续收益时恢复到默认40k：

```bash
bash tools/start_train.sh --config "$CONFIG" --resume /absolute/path/to/20k_run
```

新训练自动创建独立run，续训写原目录。20k/40k是B的预算，不能直接套给L或物理batch不同的配置；耗时以实测训练与验证速度估算。

## 6. 查看训练、重连与恢复

离开训练界面按Ctrl+B，松开后按D，不要按Ctrl+C。连接断开后只重连，不启动第二个训练：

```bash
tmux ls
tmux attach -t 实际会话名
```

在另一个终端观察指定run：

```bash
export RUN_DIR=/absolute/path/to/run
python "$MODEL_DIR/tools/monitor_training.py" --run "$RUN_DIR"
tail -f "$RUN_DIR/console.log"
nvidia-smi
```

以上是三个独立查看命令，按需运行；tail会持续占住终端。监控终端Ctrl+C不影响另一个tmux中的训练。会话存在或GPU占用不等于训练一定正常，要结合日志判断。

只有训练确实中断时才恢复：

```bash
cd "$MODEL_DIR"
bash tools/start_train.sh --config "$CONFIG" --resume "$RUN_DIR"
```

中途20k实验若仍计划停20k，恢复命令加`--cfg-options train_cfg.max_iters=20000`；不加则采用当前配置默认预算。不带路径的resume选该实验最新正式run，多实验不推荐。未保存进度不能恢复，smoke与resume不可并用。

## 7. 绘图与单模型分析

训练结束后设置实际run目录，追加总体分析：

```bash
export RUN_DIR=/absolute/path/to/run
python "$AIC_ROOT/tools/analyze_experiment.py" "$RUN_DIR"
```

也可把位置参数换成run内具体pth。脚本合并该run续训日志，只读已有验证指标，向根`result.md`追加日期、身份、配置和总体指标；不覆盖历史。官方测试没有标签，不会在这里得到官方分数。

重新绘图或持续刷新：

```bash
python "$AIC_ROOT/tools/plot_training.py" "$RUN_DIR"
python "$AIC_ROOT/tools/plot_training.py" "$RUN_DIR" --watch
```

二选一。watch默认30秒刷新，训练结束后退出。窗口指标与含义见 `model.md`，不要只用最佳mIoU判断整体表现。

## 8. 比较两个模型

```bash
python "$AIC_ROOT/tools/compare_experiments.py" \
  --baseline /absolute/path/to/baseline_run \
  --candidate /absolute/path/to/candidate_run
```

结果追加到同一result。先检查条件缺失或不一致，再看代表值、最终值、AUC和每类变化；单次信号不保证显著提升。输入具体run，不是包含多个实验的版本目录。

## 9. 预测与提交

```bash
python "$AIC_ROOT/tools/test_and_pack.py" "$RUN_DIR"
```

run默认选最佳权重并读取相邻配置。可改为具体pth；也接受配置文件或模型目录，但会自动选对应最新正式run，正式提交优先明确run或权重。无法推断配置时额外传`--config /absolute/path/to/config.py`。

产物是该run的`submission/<checkpoint_name>/`，其中保留500张PNG及同名ZIP。目录非空时拒绝覆盖，不自动清空。自动检查成功后上传ZIP，记录平台分数并绑定实际权重、配置和ZIP哈希。

## 10. 保存产物

run内保留：console、manifest、summary、全部scalars、时间戳resolved_config、曲线、最佳权重；续训另保留最近iter checkpoint和last_checkpoint。根result单独保留，版本差异更新version_notes。

tmux只防连接断开，不防关机、GPU故障或磁盘满。释放实例前备份产物并确认数据盘保留方式。本地暂无服务器日志时不伪造指标或声称完整复现。
