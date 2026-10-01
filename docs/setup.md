# 全局复现与训练后操作

适用：baseline及保持相同项目接口的改进版本。以下命令均在AutoDL Linux终端执行。本文件只维护所有版本共用的数据核验、环境复现、run分析、实验比较、测试推理和提交打包；**不维护任何版本的smoke、正式训练、恢复训练或训练监控命令**。这些命令与模型预算只写在对应版本的 `version_notes.md`。

## 1. 放置项目

推荐单卡4090/4090D 24GB、CPU至少8核、内存至少32GB、数据盘150GB；镜像Python3.10/PyTorch2.1.0/CUDA12.1。

标准项目放在 `/root/autodl-tmp/AIC`，保持 `code/`、`data/`、`tools/` 同级，MMSegmentation与各模型版本目录同级。初次部署需具备这些共享目录；标准项目已预先就位时，成员只上传自己的完整版本目录即可。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export OMP_NUM_THREADS=1
```

所用版本的 `MODEL_DIR`、`CONFIG`、训练预算及全部训练命令，以该版本 `version_notes.md` 为唯一依据。

## 2. 核验官方数据

```bash
cd "$AIC_ROOT"
python -m pip install Pillow
python tools/check_uploaded_dataset.py
echo $?
```

末尾出现PASS且退出码为0才算通过。默认完整检查训练5596对、验证1400对和测试500张，包括解码、尺寸、模式、标签范围及文件名配对；脚本不修改图片。改变 `--data-root` 只改变检查目标，不会同步修改模型配置。

## 3. 复现公共环境

先根据所用版本的 `version_notes.md` 设置 `MODEL_DIR`，再检查当前环境：

```bash
which python
python --version
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

标准环境为Python3.10、PyTorch2.1.0、torchvision0.16.0、CUDA运行时12.1，且GPU可用。只有PyTorch不匹配时安装指定版本：

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

安装公共框架及所选版本依赖：

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

通过条件：无依赖冲突，MMCV2.1.0、MMEngine0.10.7、MMSeg1.2.2可导入，且出现IMPORT PASS和CUSTOM IMPORT PASS。不要换用mmcv-lite、MMCV2.2或NumPy2；wheel安装失败时先核对Python、PyTorch、CUDA与下载源。

## 4. 指定已完成的run

训练入口会为每次实验创建独立目录。后续公共工具统一接受实际run路径，不根据版本名猜测结果：

```bash
export RUN_DIR=/absolute/path/to/completed_run
```

正式run至少应保留 `run_manifest.json`、`run_summary.json`、日志、解析配置、验证指标和checkpoint。smoke目录只用于链路检查，不能作为正式成绩或提交来源。

## 5. 绘图与单模型分析

向根 `result.md` 追加该run的总体分析：

```bash
python "$AIC_ROOT/tools/analyze_experiment.py" "$RUN_DIR"
```

重新生成已有日志的曲线：

```bash
python "$AIC_ROOT/tools/plot_training.py" "$RUN_DIR"
```

分析脚本读取run中的已有验证记录；当导出的历史run没有 `scalars.json` 时，可从原 `console.log` 回退解析，并在报告中标记来源。它不会在官方测试集上计算真实分数。代表值、最终值、AUC和每类IoU的含义见 `model.md`；不能只用最佳mIoU判断一个方案。该入口会追加 `result.md`，重复分析同一历史run前先检查是否已有记录。

## 6. 比较两个正式实验

```bash
python "$AIC_ROOT/tools/compare_experiments.py" \
  --baseline /absolute/path/to/baseline_run \
  --candidate /absolute/path/to/candidate_run
```

输入必须是两个具体run。工具会检查已记录的划分、种子、训练预算和评测方式，并把比较结果追加到根 `result.md`。条件不一致、日志缺失或训练未完成时，不能据此声称模型稳定提升。

## 7. 从run生成测试预测和ZIP

各版本训练完成后统一调用根目录入口，不需要手工进入版本目录执行预测脚本：

```bash
python "$AIC_ROOT/tools/test_and_pack.py" "$RUN_DIR"
```

该工具从run选择最佳checkpoint并读取相邻解析配置，再调用所属版本的底层推理入口。输出保存到：

```text
<RUN_DIR>/submission/<checkpoint_name>/
```

目录中包含500张预测PNG和同名ZIP。工具会检查文件名、1024尺寸、uint8灰度、PNG color type、官方标签范围和ZIP结构；目标目录非空时拒绝覆盖。无法从run推断配置时才显式指定：

```bash
python "$AIC_ROOT/tools/test_and_pack.py" /absolute/path/to/checkpoint.pth \
  --config /absolute/path/to/resolved_config.py
```

本地历史run移到 `runs/archives/v0/` 后，解析快照仍记载旧路径。对有实际checkpoint的归档，显式指定原快照和新的模型目录：

```bash
python "$AIC_ROOT/tools/test_and_pack.py" "$RUN_DIR" \
  --config /absolute/path/to/original/resolved_config.py \
  --model-dir "$MODEL_DIR"
```

本地v02、v04、v05归档没有checkpoint，不能仅凭日志中的checkpoint文件名执行推理；v03归档有实际最佳权重。命令执行前须核对归档manifest与所选模型目录的对应关系。

上传平台前记录实际run、checkpoint、配置、ZIP及其SHA256。ZIP生成成功只证明格式检查通过，不代表精度或全部赛规已经确认。

## 8. 保存与迁移实验产物

每个新正式run应保留console、manifest、summary、全部scalars、解析配置、曲线、最佳权重，以及恢复训练所需的最近checkpoint和 `last_checkpoint`。历史本地导出若缺失scalars或checkpoint，应如实列明，不能根据文件名补造。根 `result.md` 单独保留；版本差异和训练命令只更新对应 `version_notes.md`。

释放实例前备份run与提交产物，并确认数据盘保留方式。本地没有服务器日志时，不根据当前默认配置反推历史实验设置或宣称复现成功。
