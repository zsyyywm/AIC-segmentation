# RGB DenseCRF 后处理独立分支

日期：2026-09-28。只增加本目录；未修改 v1、v02、公共工具、框架、数据或默认提交入口。不训练，不自动启动服务器任务。

## 方法与边界

- 这是固定参数 RGB DenseCRF 迁移适配，不是陈瑜论文 R-ConvCRF 复现；没有热红外核和可学习参数。
- 输入是原始 1024×1024 RGB 与 MMSeg 原图对齐的 8 类 float32 softmax 概率。沿用传入配置的单模型推理（包括 slide 设置），逐图检查概率 argmax 与 MMSeg 原生预测一致；不使用彩色结果或硬标签作 unary。
- 固定 Fold0 的 1400 张：按 `SHA256('AIC-CRF-2026:'+文件名)` 最小 200 张筛选，剩余 1200 张另报。不能换划分追分。
- 固定 8 组参数：迭代5，Gaussian sxy=3，compat∈{1,3}；bilateral sxy∈{20,40}，srgb=13，compat∈{2,4}。这些是预先限定的工程候选，不是论文最佳参数。
- 筛选前二在全部1400张评测；全量和剩余1200张 mIoU 都严格提高，且两者车辆 IoU 均不下降超过1个百分点，才允许选择。通过者按全量mIoU排序；无通过者冻结 `selected=null`，保留原始 argmax。
- mIoU 使用全局8×8混淆矩阵，官方0不参与，官方1..8映射内部0..7；零并集使用 NaN 排除，与当前 MMSeg nanmean 口径一致，JSON中写null。
- 最終结果依然来自一个 checkpoint；试验不改变训练、不读取测试标签、不通过测试图片筛选参数。测试概率导出必须提供同 checkpoint、同解析配置的最终验证报告。

## 入口及产物

唯一入口 `crf_pipeline.py`，含 `export` / `evaluate` / `apply`。所有输出目录须不存在、JSON须不存在；不自动覆盖、清空或续跑半成品缓存。中断的缓存保留用于排查，重新运行换一个新路径。

缓存记录 checkpoint SHA256、继承解析后的配置哈希及快照、输入图片/标签哈希、划分、类别顺序、每张概率哈希、命令、Python/Torch版本和完成标记。评测前复核所有输入及概率内容，文件变动即拒绝；完整float32缓存1400张约43.75GiB，500张约15.63GiB，另需文件系统及临时空间。哈希复核有额外磁盘读取开销。

资源报告区分模型推理时间、整体墙钟、CRF/argmax处理时间和图像I/O；CPU RSS 峰值是操作系统记录的整个进程高水位，不能解释为某个候选独占内存。模型导出记录 CUDA allocator 峰值 allocated/reserved（不等于整卡显存）。CRF评测为CPU实现，不宣称测得GPU峰值0。

`apply` 只应用最终报告冻结的参数。测试输出复用现有 baseline 提交检查和打包工具，记录PNG/ZIP哈希；不向官网自动提交。现有版本推理入口和默认无后处理行为完全不变。

## 服务器操作示例

以下路径须替换为真实 checkpoint 与独立新输出路径。仅在现有MMSeg环境执行；本目录 requirements 的可选 C++ 后端需先安装、验证编译，未安装会明确报错，不会悄悄退化成无CRF。

```bash
cd /root/autodl-tmp/AIC
export MODEL_DIR=/root/autodl-tmp/AIC/code/v02_convnextb_d4_lovasz_gn
export CONFIG="$MODEL_DIR/configs/aic_convnext_base_upernet_768.py"
export CHECKPOINT=/absolute/path/to/best_checkpoint.pth
export CRF_RUN=/absolute/path/to/new_crf_experiment
# 仅创建此次实验容器目录；各export/apply输出子目录由程序独占创建。
mkdir "$CRF_RUN"
python code/postprocess_rgb_crf/crf_pipeline.py export --model-dir "$MODEL_DIR" --config "$CONFIG" --checkpoint "$CHECKPOINT" --split val --out "$CRF_RUN/val_cache"
python code/postprocess_rgb_crf/crf_pipeline.py evaluate --cache "$CRF_RUN/val_cache" --stage baseline --out "$CRF_RUN/baseline.json"
python code/postprocess_rgb_crf/crf_pipeline.py evaluate --cache "$CRF_RUN/val_cache" --stage screen --out "$CRF_RUN/screen.json"
python code/postprocess_rgb_crf/crf_pipeline.py evaluate --cache "$CRF_RUN/val_cache" --stage final --screen "$CRF_RUN/screen.json" --out "$CRF_RUN/final.json"
```

先核对 baseline 与同权重原生验证结果（浮点未舍入值 vs 日志两位小数），不一致时停止参数筛选。也可将 MODEL_DIR、CONFIG、CHECKPOINT 指向v1做链路验证；不能混用v1与v02报告。参数预算只8组，不通过修改常量反复追验证分。

确认最终报告后，独立应用测试集，不调参：

```bash
python code/postprocess_rgb_crf/crf_pipeline.py export --model-dir "$MODEL_DIR" --config "$CONFIG" --checkpoint "$CHECKPOINT" --split test --frozen "$CRF_RUN/final.json" --out "$CRF_RUN/test_cache"
python code/postprocess_rgb_crf/crf_pipeline.py apply --cache "$CRF_RUN/test_cache" --frozen "$CRF_RUN/final.json" --out "$CRF_RUN/test_output"
```

## 验证状态与后续验收

本地单元测试命令：`python -m unittest discover -s code/postprocess_rgb_crf/tests -v`。

已实现标签映射、概率检查、子集、筛选门槛、证据哈希和拒绝覆盖；单元测试覆盖合成数组。真实 MMSeg 模型导出、pydensecrf C++后端、1024图像资源和全量数据涨分仍须服务器验证。没有CRF实验成绩，不宣称提升。

服务器必须先核验：同checkpoint的原生验证与cache baseline一致；少量真实图的CRF正常执行；再按200/1400流程评测。CRF阈值是项目工程取舍，不是统计显著性检验；剩余1200张属于固定验证集内部确认，不是独立测试集。

## 原始依据

- Krähenbühl & Koltun, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, NIPS2011：https://arxiv.org/abs/1210.5644
- PyDenseCRF官方接口（DenseCRF2D、unary负对数概率、RGB bilateral）：https://github.com/lucasb-eyer/pydensecrf
- 陈瑜论文第三章核验摘要由用户提供：原版辐亮度核依赖热红外；本分支不声称直接复现。DOI：10.27389/d.cnki.gxadu.2025.000046。
