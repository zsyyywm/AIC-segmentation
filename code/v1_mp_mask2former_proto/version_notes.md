# MP：ConvNeXt-B + Mask2Former + 同一个多原型区域语义模块 P

日期：2026-10-01。目录：`code/v1_mp_mask2former_proto/`。主配置：`configs/mp_convnextb_mask2former_proto_768.py`。实验名：`v1_mp_convnextb_mask2former_proto_768`。

## 1. 状态与假设

代码已实现；CPU/CUDA机制检查、分类预训练核验、完整768 AMP前向/反向、1024滑窗及官方小样本本地Runner smoke已通过。没有服务器smoke、正式训练、完整1400张验证或平台提交，**没有MP精度成绩**。正式训练为条件候选，由主控结合M0/UP完整验证结果决定。

本轮由主控统一发布GitHub源码。本版本保留训练/预测、MP自检、资源比较和smoke权重状态核验入口；已删除未使用的重复M0验证工具、一次性打包工具及旧设计草稿。

2026-10-02发布核查：本版本 `runs/`、`dist/` 当前已不存在，迁移计划中的根 `runs/diagnostics/20261001_mp_feasibility/` 也不存在。下文检查结果和run路径是开发时的历史记录；来源/冻结清单及根 `runs/release_preparation/20261001_233828_github/` 中统一发布复核摘要仍在，但原始smoke日志、配置、权重、逐项报告和交付收据目前未找到，不能宣称已完整归档。保留验证工具便于重新取得证据，本次发布不启动训练。

使用前提仍是标准AIC项目：同级共享MMSeg、根官方data/与公共tools/已就位，环境符合本版本requirements。拿到这个完整版本目录不需要其他实验目录；分类预训练由配置中的公开URL下载。依赖和数据未部署的空目录不能直接训练，服务器环境尚未验证也不能写成已通过。

假设：类别绑定的多原型可补充自由区域查询，改善困难荒地外观的漏检及背景/农田裸土误报。若整体mIoU不升、关键混淆未改善，或荒地召回增加伴随明显误报，则不支持该假设；P可能与区域查询聚合重复。

精确逐像素诊断来自v03，不是v04的真实混淆矩阵，不能证明head/backbone某一方是唯一原因。历史v04验证mIoU76.32、荒地IoU51.53不属于MP。v04实际解析配置SHA256：`09646cce458c49c1d72c434b6c0cb44e116da3ab343a05198cf6ebf7ab051e5f`。

## 2. 冻结来源

| 来源 | 冻结标识或SHA256 | 本版本处理 |
|---|---|---|
| M0核心 | `m0_core_20261001_r1` | 原清单保留为`m0_source_freeze.json`；状态为冻结当时快照 |
| M0 head | `2936fbbf5590c871b6ceb00ce89ab653c85c37134a2d79766d25637263ad96d4` | `aicseg/mask2former_head.py`字节一致 |
| M0优化器构造器 | `2aa7d9c957f4c1619d3d4b7b222ff12fb8205360c213dce6c21a5117a5fa25ee` | 字节一致，保留原分组 |
| UP P契约 | `aic-p-ema-v1` | 复制冻结时的UP契约快照，保留原字节；当前UP增加换行说明，机制字段一致 |
| UP P核心 | `9a34dcc0d0816990a6852965032c8ecaa427b4f00b66a90009262bd373e714bb` | `aicseg/prototype_core.py`字节一致 |
| UP诊断Hook | `a61e295d60002e9a72153ce344f47c7b0b2163f03115b76cae9577a408c7e4a5` | `aicseg/prototype_hooks.py`字节一致 |

MP契约快照SHA256为`c0f76bce46de8213c1df6aafd2210696abb62f61a952ce90d7ce68f709cd3ab0`。当前UP契约SHA256为`43d5dfdb28aeefbd0cabebb3dda162cf4bcf3a08a3719fa43cb3b9286a9de806`，仅新增`text_eol`说明字段；去除该字段后JSON完全一致，P机制、超参数及核心/Hook源码未变。MP保留来源快照原字节，不用最新说明文件覆盖已冻结快照。

`source_manifest.json`逐文件记录复制时的原路径、上游哈希和适配。M0原配置仅修改参考实验名/work_dir；MP继承本目录副本，不跨版本继承/import。新注册、入口默认配置、run来源记录和打包名称已明示。shell仅由CRLF转LF；Git默认`-text`保留M0遗留字节，P核心/Hook与shell固定LF，避免Windows checkout改变冻结SHA。统一发布准备沿用M0的`whitespace=blank-at-eol,space-before-tab,cr-at-eol`声明，使Git正确识别保留的CRLF；仅修改Git属性，冻结源码字节未变。

本版本冻结文件见`core_freeze_manifest.json`。ZIP内`delivery_manifest.json`另外覆盖全部交付源文件。核心后续变更须重新冻结，不能继续声称与UP为同一个P。

## 3. MP相对M0的差异

新增`aicseg/prototype_mask2former_head.py`，覆写本目录M0稳定的像素解码后接口：

```text
ConvNeXt-B -> M0 pixel_decoder
  mask_features: B×256×192×192（输入768时）
  memorys: B×256×24²、48²、96²
         |
  同一个P(mask_features, 几何有效区域)，单次调用
         |
  enhanced -> 各查询层原生mask预测
  delta = enhanced - mask_features.float()
         |
  共享无bias 256→256的1×1投影，恒等初始化
         |
  双线性缩放 + FP32残差加到三层memory
         |
  原生查询解码 -> 各层类别与掩码预测
```

P同时进入mask features和query memory，能影响区域类别预测。memory适配器65,536参数，P核心196,864参数，共新增**262,400**。MP总参数107,835,977，M0参考107,573,577，约增加0.244%。

关闭P时真正旁路，保留M0原对象/dtype并跳过统计更新；冷bank也返回原对象。构造适配器时保留M0初始化随机序列，配对资源检查核验共同初始state字节SHA一致。

UP接入UPer融合特征；MP接入pixel_decoder的stride4 mask features，另投影残差到三层query memories。这是接入位置/适配器差异，**未改变P机制、数量、维度、更新和监督**。

P为8类×每类4原型、256维、temperature0.1、EMA0.99、gate_init0.001、每类最多2048统计像素、Ignore255、eps1e-6；类别绑定的确定性最远点初始化、类内最近原型EMA，无额外损失。当前forward使用旧bank的脱离梯度快照；本批原生loss计算后从同次pixel_decoder输入提交一次统计，影响下一micro batch。全Ignore/缺席类别不更新对应状态。bank、valid、counts、seen_pixels、update_steps随checkpoint保存。

前向只读特征和img_shape几何，不读GT；有效区域按完整stride4单元构造，不用标签Ignore或滑窗残留整图pad_shape。eval/predict不更新。冻结核心只支持单GPU统计更新，不可直接改多卡DDP。

保留M0：ConvNeXt-B、同一个公开分类预训练、256通道、100查询、6层pixel encoder、9层query decoder、3层memory、8stuff/0thing；原生cls2/mask5/Dice5、no-object0.1、Hungarian匹配、全部10层/30项监督及Ignore/padding/FP32点损失修复。没有G、边界、额外损失、TTA、阈值或后处理。

AdamW lr1e-4/wd0.05、主干层倍率和原生embedding免衰减与M0一致；新增矩阵用head学习率/wd0.05，1D gate免衰减，未新增另一套规则。645个可训练参数张量全部且仅一次进入优化器。

## 4. 数据、依赖与预训练

固定官方train5596/val1400，官方1..8→内部0..7、官方0→Ignore255、seed2026。保持v04/M0的resize0.5–2、768裁剪、D4、PhotoMetricDistortion和slide768/stride512。验证不训练，测试仅推理；只使用官方训练数据和允许的公开分类主干权重，不加载外部分割/检测checkpoint。

前提为标准AIC的同级`code/mmsegmentation`1.2.2、根data、根公共tools及声明的Python环境；整个目录放入code同层即可运行，无需M0/UP/v0/v1源码或配置。

requirements保留M0：MMDetection3.3.0、MMPretrain1.2.0、MMEngine0.10.7、MMCV<2.2。未升级共享环境。本地检查显式借用M0已建立的隔离环境解释器，仅作为第三方依赖环境，报告断言实际aicseg在MP目录；交付不依赖该虚拟环境路径。实际验证Python3.10.20、Torch2.0.1+cu118、MMCV2.0.0、Windows、RTX4060 Laptop8GB，pip check通过。项目标准服务器Torch2.1/CUDA12.1/MMCV2.1仍待服务器smoke核验。

公开分类文件`convnext-base_3rdparty_in21k_20220301-262fd037.pth`，439,862,065bytes，SHA256 `262fd0376855955f20f6c036aa882f5cb22b88333b766b0fa20174339c11d70d`。已逐值核对340个主干tensor。该权重无norm0..norm3的8个weight/bias，按控制/M0初始化保留，未擅自补权重。正式模型重新从这个分类权重初始化，不接v04分割权重。

本地Runner使用显式TORCH_HOME指向本版本被忽略的`runs/classification_cache/torch`；前期单项诊断显式传入M0分类缓存文件，实际路径与命令保存在报告中。运行代码不存在隐藏缓存依赖。

## 5. 验证证据

下列路径均为开发时相对本版本的原始位置；不进入源码交付，当前可用性见开头发布核查。

| 检查 | 证据目录 | 结果 |
|---|---|---|
| CPU head | `runs/validation_20261001_cpu/` | 冷bank/旁路全部10层输出一致，固定点采样RNG的30项loss逐值一致；两条路径各自梯度有限非零 |
| 完整128 CPU | `runs/validation_20261001_model128_cpu_retry01/` | 340个预训练tensor匹配；两次反向、0次优化器更新 |
| CUDA head/128 AMP | `runs/validation_20261001_cuda128/` | 机制和完整模型通过，scale128不下降 |
| 完整768 AMP | `runs/validation_20261001_cuda768_retry01/` | B1，两次反向、0更新；allocated3727.19MiB/reserved3998MiB，梯度和参数分组通过 |
| 1024 slide | `runs/validation_20261001_slide1024/` | 768/512、活跃P、无GT，输出1×8×1024²有限且bank不变 |
| 官方Runner | `runs/v1_mp_convnextb_mask2former_proto_768/20261001_221720_smoke/` | 实际8micro、B1/累积4、2次AdamW更新、8次官方样本曝光；训练/验证各前2张，保存和验证完成 |
| 同协议资源 | `runs/resources_20261001_768/` | 合成输入/标签、同随机初始化共同参数，两个独立顺序进程，768/B1/AMP，各2次预热+5次测量、0优化器更新 |
| 摘取副本 | `runs/portable_check_20261001/` | ZIP解压至新的AIC/code层级，只接入共享MMSeg和声明的依赖环境，配置/注册/CPU head全部通过 |
| GitHub清理后复核 | `runs/upload_preparation_20261001/` | 从独立Git索引摘取31个源文件，CPU机制/自身import/配置通过，训练/预测/监控入口及pip check通过；未重跑GPU训练 |

head检查还覆盖全Ignore零损失/不更新、部分Ignore、类别缺席、padding、旧整图pad_shape、eval无GT/不更新、提交统计后安全反向，以及序列化完整head恢复全部输出/buffers。分别对class和mask输出反向，P四组及memory投影均有非零梯度。旧摘取副本的报告当时保存到`runs/portable_check_20261001/evidence/`；这些报告、解压副本及其junction当前已不在版本目录，原始结果不因本次发布变为新验证。

官方smoke暂时使用训练/验证indices=[0,1]、batch1、workers0、固定loss_scale128.0、每micro日志/诊断及短调度；正式配置仍batch2/累积4、动态scale、40000micro。两张验证mIoU0、缺席类NaN是随机初始化8micro的链路输出，不记为精度成绩，短调度不能判断收敛。

`mp_checkpoint_verification.json`核验iter_8.pth：645个优化器参数状态均step2，scale128，原型更新8次、有效槽24、8条诊断；权重SHA256 `d1e1b6e2af1274ba9a5f5f1957da98bb985b4728e1000a053075b21cedaa9bf8`。少量样本缺席的类别没有伪造补齐。

本地迭代训练smoke总计**8micro/2次成功更新**；其余为0优化器更新的合成检查。累计执行计算/GPU时间远低于2小时，没有正式训练，也没有从smoke续成正式run。

保留失败：首次CPU预训练相对路径受bootstrap切目录影响，修复为按调用目录解析；一次768检查发现打包适配未写来源清单，补齐后重试；真实Runner `20261001_221141_smoke`在迭代前遇Windows GBK解码错误，PYTHONUTF8=1重试；`20261001_221653_smoke`在迭代前因整数loss_scale128被拒，改128.0。失败run不覆盖、不计作成功smoke。

## 6. 资源与待审批预算

| 同协议768/B1/AMP，不含AdamW状态 | M0 | MP | 增量 |
|---|---:|---:|---:|
| 参数 | 107,573,577 | 107,835,977 | 262,400 |
| allocated MiB | 3410.33 | 3724.68 | 314.35（9.22%） |
| reserved MiB | 3612 | 3966 | 354 |
| 前向/反向中位秒 | 0.44380 | 0.50059 | 12.79% |

共同初始state SHA256：`fe2a29166b0629bce33a9f9c1296c36b6650b644c1ef8caf107e7a49d19c952d`。P活跃时AMP保留FP32残差，不能仅用1MiB新增权重或4.5MiB亲和矩阵推算峰值。计时仅5次，笔记本温度/功率及顺序执行会影响结果，不是服务器时长保证。

实际AdamW smoke日志观察到训练allocated峰值4939MiB，训练/保存/两张验证阶段20.123秒。继承M0的RunStatsHook在验证后summary报1312.6MiB，不是整轮峰值：MMEngine Logger会重置CUDA峰值计数器。未擅自改冻结Hook；配对比较使用无Logger重置的专门工具。正式run需核对日志峰值或独立采样。

待审批正式预算：单GPUbatch2/累积4/有效batch8，10000次优化器更新、80000曝光，即40000micro；warmup750micro、每4000micro验证/保存，seed2026。AMP跳步须复核实际成功更新数。

**EMA按训练micro batch的有效统计更新，而非按优化器step更新**；全Ignore/无有效统计的批次不提交更新。后续若改变物理batch，即使维持总曝光量与优化器更新预算，原型更新频率仍可能不同。须换算总iter/warmup/间隔，并由主控同时核对UP/MP的EMA更新频率后决定比较口径；本次保持momentum0.99，不自行调整。

建议沿用M0的4090/4090D24GB、CPU8核/RAM32GB预算，先做正式尺寸batch2 smoke。M0的6–10GPU小时估计加本地约13%计算增量，可作为**7–12GPU小时初始预算建议**，不是实测；完整验证/保存及服务器环境会影响时长。batch2显存未实测，不承诺8GB卡可训，以获批服务器smoke修订。

## 7. 命令入口

标准服务器：先在独立环境装本版本requirements，不擅自升级共享环境。公共环境复现见根docs/setup.md，另须本版本MMDetection3.3.0。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR=$AIC_ROOT/code/v1_mp_mask2former_proto
export CONFIG=configs/mp_convnextb_mask2former_proto_768.py
export OMP_NUM_THREADS=1
cd "$MODEL_DIR"
python -m pip check
nvidia-smi
tmux list-sessions
python tools/validate_mp.py --device cpu --output runs/preflight_mp_unit
# 检查GPU/tmux和其他实验占用，排队后独立smoke：
bash tools/start_train.sh --config "$CONFIG" --smoke --smoke-iters 100
```

本地实际Runner命令；显式选择声明依赖的Python，复核会新建时间戳run：

```powershell
$env:PYTHONUTF8 = '1'
# 从本地标准AIC项目根目录进入：
cd code/v1_mp_mask2former_proto
$env:TORCH_HOME = Join-Path (Get-Location) 'runs/classification_cache/torch'
python tools/train.py --smoke --smoke-iters 8 --cfg-options train_dataloader.dataset.indices=[0,1] val_dataloader.dataset.indices=[0,1] train_dataloader.batch_size=1 train_dataloader.num_workers=0 train_dataloader.persistent_workers=False val_dataloader.num_workers=0 val_dataloader.persistent_workers=False optim_wrapper.loss_scale=128.0 default_hooks.logger.interval=1 custom_hooks.1.interval=1
python tools/verify_smoke.py runs/v1_mp_convnextb_mask2former_proto_768/20261001_221720_smoke
```

检查输出拒绝覆盖，复核用新目录。完整机制与资源检查：

```bash
python tools/validate_mp.py --device cuda --model-size 768 --amp --pretrained /absolute/path/to/classification_checkpoint.pth --output runs/recheck_mp_768
python tools/validate_mp.py --device cuda --amp --skip-head --slide-size 1024 --output runs/recheck_mp_slide1024
python tools/compare_resources.py --output runs/recheck_mp_resources --size 768 --repeats 5
```

以下正式/恢复命令**仅供主控审批后使用，本次未执行**。smoke不续成正式run：

```bash
bash tools/start_train.sh --config "$CONFIG"
# 仅恢复明确获批且中断的正式run：
bash tools/start_train.sh --config "$CONFIG" --resume /absolute/path/to/approved_formal_run
python tools/monitor_training.py --config "$CONFIG" --run /absolute/path/to/run --once
```

底层测试预测保持`tools/predict_and_pack.py --config "$CONFIG"`接口，显式指定checkpoint/新预测目录。公共分析和提交操作按根docs/setup.md的实际run入口。本次未运行测试提交。

## 8. 交付与待解决事项

版本开发时仅在本版本修改，采用独立Git索引交付到本地分支codex/v1-mp-mask2former-proto，首版提交61d79687beb5d614e39685e3ad62e868c1a36d67。2026-10-02由主控按用户授权统一更新main；模型、配置和冻结快照字节保持不变，原始验证证据的现存状态见开头发布核查。

GitHub只含configs/aicseg/tools、requirements、version_notes、来源/冻结清单和Git规则，不含runs/权重/图片/数据/环境/缓存/重复框架。历史源码ZIP当前已不在dist；上传不依赖ZIP或已删除的一次性打包工具。版本开发时曾摘取干净源码到临时标准层级，核验冻结哈希、配置、注册、自身import、CPU机制及训练/预测入口；只接入允许的共享MMSeg和声明环境。

清理后的GitHub上传单元为31个源文件，不包含私有机器路径或本地运行产物；实际自检工具保留便于服务器复核。上传前根README状态由主控统一更新，本版本分支不并发修改根文档，也不自行push/合并。

主控须依据M0/UP正式验证决定是否训练MP。正式验收主指标为固定全验证mIoU，保留每类IoU、荒地Precision/Recall、荒地↔背景/农田双向混淆、无荒地图像FP、冻结困难外观召回和边界8px外内部错误，失败也保留。无完整验证不宣称泛化、区域错误或赛事成绩改善。

参考：[MMDetection3.3.0 Mask2Former源码](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/mask2former_head.py)。只参考结构/源码，不加载外部分割/检测权重。
