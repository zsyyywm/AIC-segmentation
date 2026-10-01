# M0：ConvNeXt-B + Mask2Former 区域查询对照

日期：2026-10-01。版本目录为 `code/v1_m0_mask2former/`，来源为 v04 等价控制 `code/v1/`。本版本只实现完整 Mask2Former 路线，没有 P 或其他增强模块。

2026-10-02发布核查：本版本的 `runs/`、`.venv/`、`.cache/` 当前均不存在，迁移计划中的根 `runs/diagnostics/20261001_m0_feasibility/` 也不存在。下文检查结果和run路径是开发时的历史记录；冻结清单及根 `runs/release_preparation/20261001_233828_github/` 中统一发布复核摘要仍在，但原始smoke权重、日志与逐项报告目前未找到，不能宣称已完整归档。接收者可用保留的验证工具重新取得证据；本次上传没有重新训练。

## 1. 状态与假设

| 阶段 | 状态与证据 |
|---|---|
| 代码和独立配置 | 已完成；核心冻结 `m0_core_20261001_r1`，文件哈希见 `core_freeze_manifest.json`；通过GitHub交付必要源码，不额外打包源码ZIP |
| 本地正确性 | 已通过 Ignore、原生前向/损失一致性、参数分组、分类预训练、768 输入、1024 滑窗、AMP 更新和标签导出检查 |
| 本地官方数据链路 | 已完成 4 micro iterations / 1 次更新，训练与验证各取 2 张；仅作为 smoke |
| 服务器环境和 smoke | 待验证；本次 SSH 连接被拒绝，未修改服务器 |
| 正式训练、全验证成绩、平台成绩 | 未执行，无新成绩，也没有提升结论 |

主要假设：区域查询、区域类别预测及掩码约束注意力可能减少背景/农田裸土的荒地误报和真实荒地内部类别破碎。精确错误归因来自 v03，不能当成 v04 的混淆矩阵。v04 历史 mIoU76.32、荒地IoU51.53仅作历史参照。本实验与 U0 比较完整分割头及配套监督方案，收益不能完全归因于结构。

## 2. 实际结构与控制差异

唯一训练配置：`configs/m0_convnextb_mask2former_768.py`。配置完整保存在本目录内，没有 `_base_` 或其他版本依赖。

| 项目 | U0 / v04 等价控制 | M0 | 差异性质 |
|---|---|---|---|
| 主干 | ConvNeXt-B、四层128/256/512/1024、stride4/8/16/32、drop_path0.4 | 完全相同 | 固定 |
| 预训练 | 指定公开 ImageNet21K 分类权重，prefix `backbone.` | 同一权重及加载方式 | 固定 |
| 主分割头 | UPerHead512、PPM和FPN | 256维；6层多尺度可变形注意力像素编码器；100查询；9层掩码约束查询解码器 | 结构 |
| 输出 | 8类像素 logits | 9维查询分类（8有效类+no-object），查询mask组合得到8类语义分数 | 结构 |
| 监督 | 主CE1/RMI0.5；辅助FCN CE0.4 | 查询CE2 / maskBCE5 / Dice5；no-object0.1；Hungarian代价2/5/5 | 配套训练 |
| 深监督 | FCN | 初始查询输出和9层解码输出，共10组、30项损失 | 配套训练 |
| 主干LR分组 | AdamW1e-4、WD0.05、betas0.9/0.999；stage_wise0.9、num_layers12 | 每个主干参数LR/WD与控制构造器逐项一致 | 固定 |
| 新头LR/WD | 原头参数组 | 新头LR倍率1；bias/1-D norm沿用免衰减；4种query/level embedding另行免衰减 | 必要参数分组 |
| 梯度裁剪 | 无 | 首版无；已检查AMP损失、更新和参数有限性 | 固定 |
| 增强、划分、标签、调度、验证 | v04协议 | 保持一致 | 固定 |
| Ignore与padding监督 | 像素损失排除255 | 版本内修复匹配和点损失的有效采样、全Ignore图像归一化 | 必要适配 |
| 滑窗元信息 | 普通像素头 | 本版本predict纠正残留整图pad_shape | 必要尺寸适配 |

主干参数87,568,256；完整模型107,573,577。保留公开分类初始化，`load_from=None`，不接着 v04 分割权重训练。没有 RMI、Lovasz、ABL、TTA、阈值拟合或后处理。

优化器保留控制的实际层倍率，而不是另套参考配置的 backbone0.1：downsample为0.9^13，stage0..3为0.9^12..0.9^9，输出norm和新头倍率1。四种免衰减embedding是 `pixel_decoder.level_encoding`、`query_embed`、`query_feat`、`level_embed`。640个可训练参数张量全部且仅一次进入参数组。

## 3. Ignore与推理适配

新组件都在本版本 `aicseg/`。`AICMask2FormerHead`继承共享MMSeg适配头，使用MMDetection3.3.0的完整像素/查询解码器、分类/BCE/Dice、Hungarian匹配及中间监督。

有效区域同时满足内部标签不为255、位于当前 `img_shape` 内。GT类别只从有效区域产生，官方背景对应内部0，no-object对应查询分类索引8，二者不同。

全有效图像保留原生连续随机点与不确定点采样。含Ignore/训练padding的图像在有效GT像素中心有放回采样：匹配使用12544点；掩码监督保留oversample3和importance0.75。不把255变成任何类别mask的负样本。全Ignore图像的分类、BCE、Dice均为可微零；混合batch中也从分类和mask归一化分母排除该图像，避免原生sampler将零正样本计为1。

匹配与掩码监督在AMP下显式使用FP32。注意力前向保留原生padding策略；本修复针对监督，不通过GT Ignore掩码改变特征提取，推理不依赖GT。

共享 `slide_inference` 改写窗口 `img_shape` 时可能残留整图 `pad_shape`。本版本只在slide预测中复制元信息并将pad_shape改为当前窗口尺寸，使8通道分数与窗口相同；不修改共享框架，也不原地修改传入元信息。1024整图slide768/stride512已实际通过。

## 4. 依赖和预训练核验

可摘取前提：同级共享 `code/mmsegmentation`（1.2.2）、根 `data/`、根公共 `tools/` 和已准备的Python环境。每个Python进程只加载本版本aicseg。新增依赖为MMDetection3.3.0，完整声明见requirements；没有重复框架。

MMDetection3.3.0与共享MMSeg1.2.2共同支持MMCV>=2.0.0rc4,<2.2.0和MMEngine<1.0；本版本固定MMEngine0.10.7、MMPretrain1.2.0。公共服务器建议沿用项目torch2.1/cu121、MMCV2.1环境，再在隔离环境准备本版本依赖，实际服务器状态仍待确认。

本地实际通过：Python3.10.20、torch2.0.1+cu118、MMCV2.0.0、MMEngine0.10.7、MMDetection3.3.0、共享MMSeg1.2.2、MMPretrain1.2.0；RTX4060 Laptop8GB。使用本版本`.venv`安装增量依赖，没有升级共享环境，`pip check`通过。完整模块来源写入验证JSON。

分类权重来自控制指定URL，SHA256：`262fd0376855955f20f6c036aa882f5cb22b88333b766b0fa20174339c11d70d`。340个主干张量逐项等于实际分类checkpoint。四个输出norm的8个张量不在该分类backbone权重中，保持控制版本初始化；全部主干state与独立构建并初始化的控制主干逐项一致。这不等于将整个分割网络预训练。

## 5. 已执行检查和失败记录

开发时的证据目录：本版本 `runs/validation_20261001/`，不随源码交付；当前可用性见开头发布核查。

| 证据 | 核验内容 |
|---|---|
| `unit_checks_retry02.json` | GPU Ignore、混合batch、单类、单有效像素、padding、30项监督、原生前向/损失一致、stale pad_shape |
| `unit_checks_final_retry01.json` | 上述CPU复核；真实LoadAnnotations的0→255、1..8→0..7映射；真实formatter恢复1..8灰度PNG |
| `model128_backward_retry01.json` | 完整结构128输入FP32反向；主干和新头梯度有限、参数组覆盖和主干LR/WD逐项等价 |
| `model768_pretrained_retry01.json` | 分类预训练加载和独立控制初始化一致；正式尺寸特征及语义输出 |
| `amp_optimizer_step.json` | 128输入4micro/1真实AdamW更新，所有参数有限 |
| `amp768_optimizer_step.json` | 768输入4micro/1真实AdamW更新；batch1峰值allocated3759.16MiB |
| `slide1024_pretrained.json` | 原配置1024整图slide768/512，输出8×1024×1024 |

AMP诊断使用初始loss scale128和单张合成输入，生产配置仍为dynamic、batch2/累积4；诊断不代表正式预算或精度。对同分辨率人工mask logits，Ignore位置梯度为0，扰动其预测不改变匹配和损失；这不意味着全局注意力模型的原始输入Ignore位置必须没有梯度。

真实本地Runner smoke：`runs/v1_m0_convnextb_mask2former_768/20261001_214222_smoke/`。4micro、batch1/累积4、1次更新、4次官方训练样本曝光；训练和验证各限前2张，workers0、loss scale128。正式768裁剪与完整结构保持不变，保存了原生run manifest、解析配置、日志、summary及smoke权重。峰值allocated3811.81MiB、reserved4080MiB，训练阶段20.847秒。两张验证的输出只核验链路，不用于成绩比较；没有全验证或平台成绩。

保留调试失败证据：全Ignore分支最初缺少AssignResult的max_overlaps；混合batch最初受sampler正样本下限影响，现已修复。验证工具最初缺default_scope、误认为norm3会被分类权重初始化、人工Loader fixture缺元信息，均已纠正。首次真实Runner smoke `20261001_214050_smoke`在训练前遇到Windows编译器环境日志GBK解码错误；以PYTHONUTF8=1重试通过。此前失败日志/JSON不覆盖。

累计迭代式检查12micro/3次优化器更新（两次合成AMP检查各4，真实Runner4），另有不更新优化器的梯度/前向检查；没有正式长训练。已有smoke不续成正式run。

## 6. 训练、恢复和验证命令

以下为标准服务器命令。服务器依赖、GPU、tmux及目标run核验通过后按队列执行；正式训练仍需主控/用户明确授权。首次服务器smoke建议100iter，连同本地检查仍低于本版本累计300iter上限。

```bash
export AIC_ROOT=/root/autodl-tmp/AIC
export MODEL_DIR="$AIC_ROOT/code/v1_m0_mask2former"
export CONFIG=configs/m0_convnextb_mask2former_768.py
export OMP_NUM_THREADS=1
cd "$MODEL_DIR"
python -m pip check
nvidia-smi
tmux list-sessions
python tools/validate_m0.py --stage unit --device cpu --output runs/preflight/unit.json
bash tools/start_train.sh --config "$CONFIG" --smoke --smoke-iters 100
# smoke通过、明确获批后，另建正式时间戳run：
bash tools/start_train.sh --config "$CONFIG"
# 只在指定正式run确实中断且获授权时恢复：
bash tools/start_train.sh --config "$CONFIG" --resume /absolute/path/to/formal_run
python tools/monitor_training.py --config "$CONFIG" --run /absolute/path/to/run --once
```

本地实际Runner短检查可按下述命令复核；会新建独立smoke，不覆盖旧结果。仅Windows需要PYTHONUTF8设置；先激活按本版本requirements准备的环境，使用当前python，不依赖开发者保留的`.venv`。

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
cd code/v1_m0_mask2former
$env:TORCH_HOME = Join-Path (Get-Location) '.cache/torch'
python tools/train.py --smoke --smoke-iters 4 --cfg-options train_dataloader.batch_size=1 train_dataloader.num_workers=0 train_dataloader.persistent_workers=False train_dataloader.dataset.indices=2 val_dataloader.num_workers=0 val_dataloader.persistent_workers=False val_dataloader.dataset.indices=2 optim_wrapper.loss_scale=128.0
```

可重复的诊断命令（拒绝覆盖已有output，复核请使用新目录）：

```bash
python tools/validate_m0.py --stage unit --device cpu --output runs/recheck/unit.json
python tools/validate_m0.py --stage model --device cuda --size 768 --pretrained --amp --output runs/recheck/amp768.json
python tools/validate_m0.py --stage model --device cuda --size 1024 --pretrained --output runs/recheck/slide1024.json
```

正式待审批预算：单卡物理batch2、累积4、有效batch8；40000micro、10000次名义更新、80000样本曝光（约14.30epoch），seed2026；warmup750micro，随后poly1.0至40000，验证/保存每4000micro。AMP溢出跳步须在正式run中复核，不能把名义更新次数冒充实测成功更新次数。batch变化时同时换算micro总数、warmup和验证/保存间隔，维持曝光及更新口径。

资源建议：4090/4090D24GB单卡、CPU8核、RAM32GB。batch1正式尺寸本地实测约3.72GiB allocated、3.98GiB reserved；batch2训练预计6—10GiB allocated、8—12GiB reserved，建议留足运行余量。该batch2估计未经服务器测量，不是上限保证。正式40k含验证预计6—10GPU小时，也仅是预算建议；与v04历史4.8859小时以及本地短检查设备/环境不同，不能直接换算为已测时长。

正式验收须用选定checkpoint重新生成完整1400张验证预测，与U0同一评测口径比较整体mIoU、每类IoU、荒地Precision/Recall、荒地↔背景/农田两向混淆、GT无荒地图像FP、冻结困难外观分组召回和边界8像素外错误。U0完整验证预测尚待补齐。不能只报荒地召回增加；也不能依据4iter smoke淘汰或宣称Mask2Former有效。

## 7. MP接口与摘取交付

稳定扩展方法：`AICMask2FormerHead._process_pixel_decoder_outputs(mask_features, multi_scale_memorys, batch_data_samples)`。调用位置为pixel_decoder之后、query memory展平和查询解码之前；M0返回原tensor/list对象。MP可在自己的头覆写，M0核心不要再修改。

| 接口 | 768输入实测 | 1024输入实测 |
|---|---|---|
| 四层backbone | B×128×192²、B×256×96²、B×512×48²、B×1024×24² | 空间尺寸256/128/64/32 |
| mask_features | B×256×192×192，stride4 | B×256×256×256 |
| multi_scale_memorys | 从低到高：B×256×24²、48²、96²，stride32/16/8 | B×256×32²、64²、128² |
| 展平query memory | B×(24²/48²/96²)×256，随后加level embedding | B×(32²/64²/128²)×256 |
| 查询输出 | 每层B×100×9和B×100×192²，共10组 | 每层mask空间256² |
| 最终语义分数 | B×8×当前窗口H×W | 完整slide结果B×8×1024×1024 |

交付渠道为GitHub版本源码。必要文件是 `configs/`、`aicseg/`、`tools/`、`requirements.txt`、本说明、冻结清单和Git忽略/换行规则。不会上传runs、权重、图片、提交预测、`.venv`、`.cache`或重复MMSeg。整个版本目录放在标准项目code同层，按根 `docs/setup.md` 准备共享环境及官方数据后，可以使用本版本入口；无需另一个实验版本或开发者机器路径。

2026-10-01上传准备曾删除多余的源码打包工具、两份源码ZIP、重复摘取源码目录及源码字节码缓存。当时记录的两份旧摘取副本验证产物位置为 `runs/validation_20261001/preserved_portable_check_20261001/` 和 `preserved_portable_final_check_20261001/`。截至2026-10-02，这些目录及原run、smoke权重、环境和分类缓存已不在版本目录中，计划归档目标也不存在；当前未确认其他备份，保留历史路径用于追溯。环境及缓存不是交付依赖。

`tools/validate_m0.py`是可重复的运行正确性检查，保留用于他人验收；`tools/check_and_pack.py`、`tools/submission.py`和预测入口属于赛事提交链路，也保留。这些提交ZIP与已经删除的多余源码ZIP不同。

上传适配只将 `tools/start_train.sh` 改为LF，并新增本版本`.gitattributes`：默认保留源码原始字节、shell固定LF，避免Windows Git自动换行破坏冻结哈希或Linux入口。模型、损失、优化器和训练预算未改。共享MMSeg和其他版本未修改。

2026-10-01发布前复核记录：临时Git索引只收入22个版本必要文件和根README；无数据、权重、run、缓存、环境或机器私有路径。从该索引实际checkout到独立标准目录，五个冻结核心哈希与工作目录、Git存储字节逐项一致；只提供同级MMSeg和声明环境，CPU检查通过，训练/预测/监控入口可解析，Linux启动脚本语法通过。实际aicseg来自checkout副本。原证据路径为本版本 `runs/release_preflight_20261001/source_audit.json` 与 `clean_git_unit.json`，目前未找到；统一复核摘要仍在根release_preparation。本轮无新的训练更新。

继承的RunStatsHook会受MMEngine Logger重置CUDA峰值计数器影响，正式run不能只用验证后summary估算整个训练峰值；需要核对训练日志或独立资源采样。本次未改变该统计Hook或已有证据。

参考结构为共享 `mask2former_r50_8xb2-160k_ade20k-512x512.py`；前向和损失基于 [MMDetection3.3.0原实现](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/mask2former_head.py)，论文为 [Mask2Former](https://arxiv.org/abs/2112.01527)。参考的ADE20K训练数据配置、外部分割权重和参考backbone0.1学习率均未引入。
