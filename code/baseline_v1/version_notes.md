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

## 已有结果

- 首轮平台成绩：**64.4665**，来源为成员反馈。
- 首个提交包已通过500张同名预测、尺寸、8位灰度、标签范围和CRC检查。
- 实际run、配置快照、验证指标、日志和权重在服务器，本地未归档，完整关联见根 `result.md`。
- 共用工具已有语法与合成日志功能测试；这些证据不代替服务器实际训练及推理验证。
- 暂无已确认的改进结论，L是否优于B待验证。
