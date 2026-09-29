# Copyright (c) OpenMMLab. All rights reserved.
#
# Learning snapshot only. It is not imported by the AIC training code.
# Source: https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/
#         mmpretrain/models/backbones/convnext.py
# License: Apache-2.0 (upstream MMPreTrain project).
#
# The AIC server setup installs mmpretrain==1.2.0. This file contains the
# two parts used by configs/aic_convnext_large_upernet_768.py:
# ConvNeXtBlock and ConvNeXt. The network structure and forward paths match
# v1.2.0; only verbose error messages and the unused default init_cfg are
# shortened for reading. Read the methods in this order:
#   1. ConvNeXtBlock.__init__
#   2. ConvNeXtBlock.forward
#   3. ConvNeXt.__init__
#   4. ConvNeXt.forward

from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import GRN, build_norm_layer
from mmpretrain.models.backbones.base_backbone import BaseBackbone


class ConvNeXtBlock(BaseModule):
    """ConvNeXt 的最小重复单元：一个带残差连接的特征变换模块。

    输入和输出的形状相同，都是 [N, C, H, W]。因此多个 Block 可以
    首尾相接组成一个 stage。它本身不改变特征图尺寸，也不改变通道数；
    改变尺寸和通道数的工作由 ConvNeXt 中的 downsample_layers 完成。
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        #drop_path_rate：训练时随机跳过残差分支的概率
        #layer_scale_init_value：残差分支输出的初始缩放系数
        #use_grn：是否使用 ConvNeXt V2 的 GRN；本项目为 False
        #with_cp：是否用重计算换显存；本项目为 False

        # __init__ 只负责“搭积木”：创建层、参数和开关；此时不处理图像。
        super().__init__()
        self.with_cp = with_cp

        # 1. 空间混合：7x7 depthwise convolution。
        # groups=in_channels 表示每个通道各自卷积，不和其他通道混合。
        # 它负责观察一个位置周围更大的空间邻域。
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        # 2. 归一化：默认 LN2d。本项目的 ConvNeXt 默认使用 LayerNorm，
        # 不要和 UPerHead 配置中的 BatchNorm 混淆。
        self.norm = build_norm_layer(norm_cfg, in_channels)

        # 3. 通道混合：先 C -> 4C，再 4C -> C。
        # 这一步相当于逐像素执行小型 MLP；不改变 H、W，只交换通道信息。
        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.grn = GRN(mid_channels) if use_grn else None

        # 4. LayerScale：每个通道一个可学习缩放系数 gamma。
        # 当前项目把初值设为 1.0，残差分支一开始不会被刻意缩小。
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        # 5. DropPath：训练时随机让整个残差分支暂时不参与；验证/测试时关闭。
        # 它是正则化手段，避免大量 Block 过拟合。
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """让一份特征经过一个 Block，输出同形状的新特征。

        主路径是：
        x -> depthwise conv -> LayerNorm -> 通道 MLP -> LayerScale
          -> DropPath -> 与原始 x 相加。
        """

        def _inner_forward(x):
            # shortcut 保存输入，最后形成 residual connection：output = input + branch。
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                # nn.Linear 作用在最后一维，故把 NCHW 暂时排成 NHWC。
                # 形状例子：[1, 192, 192, 192] -> [1, 192, 192, 192]。
                # 数值形状在这一层恰好相同，但“第 2 维/最后一维”的语义变了。
                x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
                # 对每个空间位置的 C 维向量做 LayerNorm。
                x = self.norm(x, data_format='channel_last')
                # 每个位置独立执行 C -> 4C -> GELU -> C 的通道混合。
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                # 变回分割框架后续使用的 NCHW 格式。
                x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                # [C] 变成 [1, C, 1, 1]，再按通道广播乘到整张特征图。
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            # 残差连接让 Block 学“在输入基础上要补充什么”，而非重建全部输入。
            return shortcut + self.drop_path(x)

        if self.with_cp and x.requires_grad:
            # activation checkpoint 用算力换显存；当前配置未启用。
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


@MODELS.register_module()
class ConvNeXt(BaseBackbone):
    """把多个 ConvNeXtBlock 组装成四级特征提取主干。

    本项目选择 arch='large'，所以会构造四个 stage：
    stage 0: 3 个 Block，通道 192，分辨率为输入的 1/4；
    stage 1: 3 个 Block，通道 384，分辨率为输入的 1/8；
    stage 2: 27 个 Block，通道 768，分辨率为输入的 1/16；
    stage 3: 3 个 Block，通道 1536，分辨率为输入的 1/32。

    forward 的返回值是四张保留空间位置的特征图，供 UPerHead 解码，
    而不是分类任务中常见的一条全局特征向量。
    """

    arch_settings = {
        'atto': {'depths': [2, 2, 6, 2], 'channels': [40, 80, 160, 320]},
        'femto': {'depths': [2, 2, 6, 2], 'channels': [48, 96, 192, 384]},
        'pico': {'depths': [2, 2, 6, 2], 'channels': [64, 128, 256, 512]},
        'nano': {'depths': [2, 2, 8, 2], 'channels': [80, 160, 320, 640]},
        'tiny': {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]},
        'xlarge': {'depths': [3, 3, 27, 3], 'channels': [256, 512, 1024, 2048]},
        'huge': {'depths': [3, 3, 27, 3], 'channels': [352, 704, 1408, 2816]},
    }

    def __init__(self,
                 arch='large',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=None):
        # __init__ 负责依据配置创建四个下采样层、四个 stage 和输出归一化层。
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings
            arch = self.arch_settings[arch]
        else:
            assert 'depths' in arch and 'channels' in arch

        # depths 决定每个 stage 堆多少个 Block；channels 决定其通道数。
        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels))
        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence)
        # 例如当前配置 [0, 1, 2, 3]：四个 stage 的结果都输出给 UPerHead。
        self.out_indices = [index + 4 if index < 0 else index
                            for index in out_indices]

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # 为全部 Block 分配逐渐增大的 DropPath 概率。
        # ConvNeXt-L 共 36 个 Block，概率从 0 线性增长至配置中的 0.4。
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # downsample_layers[0] 是 stem：输入图先以 4x4、步长 4 的卷积降采样。
        # 768x768 -> 192x192，同时通道数 3 -> 192。
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # stages 中的每个元素是一串 Block；同一个 stage 内形状保持不变。
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                # stage 1/2/3 前使用 2x2、步长 2 卷积：尺寸减半、通道数增加。
                # 例如 192x192x192 -> 96x96x384。
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            # 按 depth 创建连续 Block；每个 Block 的输入输出维度均为 channels。
            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth
            self.stages.append(stage)

            if i in self.out_indices:
                # 每个需要交给解码器的 stage 都有独立的最终 LayerNorm。
                self.add_module(
                    f'norm{i}', build_norm_layer(norm_cfg, channels))

        self._freeze_stages()

    def forward(self, x):
        """从图像张量提取多尺度特征图。

        对输入 [1, 3, 768, 768]，AIC 的 ConvNeXt-L 返回：
        [1, 192, 192, 192]、[1, 384, 96, 96]、
        [1, 768, 48, 48]、[1, 1536, 24, 24]。
        """
        outs = []
        for i, stage in enumerate(self.stages):
            # 先进入该 stage 对应的下采样层，再经过该 stage 的多个 Block。
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    # 分类任务分支：空间维度 H、W 被平均成 1x1，位置丢失。
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # AIC 分割任务分支：保留 H、W，UPerHead 才能知道像素位置。
                    outs.append(norm_layer(x))
        return tuple(outs)

    def _freeze_stages(self):
        """冻结最前面的若干 stage，禁止其参数更新。

        当前配置未设置 frozen_stages，因此默认是 0，循环不会执行。
        """
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        """切换训练/验证模式后，再确保被要求冻结的 stage 保持冻结。"""
        super().train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """为某个参数返回层深度，供 layer-wise learning-rate decay 使用。

        当前配置的 LearningRateDecayOptimizerConstructor 会调用此函数：
        一般越靠前的层学习率越小，越靠后的层和解码器学习率越大。
        这不是图像前向传播路径，初读主干时可先跳过。
        """
        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:
                layer_id = max_layer_id
        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:
                layer_id = max_layer_id
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
