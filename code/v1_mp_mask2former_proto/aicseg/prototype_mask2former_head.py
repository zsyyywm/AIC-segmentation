"""MP adapters around the frozen M0 head and the byte-identical UP P core."""

import torch
from torch import nn
from torch.nn import functional as F

from mmseg.registry import MODELS

from .mask2former_head import AICMask2FormerHead
from .prototype_core import MultiPrototypeRegionContext


class Mask2FormerPrototypeAdapter(nn.Module):
    """One P call; a shared channel map aligns its delta with query memories."""

    def __init__(self, prototype_cfg):
        super().__init__()
        self.core = MultiPrototypeRegionContext(**prototype_cfg)
        self.memory_proj = nn.Conv2d(256, 256, 1, bias=False)
        with torch.no_grad():
            self.memory_proj.weight.copy_(torch.eye(256)[:, :, None, None])

    def forward(self, mask_features, memories, valid_mask):
        if not self.core.enabled or not self.core.prototype_valid.any():
            return mask_features, memories  # original objects, including dtype
        enhanced = self.core(mask_features, valid_mask)
        delta = enhanced - mask_features.float()
        # AMP must not round away the small gate before residual addition.
        aligned = self.memory_proj(delta)
        outputs = []
        for memory in memories:
            if memory.shape[1] != 256:
                raise ValueError('MP expects the frozen 256-channel memories')
            residual = F.interpolate(aligned.float(), size=memory.shape[-2:],
                                     mode='bilinear', align_corners=False)
            outputs.append(memory.float() + residual)
        return enhanced, outputs


@MODELS.register_module()
class AICPrototypeMask2FormerHead(AICMask2FormerHead):
    """M0 forward/loss unchanged except its post-pixel-decoder extension point.

    Ground truth is used only for the inherited losses and a subsequent
    statistics commit. Ordinary forward/predict never read annotations.
    """

    def __init__(self, prototype_cfg=None, feature_stride=4, **kwargs):
        if kwargs.get('feat_channels') != 256 or kwargs.get('out_channels') != 256:
            raise ValueError('MP retains M0 feat_channels/out_channels=256')
        super().__init__(**kwargs)
        if self.num_classes != 8 or feature_stride != 4:
            raise ValueError('MP retains 8 effective classes and feature stride4')
        self.feature_stride = feature_stride
        cfg = dict(prototype_cfg or {})
        cfg.setdefault('ignore_index', self.ignore_index)
        # Leave the original module initialisation RNG sequence unchanged.
        with torch.random.fork_rng(devices=[]):
            self.prototype_adapter = Mask2FormerPrototypeAdapter(cfg)
        self._collect_prototype_update = False
        self._pending_prototype_update = None

    def _geometry_mask(self, features, batch_data_samples):
        mask = torch.ones((features.shape[0], 1, *features.shape[-2:]),
                          dtype=torch.bool, device=features.device)
        if batch_data_samples is None:
            return mask
        if len(batch_data_samples) != features.shape[0]:
            raise ValueError('One metadata sample is required per feature image')
        rows = torch.arange(features.shape[-2], device=features.device)
        cols = torch.arange(features.shape[-1], device=features.device)
        for index, sample in enumerate(batch_data_samples):
            shape = sample.metainfo.get('img_shape')
            if shape is None:
                continue
            # Like UP: only wholly valid stride4 cells. Never use GT Ignore or
            # an old full-image pad_shape on the current sliding window.
            height = max(0, int(shape[0]) // self.feature_stride)
            width = max(0, int(shape[1]) // self.feature_stride)
            mask[index, 0] = (rows[:, None] < height) & (cols[None, :] < width)
        return mask

    def _process_pixel_decoder_outputs(self, mask_features,
                                       multi_scale_memorys, batch_data_samples):
        if not self.prototype_adapter.core.enabled:
            return super()._process_pixel_decoder_outputs(
                mask_features, multi_scale_memorys, batch_data_samples)
        valid_mask = self._geometry_mask(mask_features, batch_data_samples)
        if self._collect_prototype_update:
            if self._pending_prototype_update is not None:
                raise RuntimeError('MP expects one pixel decoder call per loss')
            self._pending_prototype_update = (mask_features.detach(), valid_mask)
        return self.prototype_adapter(mask_features, multi_scale_memorys, valid_mask)

    def loss(self, x, batch_data_samples, train_cfg):
        if not self.training or not self.prototype_adapter.core.enabled:
            return super().loss(x, batch_data_samples, train_cfg)
        if self._collect_prototype_update:
            raise RuntimeError('Reentrant MP loss is not supported')
        self._collect_prototype_update = True
        self._pending_prototype_update = None
        try:
            losses = super().loss(x, batch_data_samples, train_cfg)
            if self._pending_prototype_update is None:
                raise RuntimeError('M0 pixel decoder extension point was not called')
            features, valid = self._pending_prototype_update
            labels = torch.stack([sample.gt_sem_seg.data
                                  for sample in batch_data_samples])
            # Current predictions use old bank snapshots; this updates only
            # the next micro batch. The frozen core handles Ignore and AMP.
            self.prototype_adapter.core.update_from_labels(features, labels, valid)
            return losses
        finally:
            self._collect_prototype_update = False
            self._pending_prototype_update = None
