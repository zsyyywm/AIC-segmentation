"""UPer-specific 512 <-> 256 adapter around the portable P core."""

import torch
from torch import nn

from mmseg.models.decode_heads import UPerHead
from mmseg.registry import MODELS

from .prototype_core import MultiPrototypeRegionContext


class UPerPrototypeAdapter(nn.Module):
    def __init__(self, channels, prototype_cfg):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(channels, 256, 1, bias=False),
            nn.GroupNorm(32, 256), nn.GELU())
        self.core = MultiPrototypeRegionContext(**prototype_cfg)
        self.output_proj = nn.Conv2d(256, channels, 1, bias=False)

    def project(self, features):
        return self.input_proj(features)

    def enhance(self, features, projected, valid_mask=None):
        if not self.core.enabled or not self.core.prototype_valid.any():
            return features
        enhanced = self.core(projected, valid_mask)
        # No bias or second gate: exactly zero delta gives the original path.
        return features.float() + self.output_proj(enhanced - projected.float()).float()


@MODELS.register_module()
class PrototypeUPerHead(UPerHead):
    def __init__(self, prototype_cfg=None, feature_stride=4, **kwargs):
        super().__init__(**kwargs)
        if self.channels != 512 or self.num_classes != 8:
            raise ValueError('UP v1 retains channels=512 and 8 effective classes')
        self.feature_stride = int(feature_stride)
        if self.feature_stride < 1:
            raise ValueError('feature_stride must be positive')
        cfg = dict(prototype_cfg or {})
        cfg.setdefault('ignore_index', self.ignore_index)
        # Adding P must not change RNG consumption for later original modules.
        with torch.random.fork_rng(devices=[]):
            self.prototype_adapter = UPerPrototypeAdapter(self.channels, cfg)

    def _geometry_mask(self, features, metas):
        mask = torch.ones((features.shape[0], 1, *features.shape[-2:]),
                          device=features.device, dtype=torch.bool)
        for index, meta in enumerate(metas):
            shape = meta.get('img_shape')
            if shape is None:
                continue
            rows = torch.arange(features.shape[-2], device=features.device)
            cols = torch.arange(features.shape[-1], device=features.device)
            # Only wholly geometric-valid cells participate in region pooling.
            height = int(shape[0]) // self.feature_stride
            width = int(shape[1]) // self.feature_stride
            mask[index, 0] = (rows[:, None] < height) & (cols[None, :] < width)
        return mask

    def _enhance(self, features, valid_mask=None):
        if not self.prototype_adapter.core.enabled:
            return features  # true bypass, without even a channel projection
        projected = self.prototype_adapter.project(features)
        return self.prototype_adapter.enhance(features, projected, valid_mask)

    def forward(self, inputs):
        features = super()._forward_feature(inputs)
        return self.cls_seg(self._enhance(features))

    def loss(self, inputs, batch_data_samples, train_cfg):
        features = super()._forward_feature(inputs)
        adapter = self.prototype_adapter
        if not adapter.core.enabled:
            return self.loss_by_feat(self.cls_seg(features), batch_data_samples)
        mask = self._geometry_mask(features, [sample.metainfo for sample in batch_data_samples])
        projected = adapter.project(features)
        enhanced = adapter.enhance(features, projected, mask)
        logits = self.cls_seg(enhanced)
        losses = self.loss_by_feat(logits, batch_data_samples)
        # Current predictions cannot see current GT-derived bank updates.
        adapter.core.update_from_labels(projected.detach(),
                                       self._stack_batch_gt(batch_data_samples), mask)
        return losses

    def predict(self, inputs, batch_img_metas, test_cfg):
        features = super()._forward_feature(inputs)
        mask = self._geometry_mask(features, batch_img_metas)
        logits = self.cls_seg(self._enhance(features, mask))
        return self.predict_by_feat(logits, batch_img_metas)
