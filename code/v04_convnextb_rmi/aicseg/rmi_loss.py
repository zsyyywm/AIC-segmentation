"""MMSegmentation adapter; pure numerical code lives in rmi_core.py."""

import math

from torch import nn
from mmseg.registry import MODELS

from .rmi_core import regional_mutual_information


@MODELS.register_module()
class RegionMutualInformationLoss(nn.Module):
    """Regional term only; configure CrossEntropyLoss separately.

    Pixel weighting/sampling is deliberately unsupported: masking a region
    cannot be implemented by multiplying the final scalar by a pixel weight.
    """

    def __init__(self, num_classes=8, radius=3, pool_size=4,
                 covariance_epsilon=1e-6, loss_weight=0.5,
                 loss_name='loss_rmi'):
        super().__init__()
        if num_classes < 2 or int(num_classes) != num_classes:
            raise ValueError('num_classes must be an integer >= 2')
        if radius < 1 or int(radius) != radius or pool_size < 1 or int(pool_size) != pool_size:
            raise ValueError('radius and pool_size must be positive integers')
        if not math.isfinite(covariance_epsilon) or covariance_epsilon <= 0:
            raise ValueError('covariance_epsilon must be finite and positive')
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError('loss_weight must be finite and nonnegative')
        if not loss_name.startswith('loss_'):
            raise ValueError('MMSeg loss_name must start with loss_')
        self.num_classes = int(num_classes)
        self.radius = int(radius)
        self.pool_size = int(pool_size)
        self.covariance_epsilon = float(covariance_epsilon)
        self.loss_weight = float(loss_weight)
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, target, weight=None, ignore_index=255,
                reduction_override=None, avg_factor=None, **kwargs):
        if weight is not None or avg_factor is not None:
            raise ValueError('RMI does not support pixel sampler weights / avg_factor')
        if reduction_override not in (None, 'mean'):
            raise ValueError('RMI uses fixed per-image mean / per-class sum reduction')
        return self.loss_weight * regional_mutual_information(
            pred, target, num_classes=self.num_classes, radius=self.radius,
            pool_size=self.pool_size, covariance_epsilon=self.covariance_epsilon,
            ignore_index=ignore_index)
