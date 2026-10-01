"""MMSeg registry adapter; numerical implementation lives in abl_core.py."""

from mmseg.registry import MODELS

from .abl_core import ActiveBoundaryLossCore


@MODELS.register_module()
class ActiveBoundaryLoss(ActiveBoundaryLossCore):
    """Training-only loss compatible with BaseDecodeHead.loss_by_feat."""
