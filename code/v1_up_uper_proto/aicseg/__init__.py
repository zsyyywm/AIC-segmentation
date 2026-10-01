"""AIC competition extensions for MMSegmentation."""

from .dataset import AICDataset
from .hooks import AICRunStatsHook
from .metrics import AICIoUMetric
from .transforms import RandomD4
from .rmi_loss import RegionMutualInformationLoss
from .prototype_uper_head import PrototypeUPerHead
from .prototype_hooks import PrototypeDiagnosticsHook

__all__ = ['AICDataset', 'AICIoUMetric', 'AICRunStatsHook', 'RandomD4',
           'RegionMutualInformationLoss', 'PrototypeUPerHead',
           'PrototypeDiagnosticsHook']
