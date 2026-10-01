"""Dataset definition for the AIC low-altitude semantic segmentation task."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class AICDataset(BaseSegDataset):
    """AIC dataset with official label 0 ignored.

    Source masks use IDs 0..8. ``reduce_zero_label=True`` converts official
    ID 0 to 255 (ignore) and IDs 1..8 to the model's contiguous IDs 0..7.
    MMSeg's prediction formatter restores the +1 offset on export.
    """

    METAINFO = dict(
        classes=(
            'Background', 'Building', 'Road', 'Water', 'Barren',
            'Vegetation', 'Agricultural', 'Vehicle'),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
            [159, 129, 183],
            [0, 255, 0],
            [255, 195, 128],
            [0, 255, 255],
        ])

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            ignore_index=255,
            **kwargs)
