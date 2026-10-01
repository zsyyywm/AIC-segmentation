"""Metrics that expose both overall and per-class IoU to JSON logs."""

from collections import OrderedDict

import numpy as np

from mmseg.evaluation import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class AICIoUMetric(IoUMetric):
    """MMSeg IoU metric with machine-readable per-class IoU fields."""

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        if self.format_only:
            return metrics

        grouped = tuple(zip(*results))
        total_intersect = sum(grouped[0])
        total_union = sum(grouped[1])
        ious = (total_intersect / total_union).numpy() * 100
        for class_name, iou in zip(self.dataset_meta['classes'], ious):
            metrics[f'IoU/{class_name}'] = float(np.round(iou, 2))
        return OrderedDict(metrics)
