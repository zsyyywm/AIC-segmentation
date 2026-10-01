"""Aerial-image transforms for the combined V02 experiment."""

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


def apply_d4(array, transform_id):
    """Apply one of the eight square symmetries without interpolation."""
    if not 0 <= transform_id < 8:
        raise ValueError('transform_id must be in [0, 7]')

    transformed = array
    if transform_id >= 4:
        transformed = np.flip(transformed, axis=1)
    rotations = transform_id % 4
    if rotations:
        transformed = np.rot90(transformed, k=rotations, axes=(0, 1))
    return np.ascontiguousarray(transformed)


@TRANSFORMS.register_module()
class RandomD4(BaseTransform):
    """Uniformly transform an image and its masks by one D4 group element.

    The eight choices are four right-angle rotations and the same rotations
    after a left-right reflection. Identity is included. Array indexing is
    used instead of resampling, so discrete mask labels are preserved.
    """

    def transform(self, results):
        transform_id = int(np.random.randint(0, 8))
        results['img'] = apply_d4(results['img'], transform_id)
        for key in results.get('seg_fields', []):
            results[key] = apply_d4(results[key], transform_id)
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
