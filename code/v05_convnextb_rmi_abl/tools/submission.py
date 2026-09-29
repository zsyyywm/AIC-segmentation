"""Strict submission validation and deterministic ZIP creation."""

import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_SIZE = (1024, 1024)
ALLOWED_LABELS = set(range(9))
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _png_color_type(path):
    with path.open('rb') as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError(f'{path.name}: not a PNG file')
        length = struct.unpack('>I', handle.read(4))[0]
        chunk = handle.read(4)
        data = handle.read(length)
    if chunk != b'IHDR' or len(data) != 13:
        raise ValueError(f'{path.name}: invalid PNG IHDR')
    return data[9]


def validate_predictions(pred_dir, test_image_dir):
    pred_dir = Path(pred_dir)
    test_image_dir = Path(test_image_dir)
    expected = {path.name for path in test_image_dir.glob('*.png')}
    actual = {path.name for path in pred_dir.glob('*.png')}
    errors = []

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f'missing {len(missing)} files: {missing[:5]}')
    if extra:
        errors.append(f'extra {len(extra)} files: {extra[:5]}')

    for name in sorted(expected & actual):
        path = pred_dir / name
        try:
            with Image.open(path) as image:
                if image.size != EXPECTED_SIZE:
                    errors.append(f'{name}: size={image.size}, expected={EXPECTED_SIZE}')
                if image.mode != 'L':
                    errors.append(f'{name}: mode={image.mode}, expected=L')
                values = set(np.unique(np.asarray(image)).tolist())
                if not values.issubset(ALLOWED_LABELS):
                    errors.append(f'{name}: invalid labels={sorted(values - ALLOWED_LABELS)}')
            if _png_color_type(path) != 0:
                errors.append(f'{name}: PNG color type must be 0 (grayscale)')
        except Exception as exc:
            errors.append(f'{name}: {exc}')

    if len(expected) != 500:
        errors.append(f'test set has {len(expected)} PNG files, expected 500')
    if errors:
        preview = '\n'.join(f'- {item}' for item in errors[:30])
        raise ValueError(f'Submission validation failed:\n{preview}')
    return len(actual)


def make_zip(pred_dir, zip_path):
    pred_dir = Path(pred_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pred_dir.glob('*.png')):
            archive.write(path, arcname=path.name)
    return zip_path.resolve()
