"""Check uploaded PNG completeness. Prints results; never changes dataset files."""

import argparse
import sys
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data')
    root = parser.parse_args().data_root.resolve()
    expected = {'train/images': 5596, 'train/masks': 5596,
                'val/images': 1400, 'val/masks': 1400, 'test/images': 500}
    errors = []
    names = {}
    checked = 0
    print(f'DATA: {root}', flush=True)
    for folder, count in expected.items():
        files = sorted((root / folder).glob('*.png'))
        names[folder] = {path.name for path in files}
        print(f'{folder}: {len(files)} / expected {count}', flush=True)
        if len(files) != count:
            errors.append(f'{folder}: file count differs')
        mode = 'L' if folder.endswith('/masks') else 'RGB'
        for index, path in enumerate(files, 1):
            try:
                with Image.open(path) as image:
                    if image.format != 'PNG':
                        raise ValueError('Not a PNG file')
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    if image.size != (1024, 1024) or image.mode != mode:
                        raise ValueError(f'Size/mode={image.size}/{image.mode}, '
                                         f'expected (1024, 1024)/{mode}')
                    if mode == 'L':
                        low, high = image.getextrema()
                        if low < 0 or high > 8:
                            raise ValueError(f'Mask labels out of range 0..8: {low}..{high}')
            except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
                errors.append(f'{folder}/{path.name}: {exc}')
            checked += 1
            if index % 500 == 0:
                print(f'  checked {index}/{len(files)}', flush=True)
    for split in ('train', 'val'):
        images, masks = names[f'{split}/images'], names[f'{split}/masks']
        errors.extend(f'{split}: missing mask {name}' for name in sorted(images - masks))
        errors.extend(f'{split}: missing image {name}' for name in sorted(masks - images))
    if names['train/images'] & names['val/images']:
        errors.append('train/val: overlapping filenames')
    if errors:
        print(f'FAIL: {checked} PNG files checked, {len(errors)} errors')
        for error in errors[:20]:
            print(f'ERROR: {error}', file=sys.stderr)
        if len(errors) > 20:
            print('Only the first 20 errors are displayed.', file=sys.stderr)
        return 1
    print(f'PASS: {checked} PNG files complete; counts, pairs, decoding, '
          'size, mode and mask label range are valid.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
