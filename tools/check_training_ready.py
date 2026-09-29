"""Read-only, real-framework preflight; never starts a training run.

Run in the server training environment. --forward adds one synthetic AMP
forward/backward at configured batch/crop, not an optimizer update or smoke.
Only standard dependency/pretrained download caches may be written.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(cfg):
    """Check round-one invariants on a resolved config, without dependencies."""
    model = cfg['model']
    require(model['backbone']['arch'] == 'base', 'Round one requires ConvNeXt-B')
    for name in ('decode_head', 'auxiliary_head'):
        head = model[name]
        require(head['num_classes'] == 8, f'{name}: expected eight classes')
        require(head['ignore_index'] == 255, f'{name}: expected Ignore255')
        require(head['norm_cfg']['type'] == 'GN'
                and head['norm_cfg']['num_groups'] == 32, f'{name}: expected GN32')
    require(cfg['train_dataloader']['batch_size'] == 2, 'Expected physical batch2')
    require(cfg['optim_wrapper']['accumulative_counts'] == 4, 'Expected accumulation4')
    require(cfg['train_cfg']['max_iters'] == 40000, 'Expected 40k formal config')
    require(cfg['train_cfg']['val_interval'] == 4000, 'Expected validation every4k')
    require(cfg['randomness']['seed'] in (2026, 2027), 'Unexpected round-one seed')
    require(not cfg.get('resume') and cfg.get('load_from') is None,
            'Fresh formal candidates must not resume/load a trained segmentation model')
    require(tuple(model['test_cfg']['crop_size']) == (768, 768), 'Expected crop768')
    require(tuple(model['test_cfg']['stride']) == (512, 512), 'Expected stride512')
    require(model['test_cfg']['mode'] == 'slide', 'Expected sliding-window validation')
    require(cfg['train_dataloader']['dataset']['data_prefix']['img_path'] == 'train/images',
            'Training must use the official training split')
    require(cfg['val_dataloader']['dataset']['data_prefix']['img_path'] == 'val/images',
            'Validation must use the fixed validation split')
    require(any(item['type'] == 'RandomD4' for item in cfg['train_pipeline']),
            'RandomD4 is missing')


def split_inventory(root):
    result, sets = {}, {}
    for split, count in (('train', 5596), ('val', 1400), ('test', 500)):
        image_dir = root / split / 'images'
        names = sorted(path.name for path in image_dir.glob('*.png'))
        require(len(names) == count, f'{split}: expected {count} images, got {len(names)}')
        sets[split] = set(names)
        if split != 'test':
            masks = {path.name for path in (root / split / 'masks').glob('*.png')}
            require(masks == sets[split], f'{split}: image/mask names differ')
        result[split] = {'count': count, 'names_sha256': hashlib.sha256(
            '\n'.join(names).encode('utf-8')).hexdigest()}
    require(not sets['train'] & sets['val'], 'Training/validation filenames overlap')
    return result


def d4_check():
    import numpy as np
    from aicseg.transforms import apply_d4
    x = np.arange(25).reshape(5, 5)
    outputs = [apply_d4(x, index) for index in range(8)]
    require(len({item.tobytes() for item in outputs}) == 8, 'D4 is not unique')
    for out in outputs:
        require(out.flags.c_contiguous, 'D4 must produce contiguous data')
        require(np.array_equal(np.sort(out.ravel()), x.ravel()), 'D4 altered labels')


def synthetic_backward(cfg, model):
    import torch
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample
    torch.manual_seed(cfg.randomness.seed)
    model.init_weights()  # Actually check classification-pretraining initialization.
    model = model.cuda().train()
    torch.cuda.reset_peak_memory_stats()
    height, width = cfg.crop_size
    batch = cfg.train_dataloader.batch_size
    inputs, samples = [], []
    for _ in range(batch):
        inputs.append(torch.randint(0, 256, (3, height, width), dtype=torch.uint8))
        label = torch.randint(0, 8, (1, height, width))
        label[:, :8, :] = 255
        samples.append(SegDataSample(
            metainfo=dict(img_shape=(height, width), ori_shape=(height, width)),
            gt_sem_seg=PixelData(data=label)))
    data = model.data_preprocessor(dict(inputs=inputs, data_samples=samples), training=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        losses = model(**data, mode='loss')
        total, _ = model.parse_losses(losses)
    require(bool(torch.isfinite(total)), 'Non-finite combined loss')
    # No optimizer step, scheduler step or checkpoint writing in this preflight.
    total.backward()
    checked = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            require(bool(torch.isfinite(param.grad).all()), f'Non-finite gradient: {name}')
            checked += 1
    require(checked > 0, 'No gradients produced')
    values = {}
    for key, value in losses.items():
        tensors = value if isinstance(value, list) else [value]
        require(all(bool(torch.isfinite(item).all()) for item in tensors),
                f'Non-finite loss/metric: {key}')
        values[key] = sum(float(item.detach().mean()) for item in tensors)
    return dict(losses=values, gradient_tensors_checked=checked,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                scope='one synthetic backward, not a data-loader smoke or accuracy test')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--config', default='configs/aic_convnext_base_upernet_768.py')
    parser.add_argument('--forward', action='store_true')
    args = parser.parse_args()
    report = dict(status='FAIL', training_started=False)
    try:
        model_dir = Path(args.model_dir).resolve(strict=True)
        require((model_dir / 'aicseg').is_dir(), 'Not an AIC model directory')
        sys.path[:0] = [str(model_dir), str(model_dir.parent / 'mmsegmentation')]
        os.chdir(model_dir)
        versions = {}
        for name in ('torch', 'mmcv', 'mmengine', 'mmseg', 'mmpretrain'):
            module = importlib.import_module(name)
            versions[name] = module.__version__
            report['versions'] = versions.copy()
        import torch
        from mmcv.ops import nms  # noqa: F401: verify the binary ops extension
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.utils import import_modules_from_strings
        from mmseg.registry import MODELS
        require(torch.cuda.is_available(), 'CUDA required; CPU tests cannot qualify server readiness')
        report['gpu'] = torch.cuda.get_device_name()
        config = Path(args.config).resolve(strict=True)
        cfg = Config.fromfile(str(config))
        import_modules_from_strings(**cfg.custom_imports)
        init_default_scope(cfg.default_scope)
        check_config(cfg)
        report['config'] = str(config)
        report['splits'] = split_inventory(Path(cfg.data_root).resolve())
        d4_check()
        model = MODELS.build(cfg.model)
        report['parameters'] = sum(p.numel() for p in model.parameters())
        if args.forward:
            report['synthetic_backward'] = synthetic_backward(cfg, model)
        else:
            report['synthetic_backward'] = 'NOT RUN; use --forward'
        report['status'] = 'PASS'
        report['next_gate'] = '300 iter smoke with full validation; formal training needs approval'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
