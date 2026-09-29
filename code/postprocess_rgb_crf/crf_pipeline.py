"""Independent, opt-in RGB DenseCRF experiment; never trains or edits inputs."""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['Background', 'Building', 'Road', 'Water', 'Barren',
           'Vegetation', 'Agricultural', 'Vehicle']
# Predeclared engineering candidates, not paper-reproduced optimal parameters.
GRID = [dict(id=f'rgb_{i}', iterations=5, gaussian_sxy=3,
             gaussian_compat=g, bilateral_sxy=s, bilateral_srgb=13,
             bilateral_compat=b)
        for i, (g, s, b) in enumerate(
            ( (1, 20, 2), (1, 20, 4), (1, 40, 2), (1, 40, 4),
              (3, 20, 2), (3, 20, 4), (3, 40, 2), (3, 40, 4))) ]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    # Exclusive creation is deliberate: no overwriting run evidence.
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def split_names(root, split):
    images = root / 'data' / split / 'images'
    names = sorted(p.name for p in images.glob('*.png'))
    expected = 1400 if split == 'val' else 500
    if len(names) != expected:
        raise ValueError(f'{split}: expected {expected} PNG images, found {len(names)}')
    if split == 'val':
        declared = sorted((root / 'data/splits/fold_0_val.txt').read_text().splitlines())
        if names != declared:
            raise ValueError('Validation directory differs from fixed Fold0 list')
    return names


def screen_names(names):
    return sorted(sorted(names, key=lambda n: hashlib.sha256(
        ('AIC-CRF-2026:' + n).encode()).hexdigest())[:200])


def validate_probs(probs):
    if probs.ndim != 3 or probs.shape[0] != 8 or probs.dtype != np.float32:
        raise ValueError('Expected float32 probability array [8,H,W]')
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError('Probabilities must be finite and in [0,1]')
    if not np.allclose(probs.sum(axis=0), 1, atol=1e-5):
        raise ValueError('Probabilities must sum to one per pixel')


def confusion(pred, official_gt):
    if pred.shape != official_gt.shape or not np.isin(official_gt, range(9)).all():
        raise ValueError('Invalid official mask shape or IDs (expected 0..8)')
    if not np.isin(pred, range(8)).all():
        raise ValueError('Predictions must use internal IDs 0..7')
    valid = official_gt != 0
    gt = official_gt[valid].astype(np.int64) - 1
    return np.bincount(gt * 8 + pred[valid], minlength=64).reshape(8, 8)


def metrics(matrix):
    inter = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - inter
    iou = np.divide(inter * 100., union, out=np.full(8, np.nan), where=union != 0)
    return dict(mIoU=float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
                IoU={k: float(v) if np.isfinite(v) else None for k, v in zip(CLASSES, iou)},
                confusion=matrix.tolist())


def eligible(base, trial):
    for group in ('full', 'remaining'):
        a, b = base[group], trial[group]
        if a['mIoU'] is None or b['mIoU'] is None or b['mIoU'] <= a['mIoU']:
            return False
        va, vb = a['IoU']['Vehicle'], b['IoU']['Vehicle']
        if va is None or vb is None or vb < va - 1.0:
            return False
    return True


def backend():
    try:
        import pydensecrf.densecrf as dcrf
    except ImportError as exc:
        raise RuntimeError('Optional pydensecrf backend is missing; install it in the '
                           'experiment environment. No CRF result has been produced.') from exc
    return dcrf


def refine(probs, rgb, params):
    validate_probs(probs)
    if rgb.shape != (*probs.shape[1:], 3) or rgb.dtype != np.uint8:
        raise ValueError('RGB must be aligned uint8 [H,W,3]')
    if params is None:
        return probs.argmax(0).astype(np.uint8)
    if params not in GRID:
        raise ValueError('Parameters are not in the predeclared grid')
    dcrf = backend()
    h, w = rgb.shape[:2]
    model = dcrf.DenseCRF2D(w, h, 8)
    unary = np.ascontiguousarray(-np.log(np.maximum(probs, 1e-8)).reshape(8, -1))
    model.setUnaryEnergy(unary)
    model.addPairwiseGaussian(sxy=params['gaussian_sxy'], compat=params['gaussian_compat'],
                             kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    model.addPairwiseBilateral(sxy=params['bilateral_sxy'], srgb=params['bilateral_srgb'],
                              rgbim=np.ascontiguousarray(rgb), compat=params['bilateral_compat'],
                              kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    return np.asarray(model.inference(params['iterations'])).reshape(8, h, w).argmax(0).astype(np.uint8)


def resources():
    # OS-reported lifetime process high-water mark, not isolated CRF allocation.
    import psutil
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', None)
    if peak is None:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != 'darwin':
                peak *= 1024
        except ImportError:
            pass
    return dict(rss_bytes=info.rss, process_lifetime_peak_rss_bytes=peak)


def load_rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (1024, 1024):
            raise ValueError(f'Expected original RGB 1024 image: {path}')
        return np.asarray(image).copy()


def load_cache(cache, root):
    cache = Path(cache).resolve()
    m = read_json(cache / 'manifest.json')
    done = read_json(cache / 'complete.json')
    if done['manifest_sha256'] != digest(cache / 'manifest.json'):
        raise ValueError('Cache manifest changed after export')
    if m['schema'] != 1 or m['classes'] != CLASSES or m['names'] != split_names(root, m['split']):
        raise ValueError('Cache schema, classes or split mismatch')
    if m['screen_names'] != screen_names(m['names']):
        raise ValueError('Screen subset mismatch')
    for name in m['names']:
        for kind in ('images', 'masks') if m['split'] == 'val' else ('images',):
            if digest(root / 'data' / m['split'] / kind / name) != m['input_hashes'][name][kind]:
                raise ValueError(f'Input changed: {kind}/{name}')
        if digest(cache / 'probabilities' / (name + '.npy')) != done['probability_hashes'][name]:
            raise ValueError(f'Cached probabilities changed: {name}')
    return m


def frozen_params(path, identity):
    report = read_json(path)
    if report.get('stage') != 'final' or report['identity'] != identity:
        raise ValueError('Frozen report must come from full validation of this config/checkpoint')
    selected = report['selected']
    if selected is not None:
        if selected not in GRID or not eligible(report['baseline'], report['trials'][selected['id']]):
            raise ValueError('Frozen CRF parameters do not satisfy the validation gate')
    return selected


def export(args):
    config, checkpoint, model_dir = [Path(p).resolve() for p in
                                     (args.config, args.checkpoint, args.model_dir)]
    if not config.is_file() or not checkpoint.is_file() or not (model_dir / 'aicseg').is_dir():
        raise ValueError('Provide an existing explicit config, checkpoint and model version directory')
    sys.path[:0] = [str(model_dir), str(args.root / 'code/mmsegmentation')]
    from mmengine.config import Config
    from mmseg.apis import init_model, inference_model
    import torch
    cfg = Config.fromfile(str(config))
    # Resolved text captures inherited configuration, not only the top-level file.
    resolved = cfg.pretty_text
    identity = dict(checkpoint_sha256=digest(checkpoint),
                    resolved_config_sha256=hashlib.sha256(resolved.encode()).hexdigest())
    if args.split == 'test':
        if not args.frozen:
            raise ValueError('Test export requires --frozen final validation report; no test tuning')
        frozen_params(args.frozen, identity)
    names = split_names(args.root, args.split)
    if cfg.model.decode_head.num_classes != 8:
        raise ValueError('Expected an eight-class segmentation model')
    cfg.model.backbone.init_cfg = None
    cfg.load_from = None
    model = init_model(cfg, str(checkpoint), device=args.device)
    out = new_directory(args.out)
    (out / 'probabilities').mkdir()
    with (out / 'resolved_config.py').open('x', encoding='utf-8') as f:
        f.write(resolved)
    hashes = {}
    for name in names:
        kinds = ('images', 'masks') if args.split == 'val' else ('images',)
        hashes[name] = {k: digest(args.root / 'data' / args.split / k / name) for k in kinds}
    manifest = dict(schema=1, split=args.split, names=names, screen_names=screen_names(names),
                    classes=CLASSES, identity=identity, config=str(config), checkpoint=str(checkpoint),
                    input_hashes=hashes, dtype='float32', shape=[8, 1024, 1024],
                    inference='MMSeg inference_model, aligned seg_logits then float32 softmax',
                    model_test_cfg=dict(cfg.model.test_cfg), python=platform.python_version(),
                    torch=torch.__version__, command=sys.argv)
    write_json(out / 'manifest.json', manifest)
    use_cuda = str(args.device).startswith('cuda')
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    timings, probability_hashes = [], {}
    start = time.perf_counter()
    for i, name in enumerate(names):
        path = args.root / 'data' / args.split / 'images' / name
        load_rgb(path)
        if use_cuda:
            torch.cuda.synchronize(args.device)
        t = time.perf_counter()
        result = inference_model(model, str(path))
        logits = result.seg_logits.data.float()
        if tuple(logits.shape) != (8, 1024, 1024):
            raise ValueError(f'Expected full original-resolution logits, got {tuple(logits.shape)}')
        probs = logits.softmax(0).cpu().numpy()
        if use_cuda:
            torch.cuda.synchronize(args.device)
        timings.append(time.perf_counter() - t)
        validate_probs(probs)
        # Independent parity check against MMSeg native argmax output.
        if not np.array_equal(probs.argmax(0), result.pred_sem_seg.data.squeeze(0).cpu().numpy()):
            raise ValueError('Probability-cache argmax does not equal native MMSeg prediction')
        target = out / 'probabilities' / (name + '.npy')
        with target.open('xb') as f:
            np.save(f, probs, allow_pickle=False)
        probability_hashes[name] = digest(target)
        print(f'export {i+1}/{len(names)}: {name}', flush=True)
    write_json(out / 'complete.json', dict(manifest_sha256=digest(out / 'manifest.json'),
               probability_hashes=probability_hashes, inference_seconds=sum(timings),
               wall_seconds=time.perf_counter()-start, resources=resources(),
               gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if use_cuda else None,
               gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if use_cuda else None))


def evaluate(args):
    cache = Path(args.cache).resolve()
    m = load_cache(cache, args.root)
    if m['split'] != 'val':
        raise ValueError('CRF evaluation/search accepts validation caches only')
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest_hash = digest(cache / 'manifest.json')
    subset = set(m['screen_names'])
    params = []
    names = m['names']
    if args.stage == 'screen':
        params, names = GRID, sorted(subset)
    elif args.stage == 'final':
        if not args.screen:
            raise ValueError('Final evaluation requires --screen report')
        screen = read_json(args.screen)
        if (screen.get('stage') != 'screen' or screen['manifest_sha256'] != manifest_hash
                or screen['grid'] != GRID or screen['identity'] != m['identity']):
            raise ValueError('Screen report belongs to another cache or grid')
        ranked = sorted(GRID, key=lambda p: (-screen['trials'][p['id']]['full']['mIoU'], p['id']))
        params = ranked[:2]
    if params:
        backend()
    report = dict(stage=args.stage, identity=m['identity'], manifest_sha256=manifest_hash,
                  grid=GRID, subset_names=sorted(subset), command=sys.argv, trials={})
    for param in [None] + params:
        matrices = {key: np.zeros((8, 8), dtype=np.int64) for key in ('full', 'screen', 'remaining')}
        processing_seconds = 0.
        start = time.perf_counter()
        for name in names:
            probs = np.load(cache / 'probabilities' / (name + '.npy'), allow_pickle=False)
            rgb = load_rgb(args.root / 'data/val/images' / name)
            with Image.open(args.root / 'data/val/masks' / name) as im:
                gt = np.asarray(im)
            t = time.perf_counter()
            pred = refine(probs, rgb, param)
            processing_seconds += time.perf_counter() - t
            matrix = confusion(pred, gt)
            matrices['full'] += matrix
            matrices['screen' if name in subset else 'remaining'] += matrix
        result = {key: metrics(value) for key, value in matrices.items()}
        result.update(image_count=len(names), processing_seconds=processing_seconds,
                      wall_seconds=time.perf_counter()-start, resources=resources(),
                      gpu_memory_bytes=None, gpu_note='CPU-only postprocessing; no GPU allocated by evaluator')
        if param is None:
            report['baseline'] = result
        else:
            report['trials'][param['id']] = result
        print(f'{param["id"] if param else "baseline"}: {result["full"]["mIoU"]}', flush=True)
    if args.stage == 'final':
        accepted = [p for p in params if eligible(report['baseline'], report['trials'][p['id']])]
        accepted.sort(key=lambda p: (-report['trials'][p['id']]['full']['mIoU'], p['id']))
        report['selected'] = accepted[0] if accepted else None
        report['screen_report_sha256'] = digest(args.screen)
    write_json(args.out, report)


def apply_frozen(args):
    cache = Path(args.cache).resolve()
    m = load_cache(cache, args.root)
    params = frozen_params(args.frozen, m['identity'])
    if params:
        backend()
    out = new_directory(args.out)
    pred_dir = out / 'predictions'
    pred_dir.mkdir()
    start = time.perf_counter()
    for name in m['names']:
        probs = np.load(cache / 'probabilities' / (name + '.npy'), allow_pickle=False)
        rgb = load_rgb(args.root / 'data' / m['split'] / 'images' / name)
        labels = refine(probs, rgb, params) + np.uint8(1)
        Image.fromarray(labels).save(pred_dir / name)
    if m['split'] == 'test':
        helper = args.root / 'code/baseline_v1/tools/submission.py'
        spec = importlib.util.spec_from_file_location('aic_submission', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_predictions(pred_dir, args.root / 'data/test/images')
        module.make_zip(pred_dir, out / 'submission.zip')
    write_json(out / 'prediction_manifest.json', dict(identity=m['identity'], selected=params,
               frozen_report_sha256=digest(args.frozen), cache_manifest_sha256=digest(cache / 'manifest.json'),
               prediction_hashes={n: digest(pred_dir / n) for n in m['names']},
               zip_sha256=digest(out / 'submission.zip') if m['split'] == 'test' else None,
               wall_seconds=time.perf_counter()-start, resources=resources()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export')
    for name in ('config', 'checkpoint', 'model-dir', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--frozen', type=Path)
    p.set_defaults(func=export)
    p = sub.add_parser('evaluate')
    p.add_argument('--cache', required=True)
    p.add_argument('--stage', choices=('baseline', 'screen', 'final'), required=True)
    p.add_argument('--screen', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(func=evaluate)
    p = sub.add_parser('apply')
    p.add_argument('--cache', required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(func=apply_frozen)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.func(args)


if __name__ == '__main__':
    main()
