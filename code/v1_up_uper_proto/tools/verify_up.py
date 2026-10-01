"""Bounded UP development evidence: tests, pretraining, AMP and memory.

This is not a formal experiment and computes no validation score. Only official
training crops are used for the optional full-model optimiser smoke (8 steps).
"""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from _bootstrap import PROJECT_ROOT, bootstrap
bootstrap()

import numpy as np
from PIL import Image
import torch
import mmcv
import mmengine
import mmseg
import mmpretrain.models
import aicseg
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules

from run_utils import create_run_dir
from train import TeeStream
from aicseg.prototype_core import PROTOTYPE_CONTRACT_VERSION
from aicseg.prototype_uper_head import UPerPrototypeAdapter


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def official_crop(index, size=128):
    root = PROJECT_ROOT.parents[1] / 'data'
    names = sorted((root / 'train/images').glob('*.png'))
    image_path = names[index % len(names)]
    mask_path = root / 'train/masks' / image_path.name
    with Image.open(image_path) as image:
        rgb = np.array(image.convert('RGB'))
    with Image.open(mask_path) as image:
        official = np.array(image)
    # Deterministic crop, unchanged source images. Match OpenCV BGR input.
    start = (index * 97) % (rgb.shape[0] - size + 1)
    left = (index * 131) % (rgb.shape[1] - size + 1)
    rgb = rgb[start:start + size, left:left + size]
    official = official[start:start + size, left:left + size]
    internal = official.astype(np.int64) - 1
    internal[official == 0] = 255
    sample = SegDataSample(metainfo=dict(img_shape=(size, size),
        ori_shape=(size, size), pad_shape=(size, size), img_path=str(image_path)))
    sample.gt_sem_seg = PixelData(data=torch.from_numpy(internal[None].copy()))
    return dict(inputs=[torch.from_numpy(rgb[:, :, ::-1].copy()).permute(2, 0, 1)],
                data_samples=[sample]), dict(filename=image_path.name,
                crop=[left, start, size, size], split='train')


def adapter_memory_benchmark():
    adapter = UPerPrototypeAdapter(512, {}).cuda().train()
    small = torch.randn(1, 512, 8, 8, device='cuda')
    labels = torch.arange(64, device='cuda').reshape(1, 8, 8) % 8
    with torch.autocast('cuda', dtype=torch.float16):
        adapter.core.update_from_labels(adapter.project(small).detach(), labels)
    del small
    measurements = {}
    for enabled in (False, True):
        adapter.zero_grad(set_to_none=True)
        adapter.core.enabled = enabled
        torch.cuda.empty_cache()
        features = torch.randn(1, 512, 192, 192, device='cuda',
                               dtype=torch.float16, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            output = (adapter.enhance(features, adapter.project(features))
                      if enabled else features)
            loss = output.float().square().mean()
        # Keep the benchmark's scale representable; no optimiser update here.
        (loss * 65536).backward()
        torch.cuda.synchronize()
        measurements['enabled' if enabled else 'bypass'] = dict(
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
            seconds=time.perf_counter() - started)
        del features, output, loss
    measurements['increment_mb'] = (measurements['enabled']['peak_allocated_mb']
                                   - measurements['bypass']['peak_allocated_mb'])
    measurements['scope'] = 'isolated UP adapter, B1 C512 H192 W192, AMP forward+backward; not full model or server'
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-pretrained', action='store_true')
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--gpu-smoke', action='store_true')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Only use after the saved invariant tests have already passed')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    register_all_modules(init_default_scope=True)
    run = create_run_dir('v1_up_convnextb_uper_proto_768', 'verification')
    console = (run / 'console.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout = TeeStream(sys.stdout, console)
    sys.stderr = TeeStream(sys.stderr, console)
    report = dict(status='running', run_dir=str(run), command=sys.argv,
        created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        contract_version=PROTOTYPE_CONTRACT_VERSION,
        runtime=dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, mmcv=mmcv.__version__,
            mmengine=mmengine.__version__, mmseg=mmseg.__version__,
            platform=platform.platform()),
        source_hashes={str(path.relative_to(PROJECT_ROOT)): sha256(path)
                      for path in sorted((PROJECT_ROOT / 'aicseg').glob('*.py'))},
        formal_training=False, validation_score_computed=False,
        executed_micro_iterations=0, executed_optimizer_steps=0)
    report_path = run / 'verification_summary.json'
    print(f'VERIFICATION_DIR: {run}', flush=True)
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        test_results = []
        for filename in (() if args.skip_tests else ('test_prototype.py', 'test_uper_integration.py')):
            result = subprocess.run([sys.executable, '-B', str(PROJECT_ROOT / 'tests' / filename)],
                                    capture_output=True, text=True)
            (run / filename.replace('.py', '.log')).write_text(
                result.stdout + result.stderr, encoding='utf-8')
            test_results.append(dict(file=filename, exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError(f'{filename} failed; see saved log')
        report['tests'] = test_results
        report['tests_skipped'] = args.skip_tests
        save()
        cfg = Config.fromfile(str(PROJECT_ROOT / 'configs/convnextb_uper_proto_768.py'))
        url = cfg.model.backbone.init_cfg.checkpoint
        weight = args.pretrained.resolve() if args.pretrained else run / 'classification_pretrained.pth'
        if not weight.is_file():
            if not args.download_pretrained:
                raise FileNotFoundError('Provide --pretrained or --download-pretrained')
            print('DOWNLOADING_PUBLIC_CLASSIFICATION_PRETRAINED', flush=True)
            torch.hub.download_url_to_file(url, str(weight), hash_prefix='262fd037', progress=False)
        digest = sha256(weight)
        if not digest.startswith('262fd037'):
            raise RuntimeError('Classification weight hash does not match the public URL hash prefix')
        report['pretrained'] = dict(url=url, path=str(weight), sha256=digest, bytes=weight.stat().st_size)
        save()
        source = torch.load(weight, map_location='cpu')
        state = source.get('state_dict', source)
        source_backbone = {key[len('backbone.'):]: value for key, value in state.items()
                           if key.startswith('backbone.')}
        cfg.model.backbone.init_cfg.checkpoint = str(weight)
        model = MODELS.build(cfg.model)
        model.init_weights()
        actual = model.backbone.state_dict()
        common = set(actual) & set(source_backbone)
        mismatches = [name for name in common if not torch.equal(actual[name], source_backbone[name])]
        if not common or mismatches:
            raise RuntimeError(f'Pretrained backbone values mismatch: {mismatches[:10]}')
        report['pretrained_load'] = dict(exactly_matching_tensor_keys=len(common),
            exactly_matching_elements=sum(actual[name].numel() for name in common),
            backbone_keys_absent_from_checkpoint=sorted(set(actual) - set(source_backbone)),
            source_backbone_keys_not_used=sorted(set(source_backbone) - set(actual)),
            mismatches=mismatches, classification_head_not_loaded=True)
        report['parameters'] = dict(total=sum(p.numel() for p in model.parameters()),
            added=sum(p.numel() for p in model.decode_head.prototype_adapter.parameters()))
        del source, state, source_backbone, actual
        gc.collect()
        save()
        if args.gpu_smoke:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            report['runtime']['gpu'] = torch.cuda.get_device_name()
            torch.backends.cudnn.benchmark = False
            report['adapter_memory'] = adapter_memory_benchmark()
            model.cuda().train()
            # Only development smoke changes: B1, crop128, accumulation4.
            # This is not a comparable training exposure or validation run.
            model.data_preprocessor.size = (128, 128)
            cfg.model.data_preprocessor.size = (128, 128)
            cfg.train_dataloader.batch_size = 1
            cfg.train_cfg.max_iters = 8
            cfg.train_cfg.val_interval = 8
            cfg.work_dir = str(run)
            cfg.dump(str(run / 'resolved_verification_config.py'))
            wrapper = build_optim_wrapper(model, copy.deepcopy(cfg.optim_wrapper))
            wrapper.initialize_count_status(model, init_counts=0, max_counts=8)
            # Assert every added parameter is in exactly one optimiser group.
            identifiers = [id(p) for group in wrapper.optimizer.param_groups for p in group['params']]
            for name, parameter in model.decode_head.prototype_adapter.named_parameters():
                if identifiers.count(id(parameter)) != 1:
                    raise RuntimeError(f'Added parameter not assigned exactly once: {name}')
            report['optimizer_groups_verified'] = True
            report['smoke'] = dict(physical_batch=1, accumulation=4, crop=[128, 128],
                split='official_train_only', validation=False, micro_iterations=8,
                nominal_optimizer_step_points=2, comparable_to_formal_budget=False,
                records=[], input_provider='verify_up.official_crop, not the formal augmentation pipeline')
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for index in range(8):
                batch, record = official_crop(index)
                logs = model.train_step(batch, wrapper)
                if not all(torch.isfinite(value).all() for value in logs.values()):
                    raise RuntimeError('Nonfinite full-model smoke loss')
                record['micro_iteration'] = index + 1
                record['losses'] = {key: float(value) for key, value in logs.items()}
                actual_steps = max((int(state['step']) for state in wrapper.optimizer.state.values()
                                    if 'step' in state), default=0)
                record['actual_optimizer_step_count'] = actual_steps
                record['loss_scale'] = float(wrapper.loss_scaler.get_scale())
                report['smoke']['records'].append(record)
                report['executed_micro_iterations'] = index + 1
                report['executed_optimizer_steps'] = actual_steps
                save()
            if report['executed_optimizer_steps'] != 2:
                raise RuntimeError('AMP skipped a nominal update; actual optimizer count is not 2')
            report['smoke']['actual_optimizer_steps'] = report['executed_optimizer_steps']
            report['smoke']['seconds'] = time.perf_counter() - started
            report['smoke']['peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
            core = model.decode_head.prototype_adapter.core
            report['prototype_diagnostics'] = core.diagnostics()
            checkpoint = run / 'smoke_model_state.pth'
            torch.save(dict(state_dict=model.state_dict(), meta=dict(
                contract_version=PROTOTYPE_CONTRACT_VERSION, development_smoke_only=True,
                executed_micro_iterations=8, executed_optimizer_steps=2)), checkpoint)
            report['smoke_checkpoint'] = dict(path=str(checkpoint), sha256=sha256(checkpoint),
                optimizer_state_saved=False, usable_for_formal_resume=False)
            model.eval()
            batch, _ = official_crop(0)
            # Drop GT from test data: only image and geometric metadata remain.
            batch['data_samples'] = [SegDataSample(metainfo=batch['data_samples'][0].metainfo)]
            before = {key: value.clone() for key, value in core.state_dict().items()}
            with torch.no_grad():
                output = model.test_step(batch)[0]
                expected = output.seg_logits.data.clone()
            restored = torch.load(checkpoint, map_location='cpu')['state_dict']
            model.load_state_dict(restored, strict=True)
            with torch.no_grad():
                second = model.test_step(batch)[0]
            torch.testing.assert_close(expected, second.seg_logits.data, rtol=0, atol=0)
            for name, value in before.items():
                torch.testing.assert_close(value, core.state_dict()[name], rtol=0, atol=0)
            if second.pred_sem_seg.data.shape != (1, 128, 128) or second.pred_sem_seg.data.max() > 7:
                raise RuntimeError('Invalid semantic prediction shape/IDs')
            report['checkpoint_roundtrip_prediction_exact'] = True
            report['eval_without_gt_bank_frozen'] = True
        report['status'] = 'passed'
        save()
        print(f'PASS: {report_path}', flush=True)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    main()
