"""Reproducible correctness checks; synthetic diagnostics are not accuracy runs."""

import argparse
import copy
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

import torch
import numpy as np
from PIL import Image
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.optim import build_optim_wrapper
from mmengine.structures import PixelData
from mmseg.registry import MODELS, OPTIM_WRAPPER_CONSTRUCTORS
from mmseg.structures import SegDataSample
from mmseg.datasets.transforms import LoadAnnotations
from mmdet.models.dense_heads import Mask2FormerHead as NativeHead
import mmdet.models
import mmpretrain.models
import aicseg


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def sample(seg, img_shape=None):
    shape = tuple(seg.shape[-2:])
    result = SegDataSample(metainfo=dict(
        img_shape=img_shape or shape, ori_shape=shape, pad_shape=shape,
        padding_size=(0, 0, 0, 0)))
    result.gt_sem_seg = PixelData(data=seg[None])
    return result


def loss_vector(head, classes, masks, samples, native=False):
    gt, metas = head._seg_data_to_instance_data(samples)
    torch.manual_seed(2026)
    method = NativeHead._loss_by_feat_single if native else type(head)._loss_by_feat_single
    result = method(head, classes, masks, gt, metas)
    assert all(torch.isfinite(value).all() for value in result)
    return result, gt, metas


def unit_checks(cfg, device):
    head = MODELS.build(cfg.model.decode_head).to(device).eval()
    assert head.num_queries == 100 and head.cls_embed.out_features == 9
    assert len(head.pixel_decoder.encoder.layers) == 6
    assert len(head.transformer_decoder.layers) == 9
    records = {}
    partial = torch.full((32, 32), 255, dtype=torch.long, device=device)
    partial[4:28, 4:16] = 0
    partial[4:28, 16:28] = 4
    classes = torch.randn(1, 100, 9, device=device, requires_grad=True)
    masks = torch.randn(1, 100, 32, 32, device=device, requires_grad=True)
    values, gt, metas = loss_vector(head, classes, masks, [sample(partial)])
    sum(values).backward()
    invalid = partial == 255
    ignored_gradient = masks.grad[0, :, invalid].abs().max().item()
    assert ignored_gradient <= 1e-6, ignored_gradient
    changed = masks.detach().clone()
    changed[0, :, invalid] += 100
    changed_values, _, _ = loss_vector(head, classes.detach(), changed, [sample(partial)])
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-5)
               for a, b in zip(values, changed_values))
    torch.manual_seed(2026)
    before = head._get_targets_single(classes[0].detach(), masks[0].detach(), gt[0], metas[0])
    torch.manual_seed(2026)
    after = head._get_targets_single(classes[0].detach(), changed[0], gt[0], metas[0])
    assert torch.equal(before[0], after[0])
    assert masks.grad[0, :, ~invalid].abs().sum() > 0
    records['partial_ignore'] = dict(losses=[v.item() for v in values],
                                    ignored_mask_gradient_max=ignored_gradient,
                                    perturbation_loss_and_matching_invariant=True)
    cases = {
        'all_ignore': torch.full_like(partial, 255),
        'single_class': torch.full_like(partial, 4),
        'one_valid_pixel': torch.full_like(partial, 255),
    }
    cases['one_valid_pixel'][16, 16] = 4
    for name, seg in cases.items():
        c = classes.detach().clone().requires_grad_(True)
        m = masks.detach().clone().requires_grad_(True)
        losses, _, _ = loss_vector(head, c, m, [sample(seg)])
        sum(losses).backward()
        assert torch.isfinite(c.grad).all() and torch.isfinite(m.grad).all()
        if name == 'all_ignore':
            assert sum(v.item() for v in losses) == 0
            assert c.grad.abs().sum() == 0 and m.grad.abs().sum() == 0
        records[name] = dict(losses=[v.item() for v in losses], gradients_finite=True)
    full = torch.zeros_like(partial)
    full[:, 16:] = 4
    modified, _, _ = loss_vector(head, classes.detach(), masks.detach(), [sample(full)])
    original, _, _ = loss_vector(head, classes.detach(), masks.detach(), [sample(full)], native=True)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(modified, original))
    records['full_valid_native_loss_parity'] = True
    mixed_classes = classes.detach().repeat(2, 1, 1).requires_grad_(True)
    mixed_masks = masks.detach().repeat(2, 1, 1, 1).requires_grad_(True)
    mixed, _, _ = loss_vector(head, mixed_classes, mixed_masks,
                              [sample(partial), sample(cases['all_ignore'])])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(values, mixed))
    sum(mixed).backward()
    assert mixed_classes.grad[1].abs().sum() == 0
    assert mixed_masks.grad[1].abs().sum() == 0
    records['mixed_all_ignore_image_zero_contribution'] = True
    padded = torch.zeros_like(partial)
    padded[:8, :8] = 4
    padded_gt, _ = head._seg_data_to_instance_data([sample(padded, (8, 8))])
    assert padded_gt[0].labels.tolist() == [4]
    assert int(padded_gt[0].metainfo['aic_valid_mask'].sum()) == 64
    records['padding_not_class_target'] = True
    gt, metas = head._seg_data_to_instance_data([sample(partial)])
    outputs = head.loss_by_feat([classes.detach()] * 10, [masks.detach()] * 10, gt, metas)
    assert len(outputs) == 30 and all(torch.isfinite(v) for v in outputs.values())
    records['native_intermediate_supervision_loss_count'] = len(outputs)
    channels = cfg.model.decode_head.in_channels
    features = [torch.randn(1, c, 128 // s, 128 // s, device=device)
                for c, s in zip(channels, [4, 8, 16, 32])]
    with torch.no_grad():
        own = head(features, [sample(torch.zeros(128, 128, device=device, dtype=torch.long))])
        reference = NativeHead.forward(head, features, [])
        assert all(torch.equal(a, b) for ours, theirs in zip(own, reference)
                   for a, b in zip(ours, theirs))
        stale = dict(img_shape=(128, 128), pad_shape=(1024, 1024))
        scores = head.predict(features, [stale], dict(mode='slide'))
        assert scores.shape == (1, 8, 128, 128)
        assert stale['pad_shape'] == (1024, 1024)
    records['identity_hook_native_forward_bitwise_parity'] = True
    records['slide_stale_pad_shape_fixed_without_mutating_input'] = list(scores.shape)
    fixture = PROJECT_ROOT / 'runs/validation_20261001/mapping_fixture'
    fixture.mkdir(parents=True, exist_ok=True)
    raw = np.tile(np.arange(9, dtype=np.uint8), (8, 1))
    Image.fromarray(raw).save(fixture / 'official_synthetic.png')
    loaded = LoadAnnotations()(dict(
        seg_map_path=str(fixture / 'official_synthetic.png'), seg_fields=[],
        reduce_zero_label=True, label_map=None))
    expected = np.where(raw == 0, 255, raw - 1)
    assert np.array_equal(loaded['gt_seg_map'], expected)
    formatter = aicseg.AICIoUMetric(format_only=True, output_dir=str(fixture / 'export'))
    formatter.dataset_meta = aicseg.AICDataset.METAINFO
    formatter.process({}, [dict(pred_sem_seg=dict(data=torch.arange(8).reshape(1, 2, 4)),
                                img_path='synthetic.png', reduce_zero_label=True)])
    exported = Image.open(fixture / 'export/synthetic.png')
    assert exported.mode == 'L'
    assert np.array_equal(np.asarray(exported), np.arange(1, 9, dtype=np.uint8).reshape(2, 4))
    records['actual_annotation_loader_and_formatter_mapping'] = dict(
        official0_internal255=True, official1to8_internal0to7=True,
        exported_official_ids=list(range(1, 9)), grayscale_L=True)
    return records


def optimizer_checks(model, cfg):
    kwargs = dict(optim_wrapper_cfg=copy.deepcopy(cfg.optim_wrapper),
                  paramwise_cfg=copy.deepcopy(cfg.optim_wrapper.paramwise_cfg))
    kwargs['optim_wrapper_cfg'].pop('constructor')
    kwargs['optim_wrapper_cfg'].pop('paramwise_cfg')
    actual = OPTIM_WRAPPER_CONSTRUCTORS.get('AICMask2FormerOptimizerConstructor')(**kwargs)
    baseline = OPTIM_WRAPPER_CONSTRUCTORS.get('LearningRateDecayOptimizerConstructor')(**kwargs)
    custom, control = [], []
    actual.add_params(custom, model)
    baseline.add_params(control, model)
    def lookup(groups):
        result = {}
        for group in groups:
            for name, param in zip(group['param_names'], group['params']):
                assert name not in result
                result[name] = (group['lr'], group['weight_decay'], id(param))
        return result
    grouped, reference = lookup(custom), lookup(control)
    named = dict(model.named_parameters())
    assert set(grouped) == {k for k, v in named.items() if v.requires_grad}
    assert len({value[2] for value in grouped.values()}) == len(grouped)
    assert all(grouped[name][:2] == reference[name][:2]
               for name in named if name.startswith('backbone.'))
    embeddings = [name for name, child in model.named_modules()
                  if name.startswith('decode_head.') and isinstance(child, torch.nn.Embedding)]
    assert all(grouped[name + '.weight'][:2] == (1e-4, 0.) for name in embeddings)
    return dict(trainable_parameter_tensors=len(grouped), backbone_group_parity=True,
                head_embedding_no_decay=embeddings,
                groups=[{k: v for k, v in g.items() if k not in ('params', 'param_names')}
                        | dict(tensors=len(g['params'])) for g in custom])


def model_checks(cfg, device, size, pretrained, backward, amp):
    torch.hub.set_dir(str(PROJECT_ROOT / '.cache/torch/hub'))
    model_cfg = copy.deepcopy(cfg.model)
    if not pretrained:
        model_cfg.backbone.init_cfg = None
    model = MODELS.build(model_cfg).to(device)
    model.init_weights()
    result = dict(parameters=sum(p.numel() for p in model.parameters()),
                  backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
                  pretrained_loaded=pretrained, diagnostic_input_size=size)
    if pretrained:
        cached = Path(torch.hub.get_dir()) / 'checkpoints' / Path(cfg.checkpoint_file).name
        checkpoint = torch.load(cached, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        prefix = cfg.model.backbone.init_cfg.prefix
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        actual = model.backbone.state_dict()
        matched = [k for k, v in state.items() if k in actual and v.shape == actual[k].shape]
        unmatched = [k for k in actual if k not in matched]
        assert matched and all(torch.equal(actual[k].cpu(), state[k]) for k in matched)
        expected_norms = {f'norm{i}.{field}' for i in range(4) for field in ['weight', 'bias']}
        assert set(unmatched) == expected_norms, unmatched
        # The control uses this same classification checkpoint and mmpretrain
        # adapter: its four output norms are freshly initialized as well.
        control_backbone = MODELS.build(copy.deepcopy(cfg.model.backbone))
        control_backbone.init_weights()
        control_state = control_backbone.state_dict()
        assert all(torch.equal(v.cpu(), control_state[k]) for k, v in actual.items())
        del control_backbone, control_state
        result['classification_checkpoint'] = dict(
            sha256=digest(cached), matched_tensors=len(matched), unmatched_model_keys=unmatched,
            unused_backbone_checkpoint_keys=[k for k in state if k not in matched],
            exact_control_backbone_initialization_parity=True,
            classification_url=cfg.checkpoint_file)
    result['optimizer'] = optimizer_checks(model, cfg)
    model.eval()
    image = torch.randn(1, 3, size, size, device=device)
    seg = torch.zeros(size, size, dtype=torch.long, device=device)
    seg[size // 4:3 * size // 4, size // 4:3 * size // 4] = 4
    seg[:size // 8] = 255
    samples = [sample(seg)]
    with torch.no_grad():
        levels = model.extract_feat(image)
        mask, memories = model.decode_head.pixel_decoder(levels)
        identity = model.decode_head._process_pixel_decoder_outputs(mask, memories, samples)
        assert identity[0] is mask and identity[1] is memories
        result['interfaces'] = dict(backbone=[list(x.shape) for x in levels],
                                     mask_features=list(mask.shape),
                                     multi_scale_memorys=[list(x.shape) for x in memories])
        prediction = model.predict(image, samples)[0]
        assert prediction.seg_logits.data.shape == (8, size, size)
        internal = prediction.pred_sem_seg.data
        assert internal.min() >= 0 and internal.max() < 8
        result['semantic_output_shape'] = list(prediction.seg_logits.data.shape)
        result['official_label_mapping'] = [int((internal + 1).min()), int((internal + 1).max())]
    del levels, mask, memories, identity, prediction
    if backward:
        model.train()
        losses = model.loss(image, samples)
        total = sum(value for key, value in losses.items() if 'loss' in key)
        assert torch.isfinite(total)
        total.backward()
        grads = {k: v.grad for k, v in model.named_parameters() if v.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        assert any(k.startswith('backbone.') and g.abs().sum() > 0 for k, g in grads.items())
        assert any(k.startswith('decode_head.') and g.abs().sum() > 0 for k, g in grads.items())
        result['backward'] = dict(loss=total.item(), finite_gradient_tensors=len(grads),
                                  optimizer_updates=0,
                                  gradient_norm=torch.stack([g.norm() ** 2 for g in grads.values()]).sum().sqrt().item())
        model.zero_grad(set_to_none=True)
    if amp:
        assert device == 'cuda'
        model.train()
        settings = copy.deepcopy(cfg.optim_wrapper)
        # Diagnostic-only initial scale, avoiding unrelated cold-start overflow.
        # Production config remains loss_scale='dynamic'.
        settings.loss_scale = dict(init_scale=128.)
        wrapper = build_optim_wrapper(model, settings)
        wrapper.initialize_count_status(model, 0, 4)
        values = []
        for _ in range(4):
            with wrapper.optim_context(model):
                losses = model.loss(image, samples)
                total = sum(value for key, value in losses.items() if 'loss' in key)
                assert torch.isfinite(total)
            wrapper.update_params(total)
            values.append(total.item())
        states = list(wrapper.optimizer.state.values())
        assert states and all(int(s['step']) == 1 for s in states)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        result['amp_optimizer_step'] = dict(
            micro_iterations=4, optimizer_updates=1, synthetic_sample_exposures=4,
            diagnostic_input_size=size, diagnostic_initial_loss_scale=128.,
            losses=values, final_loss_scale=wrapper.loss_scaler.get_scale(),
            adam_state_tensors=len(states))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/m0_convnextb_mask2former_768.py')
    parser.add_argument('--stage', choices=['unit', 'model'], default='unit')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite validation evidence: {output}')
    package_path = Path(aicseg.__file__).resolve()
    assert package_path == PROJECT_ROOT / 'aicseg/__init__.py', package_path
    started = time.monotonic()
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.default_scope)
    result = dict(command=sys.argv, seed=2026, python=sys.version, platform=platform.platform(),
                  time=datetime.now().astimezone().isoformat(),
                  config_sha256=digest(args.config), stage=args.stage, status='started')
    result['version_package'] = str(package_path)
    result['environment'] = {
        name: dict(version=importlib.import_module(name).__version__,
                   module=importlib.import_module(name).__file__)
        for name in ['torch', 'mmcv', 'mmengine', 'mmseg', 'mmdet', 'mmpretrain']}
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    result['pip_check'] = dict(exit_code=check.returncode, output=check.stdout.strip())
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result['checks'] = (unit_checks(cfg, args.device) if args.stage == 'unit' else
                            model_checks(cfg, args.device, args.size,
                                         args.pretrained, args.backward, args.amp))
        result['status'] = 'passed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        if args.device == 'cuda':
            result['gpu'] = torch.cuda.get_device_name()
            result['peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
            result['peak_reserved_mb'] = torch.cuda.max_memory_reserved() / 1024 ** 2
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: result.get(k) for k in ['stage', 'status', 'elapsed_seconds',
                                                   'peak_allocated_mb', 'error']}))


if __name__ == '__main__':
    main()
