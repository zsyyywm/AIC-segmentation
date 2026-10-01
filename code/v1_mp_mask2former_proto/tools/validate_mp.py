"""MP correctness/resource checks, using synthetic inputs rather than scores."""

import argparse
import copy
import hashlib
import io
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

INVOCATION_DIR = Path.cwd().resolve()
bootstrap()

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
import mmdet
import mmdet.models
import mmpretrain.models
import aicseg


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_tensors(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def sample(labels=None, size=64, image_shape=None, old_pad_shape=None):
    item = SegDataSample(metainfo=dict(
        img_shape=image_shape or (size, size), ori_shape=(size, size),
        pad_shape=old_pad_shape or (size, size), padding_size=(0, 0, 0, 0)))
    if labels is not None:
        item.gt_sem_seg = PixelData(data=labels[None])
    return item


def state(core):
    return {k: v.detach().clone() for k, v in core.named_buffers()}


def same_state(before, core):
    after = state(core)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def source_and_config_checks(cfg, reference):
    manifest = json.loads((PROJECT_ROOT / 'source_manifest.json').read_text())
    unchanged = {}
    for name, entry in manifest['sources'].items():
        if not entry['changes']:
            actual = sha(PROJECT_ROOT / name)
            assert actual == entry['sha256'], name
            unchanged[name] = actual
    left, right = reference.to_dict(), cfg.to_dict()
    assert cfg.custom_hooks == reference.custom_hooks + [dict(
        type='PrototypeDiagnosticsHook', interval=50,
        core_path='decode_head.prototype_adapter.core')]
    for data in (left, right):
        data.pop('experiment_name', None)
        data.pop('work_dir', None)
        data.pop('custom_hooks', None)
    right['model']['decode_head'].pop('prototype_cfg')
    right['model']['decode_head'].pop('feature_stride')
    right['model']['decode_head']['type'] = left['model']['decode_head']['type']
    assert left == right, 'Unexpected difference outside P and identity/hooks'
    contract = json.loads((PROJECT_ROOT / 'prototype_contract.json').read_text())
    for name, value in contract['hyperparameters'].items():
        assert cfg.model.decode_head.prototype_cfg[name] == value, name
    assert cfg.model.decode_head.prototype_cfg.enabled is True
    assert Path(aicseg.__file__).resolve().is_relative_to(PROJECT_ROOT)
    assert cfg.train_dataloader.batch_size == 2
    assert cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.train_cfg.max_iters == 40000
    return dict(unchanged_source_hashes=unchanged, non_P_configuration_identical=True,
                frozen_P_hyperparameters_identical=True, version_local_import=True)


def features(device, size=64):
    return [torch.randn(1, channels, size // stride, size // stride, device=device)
            for channels, stride in zip((128, 256, 512, 1024), (4, 8, 16, 32))]


def heads(cfg, reference, device):
    torch.manual_seed(2026)
    original = MODELS.build(reference.model.decode_head).to(device)
    original.init_weights()
    torch.manual_seed(2026)
    improved = MODELS.build(cfg.model.decode_head).to(device)
    improved.init_weights()
    missing = improved.load_state_dict(original.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('prototype_adapter.') for k in missing.missing_keys)
    return original, improved


def head_checks(cfg, reference, device, output):
    base, mp = heads(cfg, reference, device)
    core = mp.prototype_adapter.core
    x = features(device)
    labels = torch.arange(64, device=device)[None].expand(64, -1) // 8
    data = [sample(labels)]
    records = {}
    base.eval(); mp.eval()
    with torch.no_grad():
        original = base(x, data)
        cold = mp(x, data)
    same_tensors(original[0], cold[0]); same_tensors(original[1], cold[1])
    records['cold_bank_native_outputs_bitwise_equal'] = True

    # Current loss sees the cold bank; its GT can only affect the NEXT batch.
    base.train(); mp.train()
    torch.manual_seed(770)
    baseline_losses = base.loss(x, data, {})
    torch.manual_seed(770)
    first_losses = mp.loss(x, data, {})
    assert baseline_losses.keys() == first_losses.keys()
    same_tensors(list(baseline_losses.values()), list(first_losses.values()))
    assert int(core.update_steps) == 1 and core.prototype_valid.any()
    assert mp._pending_prototype_update is None
    records['first_loss_uses_old_bank_and_commits_once'] = True

    # A populated bank must remain a true M0 bypass when disabled.
    core.enabled = False
    base.eval(); mp.eval()
    with torch.no_grad():
        bypass = mp(x, data)
    same_tensors(original[0], bypass[0]); same_tensors(original[1], bypass[1])
    base.train(); mp.train()
    before = state(core)
    torch.manual_seed(881); a = base.loss(x, data, {})
    torch.manual_seed(881); b = mp.loss(x, data, {})
    same_tensors(list(a.values()), list(b.values())); same_state(before, core)
    records['populated_bank_disabled_outputs_and_losses_bitwise_equal'] = True
    core.enabled = True

    # Separate class/mask gradients establish both advertised paths.
    mp.train(); mp.zero_grad(set_to_none=True)
    before = state(core)
    classes, masks = mp(x, [sample()])  # metadata only, no annotation
    same_state(before, core)
    class_objective = classes[-1].float().square().mean()
    class_objective.backward(retain_graph=True)
    class_grads = {}
    for name, parameter in mp.prototype_adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        class_grads[name] = float(parameter.grad.abs().sum())
        assert class_grads[name] > 0, name
    mp.zero_grad(set_to_none=True)
    masks[-1].float().square().mean().backward()
    mask_grads = {}
    for name, parameter in mp.prototype_adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        mask_grads[name] = float(parameter.grad.abs().sum())
        assert mask_grads[name] > 0, name
    records['separate_class_and_mask_path_gradients'] = dict(
        class_grad_l1=class_grads, mask_grad_l1=mask_grads)

    # The commit occurs before backward, with frozen-core bank snapshots.
    mp.zero_grad(set_to_none=True)
    count = int(core.update_steps)
    losses = mp.loss(x, data, {})
    assert len(losses) == 30 and int(core.update_steps) == count + 1
    sum(losses.values()).backward()
    assert all(torch.isfinite(v).all() for v in losses.values())
    for name, parameter in mp.prototype_adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    records['native_30_losses_update_then_backward'] = True

    ignore = torch.full_like(labels, 255)
    before = state(core)
    mp.zero_grad(set_to_none=True)
    losses = mp.loss(x, [sample(ignore)], {})
    assert sum(float(v) for v in losses.values()) == 0
    sum(losses.values()).backward()
    same_state(before, core)
    records['all_ignore_zero_loss_no_bank_update'] = True

    partial = labels.clone(); partial[:8] = 255
    before = int(core.update_steps)
    losses = mp.loss(x, [sample(partial)], {})
    assert all(torch.isfinite(v).all() for v in losses.values())
    assert int(core.update_steps) == before + 1
    records['partial_ignore_losses_finite_single_commit'] = True

    # Label 4 only: absent classes retain identical counters and banks.
    before = state(core)
    losses = mp.loss(x, [sample(torch.full_like(labels, 4))], {})
    keep = torch.arange(8, device=device) != 4
    for name in ('prototype_bank', 'prototype_valid', 'assignment_counts', 'class_seen_pixels'):
        torch.testing.assert_close(before[name][keep], getattr(core, name)[keep], rtol=0, atol=0)
    records['absent_classes_not_updated'] = True

    mask = mp._geometry_mask(torch.empty(1, 256, 16, 16, device=device),
                             [sample(size=64, image_shape=(48, 40), old_pad_shape=(1024, 1024))])
    assert int(mask.sum()) == 12 * 10
    full = mp._geometry_mask(torch.empty(1, 256, 16, 16, device=device),
                             [sample(size=64, old_pad_shape=(1024, 1024))])
    assert full.all()
    records['geometry_mask_ignores_stale_whole_image_pad_shape'] = True

    mp.eval(); before = state(core)
    with torch.no_grad():
        expected = mp(x, [sample()])
        repeat = mp(x, [sample()])
        prediction = mp.predict(x, [sample(old_pad_shape=(1024, 1024)).metainfo],
                                dict(mode='slide'))
    same_tensors(expected[0], repeat[0]); same_tensors(expected[1], repeat[1])
    same_state(before, core)
    assert prediction.shape == (1, 8, 64, 64)
    assert prediction.argmax(1).max() < 8
    records['eval_without_GT_no_update_and_slide_dimensions'] = True

    stream = io.BytesIO(); torch.save(mp.state_dict(), stream)
    checkpoint_hash = hashlib.sha256(stream.getvalue()).hexdigest()
    stream.seek(0)
    restored = MODELS.build(cfg.model.decode_head).to(device).eval()
    restored.load_state_dict(torch.load(stream, map_location=device), strict=True)
    same_state(before, restored.prototype_adapter.core)
    with torch.no_grad():
        actual = restored(x, [sample()])
    same_tensors(expected[0], actual[0]); same_tensors(expected[1], actual[1])
    records['serialized_checkpoint_restores_outputs_and_bank'] = checkpoint_hash
    extra = sum(p.numel() for p in mp.parameters()) - sum(p.numel() for p in base.parameters())
    assert extra == 262400, extra
    records['added_trainable_parameters'] = extra
    records['prototype_diagnostics'] = core.diagnostics()
    records['synthetic_head_size'] = 64
    records['optimizer_updates'] = 0
    (output / 'head_checks.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


def model_check(cfg, device, size, pretrained_path=None, amp=False):
    model_cfg = copy.deepcopy(cfg.model)
    if pretrained_path:
        model_cfg.backbone.init_cfg.checkpoint = str(Path(pretrained_path).resolve())
    else:
        model_cfg.backbone.init_cfg = None
    model = MODELS.build(model_cfg)
    model.init_weights()
    pretrained = None
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        raw = checkpoint.get('state_dict', checkpoint)
        loaded = model.backbone.state_dict()
        matched, unmatched = [], []
        for name, value in loaded.items():
            if 'backbone.' + name in raw:
                torch.testing.assert_close(value, raw['backbone.' + name], rtol=0, atol=0)
                matched.append(name)
            else:
                unmatched.append(name)
        assert matched and all(k.startswith(('norm0.', 'norm1.', 'norm2.', 'norm3.')) for k in unmatched)
        pretrained = dict(sha256=sha(pretrained_path), matched_keys=len(matched),
                          unmatched_output_norm_keys=unmatched,
                          source_url=cfg.checkpoint_file)
        del checkpoint, raw, loaded
    model.to(device).train()
    # Check the production constructor includes every new parameter exactly
    # once; original LR/WD rules come from the frozen, unmodified M0 source.
    from aicseg.optimizer import AICMask2FormerOptimizerConstructor
    constructor = AICMask2FormerOptimizerConstructor(
        copy.deepcopy(cfg.optim_wrapper), copy.deepcopy(cfg.optim_wrapper.paramwise_cfg))
    groups = []
    constructor.add_params(groups, model)
    grouped = {name: (group['lr'], group['weight_decay'], id(parameter))
               for group in groups
               for name, parameter in zip(group['param_names'], group['params'])}
    named = dict(model.named_parameters())
    assert len(grouped) == sum(len(group['params']) for group in groups)
    assert set(grouped) == {name for name, p in named.items() if p.requires_grad}
    assert len({item[2] for item in grouped.values()}) == len(grouped)
    for name in named:
        if name.startswith('decode_head.prototype_adapter.'):
            assert grouped[name][:2] == (1e-4, 0. if named[name].ndim == 1 else .05), name
    group_record = dict(all_trainable_parameters_once=True,
                        new_P_and_adapter_parameters_at_head_learning_rate=True,
                        groups=[dict(name=g['group_name'], lr=g['lr'],
                                     weight_decay=g['weight_decay'], tensors=len(g['params']))
                                for g in groups])
    image = torch.randn(1, 3, size, size, device=device)
    labels = torch.arange(size, device=device)[None].expand(size, -1) * 8 // size
    data = [sample(labels, size=size)]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    # Match M0's diagnostic scale; the production AmpOptimWrapper remains
    # dynamic and unmodified. These two passes do not perform optimizer steps.
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    # Cold batch warms the bank; the second checks the active P AMP pathway.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            losses = model.loss(image, data)
            total = sum(losses.values())
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(total)
        if model.decode_head.prototype_adapter.core.update_steps >= 2:
            for name, parameter in model.decode_head.prototype_adapter.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
        # No optimizer.step: these are diagnostic passes, not a training run.
        scaler.update()
    if device == 'cuda':
        torch.cuda.synchronize()
    result = dict(size=size, physical_batch=1, amp=amp, forward_backward_passes=2,
                  optimizer_updates=0, elapsed_seconds=time.perf_counter() - start,
                  diagnostic_initial_loss_scale=128. if amp else None,
                  final_loss_scale=scaler.get_scale() if amp else None,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
                  pretrained=pretrained, optimizer_group_checks=group_record, loss=float(total),
                  prototype_update_steps=int(model.decode_head.prototype_adapter.core.update_steps),
                  bank_dtype=str(model.decode_head.prototype_adapter.core.prototype_bank.dtype))
    return result


def slide_check(cfg, device, size, amp=False):
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.init_cfg = None
    model = MODELS.build(model_cfg).to(device).eval()
    model.init_weights()
    core = model.decode_head.prototype_adapter.core
    # Warm only synthetic mechanics; prediction must not modify saved state.
    core.train()
    core.update_from_labels(torch.randn(1, 256, 16, 16, device=device),
                            torch.arange(64, device=device)[None, None].expand(1, 64, -1)//8)
    core.eval()
    before = state(core)
    image = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad(), torch.autocast(device_type=device, enabled=amp):
        predicted = model.inference(image, [sample(size=size).metainfo])
    assert predicted.shape == (1, 8, size, size)
    assert torch.isfinite(predicted).all()
    same_state(before, core)
    return dict(input_size=size, output_shape=list(predicted.shape),
                window=768, stride=512, no_GT=True, prototype_buffers_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', default='runs/validation_mp')
    parser.add_argument('--model-size', type=int, default=0)
    parser.add_argument('--pretrained', default=None,
                        help='Classification checkpoint; absolute or relative to invocation directory')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--skip-head', action='store_true')
    parser.add_argument('--slide-size', type=int, default=0)
    args = parser.parse_args()
    if args.pretrained:
        args.pretrained = str((INVOCATION_DIR / args.pretrained).resolve())
    if args.amp and args.device != 'cuda':
        parser.error('--amp requires CUDA')
    if args.model_size and (args.model_size < 64 or args.model_size % 32):
        parser.error('--model-size must be a multiple of32, at least64')
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    init_default_scope('mmseg')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    cfg = Config.fromfile(str(PROJECT_ROOT/'configs/mp_convnextb_mask2former_proto_768.py'))
    reference = Config.fromfile(str(PROJECT_ROOT/'configs/m0_convnextb_mask2former_768.py'))
    cfg.dump(str(output/'resolved_config.py'))
    report = dict(created_at=datetime.now().astimezone().isoformat(), command=sys.argv,
                  seed=2026, device=args.device, python=sys.version,
                  platform=platform.platform(), torch=torch.__version__, mmdet=mmdet.__version__,
                  gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                  source=source_and_config_checks(cfg, reference),
                  verification_script_sha256=sha(__file__), status='running')
    try:
        if not args.skip_head:
            report['head'] = head_checks(cfg, reference, args.device, output)
        if args.model_size:
            report['model'] = model_check(cfg, args.device, args.model_size,
                                         args.pretrained, args.amp)
        if args.slide_size:
            report['slide'] = slide_check(cfg, args.device, args.slide_size, args.amp)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'; report['error'] = repr(error)
        raise
    finally:
        (output/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(output),
                          added_parameters=report.get('head', {}).get('added_trainable_parameters'),
                          model=report.get('model')), indent=2))


if __name__ == '__main__':
    main()
