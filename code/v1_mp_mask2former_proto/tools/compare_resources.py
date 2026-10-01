"""Paired CUDA resource measurements; synthetic checks, no optimizer steps."""

import argparse
import copy
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(args):
    import torch
    import mmdet.models
    import mmpretrain.models
    import aicseg
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.structures import PixelData
    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    init_default_scope('mmseg')
    config_name = ('mp_convnextb_mask2former_proto_768.py' if args.variant == 'MP'
                   else 'm0_convnextb_mask2former_768.py')
    cfg = Config.fromfile(str(PROJECT_ROOT / 'configs' / config_name))
    model_cfg = copy.deepcopy(cfg.model)
    # Resource measurements do not estimate accuracy. Both common networks
    # are initialized identically without loading a segmentation checkpoint.
    model_cfg.backbone.init_cfg = None
    model = MODELS.build(model_cfg)
    model.init_weights()
    common_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith('decode_head.prototype_adapter.'):
            continue
        common_hash.update(name.encode())
        common_hash.update(value.detach().contiguous().numpy().tobytes())
    parameters = sum(p.numel() for p in model.parameters())
    model = model.cuda().train()
    image = torch.randn(1, 3, args.size, args.size, device='cuda')
    labels = (torch.arange(args.size, device='cuda')[None].expand(args.size, -1)
              * 8 // args.size)
    item = SegDataSample(metainfo=dict(img_shape=(args.size, args.size),
                                     ori_shape=(args.size, args.size),
                                     pad_shape=(args.size, args.size)))
    item.gt_sem_seg = PixelData(data=labels[None])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.)

    def forward_backward(index):
        torch.manual_seed(10000 + index)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            losses = model.loss(image, [item])
            total = sum(losses.values())
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(total)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        scaler.update()
        assert scaler.get_scale() == 128.
        return float(total)

    # First warmup initializes MP's cold bank. Later timing uses active P.
    for index in range(2):
        forward_backward(index)
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    seconds = []
    for index in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = forward_backward(index + 2)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    record = dict(variant=args.variant, seed=2026, size=args.size,
                  physical_batch=1, amp=True, loss_scale=128.,
                  warmup_passes=2, timed_forward_backward_passes=args.repeats,
                  optimizer_updates=0, parameters=parameters,
                  common_initial_state_sha256=common_hash.hexdigest(),
                  seconds=seconds, median_seconds=statistics.median(seconds),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  final_synthetic_loss=loss,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  version_package=str(Path(aicseg.__file__).resolve()))
    if args.variant == 'MP':
        record['prototype_update_steps'] = int(
            model.decode_head.prototype_adapter.core.update_steps)
    (Path(args.output) / (args.variant + '.json')).write_text(
        json.dumps(record, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--variant', choices=('M0', 'MP'))
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or not 3 <= args.repeats <= 10:
        parser.error('size must be >=64 and a multiple of32; repeats must be3..10')
    output = Path(args.output).resolve()
    if args.variant:
        measure(args)
        return
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    for variant in ('M0', 'MP'):
        command = [sys.executable, '-B', str(Path(__file__).resolve()),
                   '--output', str(output), '--size', str(args.size),
                   '--repeats', str(args.repeats), '--variant', variant]
        with (output / (variant + '_console.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    m0 = json.loads((output / 'M0.json').read_text())
    mp = json.loads((output / 'MP.json').read_text())
    assert m0['common_initial_state_sha256'] == mp['common_initial_state_sha256']
    record = dict(created_at=datetime.now().astimezone().isoformat(),
                  command=sys.argv, script_sha256=digest(__file__),
                  M0=m0, MP=mp,
                  added_parameters=mp['parameters'] - m0['parameters'],
                  allocated_mib_delta=mp['peak_allocated_mib'] - m0['peak_allocated_mib'],
                  reserved_mib_delta=mp['peak_reserved_mib'] - m0['peak_reserved_mib'],
                  median_time_ratio=mp['median_seconds'] / m0['median_seconds'],
                  common_initial_parameters_identical=True,
                  limitations=['Synthetic labels/input; not an accuracy evaluation',
                               'No optimizer updates or AdamW state; B1 only',
                               'Sequential processes on a laptop GPU; short timing sample',
                               'Does not prove server B2 fits or predict a full training duration'])
    assert record['added_parameters'] == 262400
    (output / 'comparison.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
