"""Colorized terminal monitor for MMEngine JSON scalar logs."""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from mmengine.config import Config
from rich.console import Console
from rich.text import Text

from run_utils import (experiment_name_from_config, load_manifest,
                       resolve_project_path, resolve_run)


CLASS_NAMES = (
    'Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
    'Agricultural', 'Vehicle')


@dataclass
class MonitorState:
    max_iterations: int
    train_size: int
    physical_batch_size: int
    accumulation_steps: int
    best_miou: float = -math.inf
    best_iteration: int = 0


def _number(record, *names):
    for name in names:
        if name in record and isinstance(record[name], (int, float)):
            return float(record[name])
    for key, value in record.items():
        if any(key.endswith(f'/{name}') for name in names) and isinstance(
                value, (int, float)):
            return float(value)
    return None


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours:d}h{minutes:02d}m' if hours else f'{minutes:d}m{secs:02d}s'


def _append_value(text, label, value, style, fmt='.4f'):
    if value is None:
        return
    if text.plain.strip():
        text.append(' | ', style='dim')
    text.append(f'{label} ', style='bold')
    text.append(format(value, fmt), style=style)


def print_colored_training_status(record, state, console=None):
    """Print one training or validation scalar record with stable colors."""
    console = console or Console()
    step = int(_number(record, 'step', 'iter') or 0)
    miou = _number(record, 'mIoU')
    loss = _number(record, 'loss')
    is_validation = miou is not None or any(
        'IoU/' in key for key in record)

    if is_validation:
        header = Text.assemble(('[VAL] ', 'bold green'),
                               (f'Iter {step}', 'bold cyan'))
        _append_value(header, 'mIoU', miou, 'bold green', '.2f')
        _append_value(header, 'mAcc', _number(record, 'mAcc'), 'green', '.2f')
        _append_value(header, 'aAcc', _number(record, 'aAcc'), 'green', '.2f')
        console.print(header)

        classes = Text('       ')
        for class_name in CLASS_NAMES:
            value = _number(record, f'IoU/{class_name}')
            if value is not None:
                _append_value(classes, class_name, value, 'green', '.2f')
        if len(classes.plain.strip()):
            console.print(classes)

        if miou is not None and miou > state.best_miou:
            state.best_miou = miou
            state.best_iteration = step
            console.print(
                f'[BEST] mIoU {miou:.2f} @ Iter {step}', style='bold green')
        return

    if loss is None:
        return

    samples_seen = step * state.physical_batch_size
    equiv_epoch = (samples_seen / state.train_size
                   if state.train_size else 0.0)
    optimizer_step = step // max(1, state.accumulation_steps)
    progress = (100 * step / state.max_iterations
                if state.max_iterations else 0.0)
    iter_time = _number(record, 'time')
    eta = ((state.max_iterations - step) * iter_time
           if iter_time is not None and state.max_iterations else None)

    header = Text.assemble(
        ('[TRAIN] ', 'bold cyan'),
        (f'Epoch~{equiv_epoch:.2f}', 'bold cyan'),
        (' | ', 'dim'),
        (f'Iter {step}/{state.max_iterations}', 'cyan'),
        (' | ', 'dim'),
        (f'OptStep {optimizer_step}', 'cyan'),
        (' | ', 'dim'),
        (f'{progress:.2f}%', 'cyan'))
    console.print(header)

    metrics = Text('        ')
    _append_value(metrics, 'Loss', loss, 'bold yellow', '.4f')
    _append_value(
        metrics, 'Main', _number(record, 'decode.loss_ce'), 'yellow', '.4f')
    _append_value(
        metrics, 'Aux', _number(record, 'aux.loss_ce'), 'yellow', '.4f')
    _append_value(
        metrics, 'Acc', _number(record, 'decode.acc_seg'), 'yellow', '.2f')
    _append_value(metrics, 'LR', _number(record, 'lr'), 'magenta', '.3e')
    console.print(metrics)

    runtime = Text('        ')
    _append_value(runtime, 's/iter', iter_time, 'blue', '.3f')
    if runtime.plain.strip():
        runtime.append(' | ', style='dim')
    runtime.append(f'ETA {_format_duration(eta)}', style='blue')
    memory = _number(record, 'train/max_memory_reserved_mb',
                     'max_memory_reserved_mb', 'memory')
    if memory is not None:
        runtime.append(' | ', style='dim')
        runtime.append(f'VRAM {memory / 1024:.2f} GiB', style='bold blue')
    console.print(runtime)

    if not math.isfinite(loss):
        console.print('[ALERT] Loss is NaN or infinite.', style='bold red')


def _read_records(path):
    records = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _latest_scalar_file(run_dir):
    candidates = list(Path(run_dir).rglob('scalars.json'))
    return (max(candidates, key=lambda path: path.stat().st_mtime)
            if candidates else None)


def _is_validation_record(record):
    return _number(record, 'mIoU') is not None or any(
        'IoU/' in key for key in record)


def _is_training_record(record):
    return _number(record, 'loss') is not None


def _print_initial_snapshot(records, state, console):
    for record in records:
        miou = _number(record, 'mIoU')
        if miou is not None and miou > state.best_miou:
            state.best_miou = miou
            state.best_iteration = int(_number(record, 'step', 'iter') or 0)

    latest_train = next(
        (item for item in reversed(records) if _is_training_record(item)),
        None)
    latest_val = next(
        (item for item in reversed(records) if _is_validation_record(item)),
        None)
    if latest_train:
        print_colored_training_status(latest_train, state, console)
    if latest_val:
        print_colored_training_status(latest_val, state, console)
    if state.best_miou > -math.inf:
        console.print(
            f'[BEST] mIoU {state.best_miou:.2f} @ '
            f'Iter {state.best_iteration}', style='bold green')


def _state_from_run(run_dir):
    manifest = load_manifest(run_dir)
    return MonitorState(
        max_iterations=int(manifest.get('max_iterations', 0)),
        train_size=int(manifest.get('train_size', 5596)),
        physical_batch_size=int(manifest.get('physical_batch_size', 1)),
        accumulation_steps=int(
            manifest.get('gradient_accumulation_steps', 8)))


def parse_args():
    parser = argparse.ArgumentParser(description='Monitor AIC training logs')
    parser.add_argument(
        '--config', default='configs/mp_convnextb_mask2former_proto_768.py')
    parser.add_argument(
        '--run', default='latest',
        help='Run directory relative to the project, or latest')
    parser.add_argument('--refresh', type=float, default=1.0)
    parser.add_argument('--once', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run == 'latest':
        config_path = resolve_project_path(args.config)
        cfg = Config.fromfile(str(config_path))
        experiment_name = experiment_name_from_config(cfg, config_path)
        run_dir = resolve_run(args.run, experiment_name, prefer_train=False)
    else:
        run_dir = resolve_project_path(args.run)
        if not run_dir.is_dir():
            raise FileNotFoundError(f'Run directory does not exist: {run_dir}')
    state = _state_from_run(run_dir)
    # Resume creates new scalar files; retain the best result across sessions.
    for path in run_dir.rglob('scalars.json'):
        for record in _read_records(path):
            miou = _number(record, 'mIoU')
            if miou is not None and miou > state.best_miou:
                state.best_miou = miou
                state.best_iteration = int(_number(record, 'step', 'iter') or 0)
    console = Console()
    console.print(f'RUN_DIR: {run_dir}', style='bold cyan')
    console.print(
        'Epoch~ is an equivalent epoch for IterBased training; '
        'mIoU updates after validation.', style='dim')

    scalar_path = None
    seen = 0
    announced_wait = False
    while True:
        latest = _latest_scalar_file(run_dir)
        if latest != scalar_path:
            scalar_path = latest
            seen = 0
            if scalar_path:
                console.print(f'LOG: {scalar_path}', style='dim')

        if scalar_path is None:
            if args.once:
                raise FileNotFoundError(f'No scalars.json found in {run_dir}')
            if not announced_wait:
                console.print('Waiting for MMEngine scalars.json ...',
                              style='yellow')
                announced_wait = True
            time.sleep(max(0.2, args.refresh))
            continue

        records = _read_records(scalar_path)
        if seen == 0 and records:
            _print_initial_snapshot(records, state, console)
        else:
            for record in records[seen:]:
                print_colored_training_status(record, state, console)
        seen = len(records)
        if args.once:
            return
        time.sleep(max(0.2, args.refresh))


if __name__ == '__main__':
    main()
