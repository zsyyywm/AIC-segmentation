"""Shared parsing and reporting helpers for AIC experiment tools."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev


AIC_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = AIC_ROOT / 'result.md'
CLASSES = ('Background', 'Building', 'Road', 'Water', 'Barren',
           'Vegetation', 'Agricultural', 'Vehicle')


def number(record, key):
    value = record.get(key)
    if value is None:
        value = next((item for name, item in record.items()
                      if name.endswith('/' + key)), None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def _read_object(path, warnings):
    if not path.is_file():
        warnings.append(f'{path.name}: missing')
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (OSError, ValueError) as exc:
        warnings.append(f'{path.name}: {exc}')
        return {}


def _step(record):
    value = number(record, 'step')
    if value is None:
        value = number(record, 'iter')
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def resolve_run(source):
    """Resolve a run directory from a run directory, log file, or checkpoint."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Path not found: {path}')
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents):
        if (directory / 'run_manifest.json').is_file():
            return directory
        if directory == AIC_ROOT:
            break
    if start.is_dir() and list(start.rglob('scalars.json')):
        return start
    raise ValueError(f'Cannot locate a training run for: {path}')


@dataclass
class RunData:
    directory: Path
    manifest: dict
    summary: dict
    training: dict
    validations: dict
    warnings: list


def load_run(source):
    directory = resolve_run(source)
    warnings = []
    manifest = _read_object(directory / 'run_manifest.json', warnings)
    summary = _read_object(directory / 'run_summary.json', warnings)
    training, validations = {}, {}
    paths = sorted(directory.rglob('scalars.json'),
                   key=lambda path: (path.stat().st_mtime_ns, str(path)))
    if not paths:
        warnings.append('scalars.json: missing')
    for path in paths:
        bad_lines = 0
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    bad_lines += 1
                    continue
                if not isinstance(record, dict):
                    continue
                step = _step(record)
                if step is None:
                    bad_lines += 1
                    continue
                if number(record, 'mIoU') is not None:
                    validations[step] = record
                if number(record, 'loss') is not None:
                    training[step] = record
        if bad_lines:
            warnings.append(
                f'{path.relative_to(directory)}: {bad_lines} invalid lines skipped')
    if not validations:
        raise ValueError(f'No validation mIoU found in: {directory}')
    return RunData(directory, manifest, summary, training, validations,
                   warnings)


def stable_window_size(count):
    if count < 3:
        return count
    return min(5, max(3, math.ceil(count * 0.30)))


def _slope(values):
    if len(values) < 2:
        return None
    center = (len(values) - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    return (sum((index - center) * value
                for index, value in enumerate(values)) / denominator)


def _normalized_auc(points):
    if len(points) < 2:
        return None
    area = sum((right[0] - left[0]) * (left[1] + right[1]) / 2
               for left, right in zip(points, points[1:]))
    width = points[-1][0] - points[0][0]
    return area / width if width > 0 else None


def _checkpoint(run, requested=None):
    if requested:
        path = Path(requested).expanduser().resolve()
        return path if path.is_file() else None
    best = list(run.directory.rglob('best_*.pth'))
    if best:
        return max(best, key=lambda path: path.stat().st_mtime_ns)
    regular = list(run.directory.rglob('iter_*.pth'))
    return (max(regular, key=lambda path: path.stat().st_mtime_ns)
            if regular else None)


def summarize(run, requested_checkpoint=None):
    steps = sorted(run.validations)
    records = [run.validations[step] for step in steps]
    values = [number(record, 'mIoU') for record in records]
    window_n = stable_window_size(len(values))
    tail = values[-window_n:] if window_n else []
    physical_batch = number(run.manifest, 'physical_batch_size') or 1
    points = [(step * physical_batch, value)
              for step, value in zip(steps, values)]
    best_index = max(range(len(values)), key=values.__getitem__)
    class_metrics = {}
    for class_name in CLASSES:
        all_values = [number(record, f'IoU/{class_name}')
                      for record in records]
        available = [value for value in all_values[-window_n:]
                     if value is not None]
        class_metrics[class_name] = {
            'final': all_values[-1],
            'representative': median(available) if available else None,
            'std': pstdev(available) if len(available) > 1 else 0.0
            if available else None,
        }
    losses = [number(run.training[step], 'loss')
              for step in sorted(run.training)]
    losses = [value for value in losses if value is not None]
    checkpoint = _checkpoint(run, requested_checkpoint)
    slope = _slope(tail)
    tail_std = pstdev(tail) if len(tail) > 1 else 0.0
    if len(tail) < 3:
        state = '验证节点不足，当前证据不充分'
    elif slope is not None and slope > 0.10:
        state = '稳定窗口仍呈上升趋势，可能尚未充分收敛'
    elif slope is not None and slope < -0.10:
        state = '稳定窗口呈下降趋势，需要检查回落或过拟合'
    elif tail_std <= 0.25:
        state = '稳定窗口波动较小，训练基本收敛'
    else:
        state = '稳定窗口波动较大，单次最高值代表性有限'
    return {
        'steps': steps,
        'validation_count': len(steps),
        'window_size': window_n,
        'representative_miou': median(tail),
        'tail_mean': mean(tail),
        'tail_std': tail_std,
        'tail_slope_per_validation': slope,
        'final_miou': values[-1],
        'best_miou': values[best_index],
        'best_iter': steps[best_index],
        'best_final_gap': values[best_index] - values[-1],
        'normalized_auc': _normalized_auc(points),
        'class_metrics': class_metrics,
        'first_loss': losses[0] if losses else None,
        'final_loss': losses[-1] if losses else None,
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_size_mb': round(checkpoint.stat().st_size / 1024 ** 2, 2)
        if checkpoint else None,
        'state': state,
    }


def fmt(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def run_identity(run):
    manifest = run.manifest
    return {
        'run_id': run.directory.name,
        'experiment': manifest.get('experiment_name', 'N/A'),
        'config': manifest.get('resolved_config', manifest.get('config', 'N/A')),
        'model': manifest.get('model', {}),
        'max_iterations': manifest.get('max_iterations', 'N/A'),
        'equivalent_epochs': manifest.get('planned_equivalent_epochs', 'N/A'),
        'physical_batch': manifest.get('physical_batch_size', 'N/A'),
        'accumulation': manifest.get('gradient_accumulation_steps', 'N/A'),
        'effective_batch': manifest.get('effective_batch_size', 'N/A'),
        'seed': manifest.get('seed', 'N/A'),
        'pretrained': manifest.get('pretrained_checkpoint', 'N/A'),
    }


def append_result(markdown):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULT_PATH.is_file()
    with RESULT_PATH.open('a', encoding='utf-8', newline='\n') as handle:
        if new_file:
            handle.write('# AIC 实验结果记录\n\n')
            handle.write('本文件由实验分析与对比工具按调用时间追加记录。每条记录包含独立的运行目录、日期、配置、模型、训练预算和指标，历史结果不覆盖。\n\n')
        handle.write(markdown.rstrip() + '\n\n')
    return RESULT_PATH


def report_time():
    return datetime.now().astimezone().isoformat(timespec='seconds')
