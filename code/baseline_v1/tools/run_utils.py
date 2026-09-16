"""Run directory and checkpoint helpers shared by command-line tools."""

import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / 'runs'


def experiment_name_from_config(cfg, config_path):
    """Return a filesystem-safe experiment name."""
    raw_name = cfg.get('experiment_name', Path(config_path).stem)
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(raw_name)).strip('._')
    if not name:
        raise ValueError('experiment_name must contain a usable character')
    return name


def create_run_dir(experiment_name, run_type='train', now=None):
    """Create a unique timestamped directory for one run."""
    timestamp = (now or datetime.now()).strftime('%Y%m%d_%H%M%S')
    parent = RUNS_ROOT / experiment_name
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / f'{timestamp}_{run_type}'
    suffix = 1
    while candidate.exists():
        candidate = parent / f'{timestamp}_{run_type}_{suffix:02d}'
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate.resolve()


def resolve_project_path(value):
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def latest_run_dir(experiment_name, prefer_train=True):
    """Find the newest run directory for an experiment."""
    parent = RUNS_ROOT / experiment_name
    if not parent.is_dir():
        raise FileNotFoundError(f'No runs found for experiment: {experiment_name}')
    candidates = [path for path in parent.iterdir() if path.is_dir()]
    if prefer_train:
        candidates = [path for path in candidates if '_train' in path.name]
    if not candidates:
        raise FileNotFoundError(f'No run directories found in: {parent}')
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def latest_run_and_best_checkpoint(experiment_name):
    """Find the newest completed training run that has a checkpoint."""
    parent = RUNS_ROOT / experiment_name
    if not parent.is_dir():
        raise FileNotFoundError(f'No runs found for experiment: {experiment_name}')
    candidates = sorted(
        (path for path in parent.iterdir()
         if path.is_dir() and '_train' in path.name),
        key=lambda path: path.stat().st_mtime,
        reverse=True)
    for run_dir in candidates:
        try:
            return run_dir.resolve(), best_checkpoint(run_dir)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f'No training run with a checkpoint found in: {parent}')


def _checkpoint_from_marker(run_dir):
    marker = run_dir / 'last_checkpoint'
    if not marker.is_file():
        return None
    value = marker.read_text(encoding='utf-8').strip()
    if not value:
        return None
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [run_dir / path,
                                                     PROJECT_ROOT / path]
    return next((item.resolve() for item in candidates if item.is_file()), None)


def latest_checkpoint(run_dir):
    """Find the checkpoint used to resume a run."""
    run_dir = resolve_project_path(run_dir)
    marked = _checkpoint_from_marker(run_dir)
    if marked is not None:
        return marked
    candidates = list(run_dir.rglob('iter_*.pth'))
    if not candidates:
        candidates = list(run_dir.rglob('*.pth'))
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found in: {run_dir}')
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def best_checkpoint(run_dir):
    """Prefer the best validation checkpoint, then fall back to the latest."""
    run_dir = resolve_project_path(run_dir)
    candidates = list(run_dir.rglob('best_*.pth'))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()
    return latest_checkpoint(run_dir)


def resolve_run(value, experiment_name, prefer_train=True):
    """Resolve ``latest`` or an explicit run directory."""
    if value in (None, 'latest'):
        return latest_run_dir(experiment_name, prefer_train=prefer_train)
    path = resolve_project_path(value)
    if not path.is_dir():
        raise FileNotFoundError(f'Run directory does not exist: {path}')
    return path


def load_manifest(run_dir):
    path = Path(run_dir) / 'run_manifest.json'
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}
