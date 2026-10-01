"""Resolve an AIC model run, predict the official test set, and package it."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment_results import load_run, resolve_run, summarize


def read_manifest(run_dir):
    path = run_dir / 'run_manifest.json'
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def checkpoints(run_dir):
    best = list(run_dir.rglob('best_*.pth'))
    if best:
        return sorted(best, key=lambda path: path.stat().st_mtime_ns)
    regular = list(run_dir.rglob('iter_*.pth'))
    return sorted(regular, key=lambda path: path.stat().st_mtime_ns)


def selected_checkpoint(run_dir):
    available = checkpoints(run_dir)
    if not available:
        raise ValueError(f'No checkpoint found in: {run_dir}')
    try:
        best_iter = summarize(load_run(run_dir))['best_iter']
        matched = [path for path in available
                   if f'_iter_{best_iter}.pth' in path.name]
        if matched:
            return matched[-1]
    except (OSError, ValueError):
        pass
    return available[-1]


def model_root_from(path):
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents):
        if (directory / 'tools/predict_and_pack.py').is_file():
            return directory
    raise ValueError(f'Cannot find model prediction entry point for: {path}')


def explicit_model_root(value):
    if value is None:
        return None
    root = Path(value).expanduser().resolve()
    if not (root / 'tools/predict_and_pack.py').is_file():
        raise ValueError(f'Model prediction entry point not found in: {root}')
    return root


def matching_runs(model_root, config=None):
    candidates = []
    runs_root = model_root / 'runs'
    if not runs_root.is_dir():
        return candidates
    for manifest_path in runs_root.rglob('run_manifest.json'):
        run_dir = manifest_path.parent
        manifest = read_manifest(run_dir)
        if manifest.get('run_type') == 'smoke':
            continue
        if config:
            recorded = Path(str(manifest.get('config', ''))).name
            if recorded != config.name:
                continue
        available = checkpoints(run_dir)
        if available:
            candidates.append((run_dir, selected_checkpoint(run_dir)))
    return sorted(candidates, key=lambda item: item[0].stat().st_mtime_ns)


def infer_config(run_dir, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Config not found: {path}')
        return path
    manifest = read_manifest(run_dir)
    values = [run_dir / 'resolved_config.py',
              Path(str(manifest.get('resolved_config', ''))).expanduser(),
              Path(str(manifest.get('config', ''))).expanduser()]
    values.extend(run_dir.glob('*.py'))
    for path in values:
        if str(path) != '.' and path.is_file():
            return path.resolve()
    raise ValueError('Cannot infer config beside the checkpoint; pass --config')


def resolve_source(source, explicit_config=None, model_dir=None):
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Source not found: {path}')
    model_root = explicit_model_root(model_dir)
    if path.is_file() and path.suffix == '.pth':
        run_dir = resolve_run(path)
        checkpoint = path
        config = infer_config(run_dir, explicit_config)
        return run_dir, checkpoint, config, model_root or model_root_from(config)
    if path.is_file() and path.suffix == '.py':
        model_root = model_root or model_root_from(path)
        runs = matching_runs(model_root, path)
        if not runs:
            raise ValueError(f'No formal run with checkpoint matches: {path}')
        run_dir, checkpoint = runs[-1]
        return run_dir, checkpoint, path, model_root
    if path.is_dir() and (path / 'run_manifest.json').is_file():
        run_dir = path
        checkpoint = selected_checkpoint(run_dir)
        config = infer_config(run_dir, explicit_config)
        return run_dir, checkpoint, config, model_root or model_root_from(config)
    if path.is_dir() and (path / 'tools/predict_and_pack.py').is_file():
        model_root = model_root or path
        config = (Path(explicit_config).expanduser().resolve()
                  if explicit_config else None)
        runs = matching_runs(model_root, config)
        if not runs:
            raise ValueError(f'No usable formal run found in: {model_root}')
        run_dir, checkpoint = runs[-1]
        config = infer_config(run_dir, config)
        return run_dir, checkpoint, config, model_root
    raise ValueError('Source must be a run directory, checkpoint, config, or model directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--config', help='Only needed when config cannot be inferred')
    parser.add_argument('--model-dir',
                        help='Model unit containing tools/predict_and_pack.py; '
                             'required for archived runs outside code/')
    args = parser.parse_args()
    try:
        run_dir, checkpoint, config, model_root = resolve_source(
            args.source, args.config, args.model_dir)
        output_dir = run_dir / 'submission' / checkpoint.stem
        zip_path = output_dir / f'{checkpoint.stem}.zip'
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f'Output directory is not empty: {output_dir}')
        command = [
            sys.executable, str(model_root / 'tools/predict_and_pack.py'),
            str(checkpoint), '--config', str(config),
            '--pred-dir', str(output_dir), '--zip', str(zip_path),
        ]
        print(f'RUN_DIR: {run_dir}')
        print(f'CONFIG: {config}')
        print(f'CHECKPOINT: {checkpoint}')
        print(f'OUTPUT: {output_dir}')
        subprocess.run(command, cwd=model_root, check=True)
        print(f'PASS: predictions and ZIP saved in {output_dir}')
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
