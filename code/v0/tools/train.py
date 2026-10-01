"""Single-GPU training entry point with automatic run management."""

import argparse
import contextlib
import hashlib
import json
import math
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from run_utils import (create_run_dir, experiment_name_from_config,
                       latest_checkpoint,
                       resolve_project_path, resolve_run)


class TeeStream:
    """Mirror one text stream to a line-buffered log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, value):
        self.terminal.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


def parse_args():
    parser = argparse.ArgumentParser(description='Train the AIC baseline')
    parser.add_argument(
        '--config', required=True)
    parser.add_argument('--work-dir', default=None)
    parser.add_argument(
        '--resume', nargs='?', const='latest', default=None,
        help='Resume latest run, or pass an explicit run/checkpoint path')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Run the fixed short smoke-test schedule')
    parser.add_argument('--smoke-iters', type=int, default=300)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def apply_smoke_settings(cfg, max_iters):
    if max_iters < 2:
        raise ValueError('--smoke-iters must be at least 2')
    warmup_iters = min(30, max(1, max_iters // 10))
    cfg.train_cfg.max_iters = max_iters
    cfg.train_cfg.val_interval = max_iters
    cfg.default_hooks.checkpoint.interval = max_iters
    if len(cfg.param_scheduler) >= 2:
        cfg.param_scheduler[0].end = warmup_iters
        cfg.param_scheduler[1].begin = warmup_iters
        cfg.param_scheduler[1].end = max_iters


def resolve_resume(args, experiment_name):
    value = args.resume
    if value == 'latest':
        run_dir = (resolve_project_path(args.work_dir) if args.work_dir else
                   resolve_run('latest', experiment_name, prefer_train=True))
        return run_dir, latest_checkpoint(run_dir)

    path = resolve_project_path(value)
    if path.is_file():
        return path.parent, path
    if path.is_dir():
        return path, latest_checkpoint(path)
    raise FileNotFoundError(f'Resume target does not exist: {path}')


def split_hash(dataloader):
    dataset = dataloader.dataset
    while 'data_root' not in dataset and 'dataset' in dataset:
        dataset = dataset.dataset
    root = Path(dataset.get('data_root', ''))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    image_dir = root / dataset.get('data_prefix', {}).get('img_path', '')
    names = sorted(path.name for path in image_dir.glob('*.png'))
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def model_description(cfg):
    model = cfg.model
    backbone = model.get('backbone', {})
    decode = model.get('decode_head', {})
    auxiliary = model.get('auxiliary_head', {})
    return {
        'type': model.get('type'),
        'backbone': backbone.get('arch', backbone.get('type')),
        'backbone_type': backbone.get('type'),
        'decode_head': decode.get('type'),
        'auxiliary_head': auxiliary.get('type'),
        'num_classes': decode.get('num_classes'),
    }


def write_manifest(cfg, args, config_path, experiment_name, run_dir,
                   resume_from, resolved_config):
    train_dataset = cfg.train_dataloader.dataset
    train_size = int(cfg.get('train_size', 0))
    physical_batch_size = int(cfg.train_dataloader.batch_size)
    accumulation_steps = int(
        cfg.optim_wrapper.get('accumulative_counts', 1))
    max_iterations = int(cfg.train_cfg.max_iters)
    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'experiment_name': experiment_name,
        'run_type': 'smoke' if args.smoke else 'train',
        'run_dir': str(run_dir),
        'config': str(config_path),
        'resolved_config': str(resolved_config),
        'command': sys.argv,
        'resume_from': str(resume_from) if resume_from else None,
        'max_iterations': max_iterations,
        'validation_interval': int(cfg.train_cfg.val_interval),
        'train_size': train_size,
        'physical_batch_size': physical_batch_size,
        'gradient_accumulation_steps': accumulation_steps,
        'effective_batch_size': physical_batch_size * accumulation_steps,
        'planned_sample_exposures': max_iterations * physical_batch_size,
        'planned_equivalent_epochs': round(
            max_iterations * physical_batch_size / train_size, 4)
        if train_size else None,
        'planned_optimizer_steps': math.ceil(
            max_iterations / accumulation_steps),
        'model': model_description(cfg),
        'seed': cfg.get('randomness', {}).get('seed'),
        'pretrained_checkpoint': cfg.model.get(
            'backbone', {}).get('init_cfg', {}).get('checkpoint'),
        'pretraining_policy': 'public_academic_classification_backbone',
        'crop_size': list(cfg.model.get('test_cfg', {}).get('crop_size', [])),
        'validation_mode': cfg.model.get('test_cfg', {}).get('mode'),
        'validation_stride': list(
            cfg.model.get('test_cfg', {}).get('stride', [])),
        'train_split_hash': split_hash(cfg.train_dataloader),
        'val_split_hash': split_hash(cfg.val_dataloader),
        'train_data_root': str(train_dataset.get('data_root', '')),
        'train_image_path': str(
            train_dataset.get('data_prefix', {}).get('img_path', '')),
        'python': sys.version.split()[0],
        'pytorch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu': (torch.cuda.get_device_name() if torch.cuda.is_available()
                else None),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
    }
    output = Path(run_dir) / 'run_manifest.json'
    if args.resume and output.is_file():
        try:
            previous = json.loads(output.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            previous = {}
        history = list(previous.get('resume_history', []))
        history.append({
            'resumed_at': manifest['created_at'],
            'checkpoint': manifest['resume_from'],
            'resolved_config': manifest['resolved_config'],
            'command': manifest['command'],
        })
        manifest['created_at'] = previous.get(
            'created_at', manifest['created_at'])
        manifest['resume_history'] = history
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = parse_args()
    if args.smoke and args.resume:
        raise ValueError('--smoke and --resume cannot be used together')

    config_path = resolve_project_path(args.config)
    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.smoke:
        apply_smoke_settings(cfg, args.smoke_iters)

    experiment_name = experiment_name_from_config(cfg, config_path)
    resume_from = None
    if args.resume:
        run_dir, resume_from = resolve_resume(args, experiment_name)
        cfg.resume = True
        cfg.load_from = str(resume_from)
    else:
        run_dir = (resolve_project_path(args.work_dir) if args.work_dir else
                   create_run_dir(
                       experiment_name,
                       run_type='smoke' if args.smoke else 'train'))
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.resume = False

    cfg.work_dir = str(run_dir)
    config_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    resolved_config = run_dir / f'resolved_config_{config_stamp}.py'
    cfg.dump(str(resolved_config))
    write_manifest(cfg, args, config_path, experiment_name, run_dir,
                   resume_from, resolved_config)

    console_path = run_dir / 'console.log'
    with console_path.open(
            'a' if args.resume else 'w', encoding='utf-8', buffering=1) as log:
        stdout = TeeStream(sys.stdout, log)
        stderr = TeeStream(sys.stderr, log)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr):
            print(f'EXPERIMENT: {experiment_name}')
            print(f'RUN_DIR: {run_dir}')
            print(f'CONFIG: {config_path}')
            if resume_from:
                print(f'RESUME_FROM: {resume_from}')
            Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
