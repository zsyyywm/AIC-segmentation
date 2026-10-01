"""Training time, validation metric, and CUDA memory reporting."""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class AICRunStatsHook(Hook):
    """Persist a compact run summary while ordinary details stay in JSON logs."""

    priority = 'LOW'

    def before_train(self, runner):
        self.started_at = time.perf_counter()
        self.latest_metrics = {}
        self.previous_training_seconds = 0.0
        previous = Path(runner.work_dir) / 'run_summary.json'
        if runner.cfg.get('resume', False) and previous.is_file():
            try:
                old_summary = json.loads(previous.read_text(encoding='utf-8'))
                self.previous_training_seconds = float(
                    old_summary.get('training_time_seconds', 0.0))
            except (OSError, ValueError, TypeError):
                pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if torch.cuda.is_available():
            runner.message_hub.update_scalar(
                'train/max_memory_allocated_mb',
                torch.cuda.max_memory_allocated() / (1024 ** 2))
            runner.message_hub.update_scalar(
                'train/max_memory_reserved_mb',
                torch.cuda.max_memory_reserved() / (1024 ** 2))

    def after_val_epoch(self, runner, metrics=None):
        if metrics:
            self.latest_metrics = {
                key: float(value) for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        self._write_summary(runner, finished=False)
        self._update_plot(runner)

    def after_train(self, runner):
        self._write_summary(runner, finished=True)
        self._update_plot(runner)

    def _update_plot(self, runner):
        script = Path(__file__).resolve().parents[3] / 'tools/plot_training.py'
        if not script.is_file():
            runner.logger.warning('Training plot tool not found: %s', script)
            return
        result = subprocess.run(
            [sys.executable, str(script), str(runner.work_dir), '--quiet'],
            capture_output=True, text=True, check=False)
        if result.returncode:
            message = (result.stderr or result.stdout).strip()
            runner.logger.warning('Could not update training plot: %s', message)

    def _write_summary(self, runner, finished):
        elapsed = (time.perf_counter() - self.started_at
                   + self.previous_training_seconds)
        accumulation_steps = int(
            runner.cfg.optim_wrapper.get('accumulative_counts', 1))
        physical_batch_size = int(runner.cfg.train_dataloader.batch_size)
        train_size = int(runner.cfg.get('train_size', 0))
        completed_iterations = int(runner.iter)
        summary = {
            'finished': finished,
            'completed_micro_iterations': completed_iterations,
            'completed_optimizer_steps': (
                completed_iterations // accumulation_steps),
            'gradient_accumulation_steps': accumulation_steps,
            'physical_batch_size': physical_batch_size,
            'effective_batch_size': (
                physical_batch_size * accumulation_steps),
            'completed_sample_exposures': (
                completed_iterations * physical_batch_size),
            'completed_equivalent_epochs': round(
                completed_iterations * physical_batch_size / train_size, 4)
            if train_size else None,
            'training_time_seconds': round(elapsed, 3),
            'training_time_hours': round(elapsed / 3600, 4),
            'latest_validation': self.latest_metrics,
        }
        if torch.cuda.is_available():
            summary['gpu_name'] = torch.cuda.get_device_name()
            summary['peak_memory_allocated_mb'] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            summary['peak_memory_reserved_mb'] = round(
                torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
        else:
            summary['gpu_name'] = None
            summary['peak_memory_allocated_mb'] = 0.0
            summary['peak_memory_reserved_mb'] = 0.0

        output = Path(runner.work_dir) / 'run_summary.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding='utf-8')
