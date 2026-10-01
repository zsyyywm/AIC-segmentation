"""Periodic prototype evidence without adding loss terms."""

import json
from pathlib import Path

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class PrototypeDiagnosticsHook(Hook):
    priority = 'LOW'

    def __init__(self, interval=50, core_path='decode_head.prototype_adapter.core'):
        self.interval = int(interval)
        self.core_path = str(core_path)
        if self.interval < 1:
            raise ValueError('interval must be positive')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.interval:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        core = model
        for component in self.core_path.split('.'):
            core = getattr(core, component)
        stats = core.diagnostics()
        stats['micro_iteration'] = int(runner.iter + 1)
        path = Path(runner.work_dir) / 'prototype_diagnostics.jsonl'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(stats, ensure_ascii=False) + '\n')
        for name in ('gate_mean_abs', 'residual_ratio'):
            if name in stats:
                runner.message_hub.update_scalar(f'train/proto_{name}', stats[name])
        runner.message_hub.update_scalar('train/proto_valid_slots',
                                        sum(sum(row) for row in stats['valid_slots']))
