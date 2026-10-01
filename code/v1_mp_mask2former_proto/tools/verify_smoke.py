"""Audit the completed short run's optimizer and persisted prototype state."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

from _bootstrap import bootstrap

bootstrap()

import torch


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--checkpoint', default='iter_8.pth')
    parser.add_argument('--expected-micro', type=int, default=8)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    output = run / 'mp_checkpoint_verification.json'
    if output.exists():
        raise FileExistsError(output)
    path = run / args.checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    assert manifest['run_type'] == 'smoke'
    assert checkpoint['meta']['iter'] == args.expected_micro
    updates = args.expected_micro // manifest['gradient_accumulation_steps']
    steps = Counter(float(value['step'])
                    for value in checkpoint['optimizer']['state'].values())
    assert len(steps) == 1 and updates in steps, steps
    prefix = 'decode_head.prototype_adapter.core.'
    state = checkpoint['state_dict']
    assert int(state[prefix + 'update_steps']) == args.expected_micro
    assert state[prefix + 'prototype_bank'].dtype == torch.float32
    assert torch.isfinite(state[prefix + 'prototype_bank']).all()
    assert torch.isfinite(state[prefix + 'residual_gate']).all()
    diagnostics = [json.loads(line) for line in
                   (run / 'prototype_diagnostics.jsonl').read_text().splitlines()]
    assert len(diagnostics) == args.expected_micro
    assert [item['micro_iteration'] for item in diagnostics] == list(
        range(1, args.expected_micro + 1))
    assert diagnostics[-1]['update_steps'] == args.expected_micro
    log = (run / 'console.log').read_text(encoding='utf-8')
    peaks = [int(value) for value in re.findall(r'memory: (\d+)', log)]
    result = dict(status='passed', created_at=datetime.now().astimezone().isoformat(),
                  command=sys.argv, script_sha256=sha(__file__),
                  checkpoint=str(path), checkpoint_sha256=sha(path),
                  micro_iterations=args.expected_micro,
                  optimizer_parameter_step_histogram=dict(steps),
                  prototype_updates=args.expected_micro,
                  prototype_bank_dtype=str(state[prefix + 'prototype_bank'].dtype),
                  valid_slots=int(state[prefix + 'prototype_valid'].sum()),
                  diagnostics_records=len(diagnostics),
                  final_loss_scale=checkpoint['optimizer']['loss_scaler']['scale'],
                  seed=manifest['seed'],
                  logger_observed_peak_allocated_mib=max(peaks),
                  resource_note='Logger resets CUDA peak counters; the copied RunStatsHook '
                                'summary after validation is not the entire training peak',
                  formal_training=False, full_validation=False)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
