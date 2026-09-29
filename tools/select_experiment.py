"""Read-only round-one candidate selection; does not train, submit or edit result.md."""

import argparse
import json
from statistics import mean

from experiment_results import load_run, number


FAIR_KEYS = ('train_split_hash', 'val_split_hash', 'train_size',
             'physical_batch_size', 'gradient_accumulation_steps',
             'effective_batch_size', 'pretrained_checkpoint', 'pretraining_policy',
             'crop_size', 'validation_mode', 'validation_stride', 'validation_interval')


def run_score(run, seed):
    errors = list(run.warnings)
    m, s = run.manifest, run.summary
    checks = dict(run_type='train', seed=seed, max_iterations=40000,
                  physical_batch_size=2, gradient_accumulation_steps=4,
                  effective_batch_size=8, validation_interval=4000, train_size=5596)
    for key, expected in checks.items():
        if m.get(key) != expected:
            errors.append(f'{key}: expected {expected!r}, got {m.get(key)!r}')
    for key, expected in dict(finished=True, completed_micro_iterations=40000,
                              completed_sample_exposures=80000,
                              completed_optimizer_steps=10000).items():
        if s.get(key) != expected:
            errors.append(f'{key}: expected {expected!r}, got {s.get(key)!r}')
    if set(run.validations) != set(range(4000, 40001, 4000)):
        errors.append('Expected all ten validation records at 4k intervals')
    if errors:
        raise ValueError(f'{run.directory}: ' + '; '.join(errors))
    steps = sorted(run.validations)
    values = [number(run.validations[step], 'mIoU') for step in steps]
    if any(v is None or not 0 <= v <= 100 for v in values):
        raise ValueError('Missing/invalid mIoU, expected percentage units 0..100')
    best = max(range(len(values)), key=values.__getitem__)
    checkpoint = run.directory / f'best_mIoU_iter_{steps[best]}.pth'
    if not checkpoint.is_file():
        raise ValueError(f'Best validation checkpoint is missing: {checkpoint}')
    seconds = number(s, 'training_time_seconds')
    if seconds is None or seconds <= 0:
        raise ValueError('Missing valid training time for cost tie-break')
    return dict(run=str(run.directory), checkpoint=str(checkpoint), seed=seed,
                best_miou=values[best], final_miou=values[-1],
                tail3_mean=mean(values[-3:]), training_seconds=seconds)


def fair_pair(baseline, candidate):
    if baseline.directory == candidate.directory:
        raise ValueError('Baseline and candidate must be distinct runs')
    for key in FAIR_KEYS:
        left, right = baseline.manifest.get(key), candidate.manifest.get(key)
        if left is None or right is None or left != right:
            raise ValueError(f'Unverified comparison: {key}: {left!r} vs {right!r}')


def choose(abl, rmi):
    """0.3 percentage-point practical tie-break, not a significance test."""
    delta = abl['best_miou'] - rmi['best_miou']
    if abs(delta) >= 0.3 - 1e-10:
        return 'abl' if delta > 0 else 'rmi'
    if abs(abl['tail3_mean'] - rmi['tail3_mean']) > 1e-10:
        return 'abl' if abl['tail3_mean'] > rmi['tail3_mean'] else 'rmi'
    # A fully equal result keeps the predetermined first-priority ABL branch.
    return 'abl' if abl['training_seconds'] <= rmi['training_seconds'] else 'rmi'


def confirmed(deltas):
    return all(value > 0 for value in deltas) and mean(deltas) >= 0.3 - 1e-10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    screen = commands.add_parser('screen')
    screen.add_argument('--baseline', required=True)
    screen.add_argument('--abl', required=True)
    screen.add_argument('--rmi', required=True)
    confirm = commands.add_parser('confirm')
    for name in ('baseline-2026', 'candidate-2026', 'baseline-2027', 'candidate-2027'):
        confirm.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'screen':
            runs = {name: load_run(getattr(args, name)) for name in ('baseline', 'abl', 'rmi')}
            for name in ('abl', 'rmi'):
                fair_pair(runs['baseline'], runs[name])
            scores = {name: run_score(run, 2026) for name, run in runs.items()}
            winner = choose(scores['abl'], scores['rmi'])
            eligible = [name for name in ('abl', 'rmi')
                        if scores[name]['best_miou'] > scores['baseline']['best_miou']]
            # Do not spend two repeat runs on a lower-than-baseline tie-break winner.
            if winner not in eligible:
                winner = eligible[0] if eligible else None
            report = dict(status='repeat_candidate' if winner else 'no_gain_stop',
                          winner=winner, scores=scores,
                          note='Seed2027 pair still requires training approval; no automatic launch')
        else:
            runs = {name: load_run(getattr(args, name)) for name in (
                'baseline_2026', 'candidate_2026', 'baseline_2027', 'candidate_2027')}
            scores, deltas = {}, []
            for seed in (2026, 2027):
                bn, cn = f'baseline_{seed}', f'candidate_{seed}'
                fair_pair(runs[bn], runs[cn])
                scores[bn], scores[cn] = run_score(runs[bn], seed), run_score(runs[cn], seed)
                deltas.append(scores[cn]['best_miou'] - scores[bn]['best_miou'])
            for role in ('baseline', 'candidate'):
                left, right = runs[f'{role}_2026'], runs[f'{role}_2027']
                fair_pair(left, right)
                if left.manifest.get('experiment_name') != right.manifest.get('experiment_name'):
                    raise ValueError(f'{role}: repeat must use the same experiment configuration')
            report = dict(status='eligible_mainline' if confirmed(deltas) else 'unconfirmed',
                          seed_deltas_pp=deltas, mean_delta_pp=mean(deltas), scores=scores,
                          note='Audit resolved configs for algorithm identity; no statistical '
                               'significance or official-score claim. Revalidate CRF on selected checkpoint.')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(status='blocked_evidence', error=str(exc)), ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
