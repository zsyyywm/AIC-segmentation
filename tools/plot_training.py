"""Create loss and validation trend plots from an AIC run directory."""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_results import CLASSES, load_run, number, resolve_run


def plot_run(source, output=None, quiet=False):
    run = load_run(source)
    output = Path(output).expanduser().resolve() if output else (
        run.directory / 'training_curves.png')
    train_steps = sorted(run.training)
    val_steps = sorted(run.validations)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    loss_ax, score_ax, class_ax = axes
    for key, label in (('loss', 'Total'), ('decode.loss_ce', 'Main'),
                       ('aux.loss_ce', 'Aux')):
        pairs = [(step, number(run.training[step], key)) for step in train_steps]
        pairs = [(step, value) for step, value in pairs if value is not None]
        if pairs:
            loss_ax.plot([item[0] for item in pairs],
                         [item[1] for item in pairs], label=label, linewidth=1.4)
    loss_ax.set(title='Training loss', xlabel='Iteration', ylabel='Loss')
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()

    for key, label in (('mIoU', 'mIoU'), ('mAcc', 'mAcc'), ('aAcc', 'aAcc')):
        values = [number(run.validations[step], key) for step in val_steps]
        if any(value is not None for value in values):
            score_ax.plot(val_steps, values, marker='o', label=label)
    score_ax.set(title='Validation metrics', xlabel='Iteration', ylabel='Percent')
    score_ax.grid(alpha=0.25)
    score_ax.legend()

    for class_name in CLASSES:
        values = [number(run.validations[step], f'IoU/{class_name}')
                  for step in val_steps]
        if any(value is not None for value in values):
            class_ax.plot(val_steps, values, marker='.', label=class_name)
    class_ax.set(title='Per-class validation IoU', xlabel='Iteration',
                 ylabel='IoU (%)')
    class_ax.grid(alpha=0.25)
    class_ax.legend(ncol=2, fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp' + output.suffix)
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    temporary.replace(output)
    if not quiet:
        print(f'PLOT: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Run directory or checkpoint inside it')
    parser.add_argument('--output')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30.0)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        run_dir = resolve_run(args.source)
        while True:
            try:
                plot_run(run_dir, args.output, args.quiet)
            except ValueError as exc:
                if not args.watch:
                    raise
                if not args.quiet:
                    print(f'WAITING: {exc}')
            if not args.watch:
                return
            summary = run_dir / 'run_summary.json'
            if summary.is_file() and '"finished": true' in summary.read_text(
                    encoding='utf-8').lower():
                return
            time.sleep(max(2.0, args.interval))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
