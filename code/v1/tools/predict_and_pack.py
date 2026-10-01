"""Run test inference, validate all PNGs, then create the submission ZIP."""

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

from mmengine.config import Config
from mmengine.runner import Runner

from run_utils import (best_checkpoint, experiment_name_from_config,
                       latest_run_and_best_checkpoint, resolve_project_path,
                       resolve_run)
from submission import make_zip, validate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'checkpoint', nargs='?', default=None,
        help='Best checkpoint; omitted means auto-discover in the default run')
    parser.add_argument(
        '--config', default='configs/control_convnextb_rmi_768.py')
    parser.add_argument(
        '--run', default='latest',
        help='Run directory relative to the project, or latest')
    parser.add_argument('--pred-dir', default='predictions')
    parser.add_argument('--zip', default='submissions/aic_submission.zip')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)
    if any(pred_dir.iterdir()):
        raise RuntimeError(
            f'Prediction directory is not empty: {pred_dir}. '
            'Choose a new --pred-dir or clear it explicitly.')

    config_path = resolve_project_path(args.config)
    cfg = Config.fromfile(str(config_path))
    experiment_name = experiment_name_from_config(cfg, config_path)
    run_dir = None
    if args.checkpoint:
        checkpoint = resolve_project_path(args.checkpoint)
    else:
        if args.run == 'latest':
            run_dir, checkpoint = latest_run_and_best_checkpoint(
                experiment_name)
        else:
            run_dir = resolve_run(args.run, experiment_name, prefer_train=True)
            checkpoint = best_checkpoint(run_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(PROJECT_ROOT / 'runs/inference')
    cfg.test_evaluator.output_dir = str(pred_dir)
    Runner.from_cfg(cfg).test()

    test_dir = PROJECT_ROOT.parents[1] / 'data/test/images'
    count = validate_predictions(pred_dir, test_dir)
    zip_path = make_zip(pred_dir, args.zip)
    print(f'CHECKPOINT: {checkpoint}')
    if run_dir:
        print(f'RUN_DIR: {run_dir}')
    print(f'PASS: {count} compliant PNG files')
    print(f'ZIP: {zip_path}')


if __name__ == '__main__':
    main()
