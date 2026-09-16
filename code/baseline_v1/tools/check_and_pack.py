"""Validate an existing prediction directory and package it."""

import argparse

from _bootstrap import PROJECT_ROOT
from submission import make_zip, validate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', default=str(PROJECT_ROOT / 'predictions'))
    parser.add_argument(
        '--test-dir', default=str(PROJECT_ROOT.parents[1] / 'data/test/images'))
    parser.add_argument(
        '--zip', default=str(PROJECT_ROOT / 'submissions/aic_submission.zip'))
    args = parser.parse_args()
    count = validate_predictions(args.pred_dir, args.test_dir)
    output = make_zip(args.pred_dir, args.zip)
    print(f'PASS: {count} compliant PNG files')
    print(f'ZIP: {output}')


if __name__ == '__main__':
    main()

