import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS_DIR))

import run_utils


class RunUtilsTest(unittest.TestCase):

    def test_experiment_name_is_sanitized(self):
        cfg = {'experiment_name': 'B0 ConvNeXt-L / 768'}
        self.assertEqual(
            run_utils.experiment_name_from_config(cfg, 'unused.py'),
            'B0_ConvNeXt-L_768')

    def test_create_run_dir_is_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2026, 9, 13, 20, 30, 0)
            with patch.object(run_utils, 'RUNS_ROOT', Path(tmp)):
                first = run_utils.create_run_dir('exp', 'smoke', now)
                second = run_utils.create_run_dir('exp', 'smoke', now)
            self.assertEqual(first.name, '20260913_203000_smoke')
            self.assertEqual(second.name, '20260913_203000_smoke_01')

    def test_latest_train_never_selects_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'exp'
            (parent / '20260913_200000_smoke').mkdir(parents=True)
            with patch.object(run_utils, 'RUNS_ROOT', Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    run_utils.latest_run_dir('exp', prefer_train=True)

    def test_checkpoint_selection_and_latest_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / 'runs' / 'exp'
            older = parent / '20260913_200000_train'
            newer = parent / '20260913_210000_train'
            older.mkdir(parents=True)
            newer.mkdir()
            best = older / 'best_mIoU_iter_8000.pth'
            latest = older / 'iter_9000.pth'
            best.touch()
            latest.touch()
            (older / 'last_checkpoint').write_text(
                str(latest), encoding='utf-8')
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            with patch.object(run_utils, 'PROJECT_ROOT', root), patch.object(
                    run_utils, 'RUNS_ROOT', root / 'runs'):
                self.assertEqual(run_utils.latest_checkpoint(older), latest)
                self.assertEqual(run_utils.best_checkpoint(older), best)
                run_dir, checkpoint = (
                    run_utils.latest_run_and_best_checkpoint('exp'))
            self.assertEqual(run_dir, older.resolve())
            self.assertEqual(checkpoint, best.resolve())


if __name__ == '__main__':
    unittest.main()
