import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'check_training_ready.py'
SPEC = importlib.util.spec_from_file_location('training_ready', PATH)
READY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READY)


def config():
    head = dict(num_classes=8, ignore_index=255, norm_cfg=dict(type='GN', num_groups=32))
    return dict(model=dict(backbone=dict(arch='base'), decode_head=head,
                           auxiliary_head=copy.deepcopy(head),
                           test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512))),
                train_dataloader=dict(batch_size=2, dataset=dict(data_prefix=dict(img_path='train/images'))),
                val_dataloader=dict(dataset=dict(data_prefix=dict(img_path='val/images'))),
                optim_wrapper=dict(accumulative_counts=4), randomness=dict(seed=2026),
                train_cfg=dict(max_iters=40000, val_interval=4000),
                train_pipeline=[dict(type='RandomD4')], load_from=None, resume=False)


class PreflightTests(unittest.TestCase):
    def test_round_one(self):
        READY.check_config(config())

    def test_second_seed(self):
        cfg = config()
        cfg['randomness']['seed'] = 2027
        READY.check_config(cfg)

    def test_reject_smoke_as_formal(self):
        cfg = config()
        cfg['train_cfg']['max_iters'] = 300
        with self.assertRaisesRegex(ValueError, '40k'):
            READY.check_config(cfg)

    def test_reject_warmstart(self):
        cfg = config()
        cfg['load_from'] = 'smoke.pth'
        with self.assertRaisesRegex(ValueError, 'resume/load'):
            READY.check_config(cfg)

    def test_reject_ignore_as_background(self):
        cfg = config()
        cfg['model']['decode_head']['ignore_index'] = 0
        with self.assertRaisesRegex(ValueError, 'Ignore255'):
            READY.check_config(cfg)

    def test_reject_test_training(self):
        cfg = config()
        cfg['train_dataloader']['dataset']['data_prefix']['img_path'] = 'test/images'
        with self.assertRaisesRegex(ValueError, 'official training'):
            READY.check_config(cfg)


if __name__ == '__main__':
    unittest.main()
