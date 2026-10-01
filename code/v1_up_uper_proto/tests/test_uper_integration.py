"""U0 equivalence and real MMSeg head integration, CPU by default."""

import copy
from pathlib import Path
import sys
import unittest

import torch


VERSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION / 'tools'))
from _bootstrap import bootstrap
bootstrap()
import aicseg
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules

register_all_modules(init_default_scope=True)


class UPerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.control_cfg = Config.fromfile(str(VERSION / 'configs/control_convnextb_rmi_768.py'))
        cls.up_cfg = Config.fromfile(str(VERSION / 'configs/convnextb_uper_proto_768.py'))
        torch.manual_seed(2026)
        cls.control = MODELS.build(cls.control_cfg.model.decode_head)
        torch.manual_seed(2026)
        cls.up = MODELS.build(cls.up_cfg.model.decode_head)
        control_state = cls.control.state_dict()
        cls.original_weights_equal_at_construction = all(
            torch.equal(value, cls.up.state_dict()[name]) for name, value in control_state.items())
        cls.up.load_state_dict(control_state, strict=False)

    def setUp(self):
        torch.manual_seed(2026)
        self.inputs = [torch.randn(2, channels, size, size)
                       for channels, size in zip([128, 256, 512, 1024], [16, 8, 4, 2])]
        self.samples = []
        for _ in range(2):
            labels = (torch.arange(64 * 64).reshape(1, 64, 64) // 4) % 8
            labels[:, :4] = 255
            sample = SegDataSample(metainfo=dict(img_shape=(64, 64), ori_shape=(64, 64), pad_shape=(64, 64)))
            sample.gt_sem_seg = PixelData(data=labels)
            self.samples.append(sample)
        self.up.prototype_adapter.core.enabled = True

    def test_config_only_changes_declared_up_fields(self):
        first, second = copy.deepcopy(self.control_cfg.to_dict()), copy.deepcopy(self.up_cfg.to_dict())
        for config in (first, second):
            config.pop('experiment_name')
            config.pop('work_dir')
            config.pop('custom_hooks')
        second['model']['decode_head'].pop('feature_stride')
        second['model']['decode_head'].pop('prototype_cfg')
        second['model']['decode_head']['type'] = 'UPerHead'
        self.assertEqual(first, second)
        self.assertTrue(self.original_weights_equal_at_construction)

    def test_disabled_eval_logits_and_predict_equal_u0_exactly(self):
        self.control.eval()
        self.up.eval()
        self.up.prototype_adapter.core.enabled = False
        with torch.no_grad():
            torch.testing.assert_close(self.control(self.inputs), self.up(self.inputs), rtol=0, atol=0)
            metas = [sample.metainfo for sample in self.samples]
            torch.testing.assert_close(self.control.predict(self.inputs, metas, {}),
                                       self.up.predict(self.inputs, metas, {}), rtol=0, atol=0)

    def test_disabled_train_loss_equal_u0_with_matched_dropout_rng(self):
        self.control.train()
        self.up.train()
        self.up.prototype_adapter.core.enabled = False
        torch.manual_seed(123)
        first = self.control.loss(self.inputs, self.samples, {})
        torch.manual_seed(123)
        second = self.up.loss(self.inputs, self.samples, {})
        self.assertEqual(set(first), set(second))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_label_update_occurs_after_first_predictions_and_losses(self):
        self.up.train()
        self.control.train()
        core = self.up.prototype_adapter.core
        core.prototype_bank.zero_()
        core.prototype_valid.zero_()
        core.assignment_counts.zero_()
        core.update_steps.zero_()
        torch.manual_seed(123)
        baseline = self.control.loss(self.inputs, self.samples, {})
        torch.manual_seed(123)
        first = self.up.loss(self.inputs, self.samples, {})
        for key in baseline:
            torch.testing.assert_close(baseline[key], first[key], rtol=0, atol=0)
        self.assertEqual(int(core.update_steps), 1)
        self.assertTrue(core.prototype_valid.all())
        self.assertEqual(set(first), {'loss_ce', 'loss_rmi', 'acc_seg'})

    def test_full_enabled_head_backward_all_added_parameters(self):
        self.up.train()
        self.up.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = self.up._forward_feature(self.inputs)
            projected = self.up.prototype_adapter.project(features)
            self.up.prototype_adapter.core.update_from_labels(projected, self.up._stack_batch_gt(self.samples))
        losses = self.up.loss(self.inputs, self.samples, {})
        sum(value for name, value in losses.items() if name.startswith('loss_')).backward()
        for name, parameter in self.up.prototype_adapter.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_predict_needs_metadata_only_and_freezes_statistics(self):
        self.up.eval()
        core = self.up.prototype_adapter.core
        before = copy.deepcopy(core.state_dict())
        with torch.no_grad():
            predictions = self.up.predict(self.inputs, [sample.metainfo for sample in self.samples], {})
            second = self.up.predict(self.inputs, [sample.metainfo for sample in self.samples], {})
        self.assertEqual(predictions.shape, (2, 8, 64, 64))
        torch.testing.assert_close(predictions, second, rtol=0, atol=0)
        for name, value in before.items():
            torch.testing.assert_close(value, core.state_dict()[name], rtol=0, atol=0)

    def test_geometry_padding_excluded_without_gt(self):
        features = torch.randn(2, 512, 16, 16)
        mask = self.up._geometry_mask(features, [dict(img_shape=(48, 40)), dict(img_shape=(64, 64))])
        self.assertEqual(int(mask[0].sum()), 12 * 10)
        self.assertEqual(int(mask[1].sum()), 16 * 16)
        self.assertFalse(mask[0, :, 12:].any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
