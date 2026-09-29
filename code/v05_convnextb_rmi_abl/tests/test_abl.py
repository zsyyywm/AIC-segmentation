"""Pure torch tests: deliberately do NOT pretend to build MMSeg registries."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('abl_core', ROOT / 'aicseg/abl_core.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ABL = MODULE.ActiveBoundaryLossCore


class ActiveBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.loss = ABL()
        self.target = torch.zeros(2, 32, 32, dtype=torch.long)
        self.target[:, :, 16:] = 1

    def logits(self):
        return torch.randn(2, 8, 32, 32, requires_grad=True)

    def check_backward(self, pred, target, expected_zero=False):
        value = self.loss(pred, target)
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        if expected_zero:
            self.assertEqual(value.item(), 0)
            self.assertEqual(pred.grad.abs().sum().item(), 0)
        else:
            self.assertGreater(value.item(), 0)
            self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_finite_forward_backward(self):
        self.check_backward(self.logits(), self.target)

    def test_all_ignore(self):
        self.check_backward(self.logits(), self.target.fill_(255), True)

    def test_no_gt_boundary(self):
        self.check_backward(self.logits(), self.target.zero_(), True)

    def test_no_predicted_boundary(self):
        self.check_backward(torch.zeros(2, 8, 32, 32, requires_grad=True),
                            self.target, True)

    def test_ignore_not_false_boundary(self):
        target = self.target.zero_()
        target[:, 10:20, 10:20] = 255
        self.check_backward(self.logits(), target, True)

    def test_ignore_neighborhood_has_zero_gradient(self):
        self.target[:, 5:9, 5:9] = 255
        pred = self.logits()
        self.check_backward(pred, self.target)
        self.assertEqual(pred.grad[:, :, 4:10, 4:10].abs().sum().item(), 0)

    def test_ignore_logits_cannot_change_loss(self):
        self.target[:, 5:9, 5:9] = 255
        pred = self.logits()
        changed = pred.detach().clone()
        changed[:, :, 5:9, 5:9] = 1000 * torch.randn(2, 8, 4, 4)
        torch.testing.assert_close(self.loss(pred, self.target),
                                   self.loss(changed, self.target), rtol=0, atol=0)

    def test_single_pixel_object(self):
        self.target.zero_()
        self.target[:, 16, 16] = 7
        self.check_backward(self.logits(), self.target)

    def test_tiny_spatial_shape(self):
        pred = torch.randn(1, 8, 1, 2, requires_grad=True)
        self.check_backward(pred, torch.tensor([[[0, 1]]]), True)

    def test_cpu_bfloat16_autocast(self):
        pred = self.logits()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            value = self.loss(pred, self.target)
        self.assertEqual(value.dtype, torch.float32)
        value.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_half_precision_inputs_compute_fp32(self):
        pred = self.logits().detach().half().requires_grad_()
        self.check_backward(pred, self.target)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_amp(self):
        pred = self.logits().detach().cuda().requires_grad_()
        with torch.autocast('cuda', dtype=torch.float16):
            self.check_backward(pred, self.target.cuda())

    def test_geometry_distance_and_ignore(self):
        labels = self.target[0].numpy()
        distance, safe = self.loss.geometry(labels, 255)
        self.assertEqual(distance[10, 15], 0)
        self.assertEqual(distance[10, 12], 3)
        self.assertFalse(safe[0].any())
        self.assertFalse(safe[:, 0].any())
        labels[:, 10] = 255
        _, safe = self.loss.geometry(labels, 255)
        self.assertFalse(safe[:, 9:12].any())

    def test_uniform_prediction_has_no_selected_centers(self):
        pred = torch.zeros(8, 10, 10).log_softmax(0)
        centers = self.loss.predicted_boundaries(pred, torch.ones(10, 10).bool())
        self.assertFalse(centers.any())

    def test_invalid_targets_fail(self):
        with self.assertRaises(ValueError):
            self.loss(self.logits(), self.target.fill_(8))
        with self.assertRaises(ValueError):
            self.loss(self.logits(), self.target.float())

    def test_shape_and_sampler_fail(self):
        with self.assertRaises(ValueError):
            self.loss(self.logits(), self.target[:, :16])
        with self.assertRaises(ValueError):
            self.loss(self.logits(), self.target, weight=torch.ones_like(self.target))

    def test_config_parameters_fail(self):
        for kwargs in ({'max_boundary_ratio': 0}, {'max_distance': 0},
                       {'label_smoothing': 1}, {'loss_weight': -1}):
            with self.assertRaises(ValueError):
                ABL(**kwargs)

    def test_loss_weight_linear(self):
        pred = self.logits()
        base = self.loss(pred, self.target)
        doubled = ABL(loss_weight=0.2)(pred, self.target)
        torch.testing.assert_close(doubled, 2 * base)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
