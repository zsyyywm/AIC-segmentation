"""Pure torch tests: no fake MMSeg registry and no extra test dependencies."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F


CORE = Path(__file__).resolve().parents[1] / 'aicseg' / 'rmi_core.py'
SPEC = importlib.util.spec_from_file_location('aic_rmi_core_test', CORE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
rmi = MODULE.regional_mutual_information


class RMITest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(2)

    def make_pair(self, size=24, batch=2):
        logits = torch.randn(batch, 8, size, size, requires_grad=True)
        labels = torch.randint(0, 8, (batch, size, size))
        return logits, labels

    def assert_finite_backward(self, loss, logits):
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_default_finite_backward(self):
        logits, labels = self.make_pair()
        self.assert_finite_backward(rmi(logits, labels), logits)
        self.assertGreater(logits.grad.abs().sum(), 0)

    def test_all_ignore_zero_and_backward(self):
        logits, labels = self.make_pair()
        labels.fill_(255)
        value = rmi(logits, labels)
        self.assertEqual(value.item(), 0)
        self.assert_finite_backward(value, logits)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_too_small_or_single_neighborhood(self):
        for size in (1, 8, 12):
            logits, labels = self.make_pair(size=size)
            value = rmi(logits, labels)
            self.assertEqual(value.item(), 0)
            self.assert_finite_backward(value, logits)

    def test_ignored_logits_cannot_change_loss(self):
        logits, labels = self.make_pair(size=32, batch=1)
        labels[:, :4] = 255
        labels[:, 16, 16] = 255
        changed = logits.detach().clone()
        changed.masked_fill_((labels == 255)[:, None].expand_as(changed), 100)
        torch.testing.assert_close(rmi(logits, labels), rmi(changed, labels))
        self.assert_finite_backward(rmi(logits, labels), logits)
        ignored = (labels == 255)[:, None].expand_as(logits)
        self.assertEqual(logits.grad[ignored].abs().sum().item(), 0)

    def test_strict_pool_and_region_ignore_exclusion(self):
        logits, labels = self.make_pair(size=24, batch=1)
        # One ignore per 4x4 pooling tile makes every region invalid.
        labels[:, ::4, ::4] = 255
        self.assertEqual(rmi(logits, labels).item(), 0)

    def test_full_ignore_image_batch_denominator(self):
        logits, labels = self.make_pair(batch=2)
        labels[1].fill_(255)
        expected = rmi(logits[:1], labels[:1]) / 2
        torch.testing.assert_close(rmi(logits, labels), expected)

    def test_constant_region_and_saturated_logits(self):
        logits, labels = self.make_pair()
        labels.zero_()
        with torch.no_grad():
            logits.mul_(100)
        self.assert_finite_backward(rmi(logits, labels), logits)

    def test_single_pixel_vehicle_finite(self):
        logits, labels = self.make_pair()
        labels.zero_()
        labels[:, 12, 12] = 7
        self.assert_finite_backward(rmi(logits, labels), logits)

    def test_class_permutation_invariance(self):
        logits, labels = self.make_pair()
        order = torch.tensor([7, 0, 1, 3, 5, 2, 6, 4])
        inverse = torch.argsort(order)
        torch.testing.assert_close(rmi(logits, labels),
                                   rmi(logits[:, order], inverse[labels]))

    def test_direct_formula_agreement(self):
        logits, labels = self.make_pair(size=7, batch=1)
        logits = logits.detach().double().requires_grad_(True)
        eps, radius = 1e-6, 2
        probability = logits.sigmoid()
        truth = F.one_hot(labels, 8).permute(0, 3, 1, 2).double()
        values = []
        # Independent slow reference: enumerate windows and use dense inverse
        # and slogdet rather than production unfold/Cholesky solves.
        for cls in range(8):
            ys, ps = [], []
            for row in range(6):
                for col in range(6):
                    ys.append(truth[0, cls, row:row+2, col:col+2].reshape(-1))
                    ps.append(probability[0, cls, row:row+2, col:col+2].reshape(-1))
            y, p = torch.stack(ys, 1), torch.stack(ps, 1)
            y, p = y-y.mean(1, keepdim=True), p-p.mean(1, keepdim=True)
            cy, cp, cross = y@y.T/36, p@p.T/36, y@p.T/36
            eye = torch.eye(4, dtype=torch.float64)
            conditional = cy-cross@torch.linalg.inv(cp+eps*eye)@cross.T
            sign, logdet = torch.linalg.slogdet(conditional+eps*eye)
            self.assertGreater(sign, 0)
            values.append(logdet / 8)
        reference = torch.stack(values).sum()
        actual = rmi(logits, labels, radius=radius, pool_size=1)
        torch.testing.assert_close(actual, reference, atol=1e-9, rtol=1e-8)
        grad_a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        grad_b = torch.autograd.grad(reference, logits)[0]
        torch.testing.assert_close(grad_a, grad_b, atol=1e-8, rtol=1e-7)

    def test_aligned_prediction_better_than_shuffled(self):
        labels = torch.randint(0, 8, (1, 32, 32))
        exact = F.one_hot(labels, 8).permute(0, 3, 1, 2).float() * 8 - 4
        shuffled = exact.flatten(2)[..., torch.randperm(1024)].reshape_as(exact)
        self.assertLess(rmi(exact, labels), rmi(shuffled, labels))

    def test_half_input_and_cpu_autocast(self):
        logits, labels = self.make_pair()
        baseline = rmi(logits, labels)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = rmi(logits, labels)
        torch.testing.assert_close(actual, baseline)
        self.assert_finite_backward(actual, logits)
        half = logits.detach().half().requires_grad_(True)
        self.assert_finite_backward(rmi(half, labels), half)

    def test_nondivisible_spatial_shape(self):
        logits = torch.randn(1, 8, 25, 27, requires_grad=True)
        labels = torch.randint(0, 8, (1, 25, 27))
        self.assert_finite_backward(rmi(logits, labels), logits)

    def test_bad_inputs_fail(self):
        logits, labels = self.make_pair()
        with self.assertRaises(ValueError):
            rmi(logits, labels, covariance_epsilon=0)
        with self.assertRaises(ValueError):
            rmi(logits, labels, radius=1.5)
        with self.assertRaises(TypeError):
            rmi(logits, labels.float())
        with self.assertRaises(ValueError):
            rmi(logits, labels[:, :-1])
        labels[0, 0, 0] = 8
        with self.assertRaises(ValueError):
            rmi(logits, labels)
        labels[0, 0, 0] = 0
        bad = logits.detach().clone()
        bad[0, 0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            rmi(bad, labels)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
    def test_cuda_amp(self):
        logits = torch.randn(2, 8, 96, 96, device='cuda',
                             dtype=torch.float16, requires_grad=True)
        labels = torch.randint(0, 8, (2, 96, 96), device='cuda')
        with torch.autocast('cuda'):
            loss = rmi(logits, labels)
        self.assert_finite_backward(loss, logits)


if __name__ == '__main__':
    unittest.main()
