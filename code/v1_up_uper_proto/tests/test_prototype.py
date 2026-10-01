"""Meaningful P invariants; stdlib unittest, no additional test dependencies."""

import copy
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / 'aicseg/prototype_core.py'
SPEC = importlib.util.spec_from_file_location('aic_prototype_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Core = MODULE.MultiPrototypeRegionContext


class PrototypeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(2)
        self.features = torch.randn(2, 256, 8, 8)
        self.labels = torch.arange(128).reshape(2, 8, 8) % 8

    def warm_core(self):
        core = Core()
        core.train()
        self.assertTrue(core.update_from_labels(self.features, self.labels))
        return core

    def test_cold_and_disabled_are_true_identity(self):
        core = Core()
        self.assertIs(core(self.features), self.features)
        core.update_from_labels(self.features, self.labels)
        core.enabled = False
        before = copy.deepcopy(core.state_dict())
        self.assertIs(core(self.features), self.features)
        self.assertFalse(core.update_from_labels(self.features, self.labels))
        for key, value in before.items():
            torch.testing.assert_close(core.state_dict()[key], value, rtol=0, atol=0)

    def test_only_labelled_class_is_initialised(self):
        core = Core()
        core.update_from_labels(self.features, torch.full_like(self.labels, 4))
        self.assertTrue(core.prototype_valid[4].all())
        self.assertFalse(core.prototype_valid[[0, 1, 2, 3, 5, 6, 7]].any())
        self.assertEqual(core.assignment_counts.sum(), self.labels.numel())

    def test_diverse_data_uses_all_four_slots_per_class(self):
        core = self.warm_core()
        self.assertTrue(core.prototype_valid.all())
        self.assertTrue((core.assignment_counts > 0).all())
        torch.testing.assert_close(core.prototype_bank.norm(dim=-1), torch.ones(8, 4))
        diagnostics = core.diagnostics()
        self.assertEqual(len(diagnostics['max_within_class_cosine']), 8)
        self.assertTrue(all(value < 0.99 for value in diagnostics['max_within_class_cosine']))

    def test_constant_data_does_not_fabricate_diverse_slots(self):
        core = Core()
        core.update_from_labels(torch.ones_like(self.features), self.labels)
        self.assertTrue((core.prototype_valid.sum(1) == 1).all())
        self.assertTrue(torch.isfinite(core(torch.ones_like(self.features))).all())

    def test_ignore_and_padding_never_update_bank(self):
        core = Core()
        self.assertFalse(core.update_from_labels(self.features, torch.full_like(self.labels, 255)))
        self.assertFalse(core.update_from_labels(self.features, self.labels,
                         torch.zeros(2, 1, 8, 8, dtype=torch.bool)))
        self.assertFalse(core.prototype_valid.any())
        self.assertEqual(core.update_steps, 0)

    def test_any_ignore_in_pooled_cell_excludes_that_cell(self):
        core = Core()
        labels = torch.full((2, 32, 32), 4, dtype=torch.long)
        labels[:, ::4, ::4] = 255
        self.assertFalse(core.update_from_labels(self.features, labels))
        self.assertFalse(core.prototype_valid.any())

    def test_changes_to_ignored_pixels_cannot_change_bank(self):
        first, second = Core(), Core()
        second.load_state_dict(first.state_dict())
        labels = self.labels.clone()
        labels[:, :2] = 255
        changed = self.features.clone()
        changed[:, :, :2] = 1e6
        first.update_from_labels(self.features, labels)
        second.update_from_labels(changed, labels)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_absent_class_state_is_unchanged(self):
        core = self.warm_core()
        old = core.prototype_bank.clone()
        counts = core.assignment_counts.clone()
        core.update_from_labels(self.features * 2, torch.full_like(self.labels, 4))
        ids = [0, 1, 2, 3, 5, 6, 7]
        torch.testing.assert_close(core.prototype_bank[ids], old[ids], rtol=0, atol=0)
        torch.testing.assert_close(core.assignment_counts[ids], counts[ids], rtol=0, atol=0)

    def test_zero_region_mask_is_identity_without_nan(self):
        core = self.warm_core()
        valid = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
        self.assertIs(core(self.features, valid), self.features)
        valid[0, :, :4] = True
        output = core(self.features, valid)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output[~valid.expand_as(output)],
                                   self.features[~valid.expand_as(output)], rtol=0, atol=0)

    def test_eval_has_no_gt_input_and_no_persistent_mutation(self):
        core = self.warm_core().eval()
        before = copy.deepcopy(core.state_dict())
        output = core(self.features)
        torch.testing.assert_close(output, core(self.features), rtol=0, atol=0)
        self.assertFalse(core.update_from_labels(self.features, self.labels))
        self.assertEqual(list(inspect.signature(core.forward).parameters),
                         ['features', 'valid_mask'])
        for key, value in before.items():
            torch.testing.assert_close(core.state_dict()[key], value, rtol=0, atol=0)

    def test_checkpoint_roundtrip_preserves_predictions_and_statistics(self):
        core = self.warm_core().eval()
        restored = Core().eval()
        restored.load_state_dict(copy.deepcopy(core.state_dict()), strict=True)
        torch.testing.assert_close(core(self.features), restored(self.features), rtol=0, atol=0)
        self.assertEqual(core.diagnostics(), restored.diagnostics())

    def test_update_before_backward_uses_safe_snapshot_and_all_gradients(self):
        core = self.warm_core()
        features = self.features.clone().requires_grad_(True)
        output = core(features)
        core.update_from_labels(features.detach() * 1.3, self.labels)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        for name, parameter in core.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_ema_formula_for_single_supported_slot(self):
        core = Core()
        features = torch.ones_like(self.features)
        core.update_from_labels(features, torch.full_like(self.labels, 4))
        old = core.prototype_bank[4, 0].clone()
        changed = features.clone()
        changed[:, 0] = 1.01
        projected = torch.nn.functional.normalize(core.query_proj(changed).float(), dim=1)
        mean = torch.nn.functional.normalize(projected.permute(0, 2, 3, 1).reshape(-1, 256).mean(0), dim=0)
        expected = torch.nn.functional.normalize(0.99 * old + 0.01 * mean, dim=0)
        # Freeze cold slots for this isolated EMA formula check.
        core.prototype_valid[4] = True
        core.prototype_bank[4] = old
        core.update_from_labels(changed, torch.full_like(self.labels, 4))
        torch.testing.assert_close(core.prototype_bank[4, 0], expected)

    def test_invalid_label_ids_fail_instead_of_silent_remapping(self):
        core = Core()
        for bad in (-1, 8, 9):
            with self.assertRaises(ValueError):
                core.update_from_labels(self.features, torch.full_like(self.labels, bad))

    def test_core_has_no_pixel_pair_attention(self):
        core = self.warm_core()
        large = torch.randn(1, 256, 32, 40)
        self.assertEqual(core(large).shape, large.shape)
        self.assertEqual(len(core.diagnostics()['attention_mass']), 8)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for AMP check')
    def test_cuda_amp_update_and_backward_keep_bank_float32(self):
        core = Core().cuda().train()
        optimizer = torch.optim.SGD(core.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler()
        features = self.features.cuda().half().requires_grad_(True)
        labels = self.labels.cuda()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            core.update_from_labels(features.detach(), labels)
            output = core(features)
            core.update_from_labels(features.detach() * 1.1, labels)
            loss = output.float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self.assertEqual(core.prototype_bank.dtype, torch.float32)
        self.assertTrue(torch.isfinite(core.prototype_bank).all())
        for name, parameter in core.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for AMP check')
    def test_amp_small_residual_survives_output_minus_input(self):
        core = Core(gate_init=1e-4).cuda().train()
        with torch.no_grad():
            eye = torch.eye(256, device='cuda')[:, :, None, None]
            core.value_proj.weight.copy_(eye)
            core.context_proj.weight.copy_(eye)
        features = torch.ones(1, 256, 4, 4, device='cuda', dtype=torch.float16)
        labels = torch.full((1, 4, 4), 4, device='cuda', dtype=torch.long)
        core.update_from_labels(features, labels)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            enhanced = core(features)
        self.assertEqual(enhanced.dtype, torch.float32)
        self.assertTrue(((enhanced - features.float()) > 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
