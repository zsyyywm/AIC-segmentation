import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crf_pipeline as c


class CRFTests(unittest.TestCase):
    def test_official_ignore_and_mapping(self):
        gt = np.array([[0, 1, 8], [2, 2, 1]], dtype=np.uint8)
        pred = np.array([[7, 0, 7], [1, 0, 0]], dtype=np.uint8)
        matrix = c.confusion(pred, gt)
        self.assertEqual(matrix.sum(), 5)
        self.assertEqual(matrix[7, 7], 1)
        self.assertEqual(matrix[1, 0], 1)
        self.assertEqual(c.metrics(matrix)['IoU']['Vehicle'], 100.)

    def test_global_accumulation_not_image_mean(self):
        a = c.confusion(np.array([[0]], dtype=np.uint8), np.array([[1]], dtype=np.uint8))
        b = c.confusion(np.ones((1, 9), dtype=np.uint8), np.ones((1, 9), dtype=np.uint8))
        self.assertAlmostEqual(c.metrics(a + b)['IoU']['Background'], 10.)

    def test_all_ignore_and_zero_union(self):
        matrix = c.confusion(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
        self.assertIsNone(c.metrics(matrix)['mIoU'])
        self.assertTrue(all(v is None for v in c.metrics(matrix)['IoU'].values()))

    def test_baseline_identity_without_optional_backend(self):
        rng = np.random.default_rng(20)
        probs = rng.random((8, 4, 5), dtype=np.float32)
        probs /= probs.sum(0)
        actual = c.refine(probs, np.zeros((4, 5, 3), dtype=np.uint8), None)
        np.testing.assert_array_equal(actual, probs.argmax(0))

    def test_probability_validation(self):
        bad = np.full((8, 2, 2), .2, dtype=np.float32)
        with self.assertRaises(ValueError):
            c.validate_probs(bad)
        with self.assertRaises(ValueError):
            c.validate_probs(np.zeros((2, 2), dtype=np.uint8))

    def test_subset_stable_and_disjoint(self):
        names = [f'{i:04}.png' for i in range(1400)]
        chosen = c.screen_names(names)
        self.assertEqual(len(set(chosen)), 200)
        self.assertEqual(chosen, c.screen_names(list(reversed(names))))
        self.assertEqual(len(set(names) - set(chosen)), 1200)
        self.assertEqual(len(c.GRID), 8)

    def test_selection_gate(self):
        def result(score, vehicle):
            return {g: {'mIoU': score, 'IoU': {'Vehicle': vehicle}}
                    for g in ('full', 'remaining')}
        base = result(74, 77)
        self.assertTrue(c.eligible(base, result(74.1, 76)))
        self.assertFalse(c.eligible(base, result(75, 75.99)))
        self.assertFalse(c.eligible(base, result(74, 77)))
        trial = result(75, 77)
        trial['remaining']['mIoU'] = 73
        self.assertFalse(c.eligible(base, trial))

    def test_immutable_json_and_frozen_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'report.json'
            c.write_json(p, {'stage': 'final', 'identity': {'checkpoint': 'a'}, 'selected': None})
            self.assertIsNone(c.frozen_params(p, {'checkpoint': 'a'}))
            with self.assertRaises(ValueError):
                c.frozen_params(p, {'checkpoint': 'b'})
            with self.assertRaises(FileExistsError):
                c.write_json(p, {})

    def test_non_grid_params_rejected(self):
        probs = np.full((8, 2, 2), 1 / 8, dtype=np.float32)
        with self.assertRaises(ValueError):
            c.refine(probs, np.zeros((2, 2, 3), dtype=np.uint8), {'iterations': 1})


if __name__ == '__main__':
    unittest.main()
