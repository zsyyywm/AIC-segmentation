from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_experiment import choose, confirmed


def score(best, tail=74, seconds=9000):
    return dict(best_miou=best, tail3_mean=tail, training_seconds=seconds)


class SelectionTests(unittest.TestCase):
    def test_best_outside_tie(self):
        self.assertEqual(choose(score(75), score(74.6, tail=80)), 'abl')

    def test_exact_point_three(self):
        self.assertEqual(choose(score(74.9), score(74.6, tail=80)), 'abl')

    def test_tail_breaks_near_tie(self):
        self.assertEqual(choose(score(75, tail=74), score(74.9, tail=74.5)), 'rmi')

    def test_cost_breaks_equal_tail(self):
        self.assertEqual(choose(score(75), score(74.9, seconds=8000)), 'rmi')

    def test_both_seeds_must_win(self):
        self.assertFalse(confirmed([1.5, -0.1]))
        self.assertFalse(confirmed([0.6, 0]))

    def test_mean_threshold(self):
        self.assertTrue(confirmed([0.1, 0.5]))
        self.assertFalse(confirmed([0.1, 0.49]))


if __name__ == '__main__':
    unittest.main()
