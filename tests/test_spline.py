"""自然三次样条构造与求值的单元测试。"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline import eval_spline, natural_cubic_spline


def _derivatives(segment, x_left, x):
    _, b, c, d = segment
    t = x - x_left
    return b + 2.0 * c * t + 3.0 * d * t * t, 2.0 * c + 6.0 * d * t


class TestConstruction(unittest.TestCase):
    def test_two_points_are_linear(self):
        segments = natural_cubic_spline([0, 2], [1, 5])
        self.assertEqual(len(segments), 1)
        for got, want in zip(segments[0], (1.0, 2.0, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(eval_spline(segments, [0, 2], 1.0), 3.0, places=12)

    def test_three_points_known_coefficients(self):
        segments = natural_cubic_spline([0, 1, 2], [0, 1, 0])
        expected = [(0.0, 1.5, 0.0, -0.5), (1.0, 0.0, -1.5, 0.5)]
        for segment, want in zip(segments, expected):
            for got_value, want_value in zip(segment, want):
                self.assertAlmostEqual(got_value, want_value, places=12)

    def test_interpolates_nodes(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        segments = natural_cubic_spline(xs, ys)
        for x, y in zip(xs, ys):
            self.assertAlmostEqual(eval_spline(segments, xs, x), y, places=12)

    def test_rejects_bad_nodes(self):
        with self.assertRaises(ValueError):
            natural_cubic_spline([0], [1])
        with self.assertRaises(ValueError):
            natural_cubic_spline([0, 1], [1])
        with self.assertRaises(ValueError):
            natural_cubic_spline([0, 0, 1], [0, 1, 2])
        with self.assertRaises(ValueError):
            natural_cubic_spline([0, 2, 1], [0, 1, 2])


class TestSmoothness(unittest.TestCase):
    def setUp(self):
        self.xs = [0, 1, 2, 3, 4]
        self.ys = [0, 1, 0, 1, 0]
        self.segments = natural_cubic_spline(self.xs, self.ys)

    def test_natural_boundary(self):
        _, _, c_first, _ = self.segments[0]
        self.assertAlmostEqual(2.0 * c_first, 0.0, places=12)

        _, _, c_last, d_last = self.segments[-1]
        h = self.xs[-1] - self.xs[-2]
        self.assertAlmostEqual(2.0 * c_last + 6.0 * d_last * h, 0.0, places=12)

    def test_c2_continuity_at_inner_nodes(self):
        for k in range(1, len(self.xs) - 1):
            x = self.xs[k]
            left_d1, left_d2 = _derivatives(self.segments[k - 1], self.xs[k - 1], x)
            right_d1, right_d2 = _derivatives(self.segments[k], self.xs[k], x)
            self.assertAlmostEqual(left_d1, right_d1, places=10)
            self.assertAlmostEqual(left_d2, right_d2, places=10)

    def test_accuracy_on_sine(self):
        xs = [k * math.pi / 8 for k in range(9)]
        ys = [math.sin(x) for x in xs]
        segments = natural_cubic_spline(xs, ys)
        max_error = max(
            abs(eval_spline(segments, xs, k * math.pi / 64) - math.sin(k * math.pi / 64))
            for k in range(65)
        )
        self.assertLess(max_error, 1e-3)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.xs = [0, 1, 2]
        self.ys = [0, 1, 0]
        self.segments = natural_cubic_spline(self.xs, self.ys)

    def test_right_endpoint(self):
        self.assertAlmostEqual(eval_spline(self.segments, self.xs, 2.0), 0.0, places=12)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            eval_spline(self.segments, self.xs, -0.1)
        with self.assertRaises(ValueError):
            eval_spline(self.segments, self.xs, 2.1)

    def test_segment_count_mismatch(self):
        with self.assertRaises(ValueError):
            eval_spline(self.segments[:-1], self.xs, 0.5)

    def test_bad_xs(self):
        with self.assertRaises(ValueError):
            eval_spline(self.segments, [0, 2, 1], 0.5)
        with self.assertRaises(ValueError):
            eval_spline(self.segments, [0], 0.5)


if __name__ == "__main__":
    unittest.main()
