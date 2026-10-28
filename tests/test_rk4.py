"""RK4 单步与初值问题求解的单元测试。"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rk4 import rk4_solve, rk4_step


class TestStep(unittest.TestCase):
    def test_exponential_single_step(self):
        got = rk4_step(lambda x, y: y, 0.0, 1.0, 0.1)
        self.assertAlmostEqual(got, 1.1051708333333332, delta=1e-12)

    def test_zero_step_rejected(self):
        with self.assertRaises(ValueError):
            rk4_step(lambda x, y: y, 0.0, 1.0, 0.0)


class TestSolve(unittest.TestCase):
    def test_endpoint_and_accuracy(self):
        xs, ys = rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, 0.1)
        self.assertEqual(len(xs), 11)
        self.assertEqual(xs[-1], 1.0)
        self.assertIsInstance(xs[-1], float)
        self.assertLess(abs(ys[-1] - math.e), 1e-5)

    def test_non_integer_step_count(self):
        xs, ys = rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, 0.3)
        self.assertEqual(xs[-1], 1.0)
        self.assertEqual(len(xs), 5)
        self.assertLess(abs(ys[-1] - math.e), 1e-3)

    def test_step_larger_than_interval(self):
        xs, ys = rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, 5.0)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[-1], 1.0)
        # 步长超过区间时应截断为一步 h=1，而不是按 h=5 外推。
        expected = rk4_step(lambda x, y: y, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(ys[-1], expected, delta=1e-12)

    def test_backward_integration(self):
        xs, ys = rk4_solve(lambda x, y: y, 0.0, 1.0, -1.0, -0.1)
        self.assertEqual(xs[-1], -1.0)
        self.assertLess(abs(ys[-1] - math.exp(-1.0)), 1e-5)

    def test_fourth_order_convergence(self):
        errors = []
        for h in (0.2, 0.1, 0.05):
            _, ys = rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, h)
            errors.append(abs(ys[-1] - math.e))
        for first, second in zip(errors, errors[1:]):
            ratio = first / second
            self.assertGreater(ratio, 12.0)
            self.assertLess(ratio, 20.0)

    def test_tiny_interval_still_advances(self):
        xs, ys = rk4_solve(lambda x, y: 1e9, 0.0, 0.0, 1e-6, 3e-7)
        self.assertEqual(xs[-1], 1e-6)
        self.assertEqual(len(xs), 5)
        self.assertAlmostEqual(ys[-1], 1000.0, delta=1e-6)

    def test_zero_interval(self):
        xs, ys = rk4_solve(lambda x, y: y, 0.5, 2.0, 0.5, 0.1)
        self.assertEqual(xs, [0.5])
        self.assertEqual(ys, [2.0])

    def test_wrong_direction_rejected(self):
        with self.assertRaises(ValueError):
            rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, -0.1)
        with self.assertRaises(ValueError):
            rk4_solve(lambda x, y: y, 0.0, 1.0, -1.0, 0.1)

    def test_zero_step_rejected(self):
        with self.assertRaises(ValueError):
            rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
