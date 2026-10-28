"""LU 分解与线性方程组求解的单元测试。"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lu import lu_decompose, solve


def _rows(matrix):
    return matrix.to_list() if hasattr(matrix, "to_list") else matrix


def _matvec(a, x):
    return [sum(row[j] * x[j] for j in range(len(x))) for row in a]


class TestSolve(unittest.TestCase):
    def test_2x2(self):
        x = solve([[4, 3], [6, 3]], [10, 12])
        for got, want in zip(x, [1.0, 2.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_3x3(self):
        a = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        x_true = [1.0, 2.0, -1.0]
        x = solve(a, _matvec(a, x_true))
        for got, want in zip(x, x_true):
            self.assertAlmostEqual(got, want, places=10)

    def test_requires_pivoting(self):
        x = solve([[0, 1], [1, 0]], [2, 1])
        for got, want in zip(x, [1.0, 2.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_1x1(self):
        self.assertEqual(solve([[3]], [6]), [2.0])

    def test_residual_on_random_systems(self):
        rng = random.Random(2026)
        for n in range(1, 7):
            a = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                a[i][i] += 2 * n  # 严格对角占优，保证可解且条件数温和
            x_true = [rng.uniform(-3, 3) for _ in range(n)]
            b = _matvec(a, x_true)
            x = solve(a, b)
            residual = max(abs(got - want) for got, want in zip(_matvec(a, x), b))
            self.assertLess(residual, 1e-9)

    def test_singular_rejected(self):
        with self.assertRaises(ValueError):
            solve([[1, 2], [2, 4]], [1, 2])
        with self.assertRaises(ValueError):
            solve([[0, 0], [0, 0]], [0, 0])

    def test_dimension_errors(self):
        with self.assertRaises(ValueError):
            solve([[1, 2, 3], [4, 5, 6]], [1, 2])
        with self.assertRaises(ValueError):
            solve([[1, 2], [3, 4]], [1, 2, 3])
        with self.assertRaises(ValueError):
            solve([], [])


class TestDecomposition(unittest.TestCase):
    def test_pa_equals_lu(self):
        rng = random.Random(7)
        for n in range(1, 7):
            a = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                a[i][i] += 2 * n
            l, u, piv = lu_decompose(a)
            l_rows, u_rows = _rows(l), _rows(u)
            self.assertEqual(sorted(piv), list(range(n)))
            for i in range(n):
                for j in range(n):
                    got = sum(l_rows[i][k] * u_rows[k][j] for k in range(n))
                    self.assertAlmostEqual(got, a[piv[i]][j], places=10)

    def test_factor_structure(self):
        a = [[4, 3], [6, 3]]
        l, u, _ = lu_decompose(a)
        l_rows, u_rows = _rows(l), _rows(u)
        self.assertEqual(l_rows[0][0], 1.0)
        self.assertEqual(l_rows[1][1], 1.0)
        self.assertEqual(l_rows[0][1], 0.0)
        self.assertEqual(u_rows[1][0], 0.0)

    def test_permutation_tracks_row_swaps(self):
        _, _, piv = lu_decompose([[0, 1], [1, 0]])
        self.assertEqual(piv, [1, 0])

    def test_does_not_modify_input(self):
        a = [[0, 1], [1, 0]]
        before = [row[:] for row in a]
        lu_decompose(a)
        self.assertEqual(a, before)

    def test_singular_rejected(self):
        with self.assertRaises(ValueError):
            lu_decompose([[1, 2], [2, 4]])


if __name__ == "__main__":
    unittest.main()
