"""Matrix 类与 matmul 的单元测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matrix import Matrix, matmul


class TestConstruction(unittest.TestCase):
    def test_shape(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.shape, (2, 3))

    def test_copies_input(self):
        data = [[1, 2], [3, 4]]
        m = Matrix(data)
        data[0][0] = 99
        self.assertEqual(m[0, 0], 1.0)
        self.assertIsInstance(m[0, 0], float)

    def test_rejects_empty_and_ragged(self):
        for data in ([], [[]], [[1, 2], [3]]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                Matrix(data)

    def test_accepts_tuples(self):
        self.assertEqual(
            Matrix(((1, 2), (3, 4))).to_list(),
            [[1.0, 2.0], [3.0, 4.0]],
        )


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.m = Matrix([[1, 2], [3, 4]])

    def test_get(self):
        self.assertEqual(self.m[1, 0], 3.0)
        self.assertEqual(self.m[1], [3.0, 4.0])

    def test_set_element(self):
        self.m[0, 1] = 7
        self.assertEqual(self.m[0, 1], 7.0)

    def test_set_row(self):
        self.m[0] = [9, 8]
        self.assertEqual(self.m.to_list(), [[9.0, 8.0], [3.0, 4.0]])

    def test_rejects_bad_key(self):
        with self.assertRaises(TypeError):
            self.m[0, 1, 2]

    def test_rejects_wrong_row_width(self):
        with self.assertRaises(ValueError):
            self.m[0] = [1, 2, 3]


class TestRowOperations(unittest.TestCase):
    def test_swap_rows(self):
        m = Matrix([[1, 2], [3, 4]])
        m.swap_rows(0, 1)
        self.assertEqual(m.to_list(), [[3.0, 4.0], [1.0, 2.0]])

    def test_scale_row(self):
        m = Matrix([[1, 2], [3, 4]])
        m.scale_row(1, 0.5)
        self.assertEqual(m.to_list(), [[1.0, 2.0], [1.5, 2.0]])

    def test_add_row_to(self):
        m = Matrix([[1, 2], [3, 4]])
        m.add_row_to(0, 1, -2)
        self.assertEqual(m.to_list(), [[-5.0, -6.0], [3.0, 4.0]])

    def test_add_row_to_default_factor(self):
        m = Matrix([[1, 2], [3, 4]])
        m.add_row_to(0, 1)
        self.assertEqual(m.to_list(), [[4.0, 6.0], [3.0, 4.0]])


class TestTransposeAndMatmul(unittest.TestCase):
    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        t = m.T()
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.to_list(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        self.assertEqual(m.shape, (2, 3))

    def test_matmul_known_result(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        expected = [[58.0, 64.0], [139.0, 154.0]]
        self.assertEqual(matmul(a, b), expected)
        self.assertEqual((Matrix(a) @ Matrix(b)).to_list(), expected)
        self.assertEqual((Matrix(a) @ b).to_list(), expected)
        self.assertEqual(matmul(Matrix(a), b), expected)

    def test_matmul_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            matmul([[1, 2]], [[1, 2]])

    def test_matmul_result_is_independent(self):
        a = Matrix([[1, 2], [3, 4]])
        result = matmul(a, [[1, 0], [0, 1]])
        result[0][0] = 99
        self.assertEqual(a[0, 0], 1.0)


class TestDeterminant(unittest.TestCase):
    def test_2x2(self):
        self.assertAlmostEqual(Matrix([[1, 2], [3, 4]]).det(), -2.0, places=12)

    def test_3x3(self):
        m = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
        self.assertAlmostEqual(m.det(), 8.0, places=12)

    def test_row_swap_flips_sign(self):
        self.assertAlmostEqual(Matrix([[0, 1], [1, 0]]).det(), -1.0, places=12)

    def test_singular_returns_zero(self):
        self.assertEqual(Matrix([[1, 2], [2, 4]]).det(), 0.0)
        self.assertEqual(Matrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]]).det(), 0.0)

    def test_triangular(self):
        m = Matrix([[2, 5, 1], [0, -3, 4], [0, 0, 0.5]])
        self.assertAlmostEqual(m.det(), -3.0, places=12)

    def test_non_square_rejected(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2, 3], [4, 5, 6]]).det()

    def test_does_not_modify_matrix(self):
        m = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
        before = m.to_list()
        m.det()
        self.assertEqual(m.to_list(), before)

    def test_to_list_returns_copy(self):
        m = Matrix([[1, 2], [3, 4]])
        rows = m.to_list()
        rows[0][0] = 99
        self.assertEqual(m[0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
