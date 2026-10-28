"""
使用二维 list 存储的轻量矩阵类，以及独立的矩阵乘法函数。
"""
from __future__ import annotations

from typing import Any, Sequence


_EPS = 1e-12


def _copy_rectangular(data: Sequence[Sequence[float]]) -> list[list[float]]:
    if not isinstance(data, Sequence) or len(data) == 0:
        raise ValueError("输入数据必须是非空矩形")

    rows = []
    width = None
    for row in data:
        if not isinstance(row, Sequence):
            raise ValueError("输入数据必须是非空矩形")
        row_list = [float(value) for value in row]
        if width is None:
            width = len(row_list)
            if width == 0:
                raise ValueError("输入数据必须是非空矩形")
        elif len(row_list) != width:
            raise ValueError("输入数据必须是非空矩形")
        rows.append(row_list)

    return rows


def _as_list_matrix(data: "Matrix | Sequence[Sequence[float]]") -> list[list[float]]:
    if isinstance(data, Matrix):
        return data._data  # 乘法只读内部数据，避免额外拷贝。
    return _copy_rectangular(data)


class Matrix:
    """二维 list 矩阵，提供基础行操作、乘法和行列式。"""

    def __init__(self, data: Sequence[Sequence[float]]) -> None:
        self._data = _copy_rectangular(data)
        self._shape = (len(self._data), len(self._data[0]))

    @property
    def shape(self) -> tuple[int, int]:
        """返回矩阵形状 (行数, 列数)。"""
        return self._shape

    def __getitem__(self, key: Any):
        """支持 M[i, j] 与 M[i] 两种读取方式。"""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("矩阵索引必须是 M[i, j]")
            i, j = key
            return self._data[i][j]
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """支持写入单个元素，或整行替换成新的浮点列表。"""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("矩阵索引必须是 M[i, j]")
            i, j = key
            self._data[i][j] = float(value)
            return

        row = [float(item) for item in value]
        if len(row) != self._shape[1]:
            raise ValueError("输入数据必须是非空矩形")
        self._data[key] = row

    def swap_rows(self, i: int, j: int) -> None:
        """原地交换两行。"""
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def scale_row(self, i: int, c: float) -> None:
        """原地把第 i 行乘以常数 c。"""
        factor = float(c)
        self._data[i] = [factor * value for value in self._data[i]]

    def add_row_to(self, i: int, j: int, c: float = 1.0) -> None:
        """执行第 i 行 += c * 第 j 行。"""
        factor = float(c)
        source_row = self._data[j]
        target_row = self._data[i]
        self._data[i] = [
            target_value + factor * source_value
            for target_value, source_value in zip(target_row, source_row)
        ]

    def T(self) -> "Matrix":
        """返回转置后的新矩阵，原矩阵保持不变。"""
        return Matrix(list(zip(*self._data)))  # zip(*rows) 能直接按列重组数据。

    def __matmul__(self, other: Any) -> "Matrix":
        """使用 @ 调用独立的 matmul，并返回新的 Matrix。"""
        return Matrix(matmul(self, other))

    def det(self) -> float:
        """用带部分主元的高斯消元在副本上计算行列式。"""
        rows, cols = self._shape
        if rows != cols:
            raise ValueError("det 只对方阵有定义")

        work = self.to_list()  # 在副本上消元，不能改原矩阵。
        sign = 1.0

        for k in range(rows):
            pivot_row = max(range(k, rows), key=lambda i: abs(work[i][k]))  # 部分主元提升稳定性。
            pivot = work[pivot_row][k]
            if abs(pivot) < _EPS:
                return 0.0

            if pivot_row != k:
                work[k], work[pivot_row] = work[pivot_row], work[k]
                sign *= -1.0

            pivot_row_values = work[k]
            inv_pivot = 1.0 / pivot
            for i in range(k + 1, rows):
                current_row = work[i]
                factor = current_row[k] * inv_pivot
                current_row[k] = 0.0
                for j in range(k + 1, rows):
                    current_row[j] -= factor * pivot_row_values[j]

        det_value = sign
        for i in range(rows):
            det_value *= work[i][i]
        return det_value

    def to_list(self) -> list[list[float]]:
        """返回内部二维 list 的深拷贝。"""
        return [row[:] for row in self._data]


def matmul(A, B) -> list[list[float]]:
    """返回 A 与 B 的矩阵乘积，结果使用二维 list 表示。"""
    left = _as_list_matrix(A)
    right = _as_list_matrix(B)

    left_rows = len(left)
    left_cols = len(left[0])
    right_rows = len(right)
    if left_cols != right_rows:
        raise ValueError("矩阵乘法维度不匹配")

    right_t = list(zip(*right))  # 先转置右矩阵，后续每个点积都按连续内存方向读。
    return [
        [sum(a * b for a, b in zip(left_row, right_col)) for right_col in right_t]
        for left_row in left
    ]
