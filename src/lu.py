"""
带部分主元的 LU 分解，以及用它解 Ax=b。

分解满足 P A = L U，其中 L 为单位下三角矩阵，U 为上三角矩阵。
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

try:
    from .matrix import Matrix as _Matrix
except ImportError:
    try:
        from matrix import Matrix as _Matrix
    except ImportError:
        _Matrix = None


_EPS = 1e-12


def _validate_square_matrix(a: Sequence[Sequence[float]]) -> int:
    if not isinstance(a, Sequence) or len(a) == 0:
        raise ValueError("A 必须是非空方阵")

    n = len(a)
    for row in a:
        if not isinstance(row, Sequence) or len(row) != n:
            raise ValueError("A 必须是非空方阵")
    return n


def _copy_matrix(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(value) for value in row] for row in a]


def _maybe_wrap_matrix(data: List[List[float]]) -> Any:
    if _Matrix is None:
        return data

    try:
        return _Matrix(data)
    except NotImplementedError:
        return data


def _lu_decompose_lists(a: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    """在副本上做分解，返回原始二维 list 形式的 L、U、piv。"""
    n = _validate_square_matrix(a)
    u = _copy_matrix(a)
    l = [[0.0] * n for _ in range(n)]
    piv = list(range(n))

    for i in range(n):
        l[i][i] = 1.0  # L 是单位下三角矩阵。

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(u[i][k]))  # 选当前列绝对值最大的主元。
        pivot_value = u[pivot_row][k]
        if abs(pivot_value) < _EPS:
            raise ValueError("奇异矩阵：主元过小")

        if pivot_row != k:
            u[k], u[pivot_row] = u[pivot_row], u[k]  # U 整行交换。
            piv[k], piv[pivot_row] = piv[pivot_row], piv[k]
            for j in range(k):
                l[k][j], l[pivot_row][j] = l[pivot_row][j], l[k][j]  # L 只交换已写好的乘数。

        pivot_row_values = u[k]
        pivot = pivot_row_values[k]
        for i in range(k + 1, n):
            current_row = u[i]
            factor = current_row[k] / pivot
            l[i][k] = factor
            current_row[k] = 0.0
            for j in range(k + 1, n):
                current_row[j] -= factor * pivot_row_values[j]

    return l, u, piv


def lu_decompose(a: Sequence[Sequence[float]]) -> Tuple[Any, Any, List[int]]:
    """
    对方阵 a 做带部分主元的 LU 分解，满足 P A = L U。

    返回:
        (L, U, piv)
    其中 piv[i] 表示分解后第 i 行来自原矩阵的第 piv[i] 行。
    """
    l, u, piv = _lu_decompose_lists(a)
    return _maybe_wrap_matrix(l), _maybe_wrap_matrix(u), piv


def solve(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """
    先做 LU 分解，再做前代和回代，求解线性方程组 Ax=b。
    """
    n = _validate_square_matrix(a)
    if not isinstance(b, Sequence) or len(b) != n:
        raise ValueError("len(b) 必须等于阶数")

    l, u, piv = _lu_decompose_lists(a)
    pb = [float(b[piv[i]]) for i in range(n)]  # 先按 piv 把右端项做同样的行置换。

    y = [0.0] * n
    for i in range(n):
        row = l[i]
        y[i] = pb[i] - sum(row[j] * y[j] for j in range(i))  # L 的对角线恒为 1，不需要再除。

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row = u[i]
        pivot = row[i]
        if abs(pivot) < _EPS:
            raise ValueError("奇异矩阵：主元过小")
        x[i] = (y[i] - sum(row[j] * x[j] for j in range(i + 1, n))) / pivot

    return x
