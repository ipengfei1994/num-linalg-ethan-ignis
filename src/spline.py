"""
自然三次样条插值与样条求值。
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

try:
    from .lu import solve
except ImportError:
    from lu import solve


def _copy_nodes(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    """复制并校验节点，确保 xs 严格递增且与 ys 一一对应。"""
    if not isinstance(xs, Sequence) or not isinstance(ys, Sequence):
        raise ValueError("xs 和 ys 必须至少两个点")
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs 和 ys 必须至少两个点，且长度相同")

    xs_copy = [float(value) for value in xs]
    ys_copy = [float(value) for value in ys]

    for i in range(len(xs_copy) - 1):
        if xs_copy[i] >= xs_copy[i + 1]:
            raise ValueError("xs 必须严格递增")

    return xs_copy, ys_copy


def _copy_xs(xs: Sequence[float]) -> list[float]:
    """只复制并校验节点横坐标，供样条求值使用。"""
    xs_copy = [float(value) for value in xs]
    if len(xs_copy) < 2:
        raise ValueError("xs 和 ys 必须至少两个点，且长度相同")

    for i in range(len(xs_copy) - 1):
        if xs_copy[i] >= xs_copy[i + 1]:
            raise ValueError("xs 必须严格递增")

    return xs_copy


def natural_cubic_spline(xs, ys) -> list[tuple[float, float, float, float]]:
    """
    构造自然三次样条，每段返回 (a, b, c, d)。
    """
    xs_copy, ys_copy = _copy_nodes(xs, ys)
    n = len(xs_copy) - 1
    h = [xs_copy[i + 1] - xs_copy[i] for i in range(n)]
    slopes = [(ys_copy[i + 1] - ys_copy[i]) / h[i] for i in range(n)]

    m = [0.0] * (n + 1)
    if n > 1:
        system = [[0.0] * (n - 1) for _ in range(n - 1)]
        rhs = [0.0] * (n - 1)
        for i in range(1, n):
            row = system[i - 1]
            if i > 1:
                row[i - 2] = h[i - 1]
            row[i - 1] = 2.0 * (h[i - 1] + h[i])  # 三对角主对角线。
            if i < n - 1:
                row[i] = h[i]

            rhs[i - 1] = 6.0 * (slopes[i] - slopes[i - 1])  # 右端项来自相邻割线斜率之差。

        inner_m = solve(system, rhs)
        for i, value in enumerate(inner_m, start=1):
            m[i] = value

    segments = []
    for i in range(n):
        a = ys_copy[i]
        b = slopes[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0
        c = m[i] / 2.0
        d = (m[i + 1] - m[i]) / (6.0 * h[i])
        segments.append((a, b, c, d))

    return segments


def eval_spline(segments, xs, x) -> float:
    """
    在给定 x 上计算自然三次样条的值。
    """
    xs_copy = _copy_xs(xs)
    if len(segments) != len(xs_copy) - 1:
        raise ValueError("样条段数必须比节点数少 1")

    x_value = float(x)
    if x_value < xs_copy[0] or x_value > xs_copy[-1]:
        raise ValueError("x 必须落在节点区间内")

    if x_value >= xs_copy[-1]:
        idx = len(segments) - 1
    else:
        idx = bisect_right(xs_copy, x_value) - 1

    a, b, c, d = segments[idx]
    t = x_value - xs_copy[idx]  # 每一段都以左端点为局部原点。
    return a + b * t + c * t * t + d * t * t * t
