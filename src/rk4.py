"""
经典四阶 Runge-Kutta 单步与常微分方程初值问题求解。
"""
from __future__ import annotations


_EPS = 1e-12


def rk4_step(f, x, y, h) -> float:
    """
    从 (x, y) 出发，用步长 h 做一次经典四阶 RK 单步推进。
    """
    if h == 0:
        raise ValueError("步长不能为 0")

    k1 = f(x, y)
    k2 = f(x + h / 2.0, y + h * k1 / 2.0)
    k3 = f(x + h / 2.0, y + h * k2 / 2.0)
    k4 = f(x + h, y + h * k3)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rk4_solve(f, x0, y0, x_end, h) -> tuple[list[float], list[float]]:
    """
    从 (x0, y0) 积分到 x_end，返回每一步的 x、y 列表。
    """
    if h == 0:
        raise ValueError("步长不能为 0")

    if x_end == x0:
        return [x0], [y0]

    x = float(x0)
    y = float(y0)
    target = float(x_end)
    step = float(h)
    total = target - x
    if total * h < 0:
        raise ValueError("步长方向不能到达终点")

    xs = [x]
    ys = [y]

    step_count_float = total / step
    step_count_rounded = round(step_count_float)

    if (
        step_count_rounded > 0
        and abs(step_count_float - step_count_rounded) <= 1e-10
    ):
        actual_h = total / step_count_rounded  # 对接近整数步的区间，均匀重分配步长以压低累计舍入误差。

        for _ in range(step_count_rounded):
            y = rk4_step(f, x, y, actual_h)
            x += actual_h
            xs.append(x)
            ys.append(y)

        xs[-1] = target
        return xs, ys

    while abs(target - x) > _EPS:
        remaining = target - x
        if abs(remaining) <= abs(step) + _EPS:
            # 最后一步精确走到终点，保证返回值与终点对应。
            y = rk4_step(f, x, y, remaining)
            xs.append(target)
            ys.append(y)
            return xs, ys

        y = rk4_step(f, x, y, step)
        x += step
        xs.append(x)
        ys.append(y)

    if x != target:
        # 剩余量小于 _EPS 时循环不会执行，补一步避免 xs 与 ys 错位。
        y = rk4_step(f, x, y, target - x)
        xs.append(target)
        ys.append(y)

    return xs, ys
