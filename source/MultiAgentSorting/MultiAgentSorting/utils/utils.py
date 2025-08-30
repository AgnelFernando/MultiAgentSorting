from __future__ import annotations
import random
import math
from typing import Tuple, List, Optional


def _sorted_bounds(b: Tuple[float, float]) -> Tuple[float, float]:
    """Ensure (min, max) ordering for a bound tuple."""
    return (b[0], b[1]) if b[0] < b[1] else (b[1], b[0])

def sample_positions_2d(
    bound_x: Tuple[float, float],
    bound_y: Tuple[float, float],
    n: int,
    min_separation: float = 0.06,
    margin: float = 0.0,
    k: int = 30,
    seed: Optional[int] = None,
) -> List[Tuple[float, float]]:
    rng = random.Random(seed)

    # normalize and shrink by margin
    x0, x1 = sorted(bound_x); y0, y1 = sorted(bound_y)
    x0 += margin; x1 -= margin; y0 += margin; y1 -= margin
    if not (x1 > x0 and y1 > y0):
        raise ValueError("Invalid bounds after margin")

    # grid
    r = min_separation
    r2 = r * r
    cell_size = r / math.sqrt(2.0)
    # IMPORTANT: ceil, not floor
    cols = max(1, math.ceil((x1 - x0) / cell_size))
    rows = max(1, math.ceil((y1 - y0) / cell_size))
    grid = [[-1] * cols for _ in range(rows)]
    samples: List[Tuple[float, float]] = []
    active: List[int] = []

    def _rand_in_range(a: float, b: float) -> float:
        # exclude the exact upper bound to avoid gx==cols or gy==rows
        return a + (b - a) * rng.random()

    def _grid_coords(x: float, y: float) -> Tuple[int, int]:
        gx = int((x - x0) / cell_size)
        gy = int((y - y0) / cell_size)
        # clamp to valid index range
        if gx >= cols: gx = cols - 1
        if gy >= rows: gy = rows - 1
        if gx < 0: gx = 0
        if gy < 0: gy = 0
        return gx, gy

    def _in_bounds(x: float, y: float) -> bool:
        # allow at most epsilon past the edge to handle float drift
        return (x0 <= x <= x1) and (y0 <= y <= y1)

    def _far_enough(x: float, y: float) -> bool:
        gcx, gcy = _grid_coords(x, y)
        rchk = 2
        for j in range(max(0, gcy - rchk), min(rows, gcy + rchk + 1)):
            for i in range(max(0, gcx - rchk), min(cols, gcx + rchk + 1)):
                idx = grid[j][i]
                if idx != -1:
                    qx, qy = samples[idx]
                    dx = x - qx; dy = y - qy
                    if dx * dx + dy * dy < r2:
                        return False
        return True

    # seed: use open interval on the right side
    sx = _rand_in_range(x0, x1)
    sy = _rand_in_range(y0, y1)
    samples.append((sx, sy))
    gx, gy = _grid_coords(sx, sy)      # clamp BEFORE indexing
    grid[gy][gx] = 0
    active.append(0)

    # grow
    while active and len(samples) < n:
        a_idx = rng.choice(active)
        ax, ay = samples[a_idx]
        placed = False
        for _ in range(k):
            rad = r * (1.0 + rng.random())     # [r, 2r)
            ang = 2.0 * math.pi * rng.random()
            nx = ax + rad * math.cos(ang)
            ny = ay + rad * math.sin(ang)
            if _in_bounds(nx, ny) and _far_enough(nx, ny):
                samples.append((nx, ny))
                gx, gy = _grid_coords(nx, ny)  # clamp BEFORE indexing
                grid[gy][gx] = len(samples) - 1
                active.append(len(samples) - 1)
                placed = True
                break
        if not placed:
            active.remove(a_idx)

    return samples[:n]

import random
import math

def sample_points(bounds: dict[str, tuple[float, float]], n: int, min_separation: float = 0.0):
    """
    Generate n random (x, y) points within given bounds.

    :param bounds: Dictionary with "x" and "y" ranges, e.g. {"x": (-0.18, 0.25), "y": (-0.37, 0.27)}
    :param n: Number of points to sample
    :param min_separation: Minimum allowed distance between points (default 0.0 → no constraint)
    :return: List of (x, y) tuples
    """
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]

    points = []
    attempts = 0
    max_attempts = 5000

    while len(points) < n and attempts < max_attempts:
        attempts += 1
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        if min_separation > 0.0:
            too_close = any(math.hypot(x - px, y - py) < min_separation for px, py in points)
            if too_close:
                continue

        points.append((x, y))

    if len(points) < n:
        raise RuntimeError(f"Could only place {len(points)} points after {max_attempts} attempts")

    return points
