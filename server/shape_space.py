import math
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np


def shape_space_basis_func(x: float, n: int) -> List[float]:
    """Shape space basis function. Shape space and class functions combined."""
    return [math.comb(n, i) * x**(i+0.5) * (1-x)**(n-i+1) for i in range(n+1)]


def shape_space_thickness(psi: float, n: int, A: Sequence[float]) -> float:
    """Theoretical surface generated using shape space parameters A."""
    return np.dot(A, shape_space_basis_func(psi, n))


def shape_space_coords(
    upper_coeffs: Sequence[float],
    lower_coeffs: Sequence[float],
    sample_points: int = 200,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Generate aerofoil shape from shape space coefficients"""
    polynomial_order = len(upper_coeffs) - 1

    x = np.linspace(0, 1, sample_points)
    upper_coords = [shape_space_thickness(v, polynomial_order, upper_coeffs) for v in x]
    lower_coords = [shape_space_thickness(v, polynomial_order, lower_coeffs) for v in x]

    upper_coords: List[Tuple[float, float]] = list(zip(x, upper_coords))  # type: ignore
    lower_coords: List[Tuple[float, float]] = list(zip(x, lower_coords))  # type: ignore

    # trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge
    coords = upper_coords[::-1] + lower_coords[1:]

    return coords, upper_coords, lower_coords  # type: ignore
