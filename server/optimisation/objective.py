from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
from shape_space import shape_space_thickness
from shapely.geometry import asLineString
from shapely.geometry import asPoint


POLYNOMIAL_ORDER = 3
NUM_SAMPLE_POINTS = 200


"""
search vector x: [
    leading edge x,
    leading edge y,
    trailing edge x,
    trailing edge y,
    4 upper coeffs,
    4 lower coeffs
]
overall length: 12
"""


def objective_function(search_vector: Sequence[float], pointcloud_cut: np.ndarray) -> float:
    print('Start objective func')

    upper_coeffs = search_vector[4:8]
    lower_coeffs = search_vector[8:12]

    aerofoil_coords = shape_space_coords(upper_coeffs, lower_coeffs)
    aerofoil = asLineString(aerofoil_coords)

    transformed_cut = transform_points(
        pointcloud_cut,
        leading_edge_x=search_vector[0],
        leading_edge_y=search_vector[1],
        trailing_edge_x=search_vector[2],
        trailing_edge_y=search_vector[3],
    )

    ans = sum([asPoint(point).distance(aerofoil) for point in transformed_cut])
    print(ans)

    return ans


def shape_space_coords(upper_coeffs: Sequence[float], lower_coeffs: Sequence[float]) -> List[Tuple[float, float]]:
    """Generate aerofoil shape from shape space coefficients"""

    x = np.linspace(0, 1, NUM_SAMPLE_POINTS)
    upper_coords = [shape_space_thickness(v, POLYNOMIAL_ORDER, upper_coeffs) for v in x]
    lower_coords = [shape_space_thickness(v, POLYNOMIAL_ORDER, lower_coeffs) for v in x]

    # trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge
    coords = list(zip(x[::-1], upper_coords[::-1])) + list(zip(x, lower_coords))

    return coords


def transform_points(
    pointcloud_cut: np.ndarray,
    leading_edge_x: float,
    leading_edge_y: float,
    trailing_edge_x: float,
    trailing_edge_y: float,
) -> np.ndarray:
    """Scale and rotate to place leading_edge at (0,0) and trailing_edge at (1,0)."""

    transformed_cut = np.copy(pointcloud_cut)

    # Shift and scale x-axis
    transformed_cut[:, 0] = (transformed_cut[:, 0] - leading_edge_x) / max(1e-10, trailing_edge_x - leading_edge_x)

    # Shift y-axis so leading edge at (0,0)
    transformed_cut[:, 1] = transformed_cut[:, 1] - leading_edge_y

    # Rotate points so trailing edge at (1,0)
    rotate_angle = -np.arctan2(trailing_edge_y - leading_edge_y, trailing_edge_x - leading_edge_x)

    rotation_matrix = np.array([
        [np.cos(rotate_angle), -np.sin(rotate_angle)],
        [np.sin(rotate_angle), np.cos(rotate_angle)],
    ])

    transformed_cut = np.matmul(rotation_matrix, transformed_cut.T).T
    return transformed_cut
