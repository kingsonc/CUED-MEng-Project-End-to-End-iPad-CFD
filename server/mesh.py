import logging
from typing import List
from typing import NamedTuple
from typing import Sequence
from typing import Tuple

import numpy as np
import shape_space
from matplotlib import cm


LOGGER = logging.getLogger(__name__)


class Vertex(NamedTuple):
    position: Tuple[float, float, float]
    color: Tuple[float, float, float]  # RGB


def vertex_indices(M: int, N: int) -> List[int]:
    def get_index(i: int, j: int) -> int:
        return i * N + j

    indices: List[int] = []

    """
    0---3         0---3           3
    |   |  =>     |  /   and    / |
    |   |         | /          /  |
    1---2         1           1---2

    """

    for i in range(M - 1):
        for j in range(N - 1):
            indices.extend(
                [
                    get_index(i, j),  # 0
                    get_index(i + 1, j),  # 1
                    get_index(i, j + 1),  # 3
                    get_index(i, j + 1),  # 3
                    get_index(i + 1, j),  # 1
                    get_index(i + 1, j + 1),  # 2
                ],
            )

    return indices


def create_mesh(
    X: np.ndarray,
    Y: np.ndarray,
    pstat_coeff: np.ndarray,
    blade_start: Tuple[float, float],
    blade_end: Tuple[float, float],
    height: float,
) -> List[Vertex]:
    # Transformation matrix to scale coordinates
    T = np.array(
        [
            [blade_start[0] - blade_end[0], 0],
            [blade_start[1] - blade_end[1], blade_start[0] - blade_end[0]],
        ],
    )

    def make_vertex(i: int, j: int) -> Vertex:
        x, y = np.matmul(T, (float(X[i, j]), float(Y[i, j]))) - np.array(blade_start)
        return Vertex(
            position=(y, height, -x),
            color=colormap.to_rgba(pstat_coeff[i, j])[:3],
        )

    LOGGER.info('Generating vertices.')

    colormap = cm.ScalarMappable(cmap='Spectral')
    colormap.set_clim(pstat_coeff.min(), pstat_coeff.max())

    m, n = X.shape
    vertices: List[Vertex] = []

    """
    0---3         0---3           3
    |   |  =>     |  /   and    / |
    |   |         | /          /  |
    1---2         1           1---2

    013, 312
    """

    for i in range(m - 1):
        for j in range(n - 1):
            vertices.extend(
                (
                    make_vertex(i, j),  # 0
                    make_vertex(i + 1, j),  # 1
                    make_vertex(i, j + 1),  # 3
                    make_vertex(i, j + 1),  # 3
                    make_vertex(i + 1, j),  # 1
                    make_vertex(i + 1, j + 1),  # 2
                ),
            )

    LOGGER.info(f'Generated {len(vertices)} vertices.')

    return vertices


def aerofoil_mesh(
    upper_surface_coeffs: Sequence[float], lower_surface_coeffs: Sequence[float], height: float = 0.15,
) -> List[Tuple[float, float, float, float, float, float]]:
    # trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge
    coords, upper_coords, lower_coords = shape_space.shape_space_coords(upper_surface_coeffs, lower_surface_coeffs)

    """
    0---3         0---3           3
    |   |  =>     |  /   and    / |
    |   |         | /          /  |
    1---2         1           1---2

    013, 312
    """

    vertices = []

    span_color = (1, 0.75, 0.8)
    top_color = (0.75, 1, 0.82)

    # spanwise
    for i in range(len(coords) - 1):
        vertices.extend(
            [
                (*coords[i], height, *span_color),  # 0
                (*coords[i], 0, *span_color),  # 1
                (*coords[i + 1], height, *span_color),  # 3
                (*coords[i + 1], height, *span_color),  # 3
                (*coords[i], 0, *span_color),  # 1
                (*coords[i + 1], 0, *span_color),  # 2
            ],
        )

    # top surface
    for i in range(len(upper_coords) - 1):
        vertices.extend(
            [
                (*upper_coords[i], height, *top_color),  # 0
                (*lower_coords[i], height, *top_color),  # 1
                (*upper_coords[i + 1], height, *top_color),  # 3
                (*upper_coords[i + 1], height, *top_color),  # 3
                (*lower_coords[i], height, *top_color),  # 1
                (*lower_coords[i + 1], height, *top_color),  # 2
            ],
        )

    return vertices  # type: ignore
