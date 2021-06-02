from enum import IntEnum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from optimisation.objective import transform_points
from shape_space import shape_space_thickness


plt.rc('font', size=24)
plt.rcParams["figure.figsize"] = (10, 7)


class VisualiseLevel(IntEnum):
    OFF = 0
    LOW = 1
    HIGH = 2


def visualise_pointclouds(*pcds: o3d.geometry.PointCloud) -> None:
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    o3d.visualization.draw_geometries(pcds + (axes,))


def plot_naca_12(plt: plt) -> None:
    def yt(x: float, t: float) -> float:
        return 5*t*(0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    x = np.linspace(0, 1, num=200)
    y = yt(x, 0.12)
    plt.plot(x, y, 'lime', label='Theoretical NACA 0012 aerofoil', linewidth=3)
    plt.plot(x, -y, 'lime', linewidth=3)


def plot_optimised(cut: np.ndarray, search_vector: List[float]) -> None:
    transformed_cut = transform_points(
        cut,
        leading_edge_x=search_vector[0],
        leading_edge_y=search_vector[1],
        trailing_edge_x=search_vector[2],
        trailing_edge_y=search_vector[3],
    )

    plt.scatter(transformed_cut[:, 0], transformed_cut[:, 1], marker='.')

    x = np.linspace(0, 1, 1000)
    us = [shape_space_thickness(v, 3, search_vector[4:8]) for v in x]
    ls = [shape_space_thickness(v, 3, search_vector[8:12]) for v in x]

    plt.plot(x, us, 'r', label='SLSQP algorithm fitted aerofoil', linewidth=3)
    plt.plot(x, ls, 'r', linewidth=3)

    plot_naca_12(plt)

    plt.legend()
    plt.axis('equal')
    plt.show()


def pressure_contours(
    x: np.ndarray,
    y: np.ndarray,
    pitch: float,
    pstat_coeff: np.ndarray,
    pstag_coeff: np.ndarray,
    vx: np.ndarray,
    vt: np.ndarray,
) -> None:
    plt.figure()
    plt.title('Static pressure coefficients')
    plt.contourf(x, y, pstat_coeff, 21, cmap='Spectral')
    plt.contourf(x, y + pitch, pstat_coeff, 21, cmap='Spectral')
    plt.axis("equal")
    plt.colorbar()

    plt.figure()
    plt.title('Stagnation pressure coefficients')
    plt.contourf(x, y, pstag_coeff, 21, cmap='Spectral')
    plt.contourf(x, y + pitch, pstag_coeff, 21, cmap='Spectral')
    plt.axis("equal")
    plt.colorbar()

    plt.figure()
    plt.title('Axial velocity')
    plt.contourf(x, y, vx, 21, cmap='Spectral')
    plt.contourf(x, y + pitch, vx, 21, cmap='Spectral')
    plt.axis("equal")
    plt.colorbar()

    plt.figure()
    plt.title('Tangential velocity')
    plt.contourf(x, y, vt, 21, cmap='Spectral')
    plt.contourf(x, y + pitch, vt, 21, cmap='Spectral')
    plt.axis("equal")
    plt.colorbar()

    plt.show()
