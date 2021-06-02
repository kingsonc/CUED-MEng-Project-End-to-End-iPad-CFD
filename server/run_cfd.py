import logging
from typing import List
from typing import Tuple
import subprocess

import mesh
import numpy as np
import shape_space
from visualise import pressure_contours
from visualise import VisualiseLevel
import visq3d

import hpc


LOGGER = logging.getLogger(__name__)


def run_cfd(
    upper_surface_coeffs: Tuple[float],
    lower_surface_coeffs: Tuple[float],
    visualise_level: VisualiseLevel = VisualiseLevel.OFF,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    coords, *_ = shape_space.shape_space_coords(upper_surface_coeffs, lower_surface_coeffs)

    # pitch, x, y, pstat_coeff, pstag_coeff, vx, vt = _run_turbostream(coords)
    pitch, x, y, pstat_coeff, pstag_coeff, vx, vt = _run_visq3d(coords)

    if visualise_level >= VisualiseLevel.LOW:
        pressure_contours(x, y, pitch, pstat_coeff, pstag_coeff, vx, vt)

    indices = mesh.vertex_indices(*x.shape)

    return pitch, x, y, pstat_coeff, pstag_coeff, vx, vt, indices


def _run_turbostream(
    coords: List[Tuple[float, float]]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pitch, x, y, pstat_coeff, pstag_coeff, vx, vt = hpc.hpc_run(coords, fakerun=True)
    return pitch, x, y, pstat_coeff, pstag_coeff, vx, vt


def _run_visq3d(
    coords: List[Tuple[float, float]]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    visq3d.create_computational_grid(np.array(coords))
    subprocess.run(["./visq3d_solver.x"])
    pitch, x, y, pstat_coeff, pstag_coeff, vx, vt = visq3d.convert_output()

    return pitch, x, y, pstat_coeff, pstag_coeff, vx, vt
