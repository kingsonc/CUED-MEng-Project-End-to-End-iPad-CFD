import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from shape_space import shape_space_basis_func
from shape_space import shape_space_thickness
from visualise import plot_naca_12
from visualise import VisualiseLevel


plt.rc('font', size=24)
plt.rcParams["figure.figsize"] = (10, 7)


LOGGER = logging.getLogger(__name__)

SHAPE_SPACE_ORDER = 3


def least_squares(normalized_cut: np.ndarray, visualise_level: VisualiseLevel) -> Tuple[np.ndarray, np.ndarray]:
    LOGGER.info("Performing least squares fit for shape space coefficients")

    # ignore leading and trailing edge points as they are noisy
    trimmed_cut = normalized_cut[(normalized_cut[:, 0] > 0.1) & (normalized_cut[:, 0] < 0.9)]

    upper_surface_data = trimmed_cut[trimmed_cut[:, 1] > 0]
    lower_surface_data = trimmed_cut[trimmed_cut[:, 1] < 0]

    upper_surface_coeffs = least_squares_surface(upper_surface_data)
    lower_surface_coeffs = least_squares_surface(lower_surface_data)

    LOGGER.info(f"Upper surface coefficients: {upper_surface_coeffs}")
    LOGGER.info(f"Lower surface coefficients: {lower_surface_coeffs}")

    """code for finding standard deviation in data

    means = np.zeros(265)
    stds = np.zeros(265)

    scaled_us = upper_surface_data * 265
    for i in range(265):
        cut = scaled_us[(scaled_us[:,0] > (i-1)) & (scaled_us[:,0] < i+1)]
        means[i] = np.mean(cut[:,1])
        stds[i] = np.std(cut[:,1])

    breakpoint()
    """

    if visualise_level >= VisualiseLevel.LOW:
        plt.scatter(normalized_cut[:, 0], normalized_cut[:, 1], marker='.')

        x = np.linspace(0, 1, 1000)
        us = [shape_space_thickness(v, SHAPE_SPACE_ORDER, upper_surface_coeffs) for v in x]
        ls = [shape_space_thickness(v, SHAPE_SPACE_ORDER, lower_surface_coeffs) for v in x]
        plt.plot(x, us, 'r', label='Least squares fitted aerofoil', linewidth=3)
        plt.plot(x, ls, 'r', linewidth=3)

        plot_naca_12(plt)

        plt.axis('equal')
        plt.legend()
        plt.show()

    return upper_surface_coeffs, lower_surface_coeffs


def least_squares_surface(coords: np.ndarray) -> np.ndarray:
    """Use pseudo-inverse of matrix with shape space basis functions to estimate shape space
    parameters for one surface."""

    phi = [shape_space_basis_func(x, SHAPE_SPACE_ORDER) for x in coords[:, 0] if 0 < x < 1]
    b = [coords[i, 1] for i in range(coords.shape[0]) if 0 < coords[i, 0] < 1]

    psuedoinverse = np.linalg.pinv(phi)
    return np.matmul(psuedoinverse, b)
