import logging

import numpy as np
from optimisation.objective import objective_function
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from scipy.optimize import SR1
from visualise import plot_optimised


LOGGER = logging.getLogger(__name__)


bounds = [
    (0, 0.5),
    (-0.2, 0.2),
    (0.5, 1),
    (-0.2, 0.2),
    (0.1, 1),
    (0, 1),
    (0, 1),
    (0.1, 1),
    (-1, -0.1),
    (-1, 0),
    (-1, 0),
    (-1, -0.1),
]


def solver(pointcloud_cut: np.ndarray) -> None:
    # result = solver_slsqp(pointcloud_cut)
    # result = solver_trust_constr(pointcloud_cut)
    result = solver_lbfgsb(pointcloud_cut)
    plot_optimised(pointcloud_cut, result.x)

    # breakpoint()


def solver_slsqp(pointcloud_cut: np.ndarray) -> OptimizeResult:
    LOGGER.info('Start minimisation')

    result = minimize(
        objective_function,
        method='SLSQP',
        x0=[
            0, 0, 1, 0,
            0.1679916, 0.15851636, 0.1313818, 0.15751629,
            -0.1679916, -0.15851636, -0.1313818, -0.15751629,
        ],
        args=pointcloud_cut,
        bounds=bounds,
        constraints=[
            {
                'type': 'eq',
                'fun': lambda x: np.array([x[4] + x[8]]),
            },
            {
                'type': 'eq',
                'fun': lambda x: np.array([x[7] + x[11]]),
            },
        ],
        options={'disp': True},
    )

    LOGGER.info('Minimisation complete.')
    print(result)
    return result


def solver_trust_constr(pointcloud_cut: np.ndarray) -> OptimizeResult:
    LOGGER.info('Start minimisation')

    result = minimize(
        objective_function,
        method='trust-constr',
        x0=[
            0, 0, 1, 0,
            0.1679916, 0.15851636, 0.1313818, 0.15751629,
            -0.1679916, -0.15851636, -0.1313818, -0.15751629,
        ],
        args=pointcloud_cut,
        bounds=bounds,
        constraints=[
            LinearConstraint(
                A=[
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                ],
                lb=[0, 0],
                ub=[0, 0],
            ),
        ],
        hess=SR1(),
        options={'disp': True, 'verbose': 2},
    )

    LOGGER.info('Minimisation complete.')
    print(result)
    return result


def solver_lbfgsb(pointcloud_cut: np.ndarray) -> OptimizeResult:
    LOGGER.info('Start minimisation')

    result = minimize(
        objective_function,
        x0=[
            0, 0, 1, 0,
            0.1679916, 0.15851636, 0.1313818, 0.15751629,
            -0.1679916, -0.15851636, -0.1313818, -0.15751629,
        ],
        args=pointcloud_cut,
        bounds=bounds,
        options={'disp': True},
    )

    LOGGER.info('Minimisation complete.')
    print(result)
    return result
