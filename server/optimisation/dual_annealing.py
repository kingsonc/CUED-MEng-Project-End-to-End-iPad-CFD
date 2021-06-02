import numpy as np
from optimisation.objective import objective_function
from scipy.optimize import dual_annealing
from visualise import plot_optimised


def dual_annealing_solver(pointcloud_cut: np.ndarray) -> None:
    result = dual_annealing(
        objective_function,
        args=(pointcloud_cut,),
        bounds=[
            (0, 0.5),
            (-0.2, 0.2),
            (0.5, 1),
            (-0.2, 0.2),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
        ],
        x0=[
            0, 0, 1, 0,
            0.1785179, 0.04350172, 0.23296763, 0.01561675,
            -0.1443674, -0.12708204, -0.00125364, -0.24809659,
        ],
        maxfun=10,
        callback=cb,
    )

    plot_optimised(pointcloud_cut, result.x)


def cb(x: float, f: float, context: float) -> None:
    print(f'{x=}')
    print(f'{f=}')
    print(f'{context=}')
