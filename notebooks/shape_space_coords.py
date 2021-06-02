# type: ignore
import math

import numpy as np


upper_coeffs = [0.1785179, 0.04350172, 0.23296763, 0.01561675]
lower_coeefs = [-0.1443674, -0.12708204, -0.00125364, -0.24809659]


def thickness(psi, n, A):
    return np.dot(A, basis_func(psi, n))


def basis_func(x, n):
    return [math.comb(n, i) * x**(i+0.5) * (1-x)**(n-i+1) for i in range(n+1)]


x = np.linspace(0, 1, 200)
upper = [thickness(v, 3, upper_coeffs) for v in x]
lower = [thickness(v, 3, lower_coeefs) for v in x]

# trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge

coords = list(zip(x[::-1], upper[::-1])) + list(zip(x, lower))

print(coords)

with open('shape_space_coords.npy', 'wb') as f:
    np.save(f, coords)
