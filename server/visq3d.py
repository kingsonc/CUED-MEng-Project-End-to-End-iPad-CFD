from typing import Tuple

import numpy as np
from scipy.integrate import cumtrapz


def create_computational_grid(coords: np.ndarray):
    imin = np.argmin(coords[:, 0])

    x1 = np.flipud(coords[0 : imin + 1, 0])
    y1 = np.flipud(coords[0 : imin + 1, 1])

    x2 = coords[imin + 1 : :, 0]
    y2 = coords[imin + 1 : :, 1]

    # create and save grid coordinates
    # first, grid on blade
    exprat = np.zeros(81) + 1.2
    exprat[0] = 1.0
    spac = np.cumprod(exprat)
    exprat = np.zeros(20) + 1.2
    exprat[0] = 10.0
    spac2 = np.cumprod(exprat)
    spac[-20::] = np.flipud(spac2)
    spac = np.clip(spac, 0, 25)
    fx = cumtrapz(spac, initial=0)
    fx = fx / fx[-1]
    xmax = 1.0
    xmin = 0.0
    xarg = xmin + fx * (xmax - xmin)
    xcrv = x2
    ycrv = y2
    youtlo = np.interp(xarg, xcrv, ycrv)
    xcrv = x1
    ycrv = y1
    youthi = np.interp(xarg, xcrv, ycrv)

    # now, upstream grid
    exprat = np.zeros(27) + 1.25
    exprat[0] = 1.0
    spac = np.cumprod(exprat)
    spac = np.clip(spac, 0, 25)
    fx = cumtrapz(spac, initial=0)
    fx = fx / fx[-1] * (xmin + 1.0)
    xup = xmin - np.flipud(fx)
    slopeup = 0.0
    yup = (xmin - xup) * slopeup + youtlo[0]

    # now, downstream grid
    exprat = np.zeros(18) + 1.2
    exprat[0] = 1.0
    spac = np.cumprod(exprat)
    spac = np.clip(spac, 0, 10)
    fx = cumtrapz(spac, initial=0)
    fx = fx / fx[-1] * (1.0)
    xdwn = xmax + fx
    slopedwn = 0.0
    ydwn = (xdwn - xmax) * slopedwn + 0.5 * (youtlo[-1] + youthi[-1])

    # assemble geometry ready for output for visq3d
    out_x = np.concatenate((xup[0:-1:], xarg, xdwn[1::]))
    out_ysuct = np.concatenate((yup[0:-1:], youthi, ydwn[1::]))
    out_ythick = np.concatenate(
        (np.zeros(len(xup[0:-1:])), youthi - youtlo, np.zeros(len(xdwn[1::])))
    )

    np.savetxt("visq3d_tmp/visq3d_x.dat", out_x)
    np.savetxt("visq3d_tmp/visq3d_ysuct.dat", out_ysuct)
    np.savetxt("visq3d_tmp/visq3d_ythick.dat", out_ythick)


def convert_output() -> Tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    a = np.loadtxt("visq3d_tmp/visq3d.out")
    jm = 124
    im = 46
    cp = 1005.0
    ga = 1.4
    cv = cp / ga
    rgas = cp - cv
    x = np.reshape(a[:, 0], (jm, im))
    y = np.reshape(a[:, 1], (jm, im))
    pstat = np.reshape(a[:, 4], (jm, im))
    ro = np.reshape(a[:, 5], (jm, im))
    vx = np.reshape(a[:, 6], (jm, im))
    vt = np.reshape(a[:, 8], (jm, im))
    Tstat = pstat / (ro * rgas)
    eke = 0.5 * (vx ** 2 + vt ** 2)
    Mach = np.sqrt(2 * eke / (ga * rgas * Tstat))
    pstag = pstat * (1 + 0.5 * (ga - 1) * Mach ** 2) ** (ga / (ga - 1))

    pitch = y[0, -1] - y[0, 0]

    p1 = pstat[0, 0]
    p01 = pstag[0, 0]

    pstat_coeff = (pstat - p01) / (p01 - p1)
    pstag_coeff = (pstag - p01) / (p01 - p1)

    return pitch, x, y, pstat_coeff, pstag_coeff, vx, vt
