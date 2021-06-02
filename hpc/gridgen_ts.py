# type: ignore
import argparse

import numpy as np
from scipy.interpolate import interp1d
from ts import ts_tstream_default
from ts import ts_tstream_grid
from ts import ts_tstream_patch_kind
from ts import ts_tstream_type


def gridgen(blade_in, visualise=False):
    """Generate mesh for turbostream."""

    def gendist(n, spac1, spac2, frat, spacmax):
        spac = np.zeros(n - 1)
        fr = np.zeros(n)
        spac[0] = spac1
        spac[-1] = spac2
        for i in range(1, n - 2):
            spac[i] = spac[i - 1] * frat
            if spac[i] > spacmax:
                spac[i] = spacmax
        for i in range(n - 3, 1, -1):
            spacrev = spac[i + 1] * frat
            if spacrev > spacmax:
                spacrev = spacmax
            if spacrev < spac[i]:
                spac[i] = spacrev
        fr[0] = 0.0
        for i in range(n - 1):
            fr[i + 1] = fr[i] + spac[i]
        frend = fr[-1]
        fr = fr / frend
        return fr

    LE_TE = np.zeros([2, 4])
    LE_TE[0, :] = [0.0, 100.0, 1.0, 100.0]
    LE_TE[1, :] = [0.0, 100.01, 1.0, 100.01]

    nsecs = 2
    nblades = 600

    ni_blade = 101
    ni_up = 33
    ni_dwn = 33
    ni = ni_up - 1 + ni_blade + ni_dwn - 1
    nk = 37
    nj = 5

    p01 = 100000.0
    T01 = 300.0
    pexit = 99000.0
    rpm = 0.0

    x_grid = np.zeros([nk, nj, ni], dtype=np.float32)
    r_grid = np.zeros([nk, nj, ni], dtype=np.float32)
    rt_grid = np.zeros([nk, nj, ni], dtype=np.float32)

    x_grid_ns = np.zeros([nk, nsecs, ni])
    r_grid_ns = np.zeros([nk, nsecs, ni])
    rt_grid_ns = np.zeros([nk, nsecs, ni])

    x_stream_out = []
    r_stream_out = []
    x_bl_out = []
    r_bl_out = []

    npts_blade_in = np.shape(blade_in)[0]
    blade = np.ones([npts_blade_in, 3])
    blade[:, 0] = blade_in[:, 0]

    stream = np.ones([101, 2])
    stream[:, 0] = np.linspace(-1.0, 2.0, 101)

    for ns in range(nsecs):

        if ns == 0:
            rnow = 100.0
        else:
            rnow = 100.01

        stream[:, 1] = np.ones(101) * rnow

        blade[:, 1] = np.ones(npts_blade_in) * rnow
        blade[:, 2] = blade_in[:, 1] / rnow

        xle_in = LE_TE[ns, 0]
        xte_in = LE_TE[ns, 2]

        x_blade = blade[:, 0]
        r_blade = blade[:, 1]
        t_blade = blade[:, 2]

        x_bl_out.append(x_blade)
        r_bl_out.append(r_blade)

        rt_blade = r_blade * t_blade

        x_stream = stream[:, 0]
        r_stream = stream[:, 1]
        x_stream_out.append(x_stream)
        r_stream_out.append(r_stream)

        m_stream = np.zeros(len(x_stream))
        for i in range(len(x_stream) - 1):
            dx = x_stream[i + 1] - x_stream[i]
            dr = r_stream[i + 1] - r_stream[i]
            ds = np.sqrt(dx * dx + dr * dr)
            m_stream[i + 1] = m_stream[i] + ds

        xle = xle_in
        mle = np.interp(xle, x_stream, m_stream)

        xte = xte_in
        mte = np.interp(xte, x_stream, m_stream)

        m_chord = mte - mle
        m_nd = (m_stream - mle) / m_chord

        m_blade = np.interp(x_blade, x_stream, m_nd)

        indx_le = np.argmin(m_blade)

        pitch = 2.0 * np.pi / nblades
        m_blade_surf1 = np.flipud(m_blade[: indx_le + 1])
        rt_blade_surf1 = np.flipud(rt_blade[: indx_le + 1])
        m_blade_surf2 = m_blade[indx_le + 1::]
        rt_blade_surf2 = rt_blade[indx_le + 1::] + r_blade[indx_le + 1::] * pitch

        fr_blade = gendist(ni_blade, 1.0, 5.0, 1.2, 20.0)
        m0 = m_blade_surf1[0]
        m1 = m_blade_surf1[-1]
        m_blade_out = fr_blade * (m1 - m0) + m0
        # x_blade_out = np.interp(m_blade_out,m_nd,x_stream)
        # r_blade_out = np.interp(m_blade_out,m_nd,r_stream)
        # rt_blade_out_surf1 = np.interp(m_blade_out,m_blade_surf1,rt_blade_surf1)
        # rt_blade_out_surf2 = np.interp(m_blade_out,m_blade_surf2,rt_blade_surf2)
        x_blade_out = interp1d(m_nd, x_stream, kind="cubic")(m_blade_out)
        r_blade_out = interp1d(m_nd, r_stream, kind="cubic")(m_blade_out)
        rt_blade_out_surf1 = interp1d(m_blade_surf1, rt_blade_surf1, kind="cubic")(
            m_blade_out,
        )
        rt_blade_out_surf2 = interp1d(m_blade_surf2, rt_blade_surf2, kind="cubic")(
            m_blade_out,
        )

        fr_up = gendist(ni_up, 1.0, 50.0, 1.2, 50.0)
        m0 = m_blade_out[0]
        m1 = m_nd[0]
        m_up_out = fr_up * (m1 - m0) + m0
        x_up_out = np.interp(m_up_out, m_nd, x_stream)
        r_up_out = np.interp(m_up_out, m_nd, r_stream)
        theta1 = rt_blade_out_surf1[0] / r_blade_out[0]
        theta2 = theta1 + pitch
        rt_up_out_surf1 = r_up_out * theta1
        rt_up_out_surf2 = r_up_out * theta2
        m_up_out = np.flipud(m_up_out)
        x_up_out = np.flipud(x_up_out)
        r_up_out = np.flipud(r_up_out)
        rt_up_out_surf1 = np.flipud(rt_up_out_surf1)
        rt_up_out_surf2 = np.flipud(rt_up_out_surf2)

        fr_dwn = gendist(ni_dwn, 1.0, 5.0, 1.2, 20.0)
        m0 = m_blade_out[-1]
        m1 = m_nd[-1]
        m_dwn_out = fr_dwn * (m1 - m0) + m0
        x_dwn_out = np.interp(m_dwn_out, m_nd, x_stream)
        r_dwn_out = np.interp(m_dwn_out, m_nd, r_stream)
        theta1 = rt_blade_out_surf1[-1] / r_blade_out[-1]
        theta2 = theta1 + pitch
        rt_dwn_out_surf1 = r_dwn_out * theta1
        rt_dwn_out_surf2 = r_dwn_out * theta2

        m_all = np.concatenate((m_up_out[:-1], m_blade_out, m_dwn_out[1::]))
        x_all = np.concatenate((x_up_out[:-1], x_blade_out, x_dwn_out[1::]))
        r_all = np.concatenate((r_up_out[:-1], r_blade_out, r_dwn_out[1::]))
        rt_all_surf1 = np.concatenate(
            (rt_up_out_surf1[:-1], rt_blade_out_surf1, rt_dwn_out_surf1[1::]),
        )
        rt_all_surf2 = np.concatenate(
            (rt_up_out_surf2[:-1], rt_blade_out_surf2, rt_dwn_out_surf2[1::]),
        )

        fr_circ = gendist(nk, 1.0, 1.0, 1.2, 15.0)
        m_sec_grid = np.zeros([nk, ni])
        x_sec_grid = np.zeros([nk, ni])
        r_sec_grid = np.zeros([nk, ni])
        rt_sec_grid = np.zeros([nk, ni])
        for i in range(ni):
            for k in range(nk):
                m_sec_grid[k, i] = m_all[i]
                x_sec_grid[k, i] = x_all[i]
                r_sec_grid[k, i] = r_all[i]
                rt_sec_grid[k, i] = rt_all_surf1[i] + fr_circ[k] * (
                    rt_all_surf2[i] - rt_all_surf1[i]
                )
                x_grid_ns[k, ns, i] = x_sec_grid[k, i]
                r_grid_ns[k, ns, i] = r_sec_grid[k, i]
                rt_grid_ns[k, ns, i] = rt_sec_grid[k, i]

    fr_rad = gendist(nj, 1.0, 1.0, 1.0, 1.0)
    for i in range(ni):
        for k in [0, -1]:
            xline = x_grid_ns[k, :, i]
            rline = r_grid_ns[k, :, i]
            rtline = rt_grid_ns[k, :, i]
            sline = np.zeros(nsecs)
            for j in range(1, nsecs):
                dx = xline[j] - xline[j - 1]
                dr = rline[j] - rline[j - 1]
                # drt=rtline[j]-rtline[j-1]
                drt = 0.0
                ds = np.sqrt(dx ** 2 + dr ** 2 + drt ** 2)
                sline[j] = sline[j - 1] + ds
            sline = sline / sline[-1]
            # x_grid[k,:,i]=np.interp(fr_rad,sline,xline)/1000.
            # r_grid[k,:,i]=np.interp(fr_rad,sline,rline)/1000.
            # rt_grid[k,:,i]=np.interp(fr_rad,sline,rtline)/1000.
            x_grid[k, :, i] = interp1d(sline, xline, kind="linear")(fr_rad)
            r_grid[k, :, i] = interp1d(sline, rline, kind="linear")(fr_rad)
            rt_grid[k, :, i] = interp1d(sline, rtline, kind="linear")(fr_rad)
        for j in range(nj):
            for k in range(1, nk - 1):
                x_grid[k, j, i] = x_grid[0, j, i] + fr_circ[k] * (
                    x_grid[-1, j, i] - x_grid[0, j, i]
                )
                r_grid[k, j, i] = r_grid[0, j, i] + fr_circ[k] * (
                    r_grid[-1, j, i] - r_grid[0, j, i]
                )
                rt_grid[k, j, i] = rt_grid[0, j, i] + fr_circ[k] * (
                    rt_grid[-1, j, i] - rt_grid[0, j, i]
                )

    g = ts_tstream_grid.TstreamGrid()

    b = ts_tstream_type.TstreamBlock()
    bid = 0
    b.bid = bid
    b.np = 0
    b.ni = ni
    b.nj = nj
    b.nk = nk
    b.procid = 0
    b.threadid = 0
    g.add_block(b)
    g.set_bp("x", ts_tstream_type.float, bid, x_grid)
    g.set_bp("r", ts_tstream_type.float, bid, r_grid)
    g.set_bp("rt", ts_tstream_type.float, bid, rt_grid)

    ile = ni_up
    ite = ni_up + ni_blade - 1

    # Periodic patches
    p1 = ts_tstream_type.TstreamPatch()
    p1.kind = ts_tstream_patch_kind.periodic
    p1.bid = bid
    p1.ist = 0
    p1.ien = ile
    p1.jst = 0
    p1.jen = nj
    p1.kst = 0
    p1.ken = 1
    p1.nxbid = bid
    p1.nxpid = 1
    p1.idir = 0
    p1.jdir = 1
    p1.kdir = 6
    p1.pid = g.add_patch(bid, p1)

    p2 = ts_tstream_type.TstreamPatch()
    p2.kind = ts_tstream_patch_kind.periodic
    p2.bid = bid
    p2.ist = 0
    p2.ien = ile
    p2.jst = 0
    p2.jen = nj
    p2.kst = nk - 1
    p2.ken = nk
    p2.nxbid = bid
    p2.nxpid = 0
    p2.idir = 0
    p2.jdir = 1
    p2.kdir = 6
    p2.pid = g.add_patch(bid, p2)

    # Periodic patches
    p1a = ts_tstream_type.TstreamPatch()
    p1a.kind = ts_tstream_patch_kind.periodic
    p1a.bid = bid
    p1a.ist = ite
    p1a.ien = ni
    p1a.jst = 0
    p1a.jen = nj
    p1a.kst = 0
    p1a.ken = 1
    p1a.nxbid = bid
    p1a.nxpid = 3
    p1a.idir = 0
    p1a.jdir = 1
    p1a.kdir = 6
    p1a.pid = g.add_patch(bid, p1a)

    p2a = ts_tstream_type.TstreamPatch()
    p2a.kind = ts_tstream_patch_kind.periodic
    p2a.bid = bid
    p2a.ist = ite
    p2a.ien = ni
    p2a.jst = 0
    p2a.jen = nj
    p2a.kst = nk - 1
    p2a.ken = nk
    p2a.nxbid = bid
    p2a.nxpid = 2
    p2a.idir = 0
    p2a.jdir = 1
    p2a.kdir = 6
    p2a.pid = g.add_patch(bid, p2a)

    # Periodic patches
    p1b = ts_tstream_type.TstreamPatch()
    p1b.kind = ts_tstream_patch_kind.periodic
    p1b.bid = bid
    p1b.ist = 0
    p1b.ien = ni
    p1b.jst = 0
    p1b.jen = 1
    p1b.kst = 0
    p1b.ken = nk
    p1b.nxbid = bid
    p1b.nxpid = 5
    p1b.idir = 0
    p1b.jdir = 1
    p1b.kdir = 2
    p1b.pid = g.add_patch(bid, p1b)

    p2b = ts_tstream_type.TstreamPatch()
    p2b.kind = ts_tstream_patch_kind.periodic
    p2b.bid = bid
    p2b.ist = 0
    p2b.ien = ni
    p2b.jst = nj - 1
    p2b.jen = nj
    p2b.kst = 0
    p2b.ken = nk
    p2b.nxbid = bid
    p2b.nxpid = 4
    p2b.idir = 0
    p2b.jdir = 1
    p2b.kdir = 2
    p2b.pid = g.add_patch(bid, p2b)

    # Default avs
    for name in ts_tstream_default.av:
        val = ts_tstream_default.av[name]
        if isinstance(val, int):
            g.set_av(name, ts_tstream_type.int, val)
        else:
            g.set_av(name, ts_tstream_type.float, val)

    for name in ts_tstream_default.bv:
        for bid in g.get_block_ids():
            val = ts_tstream_default.bv[name]
            if isinstance(val, int):
                g.set_bv(name, ts_tstream_type.int, bid, val)
            else:
                g.set_bv(name, ts_tstream_type.float, bid, val)

    # Inlet
    pin = ts_tstream_type.TstreamPatch()
    pin.kind = ts_tstream_patch_kind.inlet
    pin.bid = 0
    pin.ist = 0
    pin.ien = 1
    pin.jst = 0
    pin.jen = nj
    pin.kst = 0
    pin.ken = nk
    pin.nxbid = 0
    pin.nxpid = 0
    pin.idir = 6
    pin.jdir = 1
    pin.kdir = 2
    pin.pid = g.add_patch(0, pin)

    pid = pin.pid

    yaw = np.zeros((nk, nj), np.float32)
    pitch = np.zeros((nk, nj), np.float32)
    pstag = np.zeros((nk, nj), np.float32)
    tstag = np.zeros((nk, nj), np.float32)

    pstag += p01
    tstag += T01
    yaw += 0.0
    pitch += 0.0

    g.set_pp("yaw", ts_tstream_type.float, bid, pid, yaw)
    g.set_pp("pitch", ts_tstream_type.float, bid, pid, pitch)
    g.set_pp("pstag", ts_tstream_type.float, bid, pid, pstag)
    g.set_pp("tstag", ts_tstream_type.float, bid, pid, tstag)
    g.set_pv("rfin", ts_tstream_type.float, bid, pid, 0.5)
    g.set_pv("sfinlet", ts_tstream_type.float, bid, pid, 1.0)

    # Outlet

    pout = ts_tstream_type.TstreamPatch()
    pout.kind = ts_tstream_patch_kind.outlet
    pout.bid = 0
    pout.ist = ni - 1
    pout.ien = ni
    pout.jst = 0
    pout.jen = nj
    pout.kst = 0
    pout.ken = nk
    pout.nxbid = 0
    pout.nxpid = 0
    pout.idir = 6
    pout.jdir = 1
    pout.kdir = 2
    pout.pid = g.add_patch(0, pout)

    g.set_pv("throttle_type", ts_tstream_type.int, pout.bid, pout.pid, 0)
    g.set_pv("ipout", ts_tstream_type.int, pout.bid, pout.pid, 0)
    g.set_pv("pout", ts_tstream_type.float, pout.bid, pout.pid, pexit)

    # mixing length
    pitchnow = 2.0 * np.pi * r_grid[0, 0, ile] / nblades
    g.set_bv("xllim", ts_tstream_type.float, bid, 0.03 * pitchnow)
    g.set_bv("fblade", ts_tstream_type.float, bid, 6.0)
    g.set_bv("nblade", ts_tstream_type.float, bid, 6.0)

    # other block variables
    g.set_bv("fmgrid", ts_tstream_type.float, bid, 0.2)
    g.set_bv("poisson_fmgrid", ts_tstream_type.float, bid, 0.05)

    # app vars
    g.set_av("restart", ts_tstream_type.int, 0)
    g.set_av("poisson_restart", ts_tstream_type.int, 0)
    g.set_av("poisson_nstep", ts_tstream_type.int, 10000)
    g.set_av("ilos", ts_tstream_type.int, 1)
    g.set_av("nlos", ts_tstream_type.int, 5)
    g.set_av("nstep", ts_tstream_type.int, 50000)
    g.set_av("nchange", ts_tstream_type.int, 2000)
    g.set_av("dampin", ts_tstream_type.float, 5.0)
    g.set_av("sfin", ts_tstream_type.float, 0.5)
    g.set_av("facsecin", ts_tstream_type.float, 0.005)
    g.set_av("cfl", ts_tstream_type.float, 0.4)
    g.set_av("poisson_cfl", ts_tstream_type.float, 0.5)
    g.set_av("fac_stmix", ts_tstream_type.float, 0.0)
    g.set_av("rfmix", ts_tstream_type.float, 0.01)
    g.set_av("viscosity", ts_tstream_type.float, 1.8e-5)
    g.set_av("viscosity_law", ts_tstream_type.int, 0)

    # initial guess
    g.set_bv("ftype", ts_tstream_type.int, bid, 0)
    g.set_bv("vgridin", ts_tstream_type.float, bid, 50.0)
    g.set_bv("vgridout", ts_tstream_type.float, bid, 50.0)
    g.set_bv("pstatin", ts_tstream_type.float, bid, 99000.0)
    g.set_bv("pstatout", ts_tstream_type.float, bid, 99000.0)
    g.set_bv("tstagin", ts_tstream_type.float, bid, 300.0)
    g.set_bv("tstagout", ts_tstream_type.float, bid, 300.0)
    g.set_bv("xllim_free", ts_tstream_type.float, bid, 0.0)
    g.set_bv("free_turb", ts_tstream_type.float, bid, 0.0)

    # Rotation
    g.set_bv("rpm", ts_tstream_type.float, bid, rpm)
    g.set_bv("rpmi1", ts_tstream_type.float, bid, rpm)
    g.set_bv("rpmi2", ts_tstream_type.float, bid, rpm)
    g.set_bv("rpmj1", ts_tstream_type.float, bid, rpm)
    g.set_bv("rpmj2", ts_tstream_type.float, bid, rpm)
    g.set_bv("rpmk1", ts_tstream_type.float, bid, rpm)
    g.set_bv("rpmk2", ts_tstream_type.float, bid, rpm)

    g.write_hdf5("/home/ksc37/fyp/hpc/in_out_tmp/input.hdf5")

    if visualise:
        import matplotlib.pyplot as plt

        plt.figure()
        for i in range(ni):
            plt.plot(x_grid[0, :, i], r_grid[0, :, i], "-k", linewidth=1)
        for j in range(nj):
            plt.plot(x_grid[0, j, :], r_grid[0, j, :], "-k", linewidth=1)
        for j in range(nsecs):
            plt.plot(x_stream_out[j], r_stream_out[j], "-r")
            plt.plot([LE_TE[j, 0]], [LE_TE[j, 1]], "og")
            plt.plot([LE_TE[j, 2]], [LE_TE[j, 3]], "og")
            plt.plot(x_bl_out[j], r_bl_out[j], "-g")

        plt.axis("equal")
        plt.xlabel("Axial distance, x")
        plt.ylabel("Radial distance, r")

        plt.figure()
        for i in range(ni):
            plt.plot(x_grid[:, -1, i], rt_grid[:, -1, i], "-k", linewidth=1)
        for k in range(nk):
            plt.plot(x_grid[k, -1, :], rt_grid[k, -1, :], "-k", linewidth=1)
        plt.axis("equal")
        plt.xlabel("Axial distance, x")
        plt.ylabel("Circ distance, rt")
        plt.title("Casing")

        plt.figure()
        for i in range(ni):
            plt.plot(x_grid[:, 0, i], rt_grid[:, 0, i], "-k", linewidth=1)
        for k in range(nk):
            plt.plot(x_grid[k, 0, :], rt_grid[k, 0, :], "-k", linewidth=1)
        plt.axis("equal")
        plt.xlabel("Axial distance, x")
        plt.ylabel("Circ distance, rt")
        plt.title("Hub")

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualise', action='store_true')
    args = parser.parse_args()

    blade_in = np.load("/home/ksc37/fyp/hpc/in_out_tmp/shape_space_coords.npy")
    gridgen(blade_in)
