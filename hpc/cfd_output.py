# type: ignore
import numpy as np
from ts import ts_tstream_cut
from ts import ts_tstream_reader


def pressures(filepath):
    pref = 100000.
    tref = 300.

    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(filepath)

    bid = 0
    b = g.get_block(bid)
    ni = b.ni
    nj = b.nj

    # cut at mid height
    jcut = (nj+1)/2

    tsc = ts_tstream_cut.TstreamStructuredCut()
    tsc.read_from_grid(g, pref, tref, bid, 0, ni, jcut, jcut+1, 0, b.nk)

    pitch = tsc.rt[-1, 0]-tsc.rt[0, 0]
    p1 = tsc.pstat[0, 0]
    p01 = tsc.pstag[0, 0]

    with open('/home/ksc37/fyp/hpc/in_out_tmp/cfd_output.npz', 'wb') as f:
        np.savez(
            f,
            x=tsc.x,
            y=tsc.rt,
            pitch=pitch,
            pstat_coeff=(tsc.pstat-p01)/(p01-p1),
            pstag_coeff=(tsc.pstag-p01)/(p01-p1),
            vx=tsc.vx,
            vt=tsc.vt,
        )


if __name__ == "__main__":
    pressures('/home/ksc37/fyp/hpc/in_out_tmp/output.hdf5')
