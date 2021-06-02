import matplotlib.pyplot as plt
from ts import ts_tstream_cut
from ts import ts_tstream_reader


pref = 100000.
tref = 300.

fname = "output.hdf5"
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read(fname)

cp = g.get_av("cp")
ga = g.get_av("ga")
rgas = cp - cp/ga

bid = 0

b = g.get_block(bid)
ni = b.ni
nj = b.nj
nk = b.nk

# cut at mid height
jcut = (nj+1)/2

tsc = ts_tstream_cut.TstreamStructuredCut()
tsc.read_from_grid(g, pref, tref, bid, 0, ni, jcut, jcut+1, 0, b.nk)

pitch = tsc.rt[-1, 0]-tsc.rt[0, 0]
p1 = tsc.pstat[0, 0]
p01 = tsc.pstag[0, 0]

plt.figure()
plt.contourf(tsc.x, tsc.rt, (tsc.pstat-p01)/(p01-p1), 21)
plt.contourf(tsc.x, tsc.rt+pitch, (tsc.pstat-p01)/(p01-p1), 21)
plt.axis("equal")
plt.colorbar()

plt.figure()
plt.contourf(tsc.x, tsc.rt, (tsc.pstag-p01)/(p01-p1), 21)
plt.contourf(tsc.x, tsc.rt+pitch, (tsc.pstag-p01)/(p01-p1), 21)
plt.axis("equal")
plt.colorbar()

plt.show()
