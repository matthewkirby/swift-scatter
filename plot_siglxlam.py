import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

rgrid, sigintrgrid, siglxlamgrid = np.loadtxt('sigmaLx_at_fixed_lamobs.txt', skiprows=1, unpack=True)

rlist = list(set(rgrid))

xs = np.linspace(0.1, 0.7, 500)
colorlist = ['r', 'orange', 'g', 'b', 'grey', 'cyan', 'purple', 'brown']

for r, c in zip(rlist, colorlist):
    idx = abs(rgrid-r) < 0.003
    sigintr = sigintrgrid[idx]
    siglxlam = siglxlamgrid[idx]
    smooth = interp1d(sigintr, siglxlam, kind='cubic')

    plt.plot(xs, smooth(xs), '-', label=r'$r = {}$'.format(r), color=c)
    plt.text(xs[0], smooth(xs[0])+0.035, r'$r = {}$'.format(r), rotation=10, color=c)

plt.xlabel(r'$\sigma_{intr}$')
plt.ylabel(r'$\sigma_{Lx|\lambda_{obs}}$')
# plt.legend()
plt.show()
