import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
from pandas import DataFrame, read_csv


def main():
    # Load the data
    fits1 = read_csv('outputs/sigma_Lx_fixed_lamobs_gaussian.txt', sep='\t', skiprows=3)
    fits2 = read_csv('outputs/sigma_Lx_fixed_lamobs_lognormal.txt', sep='\t', skiprows=3)

    # Setup
    # rlist = sorted(list(set(fits['r'].values)))
    rlist = [0.0]
    xs = np.linspace(0.1, 0.7, 500)
    colorlist = ['r', 'orange', 'g', 'b', 'yellow', 'cyan', 'purple', 'brown', 'magenta']

    fig, axes = plt.subplots(2,2, sharex=True)
    x = np.array([-0.02, 0.82])

    # Scatter
    for r, c in zip(rlist, colorlist):
        # subframe = fits1[abs(fits1['r']-r) < 0.003]
        # smooth = interp1d(subframe['sigint'], subframe['mad2'], kind='linear')
        # axes[0,0].plot(xs, smooth(xs), '-', label=r'gaussian'.format(r), color=c)
        # axes[0,0].plot(xs, smooth(xs), '-', label=r'$r = {}$'.format(r), color=c)

        subframe = fits1[abs(fits1['r']-r) < 0.003]
        axes[0,0].plot(subframe['sigint'], subframe['mad2'], 'or', label=r'gaussian, MAD'.format(r), fillstyle='none', markeredgewidth=0.5)
        axes[0,0].plot(subframe['sigint'], subframe['std2'], '^r', label=r'gaussian, STD'.format(r), fillstyle='none', markeredgewidth=0.5)
        subframe = fits2[abs(fits2['r']-r) < 0.003]
        axes[0,0].plot(subframe['sigint'], subframe['mad2'], 'ok', label=r'log-normal, MAD'.format(r), fillstyle='none', markeredgewidth=0.5)
        axes[0,0].plot(subframe['sigint'], subframe['std2'], '^k', label=r'log-normal, STD'.format(r), fillstyle='none', markeredgewidth=0.5)

    yup = np.ones(2)*(0.788+0.046)
    ydown = np.ones(2)*(0.788-0.044)
    axes[0,0].fill_between(x, yup, ydown, facecolor='k', edgecolor="none", alpha=0.25)
    axes[0,0].set_ylabel(r'$\sigma^{kMAD2}_{Lx|\lambda_{obs}}$')
    axes[0,0].set_xlim(x)


    # amplitude
    for r, c in zip(rlist, colorlist):
        # subframe = fits[abs(fits['r']-r) < 0.003]
        # smooth = interp1d(subframe['sigint'], subframe['amp2'], kind='linear')
        # axes[0,1].plot(xs, smooth(xs), '-', label=r'$r = {}$'.format(r), color=c)

        subframe = fits1[abs(fits1['r']-r) < 0.003]
        axes[0,1].plot(subframe['sigint'], subframe['amp2'], 'or', label=r'gaussian', fillstyle='none', markeredgewidth=0.5)
        subframe = fits2[abs(fits2['r']-r) < 0.003]
        axes[0,1].plot(subframe['sigint'], subframe['amp2'], '^k', label=r'log-normal', fillstyle='none', markeredgewidth=0.5)


    yup = np.ones(2)*(99.47+0.07)
    ydown = np.ones(2)*(99.47-0.07)
    axes[0,1].fill_between(x, yup, ydown, facecolor='k', edgecolor="k", alpha=0.25)
    axes[0,1].set_ylabel(r'ln Amplitude')
    axes[0,1].set_xlim(x)
    axes[0,1].yaxis.tick_right()
    axes[0,1].yaxis.set_label_position("right")


    # slope
    for r, c in zip(rlist, colorlist):
        # subframe = fits[abs(fits['r']-r) < 0.003]
        # smooth = interp1d(subframe['sigint'], subframe['slope2'], kind='linear')
        # axes[1,0].plot(xs, smooth(xs), '-', label=r'$r = {}$'.format(r), color=c)

        subframe = fits1[abs(fits1['r']-r) < 0.003]
        axes[1,0].plot(subframe['sigint'], subframe['slope2'], 'or', label=r'gaussian', fillstyle='none', markeredgewidth=0.5)
        subframe = fits2[abs(fits2['r']-r) < 0.003]
        axes[1,0].plot(subframe['sigint'], subframe['slope2'], '^k', label=r'log-normal', fillstyle='none', markeredgewidth=0.5)

    yup = np.ones(2)*(1.65+0.1)
    ydown = np.ones(2)*(1.65-0.1)
    axes[1,0].fill_between(x, yup, ydown, facecolor='k', edgecolor="k", alpha=0.25)
    axes[1,0].set_xlabel(r'$\sigma_{intr}$')
    axes[1,0].set_ylabel(r'Slope')
    axes[1,0].set_xlim(x)


    # fraccut
    for r, c in zip(rlist, colorlist):
        # subframe = fits[abs(fits['r']-r) < 0.003]
        # smooth = interp1d(subframe['sigint'], subframe['fraccut'], kind='linear')
        # axes[1,1].plot(xs, smooth(xs), '-', label=r'$r = {}$'.format(r), color=c)

        subframe = fits1[abs(fits1['r']-r) < 0.003]
        axes[1,1].plot(subframe['sigint'], subframe['fraccut'], 'or', label=r'gaussian', fillstyle='none', markeredgewidth=0.5)
        subframe = fits2[abs(fits2['r']-r) < 0.003]
        axes[1,1].plot(subframe['sigint'], subframe['fraccut'], '^k', label=r'log-normal', fillstyle='none', markeredgewidth=0.5)

    ftrue = 1./155
    yup = np.ones(2)*(ftrue + ftrue)
    ydown = np.ones(2)*(ftrue - ftrue)
    axes[1,1].fill_between(x, yup, ydown, facecolor='k', edgecolor="none", alpha=0.25)
    axes[1,1].set_xlabel(r'$\sigma_{intr}$')
    axes[1,1].set_ylabel(r'Frac Cut')
    axes[1,1].set_xlim(x)
    axes[1,1].set_ylim(1e-3, 4e-1)
    axes[1,1].set_yscale('log')
    axes[1,1].yaxis.tick_right()
    axes[1,1].yaxis.set_label_position("right")










    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5)
    fig.subplots_adjust(top=0.85, wspace=0.05, hspace=0.005)


    plt.show()





if __name__ == "__main__":
    main()
