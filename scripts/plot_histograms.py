import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench
from pytrack_analysis.viz import set_font

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(_bins, in_hists, out_hists, which, folder, binsize):
    ### calculate stats of hist
    in_mean, in_sem = np.mean(np.array(in_hists), axis=0), np.std(np.array(in_hists), axis=0)/np.sqrt(len(in_hists))
    out_mean, out_sem = np.mean(np.array(out_hists), axis=0), np.std(np.array(out_hists), axis=0)/np.sqrt(len(out_hists))
    ### plotting
    f, ax = plt.subplots(figsize=(6,4), dpi=600)
    ax.vlines(0.2, 0, 1, colors='#a8a8a8', linestyles= 'dashed', lw= 0.75)
    ax.vlines(2, 0, 1, colors='#a8a8a8', linestyles= 'dashed', lw= 0.75)
    ax.plot(_bins[:-1], in_mean, color= '#575757', ls='-', label='inside patch', lw=1) # line plot
    ax.fill_between(_bins[:-1], in_mean-in_sem, in_mean+in_sem, alpha=0.5, facecolor='#575757', lw=0)  # SEM plot
    ax.plot(_bins[:-1], out_mean, color= '#2b2b2b', ls='--', label='outside patch', lw=1)
    ax.fill_between(_bins[:-1], out_mean-out_sem, out_mean+out_sem, alpha=0.5, facecolor='#2b2b2b', lw=0)  # SEM plot
    ax.legend(loc=1, frameon=False)
    ax.set_xlim([0,2.01])
    maxy = 1.1*np.amax(out_mean)
    ax.set_ylim([-maxy/20, maxy])
    #ax.set_yticks(np.arange(0.,round(maxy, 1),.05))
    ax.set_xticks([0, 0.2, 1, 2])
    ax.set_xlabel('Head speed [mm/s]')
    ax.set_ylabel('Rel. frequency')
    if which == '':
        ax.set_title('Without Gaussian smoothing')
    if which == 'sm_':
        ax.set_title('Gaussian smoothing (width: 1.2 s)')
    if which == 'smm_':
        ax.set_title('Gaussian smoothing (width: 2.4 s)')

    ax = set_font('Quicksand-Regular', ax=ax)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'plots', 'histo_{:1.0E}_{}speed_zoomx.png'.format(binsize, which)), dpi=600)
    ax.set_xlim([0,25.1])
    ax.set_xticks([0, 2, 5, 10, 15, 20, 25])
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'plots', 'histo_{:1.0E}_{}speed_full.png'.format(binsize, which)), dpi=600)
    ax.set_yticks([0, 0.005, 0.01, 0.02])
    ax.set_ylim([-maxy/(20*30),maxy/30])
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'plots', 'histo_{:1.0E}_{}speed_zoomy.png'.format(binsize, which)), dpi=600)
    plt.cla()
    plt.clf()
    plt.close()

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)

    ### Load data
    _binsize = 0.2
    which = 'sm_'
    _bins = np.append(np.arange(0,25,_binsize),1000)
    inhistdf = pd.read_csv(os.path.join(profile.out(), experiment+'_kinematics_ihist_{}{:1.0E}.csv'.format(which, _binsize)))
    outhistdf = pd.read_csv(os.path.join(profile.out(), experiment+'_kinematics_ohist_{}{:1.0E}.csv'.format(which, _binsize)))

    out_hists = [] ##np.zeros((n_ses, len(_bins)-1))
    in_hists = []
    for each_col in inhistdf:
        if not 'bin' in each_col:
            out_hists.append(np.array(outhistdf[each_col]))
            in_hists.append(np.array(inhistdf[each_col]))

    plot_histograms(_bins, in_hists, out_hists, which, profile.out(), _binsize)
    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
