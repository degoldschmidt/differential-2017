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

def labels(ax, x='', y='', title='', xticks=None, yticks=None, legend=0):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if legend >= 0:
        ax.legend(loc=legend, frameon=False)
    ax = set_font('Quicksand-Regular', ax=ax)
    return ax

def plotrange(ax, xr, yr):
    ax.set_xlim([xr[0], xr[1]])
    ax.set_ylim([yr[0], yr[1]])
    sns.despine(ax=ax, trim=True)
    return ax


def plot_histograms(ax, _bins, data_list, colors=["#98c37e", "#5788e7", "#D66667", "#B7B7B7", "#98c37e", "#5788e7", "#D66667", "#B7B7B7"], lstyle=['-', '-', '-', '-', '--', '--', '--', '--'], labels=[]):
    ### calculate stats of hist
    for i, df in enumerate(data_list):
        tmean, tsem = np.mean(np.array(df), axis=0), np.std(np.array(df), axis=0)/np.sqrt(len(df))
        ### plotting
        ax.plot(_bins[:-1], tmean, color= colors[i], ls=lstyle[i], label=labels[i], lw=1) # line plot
        ax.fill_between(_bins[:-1], tmean-tsem, tmean+tsem, alpha=0.5, facecolor=colors[i], lw=0)  # SEM plot
    return ax


def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    outfolder = os.path.join(profile.out(), 'kinematics')

    ### Load data
    binsize = 0.2
    var = 'sm_head_speed'
    _bins = np.append(np.arange(0,25, binsize),1000)
    all_lines = []
    for each in ['in', 'out']:
        for cond in ['SAA', 'AA', 'S', 'O']:
            _file = os.path.join(outfolder, experiment+'_kinematics_hist_{}_{}_{:1.0E}_{}.csv'.format(var, each, binsize, cond))
            _df = pd.read_csv(_file)
            _list = []
            for each_col in _df:
                if not 'bin' in each_col:
                    _list.append(np.array(_df[each_col]))
            all_lines.append(_list)

    f, ax = plt.subplots(figsize=(6,4), dpi=600)
    ax.vlines(0.2, 0, 1, colors='#a8a8a8', linestyles= 'dashed', lw= 0.75)
    ax.vlines(2, 0, 1, colors='#a8a8a8', linestyles= 'dashed', lw= 0.75)
    ax = plot_histograms(ax, _bins, all_lines, labels=['SAA in', 'AA in', 'S in', 'O in', 'SAA out', 'AA out', 'S out', 'O out'])
    ax = labels(ax, x='Head speed [mm/s]', y='Rel. frequency', title='Gaussian smoothed (1.2 s)', xticks=[0,2,5,10,15,20,25], yticks=[0,0.1, 0.2, 0.3,0.4])
    ax = plotrange(ax, [0, 25], [0, 0.4001])
    plt.tight_layout()
    plt.savefig(os.path.join(profile.out(), 'plots', 'histo_cond_{:1.0E}_{}.png'.format(binsize, var)), dpi=600)
    ax = labels(ax, x='Head speed [mm/s]', y='Rel. frequency', title='Gaussian smoothed (1.2 s)', xticks=[0,0.2,1,2], yticks=[0,0.1, 0.2, 0.3,0.4])
    ax = plotrange(ax, [0, 2], [0, 0.4001])
    plt.tight_layout()
    plt.savefig(os.path.join(profile.out(), 'plots', 'histo_cond_{:1.0E}_{}_zoomx.png'.format(binsize, var)), dpi=600)
    ax = labels(ax, x='Head speed [mm/s]', y='Rel. frequency', title='Gaussian smoothed (1.2 s)', xticks=[0,2,5,10,15,20,25], yticks=[0, 0.01, 0.02])
    ax = plotrange(ax, [0, 25], [0, 0.02001])
    plt.tight_layout()
    plt.savefig(os.path.join(profile.out(), 'plots', 'histo_cond_{:1.0E}_{}_zoomy.png'.format(binsize, var)), dpi=600)
    plt.cla()
    plt.clf()
    plt.close()

    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
