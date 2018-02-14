import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench
from pytrack_analysis.viz import set_font, swarmbox

import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
#pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)
import tkinter as tk

def plot_cumulatives(data, color=None, ax=None, title=None, tiky=None, maxy=None, reduce=None):
    maxx = 108000
    dmean = np.mean(np.array(data)[:maxx,:], axis=1)
    dsem = np.std(np.array(data)[:maxx,:], axis=1) #/np.sqrt(np.array(data).shape[0])
    for col in data.columns:
        ts = np.array(data.loc[:,col])[:maxx]
        ax.plot(ts/60, color='#8c8c8c', alpha=0.5, lw=0.5)
    ax.plot(dmean/60, color=color, alpha=0.9, lw=1)
    ax.fill_between(np.arange(0,maxx), (dmean-dsem)/60, (dmean+dsem)/60, color=color, alpha=0.5, lw=0)
    ax.set_title(title)
    ax.set_xlim([-0.01*maxx,maxx])
    ax.set_xticks([0, maxx/2, maxx])
    ax.set_xticklabels(['0', '30', '60'])
    ax.set_xlabel("Time [min]")
    ax.set_ylim([-.1*maxy,1.05*maxy])
    if not reduce:
        ax.set_yticks(np.arange(0,maxy+1,tiky))
        ax.set_ylabel("Cumulative\nduration of\nmicromovements\n[min]")
        sns.despine(ax=ax, trim=True)
    else:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        sns.despine(ax=ax, left=True, trim=True)
    return ax

def main():
    """
    --- general parameters
     *
    """
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())
    sessions = db.sessions
    n_ses = len(sessions)


    conds = ["SAA", "AA", "S", "O"]
    EthoTotals = {'Yeast': {each_condition: {} for each_condition in conds}, 'Sucrose': {each_condition: {} for each_condition in conds}}
    infolder = os.path.join(profile.out(), 'classify')
    for i_ses, each in enumerate(sessions):
        ### Loading data
        try:
            meta = each.load_meta()
            csv_file = os.path.join(infolder,  each.name+'_classifier.csv')
            df = pd.read_csv(csv_file, index_col='frame')
            df['Ydur'], df['Sdur'] = df['frame_dt'], df['frame_dt']
            df.loc[df['etho'] != 4, 'Ydur'] = 0
            df.loc[df['etho'] != 5, 'Sdur'] = 0
            EthoTotals['Yeast'][meta['condition']][each.name] = np.cumsum(df['Ydur'])
            EthoTotals['Sucrose'][meta['condition']][each.name] = np.cumsum(df['Sdur'])
            print(each.name)
        except FileNotFoundError:
            pass

    ### Plotting
    colors = ["#98c37e", "#5788e7", "#D66667", "#2b2b2b"]
    maxy = {'Yeast': 2500/60, 'Sucrose': 1000/60}
    tiky = {'Yeast': 10, 'Sucrose': 5}
    for each_substr in ['Yeast', 'Sucrose']:
        f, axes = plt.subplots(1, 4, figsize=(8,3), dpi=400, sharey=True)
        for i, each_cond in enumerate(conds):
            _reduce=False
            if i>0:
                _reduce=True
            df = pd.DataFrame(EthoTotals[each_substr][each_cond])
            axes[i] = plot_cumulatives(df, color=colors[i], ax=axes[i], title=each_cond, tiky=tiky[each_substr], maxy=maxy[each_substr], reduce=_reduce)
            f.suptitle('{}'.format(each_substr), fontsize=10, fontweight='bold', x=0.05, y=0.98, horizontalalignment='left')
        ### Saving to file
        plt.subplots_adjust(top=0.8)
        plt.tight_layout()
        _file = os.path.join(profile.out(), 'plots', 'cumsum_etho_{}.png'.format(each_substr))
        print(_file)
        plt.savefig(_file, dpi=600)
        plt.cla()

    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
