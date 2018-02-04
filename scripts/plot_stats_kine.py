import os, sys
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench
from pytrack_analysis.viz import swarmbox, set_font

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

import matplotlib.pyplot as plt
import seaborn as sns

def ts_plot(_data, _file):
    f, axes = plt.subplots(8, figsize=(16,8), sharex=True)

    _data['cumsum_dist'] = np.cumsum(_data['displacements'])
    y = ['cumsum_dist', 'head_speed', 'body_speed', 'angle', 'angular', 'dcenter', 'dpatch', 'or']
    ylabels = ['cumul.\ndist.\n[mm]', 'head\nspeed\n[mm/s]', 'body\nspeed\n[mm/s]', 'angle\n[deg]', 'turn\nrate\n[deg/s]', 'dist.\n[mm]', 'dist.\n[mm]', 'length\n[mm]']
    yeast, sucrose = 6*['#ffc04c'],6*['#4c8bff']
    ycolors = [['k'], ['k', 'b', 'r'], ['k', 'b', 'r'], ['#414141', '#8a8a8a'],['#4949fb'], ['#af61f4'], yeast+sucrose, ['g', 'm']]

    for i,ax in enumerate(axes):
        matches = [each for each in _data.columns if y[i] in each]
        for j, each in enumerate(matches):
            ydata = _data[each]
            if y[i] == 'dpatch':
                ax.plot(ydata, c=ycolors[i][j], label=each, lw=1, alpha=0.9)
            else:
                ax.plot(ydata, c=ycolors[i][j], label=each, lw=.5)
                if len(matches) > 1:
                    ax.legend(fontsize=6)
        ax.set_ylabel(ylabels[i], labelpad=20, rotation_mode='anchor', rotation=0, fontsize=8)
        ax.set_xlim([_data.index[0], _data.index[-1]])
        if y[i] == 'dpatch':
            ax.set_ylim([-0.1, 5.])
        if i == len(axes)-1:
            ax.set_xlabel('frame #')
    plt.tight_layout()
    plt.savefig(_file, dpi=600)
    plt.clf()
    plt.close()

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file
    csv_file = os.path.join(profile.out(),  experiment+'_kinematics_stats.csv')
    df = pd.read_csv(csv_file, index_col='session')
    df = df.dropna().query('mistracks < 30')
    print(df)
    print(df.shape)
    print(df.daytime.value_counts())

    ## PLOTTING
    pals = {    'condition': {"SAA": "#98c37e", "AA": "#5788e7", "S":"#D66667", "O": "#B7B7B7"},
                'daytime': {8: '#445cff', 11: '#ffe11c', 14: '#ff9203', 17: '#992c03'},
                'day': None,
                'position': {"topleft": '#5F3C2B', "topright": '#AA6B46', "bottomleft": '#D3884F', "bottomright": '#DEAE95'}
            }
    xlabels = {'condition': 'Holidic medium',
               'daytime': 'Daytime',
               'day': 'Day',
               'position': 'Arena position'}
    ylabels = {'head_speed': 'Mean head speed [mm/s]',
               'body_speed': 'Mean body speed [mm/s]',
               'abs_turn_rate': 'Mean absolute turning rate [º/s]',
               'distance': 'Distance travelled [mm]',
               'dcenter': 'Mean distance to center [mm]'}
    for cat in ['condition', 'daytime', 'day', 'position']:
        my_pal = pals[cat]
        _order = None
        _morder = None
        if cat == 'condition':
            _order = ['SAA', 'AA', 'S', 'O']
            _morder = [1,3,2,0]
        if cat == 'position':
            _morder = [2,3,0,1]

        for var in ['head_speed', 'body_speed', 'abs_turn_rate', 'distance', 'dcenter']:
            print('{}_{}.png'.format(var, cat))
            f, ax = plt.subplots(figsize=(6,4), dpi=600)
            if cat == 'day':
                ax.xaxis.set_tick_params(labelrotation=70)
            if cat == 'daytime':
                ax.xaxis.set_ticklabels(['8 - 10', '11 - 13', '14 - 16', '17 - 21'])
            ax = swarmbox(x=cat, y=var, order=_order, m_order=_morder, palette=my_pal, data=df, ax=ax)
            ax = set_font('Quicksand-Regular', ax=ax)
            ax.set_title(xlabels[cat], fontsize=10, loc='center', fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel(ylabels[var])
            sns.despine(ax=ax, bottom=True, trim=True)
            plt.tight_layout()
            plt.savefig(os.path.join(profile.out(), 'plots', '{}_{}.png'.format(cat, var)), dpi=600)
            plt.close()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
