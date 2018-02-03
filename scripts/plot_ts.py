import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_arena(arena=None, spots=None, condition=None, ax=None):
    spot_colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
    cond_colors = {'SAA': '#b6d7a8', 'AA': '#A4C2F4', 'S': '#EA9999', 'O': '#CCCCCC'}
    if ax is None:
        ax = plt.gca()
    ### artists
    if arena is not None:
        ax.set_xlim([-1.1*arena.ro, 1.1*arena.ro])
        ax.set_ylim([-1.1*arena.ro, 1.1*arena.ro])
        arena_border = plt.Circle((0, 0), arena.rr, color='k', fill=False)
        ax.add_artist(arena_border)
        outer_arena_border = plt.Circle((0, 0), arena.ro, color='#aaaaaa', fill=False)
        ax.add_artist(outer_arena_border)
        ax.plot(0, 0, 'o', color='black', markersize=2)
    if spots is not None:
        for each_spot in spots:
            substr = each_spot.substrate
            spot = plt.Circle((each_spot.rx, each_spot.ry), each_spot.rr, color=spot_colors[substr], alpha=0.5)
            ax.add_artist(spot)
    if condition is not None:
        if condition in cond_colors.keys():
            spot = plt.Rectangle((-arena.ro, arena.ro-2), 5, 5, color=cond_colors[condition])
            ax.add_artist(spot)
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    return ax

def traj_plot(_data, _file, arena=None, time=None, title=None):
    f, ax = plt.subplots(1, figsize=(8,8), dpi=600)
    arena = data.arenas[fly]
    df = data.raw_data[fly]
    if time is None:
        start, end = df.index[0], df.index[-1]
    else:
        start, end = time
    ax = plot_arena(arena=arena, spots=arena.spots, condition=data.condition[fly], ax=ax)
    x = df.loc[start:end, 'body_x']
    y = df.loc[start:end, 'body_y']
    hx = df.loc[start:end, 'head_x']
    hy = df.loc[start:end, 'head_y']
    tx = df.loc[start:end, 'tail_x']
    ty = df.loc[start:end, 'tail_y']
    if only is None or only == 'body':
        ax.plot(x/scale, y/scale, c='k')
    if only is None or only == 'tail':
        ax.scatter(tx/scale, ty/scale, c='b', s=.25)
    if only is None or only == 'head':
        ax.scatter(hx/scale, hy/scale, c='r', s=.25)
    fly += 1
    plt.tight_layout()
    f.savefig(_file, dpi=600)

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
    profile = get_profile('DIFF', 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file

    ### GO THROUGH SESSIONS
    for each in db.sessions:
        print(each.name)
        df_raw, meta = each.load(VERBOSE=False)
        csv_file = os.path.join(profile.out(),  each.name+'_kinematics.csv')
        df = pd.read_csv(csv_file, index_col='frame')
        df['major'] = df_raw['major']
        df['minor'] = df_raw['minor']
        ts_plot(df, os.path.join(profile.out(),each.name+'_ts.pdf'))
        #traj_plot(df, os.path.join(profile.out(),each.name+'_traj.pdf'), arena=meta['arena'], )

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
