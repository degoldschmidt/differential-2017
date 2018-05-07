from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.plot as plot
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
from scipy.stats import ranksums
import argparse


def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ses', action="store", dest="session", type=int)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-sf', action="store", dest="startfr", type=int)
    parser.add_argument('-ef', action="store", dest="endfr", type=int)
    OVERWRITE = parser.parse_args().force
    SESSION = parser.parse_args().session
    START = parser.parse_args().startfr
    END = parser.parse_args().endfr

    #thisscript = os.path.basename(__file__).split('.')[0]
    #experiment = 'DIFF'
    #profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    BASE = '/home/degoldschmidt/Downloads/data'
    OUT = os.path.join(BASE, 'plots')
    if not os.path.isdir(OUT):
        print('mkdir')
        os.mkdir(OUT)

    conds = {'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}
    headfiles = [os.path.join(BASE,_file) for _file in os.listdir(BASE) if 'Centroids' in _file and 'Head' in _file and 'CANS' in _file]
    bodyfiles = [os.path.join(BASE,_file) for _file in os.listdir(BASE) if 'Centroids' in _file and 'Body' in _file and 'CANS' in _file]
    print(headfiles, bodyfiles)
    hx = pd.read_csv(headfiles[0], sep=';')
    hy = pd.read_csv(headfiles[1], sep=';')
    bx = pd.read_csv(bodyfiles[0], sep=';')
    by = pd.read_csv(bodyfiles[1], sep=';')

    if START is None:
        START = 1
    if END is None:
        END = int(hx.columns[-1])

    for fly in range(START, END):
        print('fly: {}'.format(fly))
        outdf = pd.DataFrame({'head_x': hx[str(fly)], 'head_y': hy[str(fly)], 'body_x': bx[str(fly)], 'body_y': by[str(fly)], 'etho': np.zeros(len(hx.index))})
        f, ax = plt.subplots(figsize=(4,4))
        # trajectory plot
        ax = plot.trajectory(xc='head_x', yc='head_y', xs='body_x', ys='body_y', data=outdf, hue='etho', no_hue=[0], to_body=[], size=1, ax=ax)
        ax.plot(np.array(outdf['head_x'])[0], np.array(outdf['head_y'])[0], '#00ff12', marker='*', markersize=3)
        ax.plot(np.array(outdf['head_x'])[-1], np.array(outdf['head_y'])[-1], '#ff0000', marker='*', markersize=3)
        #ax.add_artist(plt.Rectangle((30, 30), 2, 2, color = conds[meta['condition']]))
        #ax.set_title('{}'.format(session.name), fontweight='bold', loc='left')
        r = 275.##30.5
        ax.set_xlim([-r,r])
        ax.set_ylim([-r,r])
        # output
        plt.tight_layout()
        _file = os.path.join(OUT, "vero_traj_{:03d}_{}_{}".format(fly, START, END))
        plt.savefig(_file+'.png', dpi=300)
        plt.cla()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
