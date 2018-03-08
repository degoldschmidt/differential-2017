from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.plot as plot
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import ranksums
import argparse
import time


def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-c', nargs='+', type=str)
    OVERWRITE = parser.parse_args().force

    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())
    sessions = db.sessions
    n_ses = len(sessions)

    conds = ["SAA", "AA", "S", "O"]
    if parser.parse_args().c is not None:
        conds = parser.parse_args().c
    colormap = {'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}
    mypal = {condition: colormap[condition]  for condition in conds}

    in_suffix  =  'kinematics'
    out_suffix =  'plots'
    infolder = os.path.join(profile.out(), in_suffix)
    outfolder = os.path.join(profile.out(), out_suffix)
    outdf = {}
    _outfile = 'occupancy'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    x, y = {}, {}
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        allthem = pd.read_csv(hook_file, index_col='id')
        print(allthem.columns)
        for each in conds:
            print(each)
            x[each] = np.array(allthem[each+'_x'].dropna())
            y[each] = np.array(allthem[each+'_y'].dropna())
    else:
        print('Compute data')
        for i_ses, each in enumerate(sessions):
            try:
                print(each.name, time.clock())
                meta = each.load_meta()
                csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, in_suffix))
                kinedf = pd.read_csv(csv_file, index_col='frame')
                cond = meta['condition']
                if cond not in x.keys():
                    x[cond] = np.array(kinedf['head_x'].dropna())
                    y[cond] = np.array(kinedf['head_y'].dropna())
                else:
                    x[cond] = np.append(x[cond], kinedf['head_x'].dropna())
                    y[cond] = np.append(y[cond], kinedf['head_y'].dropna())
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')

        for each in conds:
            print("{}: dims({}, {})".format(each, x[each].shape, y[each].shape))
            outdf[each+'_x'] = pd.Series(x[each])
            outdf[each+'_y'] = pd.Series(y[each])
        outdf = pd.DataFrame(outdf)
        outdf.to_csv(hook_file, index_label='id')

    #### Plotting
    for jj in range(2):
        f, axes = plt.subplots(ncols=4,figsize=(12,3))
        # arena plot
        #for i, row in enumerate(axes):
        for j, ax in enumerate(axes):
            cond = conds[j]

            xmin = x[cond].min()
            xmax = x[cond].max()
            ymin = y[cond].min()
            ymax = y[cond].max()
            #ax = plot.arena(meta['arena'], meta['food_spots'], ax=ax)
            # hexbin plot
            if jj == 0:
                ax.hexbin(x[cond], y[cond], gridsize=(250,250), bins='log', cmap=plt.cm.inferno)
            else:
                ax.hexbin(x[cond], y[cond], gridsize=(250,250), cmap=plt.cm.inferno)
            sns.despine(ax=ax, trim=True)
            ax.axis([xmin, xmax, ymin, ymax])
            ax.set_title(cond, loc='left', fontweight='bold')
            ax.set_ylabel('y [mm]')
            ax.set_xlabel('x [mm]')
        # output
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}".format(_outfile))
        #plt.savefig(_file+'.pdf', dpi=300)
        #plt.savefig(_file+'.svg', dpi=300)
        if jj == 0:
            plt.savefig(_file+'_log.png', dpi=300)
        else:
            plt.savefig(_file+'.png', dpi=300)
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
