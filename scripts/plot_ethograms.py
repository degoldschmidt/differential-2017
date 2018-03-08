from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.plot as plot
from pytrack_analysis import Multibench
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import ranksums
import argparse
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

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
    EthoTotals = {each_condition: {} for each_condition in conds}
    _in, _in2, _out = 'classifier', 'segments', 'plots'
    infolder = os.path.join(profile.out(), _in)
    in2folder = os.path.join(profile.out(), _in2)
    outfolder = os.path.join(profile.out(), _out)
    outdf = {'session': [], 'condition': [], 'substrate': [], 'number': []}

    _outfile = 'sort_ethogram'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    f, axes = plt.subplots(ncols=len(conds), figsize=(len(conds)*3,2.5), sharey=True)
    ethocolor = {4: "#ffc04c", 5: "#4c8bff", 'NA': '#e2e2e2'}
    totaldf = { 'session': [], 'condition': [], 'totalY': [], 'totalS': [], 'sortbyY': [] }
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        allthem = pd.read_csv(hook_file, index_col='id')
    else:
        print('Compute data')
        for i_ses, each in enumerate(sessions):
            try:
                meta = each.load_meta()
                csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, _in))
                csv_file2 = os.path.join(in2folder, '{}_{}.csv'.format(each.name, _in2+'_etho'))
                ethodf = pd.read_csv(csv_file, index_col='frame')
                segmdf = pd.read_csv(csv_file2, index_col='segment')
                only_Ymm = segmdf.query("state == 4")
                only_Smm = segmdf.query("state == 5")
                totalY_mm = np.sum(only_Ymm['duration'])/60.
                totalS_mm = np.sum(only_Smm['duration'])/60.
                if np.isnan(totalY_mm): totalY_mm = 0.
                if np.isnan(totalS_mm): totalS_mm = 0.
                print(each.name, totalY_mm, totalS_mm)
                totaldf['session'].append(each.name)
                totaldf['condition'].append(meta['condition'])
                totaldf['totalY'].append(totalY_mm)
                totaldf['totalS'].append(totalS_mm)
                totaldf['sortbyY'].append(totalY_mm > totalS_mm)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        totaldf = pd.DataFrame(totaldf)
        sortY = totaldf.query('sortbyY == True').sort_values(by=['totalY'])
        sortS = totaldf.query('sortbyY == False').sort_values(by=['totalS'], ascending=False)
        allthem = pd.concat([sortS, sortY])
        allthem.to_csv(hook_file, index_label='id')
    ### sorting into condition
    sort_dict = {   'SAA': allthem.query('condition == "SAA"').reset_index(drop=True),
                    'AA': allthem.query('condition == "AA"').reset_index(drop=True),
                    'S': allthem.query('condition == "S"').reset_index(drop=True),
                    'O': allthem.query('condition == "O"').reset_index(drop=True),}
    print(sort_dict['SAA'])
    print(sort_dict['AA'])
    print(sort_dict['S'])
    print(sort_dict['O'])

    ### Plotting vlines for each session
    for i_ses, each in enumerate(sessions):
        ### Loading data
        try:
            meta = each.load_meta()
            csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, _in))
            ethodf = pd.read_csv(csv_file, index_col='frame')
            cond = meta['condition']
            panel = conds.index(cond)
            pos = sort_dict[cond].query('session == "{}"'.format(each.name)).index[0]
            print(each.name, panel, cond, pos)
            focus = [5,4]
            axes[panel].vlines(np.array(ethodf['elapsed_time'])[::10]/60.,pos,pos+1, color=ethocolor['NA'], lw=0.1)
            for i in focus:
                axes[panel].vlines(np.array(ethodf.query('etho == {}'.format(i))['elapsed_time'])/60.,pos,pos+1, color=ethocolor[i], lw=0.1)
        except FileNotFoundError:
            pass #print(csv_file+ ' not found!')

    axes[0].set_ylabel('flies')
    for i, ax in enumerate(axes):
        print(i)
        ax.set_xlabel('time [min]')
        ax.set_xlim([0,60])
        ax.set_yticks(np.arange(0,61,10))
        sns.despine(ax=ax, trim=True)
        if i > 0:
            ax.get_yaxis().set_visible(False)
            ax.spines['left'].set_visible(False)


    ### saving files
    plt.tight_layout()
    _file = os.path.join(outfolder, "{}".format(_outfile))
    plt.savefig(_file+'.pdf', dpi=300)
    plt.savefig(_file+'.png', dpi=300)
    plt.cla()

    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
