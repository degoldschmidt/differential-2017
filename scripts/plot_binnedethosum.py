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

    conds = ["SAA", "S", "AA", "O"]
    if parser.parse_args().c is not None:
        conds = parser.parse_args().c
    colormap = {'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}
    mypal = {condition: colormap[condition]  for condition in conds}

    _in, _in2, _out = 'classifier', 'segments', 'plots'
    infolder = os.path.join(profile.out(), _in)
    in2folder = os.path.join(profile.out(), _in2)
    outfolder = os.path.join(profile.out(), _out)
    outdf = {'session': [], 'condition': [], 'substrate': [], 'number': []}

    _outfile = 'binnedetho'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    ethocolor = {4: "#ffc04c", 5: "#4c8bff", 'NA': '#e2e2e2'}
    totaldf = { 'session': [], 'condition': [], 'totalY': [], 'totalS': [], 'sortbyY': [] }

    dfs = {cond: None for cond in conds}
    counts = {cond: 0 for cond in conds}
    ### Plotting vlines for each session
    for i_ses, each in enumerate(sessions):
        ### Loading data
        try:
            meta = each.load_meta()
            csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, _in))
            ethodf = pd.read_csv(csv_file, index_col='frame')
            cond = meta['condition']
            if dfs[cond] is None:
                dfs[cond] = ethodf[['elapsed_time', 'etho', 'frame_dt']]
            else:
                dfs[cond] = pd.concat([dfs[cond], ethodf[['elapsed_time', 'etho', 'frame_dt']]])
                counts[cond] += 1
            print(each.name, cond,len(dfs[cond].index), dfs[cond].columns)
        except FileNotFoundError:
            pass #print(csv_file+ ' not found!')

    f, axes = plt.subplots(ncols=len(conds), figsize=(len(conds)*2.5,3.), sharey=True)



    axes[0].set_ylabel('Mean total duration\nsucrose micromovements [min]')
    f.suptitle("pre-diet condition:", x=0.1, y=0.98)
    for i, ax in enumerate(axes):
        print(i)
        window = 5.
        time = np.arange(0, 60.+.05, 1.)
        binw = 1.
        timebins = np.arange(0, 60.+.05, binw)
        for state in [5]:
            data = np.array(dfs[conds[i]].query('etho == {}'.format(state))['elapsed_time'])/60.
            timedata = np.array(dfs[conds[i]].query('etho == {}'.format(state))['frame_dt'])/60.
            print(len(data))
            meandata = np.zeros(time.shape)
            for ti,t in enumerate(time):
                dt = max(t+window/2, 0) - max(t-window/2, 0)
                mask = (data >= t-window) & (data < t)
                #mask = (data >= t) & (data < t+binw)
                if ti == 0 and i==0:
                    print(mask)
                meandata[ti] = np.sum(timedata[mask])/counts[conds[i]]
            print(time, meandata)
            ax.plot(time, meandata, alpha=0.5, color=ethocolor[state])
        #data = np.array(dfs[conds[i]].query('etho == 5')['elapsed_time'])/60.
        #ax.hist(data, bins=np.arange(0,60.5,0.5), alpha=0.5, histtype='stepfilled', color=ethocolor[5], edgecolor='none')
        ax.set_title("{} ($n$={})".format(conds[i], counts[conds[i]]))
        ax.set_xlabel('time [min]')
        #ax.set_xlabel()
        ax.set_xlim([0,60])
        #ax.set_yticks(np.arange(0,3,1))
        #ax.set_ylim([0.,3.])
        ax.set_yticks(np.arange(0,1,.25))
        ax.set_ylim([0.,1.])
        sns.despine(ax=ax, trim=True)
        if i > 0:
            ax.get_yaxis().set_visible(False)
            ax.spines['left'].set_visible(False)


    ### saving files
    plt.tight_layout(rect=[0, -0.03, 1, 0.95])
    _file = os.path.join(outfolder, "{}".format(_outfile))
    #plt.savefig(_file+'.pdf', dpi=300)
    plt.savefig(_file+'.png', dpi=300)
    plt.cla()

    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
