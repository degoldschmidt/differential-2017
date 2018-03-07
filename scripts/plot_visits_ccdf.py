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

def plot_ccdf(data, x, y, sub, conds, mypal):
    f, ax = plt.subplots(figsize=(3.5,2.5))
    rdata = data.query('substrate == "{}"'.format(sub)).dropna()
    querystr = ''
    astr = ' or '
    for condition in conds:
        querystr += 'condition == "{}"'.format(condition)
        querystr += astr
    rdata = rdata.query(querystr[:-len(astr)])
    # swarmbox
    if len(conds) > 2:
        ax = plot.ccdf(rdata, xy=y, c=x, palette=mypal, ax=ax)
    else:
        ax = plot.ccdf(rdata, xy=y, c=x, palette=mypal, ax=ax)
    return ax

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
    outdf = {'session': [], 'condition': [], 'substrate': [], 'duration': []}

    _outfile = 'visits_ccdf'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(hook_file, index_col='id')
    else:
        print('Compute data')
        allsegments = []
        for i_ses, each in enumerate(sessions):
            ### Loading data
            try:
                meta = each.load_meta()
                print(each.name)
                csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, _in))
                csv_file2 = os.path.join(in2folder, '{}_{}.csv'.format(each.name, _in2+'_visit'))
                ethodf = pd.read_csv(csv_file, index_col='frame')
                segmdf = pd.read_csv(csv_file2, index_col='segment')
                for j, sub in enumerate(['yeast', 'sucrose']):
                    only_visits = segmdf.query("state == {}".format(j+1))
                    only_visits['session'] = each.name
                    only_visits['condition'] = meta['condition']
                    only_visits['substrate'] = sub
                    allsegments.append(only_visits)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        outdf = pd.concat(allsegments)
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    my_ylims = [12, 3]
    annos = [(13.,0.2), (2.5,0.05)]
    my_yticks = [2, 1]
    for j, sub in enumerate(['yeast', 'sucrose']):
        ax = plot_ccdf(outdf, 'condition', 'duration', sub, conds, mypal)
        ### extra stuff
        #ax.set_yticks(np.arange(0,my_ylims[j]+1,my_yticks[j]))
        ax.set_ylim([0.0001,1.5])
        sns.despine(ax=ax, trim=False)
        ax.set_xlabel('Duration $x$ [s]')
        ax.set_ylabel('p($T_{' +sub+ '}$ $\geq$ $x$)')

        ### saving files
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}_{}".format(_outfile, sub))
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
