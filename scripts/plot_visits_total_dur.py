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

def plot_swarm(data, x, y, sub, conds, mypal):
    f, ax = plt.subplots(figsize=(3,2.5))
    rdata = data.query('substrate == "{}"'.format(sub)).dropna()
    querystr = ''
    astr = ' or '
    for condition in conds:
        querystr += 'condition == "{}"'.format(condition)
        querystr += astr
    rdata = rdata.query(querystr[:-len(astr)])
    # swarmbox
    if len(conds) > 2:
        ax = plot.swarmbox(x=x, y=y, data=rdata, order=conds, palette=mypal, compare=[(conds[0], conds[1:])])
    else:
        ax = plot.swarmbox(x=x, y=y, data=rdata, order=conds, palette=mypal, compare=[(conds[0], conds[1])])
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

    _outfile = 'visits_total_duration'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(hook_file, index_col='id')
    else:
        print('Compute data')
        for i_ses, each in enumerate(sessions):
            ### Loading data
            try:
                meta = each.load_meta()
                csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, _in))
                csv_file2 = os.path.join(in2folder, '{}_{}.csv'.format(each.name, _in2+'_visit'))
                ethodf = pd.read_csv(csv_file, index_col='frame')
                segmdf = pd.read_csv(csv_file2, index_col='segment')

                for j, sub in enumerate(['yeast', 'sucrose']):
                    only_visits = segmdf.query("state == {}".format(j+1))
                    outdf['session'].append(each.name)
                    outdf['condition'].append(meta['condition'])
                    outdf['substrate'].append(sub)
                    total = np.sum(only_visits['duration'])/60.
                    if np.isnan(total):
                        total = 0.
                    outdf['duration'].append(total)
                    print(outdf['session'][-1], outdf['condition'][-1], outdf['substrate'][-1], outdf['duration'][-1])
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        outdf = pd.DataFrame(outdf)
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    my_ylims = [60, 20]
    annos = [(58,1), (21,0.4)]
    my_yticks = [15, 5]
    for j, sub in enumerate(['yeast', 'sucrose']):
        ax = plot_swarm(outdf, 'condition', 'duration', sub, conds, mypal)
        annotations = [child for child in ax.get_children() if isinstance(child, plt.Text) and ("*" in child.get_text() or 'ns' in child.get_text())]
        for each in annotations:
            y = annos[j][0]
            if 'ns' in each.get_text():
                y += annos[j][1]
            each.set_position((each.get_position()[0], y))
        print(annotations)
        ### extra stuff
        ax.set_yticks(np.arange(0,my_ylims[j]+1,my_yticks[j]))
        ax.set_ylim([-0.1*my_ylims[j],1.1*my_ylims[j]])
        sns.despine(ax=ax, bottom=True, trim=True)
        ax.set_xlabel('pre-diet condition')
        ax.set_ylabel('Total duration of\n{} visits [min]'.format(sub))

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
