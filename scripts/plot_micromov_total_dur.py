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

def plot_swarm(data, x, y, sub, conds, mypal):
    f, ax = plt.subplots(figsize=(1+2*len(conds)/4,2.5))
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

    if sub == 'yeast':
        stat, pval = ranksums(rdata.query('condition == "S"')['total_duration'], rdata.query('condition == "O"')['total_duration'])

        print(pval, 'S')
    else:
        stat, pval = ranksums(rdata.query('condition == "AA"')['total_duration'], rdata.query('condition == "O"')['total_duration'])
        print(pval, 'AA')

    return ax, pval

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-c', nargs='+', type=str)
    parser.add_argument('-suf', type=str)
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
    outdf = {'session': [], 'condition': [], 'substrate': [], 'total_duration': []}
    _outfile = 'total_micromovements'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(hook_file, index_col='id')
    else:
        print('Compute data')
        for i_ses, each in enumerate(sessions):
            ### Loading data
            try:
                print(each.name)
                meta = each.load_meta()
                csv_file = os.path.join(infolder, '{}_{}.csv'.format(each.name, _in))
                csv_file2 = os.path.join(in2folder, '{}_{}.csv'.format(each.name, _in2+'_etho'))
                ethodf = pd.read_csv(csv_file, index_col='frame')
                segmdf = pd.read_csv(csv_file2, index_col='segment')

                for j, sub in enumerate(['yeast', 'sucrose']):
                    only_mm = segmdf.query("state == {}".format(j+4))
                    totaldur = np.sum(only_mm['duration'])
                    if np.isnan(totaldur):
                        totaldur = 0.
                    print(sub,':',totaldur)
                    outdf['session'].append(each.name)
                    outdf['condition'].append(meta['condition'])
                    outdf['substrate'].append(sub)
                    outdf['total_duration'].append(totaldur/60.)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        outdf = pd.DataFrame(outdf)
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    f, ax = plt.subplots(figsize=(3,2.5))

    # swarmbox
    ymax = [60, 20]#[50, 5] ### Vero: 50, 10
    yt = [15, 5]
    annos = [(55,.8), (18.5,0.1)] ### 55 -> 48 sucrose deprived
    for j, sub in enumerate(['yeast', 'sucrose']):
        ax, pval = plot_swarm(outdf, 'condition', 'total_duration', sub, conds, mypal)
        ### moving text
        annotations = [child for child in ax.get_children() if isinstance(child, plt.Text) and ("*" in child.get_text() or 'ns' in child.get_text())]
        for each in annotations:
            y = annos[j][0]
            if 'ns' in each.get_text():
                y += annos[j][1]
            each.set_position((each.get_position()[0], y))

        ###
        #X = outdf.query('condition == "SAA" and total_duration < 4').dropna()['total_duration']
        #Y = outdf.query('condition == "S" and total_duration < 4').dropna()['total_duration']
        if sub == 'yeast':
            ax,_ = plot.annotate(1,3,pval,[40],[40], stars=True, ax=ax, align='center', _h=0.05, _ht=1.2)
        else:
            ax,_ = plot.annotate(2,3,pval,[12],[12], stars=True, ax=ax, align='center', _h=0.05, _ht=0.1)

        ### extra stuff
        ax.set_ylim([-0.05*ymax[j],ymax[j]])
        ax.set_yticks(np.arange(0, ymax[j]+1, yt[j]))
        sns.despine(ax=ax, bottom=True, trim=True)
        ax.set_xlabel('pre-diet condition')
        ax.set_ylabel('Total duration\nof {}\nmicromovements [min]'.format(sub))

        ### saving files
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}_{}".format(_outfile, sub))
        if parser.parse_args().suf is not None:
            _file += '_'+parser.parse_args().suf
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
