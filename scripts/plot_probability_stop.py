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

def reduce_data(data, sub, conds):
    rdata = data.query('substrate == "{}"'.format(sub)).dropna()
    querystr = ''
    astr = ' or '
    for condition in conds:
        querystr += 'condition == "{}"'.format(condition)
        querystr += astr
    rdata = rdata.query(querystr[:-len(astr)])
    return rdata

def plot_swarm(data, x, y, sub, conds, mypal):
    f, ax = plt.subplots(figsize=(0.5*len(conds)+1,2.5))
    rdata = reduce_data(data, sub, conds)
    # swarmbox
    if len(conds) > 2:
        ax = plot.swarmbox(x=x, y=y, data=rdata, order=conds, palette=mypal, compare=[(conds[0], conds[1:])], boxonly=True)
    else:
        ax = plot.swarmbox(x=x, y=y, data=rdata, order=conds, palette=mypal, compare=[(conds[0], conds[1])], boxonly=True)
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
    parser.add_argument('-suf', type=str)
    OVERWRITE = parser.parse_args().force

    #thisscript = os.path.basename(__file__).split('.')[0]
    #experiment = 'DIFF'
    #profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    OUT = '/home/degoldschmidt/post_tracking'
    DB = '/home/degoldschmidt/post_tracking/DIFF.yaml'
    db = Experiment(DB)
    sessions = db.sessions
    n_ses = len(sessions)

    conds = ["SAA", "S", "AA", "O"]
    if parser.parse_args().c is not None:
        conds = parser.parse_args().c
    #colormap = {'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}
    colormap = {'SAA': "#b1b1b1", 'AA': "#5788e7", 'S': "#424242", 'O': "#B7B7B7"}
    mypal = {condition: colormap[condition]  for condition in conds}
    EthoTotals = {each_condition: {} for each_condition in conds}
    _in, _in2, _out = 'classifier', 'segments', 'plots'
    infolder = os.path.join(OUT, _in)
    in2folder = os.path.join(OUT, _in2)
    outfolder = os.path.join(OUT, _out)
    outdf = {'session': [], 'condition': [], 'substrate': [], 'ratio': []}
    _outfile = 'probability_stop'
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
                csv_file2 = os.path.join(in2folder, '{}_{}.csv'.format(each.name, _in2+'_encounter'))
                ethodf = pd.read_csv(csv_file, index_col='frame')
                segmdf = pd.read_csv(csv_file2, index_col='segment')

                for j, sub in enumerate(['yeast', 'sucrose']):
                    only_encounters = segmdf.query("state == {}".format(j+1))
                    counter = 0
                    for index, row in only_encounters.iterrows():
                        pos = int(row['position'])
                        end = int(row['position']+row['arraylen'])
                        ethovec = np.array(ethodf['etho'])[pos:end]
                        has_micromovs = np.any(ethovec == j+4)
                        if has_micromovs:
                            counter += 1
                    if len(only_encounters.index) > 0:
                        ratio = counter/len(only_encounters.index)
                    else:
                        ratio = np.nan
                    #print(ratio)
                    outdf['session'].append(each.name)
                    outdf['condition'].append(meta['condition'])
                    outdf['substrate'].append(sub)
                    outdf['ratio'].append(ratio)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        outdf = pd.DataFrame(outdf)
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    ymax = [15, 20]#[50, 5] ### Vero: 50, 10
    yt = [5, 5]
    annos = [(.90,.1), (18.5,0.1)] ### 55 -> 48 sucrose deprived
    for j, sub in enumerate(['yeast', 'sucrose']):
        ax = plot_swarm(outdf, 'condition', 'ratio', sub, conds, mypal)
        annotations = [child for child in ax.get_children() if isinstance(child, plt.Text) and ("*" in child.get_text())]
        for each in annotations:
            y = annos[j][0]
            each.set_position((each.get_position()[0], y))


        if sub == 'yeast':
            l_c = 'S'
        else:
            l_c = 'AA'
        stat, pval = ranksums(outdf.query('condition == "{}" and substrate == "{}"'.format(l_c, sub))['ratio'], outdf.query('condition == "O" and substrate == "{}"'.format(sub))['ratio'])
        print(l_c, sub, pval)
        if len(conds) > 2:
            if sub == 'yeast':
                ax,_ = plot.annotate(1,3,pval,[0.75],[0.75], stars=True, ax=ax, align='center', _h=0.0, _ht=0.02)
            else:
                ax,_ = plot.annotate(2,3,pval,[0.75],[0.75], stars=True, ax=ax, align='center', _h=0.0, _ht=0.05)

        ### extra stuff
        ax.set_yticks([0,0.5,1])
        ax.set_ylim([-0.05,1.05])
        sns.despine(ax=ax, bottom=True, trim=True)
        ax.set_xticklabels(['+', '-'])
        ax.set_xlabel('Amino acids')
        ax.set_ylabel('Probability of\nstopping at a\n{} patch'.format(sub))

        ### saving files
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}_{}".format(_outfile, sub))
        if parser.parse_args().suf is not None:
            _file += '_'+parser.parse_args().suf
        #plt.savefig(_file+'.pdf', dpi=300)
        plt.savefig(_file+'.png', dpi=300)
        plt.cla()

    ### delete objects
    #del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
