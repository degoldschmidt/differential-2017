from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
from pytrack_analysis.viz import set_font, swarmbox
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import ranksums

OVERWRITE = False

def main():
    """
    --- general parameters
     *
    """
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())
    sessions = db.sessions
    n_ses = len(sessions)

    conds = ["SAA", "AA", "S", "O"]
    EthoTotals = {each_condition: {} for each_condition in conds}
    _in, _in2, _out = 'classifier', 'segments', 'plots'
    infolder = os.path.join(profile.out(), _in)
    in2folder = os.path.join(profile.out(), _in2)
    outfolder = os.path.join(profile.out(), _out)
    outdf = {'session': [], 'condition': [], 'ratio': []}
    _outfile = 'probability_stop_aa'
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
                only_yeast_encounters = segmdf.query("state == 1")
                counter = 0
                for index, row in only_yeast_encounters.iterrows():
                    pos = int(row['position'])
                    end = int(row['position']+row['arraylen'])
                    ethovec = np.array(ethodf['etho'])[pos:end]
                    has_yeast_micromov = np.any(ethovec == 4)
                    #print("Segment {:3d} at position {:6d} (len: {:4d}) has yeast micromovements: {}".format(int(index), pos, end-pos, has_yeast_micromov))
                    if has_yeast_micromov:
                        counter += 1
                ratio = counter/len(only_yeast_encounters.index)
                #print(ratio)
                outdf['session'].append(each.name)
                outdf['condition'].append(meta['condition'])
                outdf['ratio'].append(ratio)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        outdf = pd.DataFrame(outdf)
        outdf = outdf.query('condition == "SAA" or condition == "S"')
        outdf['condition'] = outdf['condition'].replace({'SAA':'+'})
        outdf['condition'] = outdf['condition'].replace({'S':'-'})
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    f, ax = plt.subplots(figsize=(2,2.5))

    # swarmbox
    ax = swarmbox(x='condition', y='ratio', data=outdf, palette={'+': '#b353b5', '-': '#cc0000'}, compare=[('+', '-')])

    #ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([0,0.5,1])
    sns.despine(ax=ax, bottom=True, trim=True)

    ax.set_xlabel('AA')
    ax.set_ylabel('Probability of\nstopping at a\nyeast patch')

    X = np.array(outdf.query('condition == "+"')['ratio'])
    Y = np.array(outdf.query('condition == "-"')['ratio'])

    stat, pval = ranksums(X, Y)
    print("Wilcoxon rank-sum test: statistic = {:1.4f}, p-value = {:1.6f}".format(stat, pval))

    plt.tight_layout()
    _file = os.path.join(outfolder, "{}.pdf".format(_outfile))
    plt.savefig(_file, dpi=300)
    plt.cla()

    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
