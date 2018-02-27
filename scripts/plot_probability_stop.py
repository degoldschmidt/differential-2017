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
import argparse

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    OVERWRITE = parser.parse_args().force

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
    for j, sub in enumerate(['yeast', 'sucrose']):
        f, ax = plt.subplots(figsize=(3,2.5))
        data = outdf.query('substrate == "{}"'.format(sub)).dropna()
        # swarmbox
        ax = swarmbox(x='condition', y='ratio', data=data, order=['SAA', 'AA', 'S', 'O'], palette={'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}, compare=[('SAA', ('AA', 'S', 'O'))])
        ax.set_yticks([0,0.5,1])
        ax.set_ylim([0,1.2])
        sns.despine(ax=ax, bottom=True, trim=True)
        ax.set_xlabel('pre-diet condition')
        #ax.set_ylabel('Probability of\nstopping at a\nyeast patch')
        ax.set_ylabel('Probability of\nstopping at a\nsucrose patch')
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}_{}.pdf".format(_outfile, sub))
        plt.savefig(_file, dpi=300)
        plt.cla()

    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
