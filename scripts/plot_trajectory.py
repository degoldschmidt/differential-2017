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
    parser.add_argument('-ses', action="store", dest="session", type=int)
    parser.add_argument('--force', action='store_true')
    OVERWRITE = parser.parse_args().force
    SESSION = parser.parse_args().session

    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())

    conds = ["SAA", "AA", "S", "O"]
    in_suffix  =  ['kinematics', 'classifier']
    out_suffix =  'plots'
    infolder = [os.path.join(profile.out(), suf) for suf in in_suffix]
    outfolder = os.path.join(profile.out(), out_suffix)
    outdf = {'session': [], 'condition': [], 'ratio': []}
    _outfile = 'trajectory'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(hook_file, index_col='id')
    else:
        print('Compute data for session {}_{:03}'.format(experiment, SESSION))
        ### Loading data
        try:
            session = db.sessions[SESSION]
            meta = session.load_meta()
            csv_file = [os.path.join(infolder[j], '{}_{}.csv'.format(session.name, suf)) for j, suf in enumerate(in_suffix)]
            dfs = [pd.read_csv(each_file, index_col='frame') for each_file in csv_file]
        except FileNotFoundError:
            print(csv_file[0] + ' not found!')
    """
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
    """
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
