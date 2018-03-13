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

def bootstrap_data(data, y, sub, conds, ssample, times):
    pvaldf = {'condition': [], 'pval': []}
    print(sub, conds, len(data.index))
    for t in range(times):
        if t%100 == 0:
            print(t)
        if ssample is not None:
            ssdata = pd.DataFrame(columns=data.columns)
            for condition in conds:
                M = len(data.query('condition == "{}"'.format(condition)).index)
                seeds = np.random.randint(0,high=10000000)
                #print(seeds)
                np.random.seed(seeds)
                random_choices = np.random.choice(M,ssample)
                #print(random_choices)
                samples = data.query('condition == "{}"'.format(condition)).iloc[random_choices]
                ssdata = ssdata.append(samples)

        for each in conds[1:]:
            X, Y = ssdata.query('condition == "{}"'.format(conds[0]))[y], ssdata.query('condition == "{}"'.format(each))[y]
            _, pval = ranksums(np.array(X),np.array(Y))
            if sub == 'yeast':
                if each == 'S' or each == 'O':
                    pvaldf['condition'].append(each)
                    pvaldf['pval'].append(pval)
            else:
                if each == 'AA' or each == 'O':
                    pvaldf['condition'].append(each)
                    pvaldf['pval'].append(pval)
    pvaldf = pd.DataFrame(pvaldf)
    print(pvaldf.head(10))
    return pvaldf


def plot_pval_distr(data_list, color_list):
    f, ax = plt.subplots(figsize=(4,2))
    for data, color in zip(data_list, color_list):
        ax.hist(data, bins=np.logspace(np.log10(0.00001),np.log10(10.0), 50), density=False, color=color, alpha=0.5)
    ymax = ax.get_ylim()[1]
    for data, color in zip(data_list, color_list):
        ax.vlines(np.median(data), 0, ymax, color=color, lw=1, linestyles='dashed')
        print(np.median(data))
    ax.set_ylim([0, ymax])
    ax.set_xscale("log")
    ax.set_xlabel('p-value (log-scale)')
    ax.set_ylabel('counts')

    sns.despine(ax=ax, trim=True)
    return ax

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('subsample_size', type=int)
    parser.add_argument('times', type=int)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-c', nargs='+', type=str)
    parser.add_argument('-suf', type=str)
    args = parser.parse_args()
    OVERWRITE = args.force
    N = args.subsample_size
    M = args.times
    _infile = args.input

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

    outfolder = os.path.join(profile.out(), 'plots')
    hook_file = os.path.join(outfolder, "{}.csv".format(_infile))
    try:
        df = pd.read_csv(hook_file, index_col='id')
    except FileNotFoundError:
        print('Compute data first')
        raise FileNotFoundError

    #### Plotting
    for j, sub in enumerate(['yeast', 'sucrose']):
        pval_file = os.path.join(outfolder, "pvals_{}.csv".format(sub))
        if os.path.isfile(pval_file) and not OVERWRITE:
            pvaldata = pd.read_csv(pval_file, index_col='id')
        else:
            rdata = reduce_data(df, sub, conds)
            pvaldata = bootstrap_data(df, 'ratio', sub, conds, N, M)
            pvaldata.to_csv(pval_file, index_label='id')
        if sub == 'yeast': fconds = ['S', 'O']
        else: fconds = ['AA', 'O']
        data_list = [pvaldata.query('condition == "{}"'.format(each))['pval'] for each in fconds]
        color_list = [colormap[each] for each in fconds]
        print(data_list)
        ax = plot_pval_distr(data_list, color_list)

        ### saving files
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}_{}".format(_infile, sub))
        _file += '_bootstrap'
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
