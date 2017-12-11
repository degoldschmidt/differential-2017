import os
from pytrack_analysis.profile import *
import numpy as np
import pandas as pd

### plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn
import seaborn as sns; sns.set(color_codes=True)
sns.set_style('ticks')

def swarmbox(x=None, y=None, hue=None, data=None, order=None, hue_order=None, multi=False,
                dodge=False, orient=None, color=None, palette=None, table=False,
                size=5, edgecolor="gray", linewidth=0, colors=None, ax=None, **kwargs):
    # default parameters
    defs = {
                'ps':   2,          # pointsize for swarmplot (3)
                'pc':   '#666666',  # pointcolor for swarmplot
                'w':    .5,         # boxwidth for boxplot (0.35)
                'lw':   0.0,        # linewidth for boxplot
                'sat':  1.,         # saturation for boxplot
                'mlw':  0.3,        # width for median lines
    }

    # axis dimensions
    #ax.set_ylim([-2.,max_dur + 2.]) # this is needed for swarmplot to work!!!

    # actual plotting using seaborn functions
    # boxplot
    my_pal = {"SAA": "#98c37e", "AA": "#5788e7", "S":"#D66667", "O": "#B7B7B7"}
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        orient=orient, color=color, palette=my_pal, saturation=defs['sat'],
                        width=defs['w'], linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), showfliers=False, **kwargs)
    #ax = sns.boxplot(x=x, y=y, hue=hue, data=data, palette=my_pal, showfliers=False, boxprops=dict(lw=1))
    # swarmplot
    ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, dodge=True,
                     orient=orient, color=defs['pc'], palette=palette, size=defs['ps'], ax=ax, **kwargs)
    # median lines
    medians = data.groupby(x)[y].median()
    dx = defs['mlw']
    new = [1, 3, 2, 0]
    for pos, median in enumerate(medians):
        ax.hlines(median, new[pos]-dx, new[pos]+dx, lw=1.5, zorder=10)

    ## figure aesthetics
    #ax.set_yticks(np.arange(0, max_dur+1, div))
    sns.despine(ax=ax, bottom=True, trim=True)
    #ax.get_xaxis().set_visible(False)
    ax.tick_params('x', length=0, width=0, which='major')

    # Adjust layout to make room for the table:
    #plt.subplots_adjust(top=0.9, bottom=0.05*nrows, hspace=0.15*nrows, wspace=1.)
    return ax


thisscript = os.path.basename(__file__).split('.')[0]
profile = get_profile('Differential 2017', 'degoldschmidt', script=thisscript)
outdir = get_out(profile)
data = pd.read_csv(os.path.join(outdir, 'etho_lens.csv'), sep="\t")

data['yeast'] *= 0.0333
data['sucrose'] *= 0.0333

import scipy.stats as sts

Y_SAA = data.query("metabolic == 'SAA'")['yeast']
Y_AA = data.query("metabolic == 'AA'")['yeast']
Y_S = data.query("metabolic == 'S'")['yeast']
Y_O = data.query("metabolic == 'O'")['yeast']

S_SAA = data.query("metabolic == 'SAA'")['sucrose']
S_AA = data.query("metabolic == 'AA'")['sucrose']
S_S = data.query("metabolic == 'S'")['sucrose']
S_O = data.query("metabolic == 'O'")['sucrose']

print('SAA, S', sts.ranksums(Y_SAA, Y_S)[1])
print('SAA, O', sts.ranksums(Y_SAA, Y_O)[1])
print('AA, S', sts.ranksums(Y_AA, Y_S)[1])
print('AA, O', sts.ranksums(Y_AA, Y_O)[1])
print('S, O', sts.ranksums(Y_S, Y_O)[1])
print('SAA, AA', sts.ranksums(Y_SAA, Y_AA)[1])

print('SAA, AA', sts.ranksums(S_SAA, S_AA)[1])
print('SAA, S', sts.ranksums(S_SAA, S_S)[1])
print('SAA, 0', sts.ranksums(S_SAA, S_O)[1])
print('AA, O', sts.ranksums(S_AA, S_O)[1])
print('AA, S', sts.ranksums(S_AA, S_S)[1])
print('S, O', sts.ranksums(S_S, S_O)[1])



lims = [[-50,1000], [-20,600]]
for i, each in enumerate(['yeast', 'sucrose']):
    f, ax = plt.subplots(figsize=(6,4))
    ax = swarmbox(x="metabolic", y=each, data=data, ax=ax)
    ax.set_ylim(lims[i])
    sns.despine(ax=ax, bottom=True, trim=True)
    ax.set_xlabel("Holidic medium")
    ax.set_ylabel("Total "+each+" micromovements [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, each+'.png'), dpi=600)
