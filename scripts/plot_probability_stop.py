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
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    f, ax = plt.subplots(figsize=(3,2.5))

    # swarmbox
    ax = swarmbox(x='condition', y='ratio', data=outdf, palette={'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}, compare=[('SAA', ('AA', 'S', 'O'))])

    ax.set_yticks([0,0.5,1])
    sns.despine(ax=ax, bottom=True, trim=True)

    ax.set_xlabel('pre-diet condition')
    ax.set_ylabel('Probability of\nstopping at a\nyeast patch')

    plt.tight_layout()
    _file = os.path.join(outfolder, "{}.pdf".format(_outfile))
    plt.savefig(_file, dpi=300)
    plt.cla()

    ### delete objects
    del profile

    #         df['Ydur'], df['Sdur'] = df['frame_dt'], df['frame_dt']
    #         df.loc[df['etho'] != 4, 'Ydur'] = 0
    #         df.loc[df['etho'] != 5, 'Sdur'] = 0
    #         EthoTotals['Yeast'][meta['condition']][each.name] = np.cumsum(np.array(df['Ydur']))[:108000]
    #         EthoTotals['Sucrose'][meta['condition']][each.name] = np.cumsum(np.array(df['Sdur']))[:108000]
    #         print(each.name)
    #     except FileNotFoundError:
    #         pass
    #
    # ### Plotting
    # colors = ["#98c37e", "#5788e7", "#D66667", "#2b2b2b"]
    # maxy = {'Yeast': 50, 'Sucrose': 25}
    # tiky = {'Yeast': 10, 'Sucrose': 5}
    # for each_substr in ['Yeast', 'Sucrose']:
    #     f, axes = plt.subplots(1, 4, figsize=(8,3), dpi=400, sharey=True)
    #     for i, each_cond in enumerate(conds):
    #         _reduce=False
    #         if i>0:
    #             _reduce=True
    #         df = pd.DataFrame(EthoTotals[each_substr][each_cond])
    #         axes[i] = plot_cumulatives(df, color=colors[i], ax=axes[i], title=each_cond, tiky=tiky[each_substr], maxy=maxy[each_substr], reduce=_reduce)
    #         f.suptitle('{}'.format(each_substr), fontsize=10, fontweight='bold', x=0.05, y=0.98, horizontalalignment='left')
    #     ### Saving to file
    #     plt.subplots_adjust(top=0.8)
    #     plt.tight_layout()
    #     _file = os.path.join(profile.out(), 'plots', 'cumsum_etho_{}.png'.format(each_substr))
    #     print(_file)
    #     plt.savefig(_file, dpi=600)
    #     plt.cla()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
