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

OVERWRITE = True

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
    ### inputs
    _in = ['kinematics', 'classifier']
    infolder = [os.path.join(profile.out(), each) for each in _in]
    ### outputs
    _out = 'plots'
    outfolder = os.path.join(profile.out(), _out)
    outdf = {'session': [], 'condition': [], 'same_patch': [], 'near_patch': [], 'far_patch': []}
    _outfile = 'probability_stop'
    data_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(data_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(data_file, index_col='id')
    else:
        print('Compute data')
        for i_ses, each in enumerate(sessions[:1]):
            ### Loading data
            try:
                ### Loading data
                meta = each.load_meta()
                csv_file = [os.path.join(infolder[i_in], '{}_{}.csv'.format(each.name, each_in)) for i_in, each_in in enumerate(_in)]
                dfs = [pd.read_csv(_file, index_col='frame') for _file in csv_file]
                per_frame_df = pd.concat(dfs, axis=1)

                segment_cols = ['visit', 'visit_index']
                segmfolder = os.path.join(profile.out(), 'segments')
                csv_file = [os.path.join(segmfolder, '{}_{}_{}.csv'.format(each.name, 'segments', segm)) for segm in segment_cols]
                dfs = [pd.read_csv(_file, index_col='segment') for _file in csv_file]
                per_segment_df = pd.concat(dfs, axis=1)

                print(per_frame_df.head(5))
                print(per_segment_df.head(5))
                """
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
                """
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        """
        outdf = pd.DataFrame(outdf)
        outdf.to_csv(data_file, index_label='id')
    print(outdf)
        """

    """
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
    """
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
