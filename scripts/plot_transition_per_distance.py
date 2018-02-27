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
onlyAA = True

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
    outdf = {'session': [], 'condition': [], 'same_patch': [], 'near_patch': [], 'far_patch': [], 'total': []}
    _outfile = 'transition_per_distance'
    data_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(data_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(data_file, index_col='id')
    else:
        print('Compute data')
        for i_ses, each in enumerate(sessions):
            ### Loading data
            try:
                ### Loading data
                print(each.name)
                meta = each.load_meta()
                csv_file = [os.path.join(infolder[i_in], '{}_{}.csv'.format(each.name, each_in)) for i_in, each_in in enumerate(_in)]
                dfs = [pd.read_csv(_file, index_col='frame') for _file in csv_file]
                per_frame_df = pd.concat(dfs, axis=1)

                segment_cols = ['visit', 'visit_index']
                segmfolder = os.path.join(profile.out(), 'segments')
                csv_file = [os.path.join(segmfolder, '{}_{}_{}.csv'.format(each.name, 'segments', segm)) for segm in segment_cols]
                dfs = [pd.read_csv(_file, index_col='segment') for _file in csv_file]
                per_segment_df = dfs[0]
                per_segment_df['spot'] = dfs[1]['state']

                only_food_visits = per_segment_df.query('state > 0')
                states = np.array(only_food_visits['state'])
                spots = np.array(only_food_visits['spot'])
                pos = np.array(only_food_visits['position'])
                lens = np.array(only_food_visits['arraylen'])
                ### diff: 2->1 -1; 1->2 1;
                visit_transitions = np.diff(states) + 2*states[:-1] ##  3 (diff) 2 (yy) 4 (ss)

                #print(only_food_visits[['state', 'spot']])
                distances = np.arange(4,36,4)
                same, other = np.zeros(distances.shape, dtype=np.int32), np.zeros(distances.shape, dtype=np.int32)

                transitions = [np.where(visit_transitions == 2)[0], np.where(visit_transitions == 4)[0]]
                Ntrans = [len(transition) for transition in transitions]
                for i, sub in enumerate(['yeast', 'sucrose']):
                    for index in transitions:
                        print("{}->{} @ {}-{}".format(spots[index], spots[index+1], pos[index]+lens[index], pos[index+1]))
                        dist_to_pre = per_frame_df.iloc[pos[index]+lens[index]:pos[index+1]]['dpatch_{}'.format(spots[index])]
                        if np.any(dist_to_pre>16):
                            if spots[index]==spots[index+1]:
                                same += 1
                            else:
                                dist_to_pre = per_frame_df.iloc[pos[index]+lens[index]:pos[index+1]]['dpatch_{}'.format(spots[index])]

                if same+near+far != Ntrans: print('we have a problem')
                if Ntrans > 0:
                    same /= Ntrans
                    near /= Ntrans
                    far /= Ntrans
                #{'session': [], 'condition': [], 'same_patch': [], 'near_patch': [], 'far_patch': []}
                outdf['session'].append(each.name)
                outdf['condition'].append(meta['condition'])
                outdf['same_patch'].append(same)
                outdf['near_patch'].append(near)
                outdf['far_patch'].append(far)
                outdf['total'].append(Ntrans)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')

        outdf = pd.DataFrame(outdf)
        print("Saving to {}".format(data_file))
        outdf.to_csv(data_file, index_label='id')
    print(outdf)
    if onlyAA:
        outdf = outdf.query('condition == "SAA" or condition == "S"')
        outdf['condition'] = outdf['condition'].replace({'SAA':'+'})
        outdf['condition'] = outdf['condition'].replace({'S':'-'})
        newfile = os.path.join(outfolder, "{}_aa.csv".format(_outfile))
        outdf.to_csv(newfile, index_label='id')

    #### Plotting
    if onlyAA: width = 6
    else: width = 9
    f, axes = plt.subplots(1,3,figsize=(width,2.5))

    # swarmbox
    patchid = ['far_patch', 'near_patch', 'same_patch']
    patchlab = ['distant yeast', 'adjacent yeast', 'same yeast']
    for i, ax in enumerate(axes):
        if onlyAA: ax = swarmbox(ax=ax, x='condition', y=patchid[i], data=outdf, palette={'+': '#b353b5', '-': '#cc0000'}, compare=[('+', '-')])
        else: ax = swarmbox(ax=ax, x='condition', y=patchid[i], data=outdf, palette={'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}, compare=[('SAA', ('AA', 'S', 'O'))])

        ax.set_yticks([0,0.5,1])
        ax.set_ylim([-.05, 1.25])
        sns.despine(ax=ax, bottom=True, trim=True)

        xlabel= ''
        if onlyAA: xlabel= 'AA'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Transition\nprobability to\n{}'.format(patchlab[i]))

    plt.tight_layout()
    suffix = ''
    if onlyAA: suffix= '_aa'
    _file = os.path.join(outfolder, "{}.pdf".format(_outfile+suffix))
    plt.savefig(_file, dpi=300)
    plt.cla()
    plt.clf()

    # Data to plot
    for each in ['+', '-']:
        plt.figure(figsize=(2,2))
        sizes = [np.median(outdf.query('condition == "{}"'.format(each))[jj]) for jj in patchid]
        sizes /= np.sum(sizes)
        print(sizes)
        colors = ['#000000', '#0072b2', '#d55e00']
        explode = (0.0, 0.1, 0.1)  # explode 1st slice

        # Plot
        plt.pie(sizes, labels=['', '', ''], explode=explode, colors=colors, autopct='', shadow=False, startangle=90)
        plt.axis('equal')
        _file = os.path.join(outfolder, "pie_{}_{}.pdf".format(_outfile, each))
        plt.savefig(_file, dpi=300)
        plt.cla()
        plt.clf()
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
