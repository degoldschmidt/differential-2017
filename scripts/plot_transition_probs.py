from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
from pytrack_analysis.plot import set_font, swarmbox
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import ranksums
import argparse

OVERWRITE = False
onlyAA = False

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-c', nargs='+', type=str)
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
    substrates = ['yeast', 'sucrose']
    ### inputs
    _in = ['kinematics', 'classifier']
    infolder = [os.path.join(profile.out(), each) for each in _in]
    ### outputs
    _out = 'plots'
    outfolder = os.path.join(profile.out(), _out)
    outdf = {   'session': [], 'condition': [], 'substrate': [], 'same_patch': [], 'near_patch': [], 'far_patch': [], 'total': []}
    _outfile = 'transition_probability'
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
                visit_transitions = np.diff(states) + 2*states[:-1]

                #print(only_food_visits[['state', 'spot']])
                for j, substr in enumerate(substrates):
                    transitions = np.where(visit_transitions == 2*(j+1))[0]
                    print(2*(j+1), len(transitions))
                    Ntrans = len(transitions)
                    same, near, far = 0, 0, 0
                    for index in transitions:
                        if spots[index]==spots[index+1]:
                            dist_to_pre = per_frame_df.iloc[pos[index]+lens[index]:pos[index+1]]['dpatch_{}'.format(spots[index])]
                            if np.any(dist_to_pre>16):
                                far += 1
                            else:
                                same += 1
                        else:
                            dist_to_pre = per_frame_df.iloc[pos[index]+lens[index]:pos[index+1]]['dpatch_{}'.format(spots[index])]
                            if np.any(dist_to_pre>16):
                                far += 1
                            else:
                                near += 1
                    print(substr, ':', same, near, far)
                    if same+near+far != Ntrans: print('we have a problem')
                    if Ntrans > 0:
                        same /= Ntrans
                        near /= Ntrans
                        far /= Ntrans
                    #{'session': [], 'condition': [], 'same_patch': [], 'near_patch': [], 'far_patch': []}
                    outdf['session'].append(each.name)
                    outdf['condition'].append(meta['condition'])
                    outdf['substrate'].append(substr)
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
    # swarmbox
    for j, each in enumerate(substrates):
        if onlyAA: width = 6
        else: width = 2.*len(conds)+1
        f, axes = plt.subplots(1,3,figsize=(width,2.5))
        data = outdf.query('substrate == "{}" and total > 4'.format(each))
        querystr = ''
        astr = ' or '
        for condition in conds:
            querystr += 'condition == "{}"'.format(condition)
            querystr += astr
        rdata = data.query(querystr[:-len(astr)])
        patchid = ['far_patch', 'near_patch', 'same_patch']
        patchlab = ['distant {}'.format(each), 'adjacent {}'.format(each), 'same {}'.format(each)]
        for i, ax in enumerate(axes):
            print(conds)
            if onlyAA: ax = swarmbox(ax=ax, x='condition', y=patchid[i], data=rdata, palette={'+': '#b353b5', '-': '#cc0000'}, compare=[('+', '-')])
            else:
                if len(conds) > 2:
                    ax = swarmbox(ax=ax, x='condition', y=patchid[i], data=rdata, order=conds, palette=mypal, compare=[(conds[0], conds[1:])])
                else:
                    ax = swarmbox(ax=ax, x='condition', y=patchid[i], data=rdata, order=conds, palette=mypal, compare=[(conds[0], conds[1])])

            annotations = [child for child in ax.get_children() if isinstance(child, plt.Text) and ("*" in child.get_text())]
            for an in annotations:
                an.set_position((an.get_position()[0], an.get_position()[1]+0.09))
            print(annotations)
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
        _file = os.path.join(outfolder, "{}_{}.png".format(_outfile+suffix, each))
        plt.savefig(_file, dpi=300)
        plt.cla()
        plt.clf()

        """ This is for pie plots
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
        """
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
