from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
from pytrack_analysis.viz import set_font, swarmbox, plot_along
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import ranksums
import argparse

onlyAA = True

def get_ratios(data, ds):
    count = np.zeros(ds.shape, dtype=np.int32)
    samecount = np.zeros(ds.shape, dtype=np.int32)
    elsecount = np.zeros(ds.shape, dtype=np.int32)
    for index, row in data.iterrows():
        d = row['max_distance']
        #print(d)
        i = np.where(ds > d)[0][0]-1
        #print(ds[i])
        count[i] += 1
        if row['to_same']:
            samecount[i] += 1
        else:
            elsecount[i] += 1
    ratios = np.divide(samecount, count)
    oratios = np.divide(elsecount, count)
    return ratios, oratios

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--check', action='store_true')
    OVERWRITE = parser.parse_args().force
    CHECK_ALL = parser.parse_args().check

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
    outdf = {'session': [], 'condition': [], 'substrate': [], 'max_distance': [], 'to_same': [], 'start': [], 'end': []}
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

                for i, sub in enumerate(['yeast', 'sucrose']):
                    transitions = np.where(visit_transitions == 2*(i+1))[0]
                    Ntrans = len(transitions)
                    for index in transitions:
                        dist_to_pre = per_frame_df.iloc[pos[index]+lens[index]:pos[index+1]]['dpatch_{}'.format(spots[index])]
                        max_dist = np.max(dist_to_pre)
                        outdf['session'].append(each.name)
                        outdf['condition'].append(meta['condition'])
                        outdf['substrate'].append(sub)
                        outdf['max_distance'].append(max_dist)
                        outdf['to_same'].append((spots[index]==spots[index+1]))
                        outdf['start'].append(pos[index]+lens[index])
                        outdf['end'].append(pos[index+1])
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')

        outdf = pd.DataFrame(outdf)
        print("Saving to {}".format(data_file))
        outdf.to_csv(data_file, index_label='id')
    print(outdf)

    print(np.min(outdf['max_distance']))

    data1 = outdf.query('substrate == "yeast"').query('condition == "S"')
    data2 = outdf.query('substrate == "yeast"').query('condition == "SAA"')
    ds = np.arange(0,53,1)
    f, ax = plt.subplots(figsize=(3,3))
    ratios, oratios = get_ratios(data1, ds)
    colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
    ax.plot(ds, ratios, 'r-')
    ratios, oratios = get_ratios(data2, ds)
    ax.plot(ds, ratios, 'g-')
    #plt.plot(ds, oratios, 'r-')
    plot_along(f, ax)

    if CHECK_ALL:
        _id = -1
        for index, row in outdf.iterrows():
            f, ax = plt.subplots(figsize=(3,3))
            print(row)
            if _id != row['session']:
                _id = int(row['session'][-3:])
                session = db.sessions[_id]
                meta = session.load_meta()
                kinefolder = os.path.join(profile.out(), 'kinematics')
                csv_file = os.path.join(kinefolder, '{}_{}.csv'.format(session.name, 'kinematics'))
                df = pd.read_csv(csv_file, index_col='frame')
            colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
            for spot in meta['food_spots']:
                ax.add_artist(plt.Circle((spot['x'], spot['y']), radius=1.5, color=colors[spot['substr']]))
                ax.add_artist(plt.Circle((spot['x'], spot['y']), radius=2.5, color=colors[spot['substr']], lw=1, ls='--', fill=False))
                start, end = row['start'], row['end']
                ax.plot(df.iloc[start:end]['head_x'], df.iloc[start:end]['head_y'], lw=0.5)
            ax.set_xlim([-30,30])
            ax.set_ylim([-30,30])
            ax.set_aspect('equal')
            plot_along(f, ax, fullscreen=False)
            plt.clf()
            plt.cla()
            plt.close()

    """
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
    """
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
