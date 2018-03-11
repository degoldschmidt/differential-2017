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


def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ses', action="store", dest="session", type=int)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-sf', action="store", dest="startfr", type=int)
    parser.add_argument('-ef', action="store", dest="endfr", type=int)
    OVERWRITE = parser.parse_args().force
    SESSION = parser.parse_args().session
    START = parser.parse_args().startfr
    END = parser.parse_args().endfr

    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())

    conds = {'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}
    in_suffix  =  ['kinematics', 'classifier']
    out_suffix =  'plots'
    infolder = [os.path.join(profile.out(), suf) for suf in in_suffix]
    outfolder = os.path.join(profile.out(), out_suffix)
    outdf = {'session': [], 'condition': [], 'ratio': []}
    _outfile = 'tseries'
    if SESSION is None: listsession = db.sessions
    else: listsession = db.sessions[SESSION:SESSION+1]
    for session in listsession:
        noPlot = False
        meta = session.load_meta()
        if START is None: START = meta['video']['first_frame']
        if END is None: END = meta['video']['last_frame']
        print('First frame: {}, last frame: {}'.format(START, END))
        hook_file = os.path.join(outfolder, "{}_{}.csv".format(_outfile, session.name))
        ### take data from input files or hook
        if os.path.isfile(hook_file) and not OVERWRITE:
            print('Found data hook for session {}'.format(session.name))
            outdf = pd.read_csv(hook_file, index_col='id')
        else:
            print('Compute data for session {}'.format(session.name))
            ### Loading data
            try:
                csv_file = [os.path.join(infolder[j], '{}_{}.csv'.format(session.name, suf)) for j, suf in enumerate(in_suffix)]
                dfs = [pd.read_csv(each_file, index_col='frame') for each_file in csv_file]
                outdf = pd.concat(dfs, axis=1).loc[START:END]
                outdf = outdf.loc[:,~outdf.columns.duplicated()]
                print(outdf)
                ### save plotted data
                outdf.to_csv(hook_file, index_label='id')
            except FileNotFoundError:
                print(csv_file[0] + ' not found!')
                noPlot = True

        #### Plotting
        if not noPlot:
            f, axes = plt.subplots(5, figsize=(7,6), sharex=True)
            palette = {    -1: '#ff00fc',
                            0: '#ff00c7',
                            1: '#c97aaa',
                            2: '#000000',
                            3: '#30b050',
                            4: '#ffc04c',
                            5: '#4c8bff',
                            6: '#ff1100'}
            visitpal = {    0: '#ffffff',
                            1: '#ffc04c',
                            2: '#4c8bff'}

            t0 = outdf.loc[START,'elapsed_time']
            time = np.array(outdf['elapsed_time']) - t0
            outdf['min_patch'] = outdf.loc[:,['dpatch_{}'.format(i) for i in range(11)]].min(axis=1)

            # ROW 0: min distance to patch (TOP)
            axes[0].plot(time, outdf['min_patch'], 'k-')
            axes[0].set_ylabel('Min. distance\nto patch [mm]')
            axes[0].set_xticks([])
            axes[0].get_xaxis().set_visible(False)
            axes[0].set_ylim([0,10])
            axes[0].set_yticks(np.arange(0,11,2.5))
            sns.despine(ax=axes[0], bottom=True, trim=True)

            # ROW 1: linear speed
            axes[1].plot(time, outdf['sm_head_speed'], '-', color='#fb8072', label='head')
            axes[1].plot(time, outdf['sm_body_speed'], '-', color='#1f5fd5', label='body')
            axes[1].set_ylabel('Linear\nspeed [m/s]', labelpad=15)
            axes[1].set_xticks([])
            axes[1].get_xaxis().set_visible(False)
            axes[1].set_yticks([0, 2, 5, 10, 15])
            axes[1].set_ylim([0,20])
            axes[1].legend()
            sns.despine(ax=axes[1], bottom=True, trim=True)

            # ROW 2: angular speed
            axes[2].plot(time, outdf['angular_speed'], 'k-')
            axes[2].set_ylim([-505,505])
            axes[2].set_ylabel('Angular\nspeed [º/s]')
            axes[2].get_xaxis().set_visible(False)
            axes[2].set_xticks([])
            axes[2].set_yticks([-500,-125,0,125,500])
            sns.despine(ax=axes[2], bottom=True, trim=True)

            # ROW 3: ethogram
            axes[3].vlines(time, 0, 1, colors=outdf['etho'].apply(lambda x: palette[x]))
            axes[3].set_ylim([0,1])
            axes[3].set_ylabel('Ethogram', labelpad=40)
            axes[3].set_xticks([])
            axes[3].get_xaxis().set_visible(False)
            axes[3].set_yticks([])
            #axes[3].set_xlim([0, 30.])
            sns.despine(ax=axes[3], bottom=True, left=True, trim=True)


            # ROW 4: visits (BOTTOM)
            axes[4].vlines(time, 0, 1, colors=outdf['visit'].apply(lambda x: visitpal[x]))
            ii = outdf.query('frame_dt > 0.1').index[0]
            for t in np.arange(outdf.loc[ii,'elapsed_time']-t0, outdf.loc[ii+1,'elapsed_time']-t0, 0.033):
                print(t)
                axes[4].vlines(t, 0, 1, colors=visitpal[1])
            print(ii)
            axes[4].set_ylim([-0.5,1.5])
            axes[4].set_ylabel('Visits', labelpad=40)
            axes[4].set_yticks([])
            axes[4].set_xlim([0, 30.])
            axes[4].set_xticks(np.arange(0,31,5))
            axes[4].set_xlabel('Time [s]')
            sns.despine(ax=axes[4], left=True, trim=True)


            axes[0].hlines(2.5, 0, 30, colors="#aaaaaa", linestyles='dashed', lw=1)
            axes[0].hlines(5, 0, 30, colors="#aaaaaa", linestyles='dashed', lw=1)
            axes[1].hlines(2., 0, 30, colors="#aaaaaa", linestyles='dashed', lw=1)
            #ax_tseries[3].hlines(4., 0, 30, colors="#aaaaaa", linestyles='dashed')
            axes[2].hlines(125., 0, 30, colors="#aaaaaa", linestyles='dashed', lw=1)
            axes[2].hlines(-125., 0, 30, colors="#aaaaaa", linestyles='dashed', lw=1)


            VISIT = True
            ANNOS = True
            xend = 8.89
            xends = [0.735, 1.35, 4.26, 5.3, 8.89, 19.95]
            if ANNOS:
                for ax in axes:
                    if VISIT:
                        ax.axvline(x=xend-0.05,ymin=-0,ymax=1.2,c="#4f4f4f",linewidth=1, ls='dashed',zorder=10, clip_on=False)
                        ax.axvline(x=xends[-1]-0.05,ymin=-0,ymax=1.2,c="#4f4f4f",linewidth=1, ls='dashed',zorder=10, clip_on=False)
                    for e in xends:
                        ax.axvline(x=e-0.05,ymin=-0,ymax=1.2,c="#4f4f4f",linewidth=1, ls='dotted',zorder=10, clip_on=False)

            # output
            #plt.tight_layout()
            _file = os.path.join(outfolder, "{}_{}".format(_outfile, session.name))
            #plt.savefig(_file+'.pdf', dpi=300)
            #plt.savefig(_file+'.svg', dpi=300)
            plt.savefig(_file+'.png', dpi=300)
            #plt.show()
            plt.cla()
            plt.clf()
            plt.close()

    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
