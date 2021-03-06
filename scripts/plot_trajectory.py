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

    #thisscript = os.path.basename(__file__).split('.')[0]
    #experiment = 'DIFF'
    #profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    OUT = '/home/degoldschmidt/post_tracking'
    DB = '/home/degoldschmidt/post_tracking/DIFF.yaml'
    db = Experiment(DB)

    conds = {'SAA': "#98c37e", 'AA': "#5788e7", 'S': "#D66667", 'O': "#B7B7B7"}
    in_suffix  =  ['kinematics', 'classifier']
    out_suffix =  'plots'
    infolder = [os.path.join(OUT, suf) for suf in in_suffix]
    outfolder = os.path.join(OUT, out_suffix)
    outdf = {'session': [], 'condition': [], 'ratio': []}
    _outfile = 'trajectory'
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
            outdf = outdf.loc[START:END]
        else:
            print('Compute data for session {}'.format(session.name))
            ### Loading data
            try:
                csv_file = [os.path.join(infolder[j], '{}_{}.csv'.format(session.name, suf)) for j, suf in enumerate(in_suffix)]
                dfs = [pd.read_csv(each_file, index_col='frame') for each_file in csv_file]
                outdf = pd.concat(dfs, axis=1).loc[START:END]
                ### save plotted data
                outdf.to_csv(hook_file, index_label='id')
            except FileNotFoundError:
                print(csv_file[0] + ' not found!')
                noPlot = True

        #### Plotting
        if not noPlot:
            f, ax = plt.subplots(figsize=(4,4))
            # arena plot
            ax = plot.arena(meta['arena'], meta['food_spots'], ax=ax)
            for i in [1,3,9]:
                x,y = meta['food_spots'][i]['x'], meta['food_spots'][i]['y']
                ax.add_artist(plt.Circle((x,y), 2.5, ls='dashed', color='#818181', fill=False, lw=1.5))
                ax.add_artist(plt.Circle((x,y), 3, ls='dotted', color='#818181', fill=False, lw=1.5))
                if i == 1:
                    ax.add_artist(plt.Circle((x,y), 5, ls='dotted', color='#818181', fill=False, lw=1))

            # trajectory plot
            ax = plot.trajectory(xc='head_x', yc='head_y', xs='body_x', ys='body_y', data=outdf, hue='etho', no_hue=[0, 1, 2, 3, 6], to_body=[], size=1, ax=ax)
            ax.plot(np.array(outdf['head_x'])[0], np.array(outdf['head_y'])[0], '#00ff12', marker='*', markersize=3)
            intval = 200
            timepoints = np.array(outdf['elapsed_time'])[intval::intval]
            #ax.scatter(np.array(outdf['head_x'])[intval::intval], np.array(outdf['head_y'])[intval::intval], c=timepoints, cmap=plt.get_cmap('viridis'), marker='s', s=1, zorder=10)
            ax.plot(np.array(outdf['head_x'])[-1], np.array(outdf['head_y'])[-1], '#ff0000', marker='*', markersize=3)
            ax.add_artist(plt.Rectangle((30, 30), 2, 2, color = conds[meta['condition']]))
            ax.text(30, 35, meta['condition'])
            #ax.set_title('{}'.format(session.name), fontweight='bold', loc='left')
            ax.set_xlim([2.5,27.5])
            ax.set_ylim([-13.5,11.5])
            ax.set_xlim([0.5,15.5])
            ax.set_ylim([-3.,12.])
            r = 30.5
            ax.set_xlim([-r,r])
            ax.set_ylim([-r,r])
            # output
            plt.tight_layout()
            _file = os.path.join(outfolder, "{}_{}_{}_{}".format(_outfile, session.name, START, END))
            _file = os.path.join(outfolder, "{}_{}".format(_outfile, session.name))
            #plt.savefig(_file+'.pdf', dpi=300)
            #plt.savefig(_file+'.svg', dpi=300)
            plt.savefig(_file+'.png', dpi=300)
            plt.cla()
            plt.clf()
            plt.close()

    ### delete objects
    #del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
