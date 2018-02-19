import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench
from pytrack_analysis.viz import set_font, swarmbox

import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
#pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)
import tkinter as tk
RUN_STATS = True

def main():
    """
    --- general parameters
     * thisscript: scriptname
     * experiment: experiment id (4 letters)
     * profile:    profile for given experiment, user and scriptname (from profile file)
     * db:         database from file
     * _sessions:  which sessions to process
     * n_ses:      number of sessions
     * stats:      list for stats
    """
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())
    sessions = db.sessions
    n_ses = len(sessions)
    stats = []
    infolder = os.path.join(profile.out(), 'kinematics')
    outfolder = os.path.join(profile.out(), 'classify')

    ### GO THROUGH SESSIONS
    #totals = {'session': [], 'totals_Y': [], 'totals_S': [], 'condition': []}
    ethoY, ethoS = [{} for i in range(4)], [{} for i in range(4)]
    for i_ses, each in enumerate(sessions):
        csv_file = os.path.join(infolder,  each.name+'_kinematics.csv')
        df = pd.read_csv(csv_file, index_col='frame')
        rawdf, meta = each.load(VERBOSE=False)
        df['major'] = rawdf['major']
        df['minor'] = rawdf['minor']
        first = meta['video']['first_frame']
        last = meta['video']['last_frame']
        df['time'] = rawdf['elapsed_time'] - rawdf.loc[first,'elapsed_time']
        total_time = df.loc[last, 'time']

        ## total micromoves
        nancheck = df['sm_head_speed'].isnull().values.any()
        if not (meta['flags']['mistracked_frames'] > 30 or meta['condition'] =='NA' or nancheck or len(meta['food_spots']) == 0):
            classify = Classifier(df, meta)
            df['etho'], df['visit'], df['encounter'], df['encounter_index'] = classify.run()
            outdf = df.loc[:, ['time', 'frame_dt', 'etho', 'visit', 'encounter']]
            outfile = os.path.join(outfolder, each.name+'_'+classify.name+'.csv')
            outdf.to_csv(outfile, index_label='frame')

            #sumsY = df.groupby(['etho'])['frame_dt'].sum().loc[4]
            #sumsS = df.groupby(['etho'])['frame_dt'].sum().loc[5]
            #print(sumsY/60, sumsS/60)

            #nfrs = meta['video']['nframes']
            #ytotal = (total_time/60)*np.sum(np.array(df['etho'])==4)/nfrs
            #stotal = (total_time/60)*np.sum(np.array(df['etho'])==5)/nfrs
            #print(ytotal, stotal)
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
