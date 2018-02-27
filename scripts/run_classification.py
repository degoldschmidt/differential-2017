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
pd.set_option('display.max_rows', 10000)
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
    _in, _out = 'kinematics', 'classifier'
    output = profile.out()
    infolder = os.path.join(output, _in)
    outfolder = os.path.join(output, _out)

    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sessions):
        csv_file = os.path.join(infolder,  each.name+'_'+_in+'.csv')
        df = pd.read_csv(csv_file, index_col='frame')
        meta = each.load_meta(VERBOSE=False)
        ## total micromoves
        nancheck = df['sm_head_speed'].isnull().values.any()
        if not (meta['flags']['mistracked_frames'] > 30 or meta['condition'] =='NA' or nancheck or len(meta['food_spots']) == 0):
            classify = Classifier(df, meta)
            odf = classify.run(save_as=outfolder, ret=True)
    #print(odf.iloc[1924:1926])
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
