import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Segments
from pytrack_analysis import Multibench

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
    _in, _out = 'classifier', 'segments'
    infolder = os.path.join(profile.out(), _in)
    outfolder = os.path.join(profile.out(), _out)

    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sessions):
        ### Loading data
        try:
            csv_file = os.path.join(infolder,  each.name+'_' + _in +'.csv')
            df = pd.read_csv(csv_file, index_col='frame')
            meta = each.load_meta(VERBOSE=False)
            segm = Segments(df, meta)
            dfs = segm.run(save_as=outfolder, ret=True)
        except FileNotFoundError:
            print(csv_file+ ' not found!')
    for each in dfs.keys():
        print(each)
        print(dfs[each].head(10))
        print()
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
