import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench

import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

import matplotlib.pyplot as plt

def plotting(data):
    plt.plot(data['displacements'], 'b-')
    plt.plot(data['head_speed'], 'k-')
    plt.plot(data['sm_head_speed'], 'b-')
    plt.plot(data['smm_head_speed'], 'r-')
    plt.show(block=False)
    plt.pause(30)
    plt.close()

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('DIFF', 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file

    ### GO THROUGH SESSIONS
    for each in db.sessions[:1]:
        df, meta = each.load(VERBOSE=False)
        kine = Kinematics(df, meta)
        outdf = kine.run()#save_as=profile.out())
        print(outdf.columns)
    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
