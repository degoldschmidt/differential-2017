import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
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

def get_day(_int):
    """
    if _int == 8 or _int == 9 or _int == 10:
        return 9
    elif _int == 11 or _int == 12:
        return 11
    elif _int == 13 or _int == 14:
        return 13
    elif _int == 15 or _int == 16:
        return 15
    elif _int == 17 or _int == 18:
        return 17
    elif _int == 19 or _int == 20 or _int == 21 or _int == 22:
        return 19
    """
    if _int >= 8 and _int < 11:
        return 8
    elif _int >= 11 and _int < 14:
        return 11
    elif _int >= 14 or _int < 17:
        return 14
    elif _int >= 17 or _int < 20:
        return 17
    elif _int >= 20:
        return 20

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('DIFF', 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file

    ### GO THROUGH SESSIONS
    statdict = {'session': [], 'day': [], 'daytime': [], 'condition': [], 'position': [], 'head_speed': [], 'body_speed': [], 'distance': [], 'abs_turn_rate': [], 'major': [], 'minor': []}
    for each in db.sessions:
        df, meta = each.load(VERBOSE=False)
        kine = Kinematics(df, meta)
        outdf = kine.run() #save_as=profile.out())
        statdict['session'].append(each.name)
        statdict['day'].append(meta['datetime'].date())
        statdict['daytime'].append(get_day(meta['datetime'].hour))
        statdict['condition'].append(meta['condition'])
        statdict['position'].append(meta['arena']['name'])
        statdict['head_speed'].append(outdf['smm_head_speed'].mean())
        statdict['body_speed'].append(outdf['smm_body_speed'].mean())
        statdict['distance'].append(np.cumsum(np.array(outdf['displacements']))[-1])
        statdict['abs_turn_rate'].append(np.abs(outdf['angular_speed']).mean())
        statdict['major'].append(df['major'].mean())
        statdict['minor'].append(df['minor'].mean())
    statdf = pd.DataFrame(statdict)
    statdf = statdf.reindex(columns=['session', 'day', 'daytime', 'condition', 'position', 'head_speed', 'body_speed', 'distance', 'abs_turn_rate', 'major', 'minor'])
    print(statdf)
    print(statdf.daytime.value_counts())
    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
