import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench
from pytrack_analysis.viz import set_font

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

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
    if _int >= 11 and _int < 14:
        return 11
    if _int >= 14 and _int < 17:
        return 14
    if _int >= 17:
        return 17

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file
    #db.dropna(mistracks=30)

    ### GO THROUGH SESSIONS
    statdict = {'session': [], 'day': [], 'daytime': [], 'condition': [], 'position': [], 'head_speed': [], 'body_speed': [], 'distance': [], 'dcenter': [], 'abs_turn_rate': [], 'major': [], 'minor': [], 'mistracks': []}
    _sessions = db.sessions
    n_ses = len(_sessions)
    for each_binsize in [0.01, 0.05, 0.1, 0.2]:
        _binsize = each_binsize
        _bins = np.append(np.arange(0,25,_binsize),1000)
        for each_sm in ['', 'sm_', 'smm_']:
            out_hists = pd.DataFrame()
            in_hists = pd.DataFrame()
            out_hists['bin'], in_hists['bin'] = _bins[:-1], _bins[:-1]
            which = each_sm
            for i_ses, each in enumerate(_sessions):
                df, meta = each.load(VERBOSE=False)
                kine = Kinematics(df, meta)
                outdf = kine.run(save_as=profile.out(), ret=True, _VERBOSE=False)

                ### stats (TODO: move to kinematics)
                statdict['session'].append(each.name)
                statdict['day'].append(meta['datetime'].date())
                statdict['daytime'].append(get_day(meta['datetime'].hour))
                statdict['condition'].append(meta['condition'])
                statdict['position'].append(meta['arena']['name'])
                statdict['head_speed'].append(outdf['smm_head_speed'].mean())
                statdict['body_speed'].append(outdf['smm_body_speed'].mean())
                statdict['distance'].append(np.cumsum(np.array(outdf['displacements']))[-1])
                statdict['dcenter'].append(outdf['dcenter'].mean())
                statdict['abs_turn_rate'].append(np.abs(outdf['angular_speed']).mean())
                statdict['major'].append(df['major'].mean())
                statdict['minor'].append(df['minor'].mean())
                statdict['mistracks'].append(meta['flags']['mistracked_frames'])

                ### Histograms
                nancheck = outdf[which+'head_speed'].isnull().values.any()
                if not (statdict['mistracks'][-1] > 30 or meta['condition'] =='NA' or nancheck or len(meta['food_spots']) == 0):
                    out_speed = np.array(outdf.query('min_dpatch > 2.5')[which+'head_speed'])
                    in_speed = np.array(outdf.query('min_dpatch <= 2.5')[which+'head_speed'])
                    out_hist, _ = np.histogram(out_speed, bins=_bins)  # arguments are passed to np.histogram
                    out_hist = out_hist/np.sum(out_hist)  # normalize
                    out_hists[each.name] = out_hist
                    in_hist, _ = np.histogram(in_speed, bins=_bins)  # arguments are passed to np.histogram
                    in_hist = in_hist/np.sum(in_hist)  # normalize
                    in_hists[each.name] = in_hist
                #else:
                    #print('Exclude: {} (mistracks: {}, condition: {}, NaNs: {}, #food_spots: {})'.format(each.name, statdict['mistracks'][-1], meta['condition'], nancheck, len(meta['food_spots'])))

            ### save Histograms
            print(which, _binsize)
            out_hist_file = os.path.join(profile.out(), experiment+'_kinematics_ohist_{}{:1.0E}.csv'.format(which, _binsize))
            out_hists.to_csv(out_hist_file, index=False)
            in_hist_file = os.path.join(profile.out(), experiment+'_kinematics_ihist_{}{:1.0E}.csv'.format(which, _binsize))
            in_hists.to_csv(in_hist_file, index=False)
            del out_hists
            del in_hists

    statdf = pd.DataFrame(statdict)
    statdf = statdf.reindex(columns=['session', 'day', 'daytime', 'condition', 'position', 'head_speed', 'body_speed', 'distance', 'dcenter', 'abs_turn_rate', 'major', 'minor', 'mistracks'])
    print(statdf)
    print(statdf.daytime.value_counts())
    statfile = os.path.join(profile.out(), experiment+'_kinematics_stats.csv')
    statdf.to_csv(statfile, index=False)
    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
