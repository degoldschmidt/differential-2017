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

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file
    #db.dropna(mistracks=30)

    ### GO THROUGH SESSIONS
    ### Containers for stats and histograms
    stats = []
    allhists = {}
    ### conditions
    binsizes = [0.2] #[0.01, 0.05, 0.1, 0.2]:
    dfvars = ['sm_head_speed']
    conditionals = ['total', 'in', 'out']
    ### sessions from db
    _sessions = db.sessions
    n_ses = len(_sessions)
    for i_ses, each in enumerate(_sessions):
        df, meta = each.load(VERBOSE=False)
        kine = Kinematics(df, meta)
        outfolder = os.path.join(profile.out(), kine.name)
        ### run kinematics
        outdf = kine.run(save_as=outfolder, ret=True)
        ### get stats and append to list
        stats.append(kine.stats())
        ### go through histogram parameters
        for each_binsize in binsizes:
            _bins = np.append(np.arange(0,25,each_binsize),1000)
            for each_var in dfvars:
                for each_cond in conditionals:
                    this_key = str((each_binsize, each_var, each_cond))
                    allhists[this_key] = pd.DataFrame()
                    this_entry = allhists[this_key]
                    this_entry['bin'] = _bins[:-1]
                    nancheck = outdf['sm_head_speed'].isnull().values.any()
                    if not (stats[-1].loc[0,'mistracks'] > 30 or meta['condition'] =='NA' or nancheck or len(meta['food_spots']) == 0):
                        if each_cond == 'in':
                            signal = np.array(outdf.query('min_dpatch > 2.5')[each_var])
                        elif each_cond == 'out':
                            signal = np.array(outdf.query('min_dpatch <= 2.5')[each_var])
                        else:
                            signal = np.array(outdf[each_var])
                        hist, _ = np.histogram(signal, bins=_bins)  # arguments are passed to np.histogram
                        hist = hist/np.sum(hist)  # normalize
                        this_entry[each.name] = hist
                    else:
                        print('Exclude: {} (mistracks: {}, condition: {}, NaNs: {}, #food_spots: {})'.format(each.name, stats[-1].loc[0,'mistracks'], meta['condition'], nancheck, len(meta['food_spots'])))

    ### save stats
    statdf = pd.concat(stats, ignore_index=True)
    print(statdf)
    statfile = os.path.join(outfolder, experiment+'_kinematics_stats.csv')
    statdf.to_csv(statfile, index=False)

    ### save Histograms
    for each_binsize in binsizes:
        for each_var in dfvars:
            for each_cond in conditionals:
                print('Save {} histograms for {} with binsize {}'.format(each_cond, each_var, each_binsize))
                hist_file = os.path.join(outfolder, experiment+'_kinematics_hist_{}_{}_{:1.0E}.csv'.format(each_cond, each_var, each_binsize))
                this_key = str((each_binsize, each_var, each_cond))
                allhists[this_key].to_csv(hist_file, index=False)

    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
