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

def get_hist(data, bins=None):
    hist, _ = np.histogram(data, bins=bins)  # arguments are passed to np.histogram
    hist = hist/np.sum(hist)  # normalize
    return hist

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file
    ### Containers for histograms
    allhists = {}
    ### conditions (setting up dataframes)
    dfvars = ['sm_head_speed']#, 'angular_speed']
    dflist = []
    binsizes = {'sm_head_speed': [0.2], 'angular_speed': [1]} #[0.01, 0.05, 0.1, 0.2]:
    binrange = {'sm_head_speed': [0,25], 'angular_speed': [0, 100]} #[0.01, 0.05, 0.1, 0.2]:
    symmetric = {'sm_head_speed': False, 'angular_speed': True}
    ### sessions from db
    _sessions = db.sessions
    n_ses = len(_sessions)

    ### GO THROUGH CONDITIONS
    for each_var in dfvars:
        print("Histograms for {}".format(each_var))
        for each_binsize in binsizes[each_var]:
            ### GO THROUGH SESSIONS
            _bins = np.append(np.arange(binrange[each_var][0],binrange[each_var][1],each_binsize), 10000)
            _total, _in, _out = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            _total['bin'], _in['bin'], _out['bin'] = _bins[:-1], _bins[:-1], _bins[:-1]
            for i_ses, each in enumerate(_sessions):
                outfolder = os.path.join(profile.out(), 'kinematics')
                csv_file = os.path.join(outfolder,  each.name+'_kinematics.csv')
                df = pd.read_csv(csv_file, index_col='frame')
                meta = each.load_meta(VERBOSE=False)

                ### go through histogram parameters
                nancheck = df['sm_head_speed'].isnull().values.any()
                #if meta['condition'] =='O':
                if not (meta['flags']['mistracked_frames'] > 30 or meta['condition'] =='NA' or nancheck or len(meta['food_spots']) == 0):
                    ### query data by food patch distance
                    out_signal = np.array(df.query('min_dpatch > 2.5')[each_var])
                    in_signal = np.array(df.query('min_dpatch <= 2.5')[each_var])
                    signal = np.array(df[each_var])
                    if symmetric[each_var]:
                        _total[each.name+'_pos'] = get_hist(signal, bins=_bins)
                        _total[each.name+'_neg'] = get_hist(-signal, bins=_bins)
                        _in[each.name+'_pos'] = get_hist(in_signal, bins=_bins)
                        _in[each.name+'_neg'] = get_hist(-in_signal, bins=_bins)
                        _out[each.name+'_pos'] = get_hist(out_signal, bins=_bins)
                        _out[each.name+'_neg'] = get_hist(-out_signal, bins=_bins)
                    else:
                        _total[each.name] = get_hist(signal, bins=_bins)
                        _in[each.name] = get_hist(in_signal, bins=_bins)
                        _out[each.name] = get_hist(out_signal, bins=_bins)
                else:
                    pass
                    #print('Exclude: {} (mistracks: {}, condition: {}, NaNs: {}, #food_spots: {})'.format(each.name, meta['flags']['mistracked_frames'], meta['condition'], nancheck, len(meta['food_spots'])))
            ### save Histograms
            print(_total.head(1))
            print(_in.head(1))
            print(_out.head(1))
            print('Save {} histogram with binsize {}'.format(each_var, each_binsize))
            hist_file = os.path.join(outfolder, experiment+'_kinematics_hist_{}_{}_{:1.0E}.csv'.format(each_var, 'total', each_binsize))
            _total.to_csv(hist_file, index=False)
            hist_file = os.path.join(outfolder, experiment+'_kinematics_hist_{}_{}_{:1.0E}.csv'.format(each_var, 'in', each_binsize))
            _in.to_csv(hist_file, index=False)
            hist_file = os.path.join(outfolder, experiment+'_kinematics_hist_{}_{}_{:1.0E}.csv'.format(each_var, 'out', each_binsize))
            _out.to_csv(hist_file, index=False)
    ### delete objects
    del profile


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
