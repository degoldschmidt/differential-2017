import os
import numpy as np
import pandas as pd

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import RawData, get_session_list
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import frameskips, get_displacements, mistracks, get_head_tail, get_pixel_flip
from pytrack_analysis.viz import plot_along, plot_fly, plot_interval, plot_overlay, plot_ts
import matplotlib.pyplot as plt

experiment = 'DIFF'
user = 'degoldschmidt'
ascript = '001-post_tracking.ipynb'

profile = get_profile(experiment, user, script=ascript)
folders = profile.get_folders()

colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
colunits = ['Datetime', 's',            's',        'px',       'px',       'rad',      'px',       'px']
raw_data = RawData(experiment, folders, columns=colnames, units=colunits, noVideo=False)

### go through sessions
for session_id in range(raw_data.nvids):
    raw_data.get_session(session_id)
    load_from_file = False
    ### for each arena
    for i_arena, each_df in enumerate(raw_data.get_data()):
        ### compute head and tail positions
        each_df['head_x'], each_df['head_y'], each_df['tail_x'], each_df['tail_y'] = get_head_tail(each_df, x='body_x', y='body_y', angle='angle', major='major')
    if not load_from_file:
        flips, headpxs, tailpxs = get_pixel_flip(raw_data.get_data(), hx='head_x', hy='head_y', tx='tail_x', ty='tail_y', video=raw_data.video_file, start=raw_data.first_frame)
    for i_arena, each_df in enumerate(raw_data.get_data()):
        file_id = 4 * session_id + i_arena
        _file = os.path.join(folders['processed'],'pixeldiff','{}_{:03d}.csv'.format(experiment, file_id))
        each_df['flip'] = flips[i_arena]
        each_df['headpx'] = headpxs[i_arena]
        each_df['tailpx'] = tailpxs[i_arena]
        new_df = each_df.loc[:, ['flip', 'headpx', 'tailpx']]
        print("Saving data to", _file)
        new_df.to_csv(_file, index_label='frame')
