import os
import numpy as np
import pandas as pd

from pytrack_analysis import Multibench
from pytrack_analysis.cli import get_args
from pytrack_analysis.dataio import RawData, get_session_list
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import frameskips, get_displacements, get_head_tail, mistracks, get_pixel_flip
from pytrack_analysis.viz import plot_along, plot_fly, plot_overlay, plot_ts

def main():
    var_args =  [['exp', 'exp', 'Select experiment by four-letter ID'],
                        ['from', 'sfrom', 'Select session from where to start analyzing'],
                        ['not', 'snot', 'Select session which not to analyze'],
                        ['only', 'sonly', 'Select session which only to analyze'],
                        ['to', 'sto', 'Select session where to end analyzing'],
                        ['u', 'user', 'Select user']]
    opt_args = [['load', 'Option whether to load file'],
                ['novideo', 'Option whether to use videos'],
                ['plot', 'Option whether to plot']]
    args = get_args(var_args, opt_args, SILENT=True)
    profile = get_profile(args.exp, args.user, script=get_scriptname(__file__))
    load_from_file = args.load
    if args.novideo:
        load_from_file = True
    folders = profile.get_folders()

    ### go through video sessions
    for each_session in get_session_list(profile.Nvids(), args.sfrom, args.sto, args.snot, args.sonly):

        ### load video session data and metadata
        colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
        colunits = ['Datetime', 's',            's',        'px',       'px',       'rad',      'px',       'px']
        session_data = RawData(args.exp, each_session, folders, columns=colnames, units=colunits, noVideo=args.novideo)
        ### scale trajectories to mm
        scale = 8.543
        session_data.set_scale('fix_scale', scale, unit='mm')
        ### detect frameskips
        frameskips(session_data, dt='frame_dt')
        for i_arena, each_df in enumerate(session_data.raw_data):
            ### compute frame-to-frame displacements
            each_df['displacement'] = get_displacements(each_df, x='body_x', y='body_y')
            ### detect mistracked frames
            each_df = mistracks(each_df, i_arena, dr='displacement', major='major', thresholds=(4, 5))
            ### compute head and tail positions
            head_tails = get_head_tail(each_df, x='body_x', y='body_y', angle='angle', major='major')
            each_df['head_x'] = scale * head_tails[0] + session_data.arenas[i_arena].x
            each_df['head_y'] = scale * head_tails[1] + session_data.arenas[i_arena].y
            each_df['tail_x'] = scale * head_tails[2] + session_data.arenas[i_arena].x
            each_df['tail_y'] = scale * head_tails[3] + session_data.arenas[i_arena].y
            ### detect head flips
            file_id = 4 * (each_session-1) + i_arena + 1
            _file = os.path.join(folders['processed'],'post_tracking','{}_{:03d}.csv'.format(args.exp, file_id))
            if load_from_file:
                print("Loading data from", _file)
                df = pd.read_csv(_file, index_col='frame')
                each_df['flip'] = df['flip']
                each_df['headpx'] = df['headpx']
                each_df['tailpx'] = df['tailpx']
            else:
                flip, headpx, tailpx = get_pixel_flip(each_df, hx='head_x', hy='head_y', tx='tail_x', ty='tail_y', video=session_data.video_file, start=session_data.first_frame)
                each_df['flip'] = flip
                each_df['headpx'] = headpx
                each_df['tailpx'] = tailpx
                print("Saving data to", _file)
                each_df.to_csv(_file, index_label='frame')



        if args.plot:
            #f, ax = plot_fly(session_data.raw_data[0], x='body_x', y='body_y', hx='head_x', hy='head_y')
            #plot_along(f, ax)
            for i in range(4):
                f, ax = plot_ts(session_data.raw_data[i], x='frame', y=['frame_dt', 'angle', 'major', 'minor', 'displacement', 'flip'], units=['s', 'rad', 'mm', 'mm', 'mm/frame', ''])
                plot_along(f, ax)

        ### detect jumps

        ### detect correct head positions

        ### analyze general statistics from trajectory

        ### save fly data and metadata files

        print()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
