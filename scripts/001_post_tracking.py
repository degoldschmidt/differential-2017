import os
import numpy as np

from pytrack_analysis import Multibench
from pytrack_analysis.cli import get_args
from pytrack_analysis.dataio import RawData, get_session_list
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import frameskips, mistracks
from pytrack_analysis.viz import plot_along, plot_fly, plot_overlay, plot_ts

def main():
    args = get_args(    ['exp', 'exp', 'Select experiment by four-letter ID'],
                        ['from', 'sfrom', 'Select session from where to start analyzing'],
                        ['not', 'snot', 'Select session which not to analyze'],
                        ['only', 'sonly', 'Select session which only to analyze'],
                        ['to', 'sto', 'Select session where to end analyzing'],
                        ['u', 'user', 'Select user'],
                        SILENT=True)
    profile = get_profile(args.exp, args.user, script=get_scriptname(__file__))
    folders = profile.get_folders()

    ### go through video sessions
    for each_session in get_session_list(profile.Nvids(), args.sfrom, args.sto, args.snot, args.sonly):

        ### load video session data and metadata
        colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
        colunits = ['Datetime', 's',            's',        'px',       'px',       'rad',      'px',       'px']
        session_data = RawData(args.exp, each_session, folders, columns=colnames, units=colunits)

        ### scale trajectories to mm
        session_data.set_scale('fix_scale', 8.543, unit='mm')

        ### detect frameskips
        frameskips(session_data, dt='frame_dt')

        ### PLOT trajectories HERE
        #f2, ax2 = plot_overlay(session_data.raw_data, 37597, x='body_x', y='body_y', arena=session_data.arenas, scale=8.543, trace=40, video=session_data.video_file)
        #plot_along(f2, ax2)
        #f1, ax1 = plot_fly(session_data.raw_data[0], x='body_x', y='body_y')
        #plot_along(f1, ax1)
        for i in range(4):
            f, ax = plot_ts(session_data.raw_data[i], x='frame', y=['frame_dt', 'angle', 'major', 'minor'], units=['s', 'rad', 'mm', 'mm'])
            plot_along(f, ax)
        mistracks(session_data, x='body_x', y='body_y', major='major', thresholds=(4, 5))
        f2, ax2 = plot_overlay(session_data.raw_data, session_data.first_frame+108000, x='body_x', y='body_y', arena=session_data.arenas, scale=8.543, trace=108000, video=session_data.video_file)
        plot_along(f2, ax2)
        for i in range(4):
            f, ax = plot_ts(session_data.raw_data[i], x='frame', y=['frame_dt', 'angle', 'major', 'minor'], units=['s', 'rad', 'mm', 'mm'])
            plot_along(f, ax)


        ### detect mistracked frames
        #detect_mistrack(session_data)

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
