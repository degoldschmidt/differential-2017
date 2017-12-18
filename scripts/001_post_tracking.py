import os

from pytrack_analysis import Multibench
from pytrack_analysis.cli import get_args
from pytrack_analysis.dataio import RawData, get_session_list
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import frameskips, mistracks

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
        print(session_data.raw_data[3].head())

        ### scale trajectories to mm
        session_data.set_scale('diameter', 49.75, unit='mm')

        ### detect frameskips
        frameskips(session_data, dt='frame_dt')

        ### detect mistracked frames
        #detect_mistrack(session_data)

        ### detect jumps

        ### detect correct head positions

        ### analyze general statistics from trajectory

        ### save fly data and metadata files

        """
        #f, ax = plot_overlay(allfiles['video'], arenas=geom_data, spots=food_data)
        #plot_along(f, ax)

        ### getting metadata
        meta = get_meta(allfiles, dtstamp, manual)
        meta['datadir'] = session_folder.split('/')[-2:]
        meta['experiment'] = EXP_ID
        meta['num_frames'] = get_num_frames(raw_data)
        for each_condition in meta['variables']:
            for each_file in meta["files"]:
                if each_condition in each_file:
                    conditions = pd.read_csv(each_file, sep='\t', index_col='ix')
        """
        print()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
