import os

from pytrack_analysis import Multibench
from pytrack_analysis.cli import get_args
from pytrack_analysis.dataio import get_data
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import get_files, get_frame_skips, get_session_list

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
    raw = folders['raw']
    videos = folders['videos']

    ### go through SESSIONS
    for each_session in get_session_list(profile.Nvids(), args.sfrom, args.sto, args.snot, args.sonly):
        ### get timestamp and all files from session folder
        allfiles, dtstamp, timestr = get_files(raw, each_session, videos)

        ### load raw data
        raw_data, data_units = get_data(allfiles['data'])

        ### analyze frameskips
        skips, bskips, maxskip, max_skip_arg = get_frame_skips(raw_data, println=True)

        ### detect arena geometry
        arena = get_arena_geometry()

        ### detect food spots
        food_spots = get_food_spots()

        ### translate to start position

        ### translate trajectories ot arena center

        ### scale trajectories to mm

        ### detect mistracked frames

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
