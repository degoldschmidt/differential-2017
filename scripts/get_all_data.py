import os
import numpy as np
import pandas as pd

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import RawData, get_session_list
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import frameskips, get_displacements, mistracks, get_head_tail, get_corrected_flips
from pytrack_analysis.viz import plot_along, plot_fly, plot_interval, plot_overlay, plot_ts
import matplotlib.pyplot as plt

def plot_arena(arena=None, spots=None, condition=None, ax=None):
    spot_colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
    cond_colors = {'SAA': '#b6d7a8', 'AA': '#A4C2F4', 'S': '#EA9999', 'O': '#CCCCCC'}
    if ax is None:
        ax = plt.gca()
    ### artists
    if arena is not None:
        ax.set_xlim([-1.1*arena.ro, 1.1*arena.ro])
        ax.set_ylim([-1.1*arena.ro, 1.1*arena.ro])
        arena_border = plt.Circle((0, 0), arena.rr, color='k', fill=False)
        ax.add_artist(arena_border)
        outer_arena_border = plt.Circle((0, 0), arena.ro, color='#aaaaaa', fill=False)
        ax.add_artist(outer_arena_border)
        ax.plot(0, 0, 'o', color='black', markersize=2)
    if spots is not None:
        for each_spot in spots:
            substr = each_spot.substrate
            spot = plt.Circle((each_spot.rx, each_spot.ry), each_spot.rr, color=spot_colors[substr], alpha=0.5)
            ax.add_artist(spot)
    if condition is not None:
        if condition in cond_colors.keys():
            spot = plt.Rectangle((-arena.ro, arena.ro-2), 5, 5, color=cond_colors[condition])
            ax.add_artist(spot)
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    return ax

def plot_traj(data, scale, time=None, only=None):
    f, axes = plt.subplots(2, 2, figsize=(20,20))
    fly = 0
    for each_row in axes:
        for ax in each_row:
            arena = data.arenas[fly]
            df = data.raw_data[fly]
            if time is None:
                start, end = df.index[0], df.index[-1]
            else:
                start, end = time
            ax = plot_arena(arena=arena, spots=arena.spots, condition=data.condition[fly], ax=ax)
            x = df.loc[start:end, 'body_x']
            y = df.loc[start:end, 'body_y']
            hx = df.loc[start:end, 'head_x']
            hy = df.loc[start:end, 'head_y']
            tx = df.loc[start:end, 'tail_x']
            ty = df.loc[start:end, 'tail_y']
            if only is None or only == 'body':
                ax.plot(x/scale, y/scale, c='k')
            if only is None or only == 'tail':
                ax.scatter(tx/scale, ty/scale, c='b', s=1)
            if only is None or only == 'head':
                ax.scatter(hx/scale, hy/scale, c='r', s=1)
            fly += 1
    plt.tight_layout()
    plt.show()

def main():
    experiment = 'DIFF'
    user = 'degoldschmidt'
    ascript = __file__
    profile = get_profile(experiment, user, script=ascript)
    folders = profile.get_folders()
    colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
    colunits = ['Datetime', 's',            's',        'px',       'px',       'rad',      'px',       'px']
    raw_data = RawData(experiment, folders, columns=colnames, units=colunits, noVideo=False)
    ### go through all session
    for i_session in range(raw_data.nvids):
        raw_data.get_session(i_session)
        raw_data.center()
        mistrk_list = []
        ### for each arena
        for i_arena, each_df in enumerate(raw_data.get_data()):
            ### compute head and tail positions
            each_df['head_x'], each_df['head_y'], each_df['tail_x'], each_df['tail_y'] = get_head_tail(each_df, x='body_x', y='body_y', angle='angle', major='major')
            ### compute frame-to-frame displacements
            arena = raw_data.arenas[i_arena]
            each_df['displacement'], each_df['dx'], each_df['dy'], each_df['mov_angle'], each_df['align'], each_df['acc'] = get_displacements(each_df, x='body_x', y='body_y', angle='angle')
            ### detect mistracked frames
            each_df, mistr = mistracks(each_df, i_arena, dr='displacement', major='major', thresholds=(4*8.543, 5*8.543))
            mistrk_list.append(len(mistr))

            file_id = 4 * (i_session) + i_arena
            _file = os.path.join(folders['processed'],'pixeldiff','{}_{:03d}.csv'.format(experiment, file_id))
            ### flips START-----
            df = pd.read_csv(_file, index_col='frame')
            each_df['headpx'], each_df['tailpx'] = df['headpx'], df['tailpx']
            each_df = get_corrected_flips(each_df)
        ### scale trajectories to mm
        #print(raw_data.get_data(0).head(3))
        scale = 8.543
        raw_data.set_scale('fix_scale', scale, unit='mm')
        raw_data.flip_y()
        print(mistrk_list)
        #print(raw_data.get_data(0).head(3))
        #plot_traj(raw_data, scale, time=(raw_data.first_frame, raw_data.last_frame), only='tail')
        for i_arena, each_df in enumerate(raw_data.get_data()):
            file_id = 4 * i_session + i_arena
            _file = os.path.join(folders['processed'], 'post_tracking','{}_{:03d}.csv'.format(experiment, file_id))
            out_df = each_df[['datetime', 'elapsed_time', 'frame_dt', 'body_x', 'body_y', 'head_x', 'head_y', 'tail_x', 'tail_y', 'angle', 'major', 'minor', 'flipped']]
            out_df.to_csv(_file, index_label='frame')
            meta_dict = {}
            arena = raw_data.arenas[i_arena]

            #### meta_dict save
            import yaml
            import io
            with open(os.path.join(folders['manual'],'conditions.yaml'), 'r') as stream:
                try:
                    conds = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            meta_dict['arena'] = {'x': float(arena.x), 'y': float(arena.y), 'layout': conds['arena_layout'], 'name': arena.name, 'outer': float(arena.outer), 'radius': float(arena.r), 'scale': arena.pxmm}
            meta_dict['condition'] = raw_data.condition[i_arena]
            meta_dict['datafile'] = _file
            meta_dict['datetime'] = raw_data.timestamp
            meta_dict['flags'] = {'mistracked_frames': mistrk_list[i_arena]}
            spots = arena.spots
            meta_dict['food_spots'] = [{'x': float(each.rx), 'y': float(each.ry), 'r': 1.5, 'substr': each.substrate} for each in spots]
            meta_dict['fly'] = {'genotype': conds['genotype'], 'mating': conds['mating'], 'metabolic': raw_data.condition[i_arena], 'n_per_arena': conds['num_flies'], 'sex': conds['sex']}
            meta_dict['setup'] = {'light': conds['light'], 'humidity': conds['humidity'], 'name': conds['setup'], 'room': 'behavior room', 'temperature': '25C'}
            meta_dict['video'] = {'dir': folders['videos'], 'file': raw_data.video_file, 'first_frame': int(raw_data.first_frame), 'last_frame': int(raw_data.last_frame), 'nframes': len(each_df.index), 'start_time': raw_data.starttime}
            _yaml = _file[:-4]+'.yaml'
            with io.open(_yaml, 'w', encoding='utf8') as f:
                yaml.dump(meta_dict, f, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
