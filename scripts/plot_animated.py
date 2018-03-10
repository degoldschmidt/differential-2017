"""
===============
Demo Animation
===============
"""
from pytrack_analysis.profile import get_profile
import seaborn as sns
import numpy as np
import imageio
import pandas as pd
import os

from pytrack_analysis.plot import set_font, swarmbox
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as anim
import argparse

ONLY_VIDEO = False
ONLY_TRAJ = True
NO_ANNOS = True

def get_tseries_axes(grid, position, sharex=None):
    row=position[0]
    col=position[1]
    if sharex is None:
        ax = plt.subplot(grid[row, col:], sharex=sharex)
    else:
        ax = plt.subplot(grid[row, col:])
        ax.xaxis.set_visible(False)
    line, = ax.plot([], [])
    return ax, line

palette = {    -1: '#ff00fc',
                0: '#ff00c7',
                1: '#c97aaa',
                2: '#000000',
                3: '#30b050',
                4: '#ff7f00',
                5: '#1f78b4',
                6: '#ff1100'}
palette2 ={     0: '#ffffff',
                1: '#ff7f00',
                2: '#1f78b4'}
# animate time series
def animate(frame, *args):
    lines = args[0]
    xarray = args[1]
    yarrays = args[2]
    ax_ts = args[3]
    xdata = np.array(xarray.loc[:frame+1])
    ydata = [None for i in range(5)]
    for j, each in enumerate(yarrays.columns):
        if j < 2:
            ydata[j] = int(yarrays.loc[frame, each])
        else:
            ydata[j] = np.array(yarrays.loc[:frame+1, each])
    # update the data of both line objects
    for j,each_line in enumerate(lines):
        if j < 2:
            if j == 1:
                ax_ts[j].vlines(xarray.loc[frame], 0, 1, color=palette[ydata[j]])
            if j == 0:
                ax_ts[j].vlines(xarray.loc[frame], 0, 1, color=palette2[ydata[j]])
                if frame-1 in xarray.index:
                    if xarray.loc[frame] - xarray.loc[frame-1] > 0.1:
                        a = xarray.loc[frame]-0.0333
                        while a > xarray.loc[frame-1]:
                            ax_ts[j].vlines(a, 0, 1, color=palette2[ydata[j]])
                            a -= 0.0333
        else:
            each_line.set_data(xdata, ydata[j])
    return lines

# animate video
radius = int(8.543 * 12.5)
print(radius) ##100
def updatefig(i, *image):
    xpos, ypos = image[3][0][i-4970], image[3][1][i-4970]
    etho = image[3][2][i-4970]
    bxpos, bypos = image[3][3][i-4970], image[3][4][i-4970]
    y = int(359.3853 - 8.543 * (-1.))##image[3][1][i-6101]
    x = int(366.1242 + 8.543 * 15.)##image[3][0][i-6101]
    a, b = (x-radius), (y-radius)
    if i%100==0:
        print(i, xpos, ypos, etho)

    ### update plots
    if len(image[2].get_lines()) == 0:
        image[2].plot([xpos-a, bxpos-a], [ypos-b, bypos-b], color='#ff228c', ls='-', lw=1)
    image[0].set_array(image[1].get_data(i)[y-radius:y+radius, x-radius:x+radius])
    image[2].set_title("frame #{}".format(i))
    if not NO_ANNOS:
        image[2].plot(xpos-a, ypos-b, color=palette[etho], marker='.', markersize=5)
    else:
        image[2].plot(xpos-a, ypos-b, color='#ff228c', marker='.', markersize=5)
        image[2].get_lines()[0].set_data([xpos-a, bxpos-a], [ypos-b, bypos-b])
    return image[0],

def init_animation(data, time=None, cols=None, video=None, figsize=(8,6), interval=None, playback=1, meta=None):
    ts_colors = [   '#8dd3c7',
                    '#fcfc2f',
                    '#bebada',
                    '#fb8072',
                    '#80b1d3',]
    spot_colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
    fig = plt.figure(figsize=figsize)
    ## gridspec
    N = len(cols)   # number of rows
    if interval is None:
        T = np.array(data.index)
    else:
        T = np.arange(interval[0],interval[1],dtype=np.int32)
    xarray = data.loc[T[0]:T[-1], time] - data.loc[T[0], time]
    yarrays = data.loc[T[0]:T[-1], cols]
    gs = GridSpec(N, 2*N, height_ratios=[2,2,2,1,1])
    gs.update(wspace=2)
    ## video axis
    vid = imageio.get_reader(video)
    if ONLY_VIDEO:
        fig, ax_video = plt.subplots(figsize=(6,6))
    else:
        ax_video = plt.subplot(gs[:,:N]) ### N == 5
    ax_video.set_aspect('equal')
    sns.despine(ax=ax_video, bottom=True, left=True)
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    # initial frame
    scale = meta['arena']['scale']
    radius = int(scale * 12.5)
    x0, y0 = meta['arena']['x'], meta['arena']['y']
    x, y = int(x0 + scale * 15.), int(y0 - scale * (-1.))

    im = ax_video.imshow(vid.get_data(interval[0])[y-radius:y+radius, x-radius:x+radius], animated=True)
    if not ONLY_TRAJ:
        for ii,each in enumerate(meta['food_spots']):
            sx, sy = scale * each['x'] + x0 - (x-radius), -scale * each['y'] + y0 - (y-radius)
            ax_video.add_artist(plt.Circle((sx,sy), scale*1.5, color=spot_colors[each['substr']], lw=2.5, alpha=0.5, fill=False, zorder=100))
            if ii in [1, 3, 9]:
                ax_video.add_artist(plt.Circle((sx,sy), scale*2.5, color='#ffffff', ls='dashed', lw=1.5, alpha=0.5, fill=False, zorder=100))
            if ii == 1:
                ax_video.add_artist(plt.Circle((sx,sy), scale*5, color='#ffffff', ls='dotted', lw=1, alpha=0.5, fill=False, zorder=100))
            #ax_video.text(sx,sy, "{}".format(ii), zorder=100)

    ax_tseries, lines = None, None
    if not ONLY_VIDEO:
        ax_tseries, lines = [], []
        ax, l = get_tseries_axes(gs, [4, 5])
        l.set_color(ts_colors[0])
        ax.set_xlim(0, 1.1*np.max(xarray))
        if np.min(data[cols[-1]]) < 0:
            ax.set_ylim(1.1*np.min(data[cols[-1]]), 1.1*np.max(data[cols[-1]]))
        else:
            ax.set_ylim(0, 1.1*np.max(data[cols[-1]]))
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel(time)
        ax.set_ylabel(cols[-1])
        ax_tseries.append(ax)
        lines.append(l)
        for i in range(1,5):
            ax, l = get_tseries_axes(gs, [4-i, 5], sharex=ax_tseries[0])
            l.set_color(ts_colors[i])
            ax.set_xlim(0, 1.1*np.max(xarray))
            ax.set_ylabel(cols[-i-1])
            if np.min(data[cols[-i-1]]) < 0:
                ax.set_ylim(1.1*np.min(data[cols[-i-1]]), 1.1*np.max(data[cols[-i-1]]))
            else:
                ax.set_ylim(0, 1.1*np.max(data[cols[-i-1]]))
            sns.despine(ax=ax, bottom=True, trim=True)
            ax_tseries.append(ax)
            lines.append(l)

        ### thresholds
        ax_tseries[4].hlines(2.5, ax_tseries[4].get_xlim()[0], ax_tseries[4].get_xlim()[1], colors="#aaaaaa", linestyles='dashed')
        ax_tseries[4].hlines(5, ax_tseries[4].get_xlim()[0], ax_tseries[4].get_xlim()[1], colors="#aaaaaa", linestyles='dashed')

        ax_tseries[3].hlines(2., ax_tseries[4].get_xlim()[0], ax_tseries[4].get_xlim()[1], colors="#aaaaaa", linestyles='dashed')
        #ax_tseries[3].hlines(4., ax_tseries[4].get_xlim()[0], ax_tseries[4].get_xlim()[1], colors="#aaaaaa", linestyles='dashed')

        ax_tseries[2].hlines(125., ax_tseries[4].get_xlim()[0], ax_tseries[4].get_xlim()[1], colors="#aaaaaa", linestyles='dashed')
        ax_tseries[2].hlines(-125., ax_tseries[4].get_xlim()[0], ax_tseries[4].get_xlim()[1], colors="#aaaaaa", linestyles='dashed')

    return fig, ax_video, ax_tseries, T, xarray, yarrays, im, vid, lines

def run_animation(fig, frames, xarray, yarrays, lines, im, vid, ax_video, ax_ts, pixelpos, factor=1, outfile="out.mp4"):
    myinterval = 1000./(30*factor)
    print("Interval between frame: {}".format(myinterval))
    if not ONLY_VIDEO:
        ani_lines = anim.FuncAnimation(fig, animate, frames, blit=True, fargs=(lines, xarray, yarrays, ax_ts), interval=myinterval)
    ani_image = anim.FuncAnimation(fig, updatefig, frames, blit=True, fargs=(im, vid, ax_video, pixelpos), interval=myinterval)
    #plt.tight_layout()
    if ONLY_VIDEO:
        ani_image.save(outfile, writer='ffmpeg', dpi=300)
    else:
        ani_lines.save(outfile, extra_anim=[ani_image], writer='ffmpeg', dpi=300)

def respine(ax, interval, tickint, bottom):
    if interval is None:
        ax.set_ylim([0, 1/tickint])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        sns.despine(ax=ax, left=True, bottom=bottom, trim=True)
    else:
        if type(tickint) is list:
            ax.set_yticks(tickint)
        else:
            ax.set_yticks(np.arange(interval[0], interval[1]+1,tickint))
        ax.set_ylim(interval)
        sns.despine(ax=ax, bottom=bottom, trim=True)
    return ax

def main():
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ses', action="store", dest="session", type=int)
    parser.add_argument('-sf', action="store", dest="startfr", type=int)
    parser.add_argument('-ef', action="store", dest="endfr", type=int)
    SESSION = parser.parse_args().session
    START = parser.parse_args().startfr
    END = parser.parse_args().endfr

    # profile
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)

    ### input data
    _in, _in2 = 'kinematics', 'classifier'
    _out = 'plots'
    infolder = os.path.join(profile.out(), _in)
    infolder2 = os.path.join(profile.out(), _in2)
    outfolder = os.path.join(profile.out(), _out)
    _outfile = 'video'
    db = Experiment(profile.db())
    session = db.sessions[SESSION]
    meta = session.load_meta(VERBOSE=False)
    if os.name == 'posix':
        _file = meta['video']['file'].split('\\')[-1]
        video_file = os.path.join("/Volumes/DATA_BACKUP/data/tracking/all_videos", _file)
        print("MacOSX:", video_file)
    else:
        video_file = meta['video']['file']
    first = meta['video']['first_frame']
    csv_file = os.path.join(infolder,  '{}_{}.csv'.format(session.name, _in))
    csv_file2 = os.path.join(infolder2,  '{}_{}.csv'.format(session.name, _in2))
    kinedf = pd.read_csv(csv_file, index_col='frame')
    ethodf = pd.read_csv(csv_file2, index_col='frame')
    kinedf['min_patch'] = kinedf.loc[:,['dpatch_{}'.format(i) for i in range(11)]].min(axis=1)
    df = pd.concat([kinedf[['elapsed_time', 'sm_head_speed', 'angular_speed', 'min_patch']], ethodf[['etho', 'visit']]], axis=1)
    data_cols = ['visit', 'etho', 'angular_speed', 'sm_head_speed', 'min_patch']
    fig, ax_video, ax_tseries, frames, xarray, yarrays, im, vid, lines = init_animation(df, time='elapsed_time', cols=data_cols, video=video_file, figsize=(16,6), interval=[START,END], meta=meta)
    if not ONLY_VIDEO:
        lines[0].set_color('#222222')
        ax_tseries[0].set_xlabel('Time [s]')
        ax_tseries[4] = respine(ax_tseries[4], [0,10], 2.5, True)
        ax_tseries[3] = respine(ax_tseries[3], [0,20], [0,2,5,10,15], True)
        ax_tseries[2] = respine(ax_tseries[2], [-600,600], [-500,-125,0,125,500], True)
        ax_tseries[1] = respine(ax_tseries[1], None, 1, True)
        ax_tseries[0] = respine(ax_tseries[0], None, 0.5, False)
        ax_tseries[4].set_ylabel('Min. distance\nto patch [mm]')
        ax_tseries[3].set_ylabel('Linear\nspeed [mm/s]')
        ax_tseries[2].set_ylabel('Angular\nspeed [º/s]')
        ax_tseries[1].set_ylabel('Ethogram', labelpad=40)
        ax_tseries[0].set_ylabel('Visits', labelpad=40)
    if NO_ANNOS:
        ax_tseries[1].set_visible(False)
        ax_tseries[0].set_visible(False)

    pixelpos = [np.zeros(frames.shape), np.zeros(frames.shape), np.zeros(frames.shape), np.zeros(frames.shape), np.zeros(frames.shape)]
    scale, x0, y0 = meta['arena']['scale'], meta['arena']['x'], meta['arena']['y']
    pixelpos[0] = (scale*np.array(kinedf['head_x']) + x0).astype(int)
    pixelpos[1] = (-scale*np.array(kinedf['head_y']) + y0).astype(int)
    pixelpos[2] = np.array(ethodf['etho'])
    pixelpos[3] = (scale*np.array(kinedf['body_x']) + x0).astype(int)
    pixelpos[4] = (-scale*np.array(kinedf['body_y']) + y0).astype(int)

    ### save animation to file
    _file = os.path.join(outfolder, "{}_{}.mp4".format(_outfile, session.name))
    run_animation(fig, frames, xarray, yarrays, lines, im, vid, ax_video, ax_tseries, pixelpos, outfile=_file)
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
