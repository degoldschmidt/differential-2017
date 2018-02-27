"""
===============
Demo Animation
===============
"""
from pytrack_analysis.profile import get_profile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as anim
import seaborn as sns
import numpy as np
import imageio
import pandas as pd
import os

from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench
from pytrack_analysis.viz import set_font, swarmbox


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

# animate time series
def animate(frame, *args):
    if frame%10==0:
        print(frame)
    lines = args[0]
    xarray = args[1]
    yarrays = args[2]
    xdata = np.array(xarray.loc[:frame+1])
    ydata = [None for i in range(5)]
    for j, each in enumerate(yarrays.columns):
        ydata[j] = np.array(yarrays.loc[:frame+1, each])
    # update the data of both line objects
    for j,each_line in enumerate(lines):
        each_line.set_data(xdata, ydata[j])
    return lines

# animate video
radius = 100
def updatefig(i, *image):
    y = image[3][1][i-6101]
    x = image[3][0][i-6101]
    image[0].set_array(image[1].get_data(i)[y-radius:y+radius, x-radius:x+radius])
    image[2].set_title(i)
    return image[0],

def init_animation(data, time=None, cols=None, video=None, figsize=(8,6), interval=None, playback=1):
    ts_colors = [   '#8dd3c7',
                    '#fcfc2f',
                    '#bebada',
                    '#fb8072',
                    '#80b1d3',]
    fig = plt.figure(figsize=figsize)
    ## gridspec
    N = len(cols)   # number of rows
    if interval is None:
        T = np.array(data.index)
    else:
        T = np.array(data.index)[interval[0]:interval[1]]
    xarray = df.loc[T[0]:T[-1]-1, time]
    yarrays = df.loc[T[0]:T[-1]-1, cols]
    gs = GridSpec(N, 2*N, height_ratios=[2,2,2,1,1])
    gs.update(wspace=2)
    ## video axis
    vid = imageio.get_reader(video)
    ax_video = plt.subplot(gs[:,:N]) ### N == 5
    ax_video.set_aspect('equal')
    sns.despine(ax=ax_video, bottom=True, left=True)
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    # initial frame
    im = plt.imshow(random_stack[0], animated=True)

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
    return fig, ax_video, ax_tseries, T, xarray, yarrays, im, vid, lines

def run_animation(fig, frames, xarray, yarrays, lines, im, vid, ax_video, pixelpos, factor=1):
    myinterval = 1000./(30*factor)
    print("Interval between frame: {}".format(myinterval))
    ani_lines = anim.FuncAnimation(fig, animate, frames, blit=True, fargs=(lines, xarray, yarrays), interval=myinterval)
    ani_image = anim.FuncAnimation(fig, updatefig, frames, blit=True, fargs=(im, vid, ax_video, pixelpos), interval=myinterval)
    #plt.tight_layout()
    ani_lines.save('out.mp4', extra_anim=[ani_image], writer='ffmpeg', dpi=300)

def respine(ax, interval, bottom):
    ax.set_ylim(interval)
    sns.despine(ax=ax, bottom=bottom, trim=True)
    return ax

if __name__ == '__main__':
    # profile
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)

    ### input data
    T = 1000
    random_stack = np.random.random((T,100,100))
    """
    df = pd.DataFrame({ 'time': 5.*np.arange(T),
                        'speed': np.random.rand(T),
                        'angular speed': np.random.rand(T),
                        'distance': np.random.rand(T),
                        'more stuff': np.random.rand(T),
                        'even more': np.random.rand(T),
                       })
    data = [col for col in df.columns if not col == 'time']
    """
    _in, _in2 = 'kinematics', 'classifier'
    infolder = os.path.join(profile.out(), _in)
    infolder2 = os.path.join(profile.out(), _in2)
    db = Experiment(profile.db())
    sessions = db.sessions[:1]
    for i_ses, each in enumerate(sessions):
        meta = each.load_meta(VERBOSE=False)
        video_file = meta['video']['file']
        first = meta['video']['first_frame']
        csv_file = os.path.join(infolder,  '{}_{}.csv'.format(each.name, _in))
        csv_file2 = os.path.join(infolder2,  '{}_{}.csv'.format(each.name, _in2))
        kinedf = pd.read_csv(csv_file, index_col='frame')
        ethodf = pd.read_csv(csv_file2, index_col='frame')
        df = pd.concat([kinedf[['elapsed_time', 'sm_head_speed', 'angular_speed', 'dcenter']], ethodf[['etho', 'visit']]], axis=1)
        data_cols = ['dcenter', 'sm_head_speed', 'angular_speed', 'etho', 'visit']
        fig, ax_video, ax_tseries, frames, xarray, yarrays, im, vid, lines = init_animation(df, time='elapsed_time', cols=data_cols, video=video_file, figsize=(16,6), interval= [0,5*30])
        lines[0].set_color('#222222')
        ax_tseries[0].set_xlabel('Time [s]')

        ax_tseries[4].set_ylabel('Min. distance\nto patch [mm]')
        ax_tseries[3].set_ylabel('Linear\nspeed [mm/s]')
        ax_tseries[2].set_ylabel('Angular\nspeed [º/s]')
        ax_tseries[1].set_ylabel('Ethogram')
        ax_tseries[0].set_ylabel('Visits')

        ax_tseries[4] = respine(ax_tseries[4], [0,15], True)
        ax_tseries[3] = respine(ax_tseries[3], [0,8], True)
        ax_tseries[2] = respine(ax_tseries[2], [-600,600], True)
        ax_tseries[1] = respine(ax_tseries[1], [0,1], True)
        ax_tseries[0] = respine(ax_tseries[0], [0,1], False)

        pixelpos = [np.zeros(frames.shape), np.zeros(frames.shape)]
        scale, x0, y0 = meta['arena']['scale'], meta['arena']['x'], meta['arena']['y']
        pixelpos[0] = (scale*np.array(kinedf['body_x']) + x0).astype(int)
        pixelpos[1] = (-scale*np.array(kinedf['body_y']) + y0).astype(int)
        run_animation(fig, frames, xarray, yarrays, lines, im, vid, ax_video, pixelpos)
    ### delete objects
    del profile
