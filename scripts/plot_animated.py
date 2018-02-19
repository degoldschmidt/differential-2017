"""
===============
Demo Gridspec03
===============

"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as anim
import seaborn as sns
import numpy as np
import pandas as pd


def get_tseries_axes(grid, position, sharex=None):
    row=position[0]
    col=position[1]
    if sharex is None:
        ax = plt.subplot(grid[row, col:], sharex=sharex)
    else:
        ax = plt.subplot(grid[row, col:])
        ax.xaxis.set_visible(False)
    ax.set_ylim(-1, 2)
    line, = ax.plot([], [])
    return ax, line

# animate time series
def animate(frame, *lines):
    xdata = np.array(df.loc[:frame+1, 'time'])
    ydata = [None for i in range(5)]
    for j, each in enumerate(data):
        ydata[j] = np.array(df.loc[:frame+1, each])
    # update the data of both line objects
    for j,each_line in enumerate(lines):
        each_line.set_data(xdata, ydata[j])
    return lines

# animate video
def updatefig(i, *image):
    image[0].set_array(image[1][i])
    return image[0],

def animate_time_series_with_video(data, time=None, cols=None, video=None, figsize=(8,6), interval=None):
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
        T = np.array(data.index)[interval[0], interval[1]]
    xarray = np.array(df['time'])
    gs = GridSpec(N, N)
    gs.update(wspace=2)
    ## video axis
    ax_video = plt.subplot(gs[1:N-1, 0:2]) ### N == 5
    ax_video.set_aspect('equal')
    sns.despine(ax=ax_video, bottom=True, left=True)
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    # initial frame
    im = plt.imshow(random_stack[0], animated=True)

    print(cols)
    ax_tseries, lines = [], []
    ax, l = get_tseries_axes(gs, [4, 2])
    l.set_color(ts_colors[0])
    ax.set_xlim(0, np.max(xarray)+10)
    sns.despine(ax=ax, trim=True)
    ax.set_xlabel(df.columns[-1])
    ax.set_ylabel(cols[-1])
    ax_tseries.append(ax)
    lines.append(l)
    for i in range(1,5):
        ax, l = get_tseries_axes(gs, [4-i, 2], sharex=ax_tseries[0])
        l.set_color(ts_colors[i])
        ax.set_ylabel(cols[-i-1])
        ax.set_xlim(0, np.max(xarray)+10)
        sns.despine(ax=ax, bottom=True, trim=True)
        ax_tseries.append(ax)
        lines.append(l)

    ani_lines = anim.FuncAnimation(fig, animate, T, blit=True, fargs=lines, interval=1000/25)
    ani_image = anim.FuncAnimation(fig, updatefig, T, blit=True, fargs=(im, video), interval=1000/25)
    #plt.tight_layout()
    ani_lines.save('out.mp4', extra_anim=[ani_image], writer='ffmpeg', fps=25, dpi=300)

if __name__ == '__main__':
    ### input data
    T = 100
    random_stack = np.random.random((T,100,100))
    df = pd.DataFrame({ 'time': 5.*np.arange(T),
                        'speed': np.random.rand(T),
                        'angular speed': np.random.rand(T),
                        'distance': np.random.rand(T),
                        'more stuff': np.random.rand(T),
                        'even more': np.random.rand(T),
                       })
    data = [col for col in df.columns if not col == 'time']
    animate_time_series_with_video(df, time='time', cols=data, video=random_stack, figsize=(16,6))
