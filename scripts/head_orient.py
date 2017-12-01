### imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from scipy import signal
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("index", help="display a frame of a given index",
                    type=int)
args = parser.parse_args()

def get_distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0]) + (pt1[1]-pt2[1])*(pt1[1]-pt2[1]))

def get_disconts(data, thr_disc=80., thr_coh=0.001):        ### TODO: explicit!!!
    data['flags'] = pd.Series(np.zeros(len(data.index)), index=data.index)
    data['last_index'] = pd.Series(np.zeros(len(data.index)), index=data.index)
    data['last_distance'] = pd.Series(np.zeros(len(data.index)), index=data.index)
    flag = 0
    for index, row in data.iterrows():
        if row['distance'] > thr_disc or flag == 1:
            if index == 0:
                print('First frame breaker!')
            elif flag == 0:
                data.loc[index, 'flags'] = 1
                flag = 1
                data.loc[index, 'last_index'] = index-1
            else:
                data.loc[index, 'last_index'] = data['last_index'].loc[index-1]
                this_last = data.loc[index, 'last_index']
                dist = get_distance((data.loc[index, 'body_x'], data.loc[index, 'body_y']), (data.loc[this_last, 'body_x'], data.loc[this_last, 'body_y']))
                data.loc[index, 'last_distance'] = dist
                if dist < thr_disc or row['distance'] < thr_coh:
                    data.loc[index, 'flags'] = 0
                    flag = 0
                else:
                    data.loc[index, 'flags'] = 1
    return data

def get_files(timestampstring, basedir, videos):
    print(timestampstring)
    return {
                    "data" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "fly" in eachfile and timestampstring in eachfile and "cleaned" not in eachfile],
                    "food" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "food" in eachfile and timestampstring in eachfile],
                    "geometry" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "geometry" in eachfile and timestampstring in eachfile][0],
                    "timestart" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile and timestampstring in eachfile][0],
                    "video" : [os.path.join(videos, eachfile) for eachfile in os.listdir(videos) if ".avi" in eachfile and timestampstring in eachfile][0],
    }

def get_frame_skips(data, println=False):
    secs = data['elapsed_time_[s]']
    frameskips = data['frame_dt_[s]']
    total = frameskips.shape[0]
    strict_skips = np.sum(frameskips > (1/30)+(1/30))
    easy_skips = np.sum(frameskips > (1/30)+(1/3))
    if println:
        print('Skips of more than 1 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/30), strict_skips, 100*strict_skips/total))
        print('Skips of more than 10 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/3), easy_skips, 100*easy_skips/total))
    return strict_skips, easy_skips

def get_patch_average(image, pt, radius=1):
    pxls = []
    for x in range(-radius, radius+1):
        yr = radius-abs(x)
        for y in range(-yr, yr+1):
            pxls.append(image[int(pt[1])+y, int(pt[0])+x, 0])
    return np.mean(np.array(pxls))

def get_sign(image, frame, x, y, major, orird):
    headx, heady = x.loc[frame]+0.5*major.loc[frame]*np.cos(orird.loc[frame]), y.loc[frame]+0.5*major.loc[frame]*np.sin(orird.loc[frame])
    tailx, taily = x.loc[frame]-0.5*major.loc[frame]*np.cos(orird.loc[frame]), y.loc[frame]-0.5*major.loc[frame]*np.sin(orird.loc[frame])
    headpx = get_patch_average(image, (headx, heady))
    tailpx = get_patch_average(image, (tailx, taily))
    pixeldiff = tailpx - headpx
    return np.sign(pixeldiff)

def get_signs(data, vidfile, space=300):
    import imageio
    vid = imageio.get_reader(vidfile,  'ffmpeg')
    signs = np.zeros(data['angle'].shape)
    x = data['body_x']
    y = data['body_y']
    orird = data['angle']
    major = data['major']
    for jx in range(0, len(data.index), space):
        frame = data.index[jx]
        image = vid.get_data(frame)
        signs[jx] = get_sign(image, frame, x, y, major, orird)
        if jx > 0:
            if signs[jx-space] == signs[jx]:
                signs[jx-space:jx] = signs[jx]
            elif signs[jx-space] != signs[jx]:
                second = int(space/10)
                for kx in range(1,space-1,second):
                    frame = data.index[jx-kx]
                    image = vid.get_data(frame)
                    signs[jx-kx] = get_sign(image, frame, x, y, major, orird)
                    if signs[jx-kx] == signs[jx-kx+second]:
                        signs[jx-kx+1:jx-kx+second] = signs[jx-kx]
                    else:
                        for lx in range(jx-kx+1,jx-kx+second):
                            frame = data.index[lx]
                            image = vid.get_data(frame)
                            signs[lx] = get_sign(image, frame, x, y, major, orird)

    return signs

def load_data(filename):
    ## test, whether header is provided as first row in file
    test_header = pd.read_csv(filename, sep="\s+", nrows=0)
    test_header = [col for col in test_header.columns]
    # if header contains 'NaN', it is probably data
    use_header = None
    renaming = ['datetime_[]', 'elapsed_time_[s]', 'frame_dt_[s]', 'body_x', 'body_y', 'angle', 'major', 'minor']
    if 'NaN' in test_header:
        skipr = 0
    else:
        skipr = 1
    ## load data
    raw_data = pd.read_csv(filename, sep="\s+", header=use_header, skiprows=skipr) ### TODO: HEADER for centroid data
    new_datetimes = raw_data[0]
    new_datetimes = pd.to_datetime(new_datetimes)
    raw_data[0] = new_datetimes
    # renaming columns with standard header
    if use_header is None:
        raw_data.columns = renaming
    return raw_data

def gaussian_filter(_df, _len=16, _sigma=None):
    cols = np.empty((len(_df.index), len(_df.columns)))
    cols.fill(np.nan)
    if _sigma is None:
        _sigma = _len/10
    header = []
    for column in _df:
        header.append(column)
        cols[:,len(header)-1] = gaussian_filtered(_df[column], _len=_len, _sigma=_sigma)
    return pd.DataFrame(cols, columns=header)

def gaussian_filtered(_X, _len=16, _sigma=1.6):
    norm = np.sqrt(2*np.pi)*_sigma ### Scipy's gaussian window is not normalized
    window = signal.gaussian(_len+1, std=_sigma)/norm
    return np.convolve(_X, window, "same")

def translate_to(data, filename):
    filestart = np.loadtxt(allfiles['timestart'], dtype=bytes).astype(str)
    filestart = datetime.datetime.strptime(filestart[1][:19], '%Y-%m-%dT%H:%M:%S')
    ##filestart = filestart - datetime.timedelta(hours=1, seconds=5) ## needed for test/01
    mask = (data['datetime_[]'] > filestart)
    data = data.loc[mask]
    return data, data.index[0]


####PLOTTING test
def plotting(data, flies, vidfile):
    import imageio
    vid = imageio.get_reader(vidfile,  'ffmpeg')
    dur = 1800 ### = 1 minute
    f, axes = plt.subplots(15, 4, figsize=(45,135))
    start, end = first_frame, len(data[0].index)
    chosen_frames = np.random.choice(np.arange(start, end), 60, replace=False)
    counter = 0

    for row in axes:
        for ax in row:
            index = chosen_frames[counter]
            image = vid.get_data(index)
            secs = data[0].loc[index,'elapsed_time_[s]'] - data[0].loc[first_frame,'elapsed_time_[s]']
            ax.set_title('frame #{}, {:6.2f} secs'.format(index, secs), fontsize=32)
            ax.imshow(image)
            for fly in flies:
                x = data[fly]['body_x']
                y = data[fly]['body_y']
                oridg = np.degrees(data[fly]['angle'])
                orird = data[fly]['angle']
                major = data[fly]['major']
                minor = data[fly]['minor']
                #ax.plot(x0.loc[start:start+dur], y0.loc[start:start+dur], 'k-', lw=0.2)#, markersize=0.75)
                #e = mpatches.Ellipse((x.loc[index], y.loc[index]), major.loc[index], minor.loc[index], angle=oridg[index], edgecolor="#6bf9b5", lw=0.2, facecolor='none', alpha=0.4)
                #ax.plot((x.loc[index], x.loc[index]+0.5*major.loc[index]*np.cos(orird.loc[index])),(y.loc[index], y.loc[index]+0.5*major.loc[index]*np.sin(orird.loc[index])),'w-', lw=0.2, alpha=0.4)
                ax.plot(x.loc[index]+0.5*major.loc[index]*np.cos(orird.loc[index]), y.loc[index]+0.5*major.loc[index]*np.sin(orird.loc[index]), marker='.', markersize=1, color="#6bf9b5", alpha=0.4)
                #ax.add_artist(e)
                """
                try:
                    ax.plot(x.loc[index], y.loc[index], color='m', marker='.', markersize=1)
                except KeyError:
                    print("Start point is flagged")
                """
            counter +=1

    return f, axes



if __name__ == '__main__':

    ## Define full directory path and filenames
    tk.Tk().withdraw()
    ##openfilename = filedialog.askopenfilename()
    #openfilename = "/Users/degoldschmidt/Google Drive/PhD Project/Tracking Analysis/tracking test/data/02/cam01_2017-11-21T14_10_06.avi"
    folder = "E:/Dennis/Google Drive/PhD Project/Experiments/001-DifferentialDeprivation/data/raw/"
    videos = "E:/Dennis/Google Drive/PhD Project/Experiments/001-DifferentialDeprivation/data/videos/"
    tfile = 2
    for each in range(tfile,tfile+1):
        openfilename = folder + "{:02d}".format(each) +'/'
        print(openfilename)
        if '.avi' in openfilename:
            basedir = os.path.dirname(openfilename)
            filetimestamp = os.path.basename(openfilename).split('.')[0].split('_')[1]+os.path.basename(openfilename).split('.')[0][-6:-3]
        else:
            basedir = openfilename
            filetimestamp = [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile][0].split('.')[0].split('_')[2]+[os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile][0].split('.')[0][-6:-3]
        allfiles = get_files(filetimestamp, basedir, videos)
        filenames = allfiles['data']

        alldata = []
        skip = 0

        colors = ["#ff7744", "#77ff77", "#7777ff", "#ffaa00"]
        labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
        f, ax = plt.subplots(2, sharex=True)
        skip = 0 # TODO fix this ---> fly
        only = 1
        for ix, eachfile in enumerate(filenames[skip:only]):
            print("Start anaylzing fly {} ({})...".format(ix, labels[ix]) )
            ### 1) Load raw data file
            raw_data = load_data(eachfile)
            ### 2) Check frame skips
            skips, bskips = get_frame_skips(raw_data)
            ### 3) Get start timestamp --> startpos
            raw_data, first_frame = translate_to(raw_data, allfiles['timestart'])
            index = np.arange(0, 2000, 100) + first_frame##args.index ## from args
            index = index[:10]
            ### 4) discontinuities
            frames = np.array(raw_data.index)
            raw_data['dx'] = np.append(0.,np.diff(raw_data['body_x']))
            raw_data['dy'] = np.append(0.,np.diff(raw_data['body_y']))
            dx, dy = raw_data['dx'], raw_data['dy']
            raw_data['distance'] = pd.Series(np.sqrt( dx*dx + dy*dy ), index=raw_data.index)
            ### 5) check orientation
            filtered = gaussian_filter(raw_data.loc[:,['dx', 'dy', 'distance', 'angle']], _len=30)
            mov_dir = np.arctan2(filtered['dy'], filtered['dx'])
            diffthr = 0.5
            alignm = np.cos(mov_dir - filtered['angle'])[filtered.query('distance > '+str(diffthr)).index]  ### positive if aligned
            wlen = 300 # 10 seconds window
            if alignm.shape[0] < wlen:
                wlen = alignm.shape[0]
            if not np.isnan(np.mean(alignm)):
                rmean = np.convolve(alignm, np.ones((wlen,))/wlen, mode='same')
            else:
                rmean = alignm

            ### get pixel values for head & tail
            print("Start pixel intensity algorithm...")
            signs = get_signs(raw_data, allfiles['video'], space=300)
            print("Done.")
            raw_data.loc[signs==-1, 'angle'] += np.pi

            for jx in index:
                ax[0].vlines(jx, np.amin(rmean), np.amax(rmean), colors='r', linestyles='dashed')
            ax[0].plot(filtered.query('distance > '+str(diffthr)).index, rmean, '-', color=colors[ix], label="fly " + str(ix+1))
            ax[0].plot(raw_data.index, signs, '--', color=colors[ix], label="fly " + str(ix+1))
            ax[0].set_ylabel('Alignment r. avg')
            ax[0].set_xlim([first_frame,len(raw_data.index)])
            ax[1].plot(filtered['distance'], '--', color=colors[ix], label="fly " + labels[ix])
            ax[1].plot(raw_data['distance'], '-', color=colors[ix], label="fly " + labels[ix])
            ax[1].set_ylabel('Displacement [px]')
            ax[1].set_xlabel('#frame')
            ax[1].set_xlim([first_frame,len(raw_data.index)])
            raw_data = get_disconts(raw_data)
            for each in range(skip):
                alldata.append([])
            alldata.append(raw_data)

        #for ix, df in enumerate(alldata):
            #df.to_csv(os.path.join(basedir, filenames[ix].split('.')[0]+'_cleaned.csv'), sep=' ')
        """
        for eachfile in filenames:
            filename = os.path.join(basedir, eachfile.split('.')[0]+'_cleaned.csv')
            alldata.append(pd.read_csv(filename, header=None))

        first_frame = alldata[0].index[0]
        print(first_frame)
        """
        plt.legend()
        plt.savefig(os.path.join(basedir,filetimestamp+'_alignment.pdf'), dpi=600)
        filename = allfiles['video']

        import imageio
        vid = imageio.get_reader(filename,  'ffmpeg')
        start = first_frame
        end = alldata[0].index[-1]
        chosen_frames = np.random.choice(np.arange(start, end), 5, replace=False)
        for ix in chosen_frames:
            f, ax = plt.subplots()
            image = vid.get_data(ix)
            secs = alldata[0].loc[ix,'elapsed_time_[s]'] - alldata[0].loc[first_frame,'elapsed_time_[s]']
            plt.title('frame #{}, {:6.2f} secs'.format(ix, secs), fontsize=32)
            fly = 0
            x = alldata[fly]['body_x']
            y = alldata[fly]['body_y']
            oridg = np.degrees(alldata[fly]['angle'])
            orird = alldata[fly]['angle']
            major = alldata[fly]['major']
            minor = alldata[fly]['minor']
            headx, heady = x.loc[ix]+0.5*major.loc[ix]*np.cos(orird.loc[ix]), y.loc[ix]+0.5*major.loc[ix]*np.sin(orird.loc[ix])
            tailx, taily = x.loc[ix]-0.5*major.loc[ix]*np.cos(orird.loc[ix]), y.loc[ix]-0.5*major.loc[ix]*np.sin(orird.loc[ix])


            headpx = get_patch_average(image, (headx, heady))
            tailpx = get_patch_average(image, (tailx, taily))
            pixeldiff = tailpx - headpx
            de = 20
            x0 = int(x[ix]-de)
            x1 = int(x[ix]+de)
            y0 = int(y[ix]+de)
            y1 = int(y[ix]-de)


            ax.imshow(image)
            ax.set_xlim([x0,x1])
            ax.set_ylim([y0,y1])
            #ax.plot(x0.loc[start:start+dur], y0.loc[start:start+dur], 'k-', lw=0.2)#, markersize=0.75)
            e = mpatches.Ellipse((x.loc[ix], y.loc[ix]), major.loc[ix], minor.loc[ix], angle=oridg[ix], edgecolor="#6bf9b5", lw=2, facecolor='none', alpha=0.4)
            #ax.plot((x.loc[ix], x.loc[ix]+0.5*major.loc[ix]*np.cos(orird.loc[ix])),(y.loc[ix], y.loc[ix]+0.5*major.loc[ix]*np.sin(orird.loc[ix])),'w-', lw=0.2, alpha=0.4)
            ax.plot(headx, heady, marker='.', markersize=20, color="#f9e26b", alpha=0.7)
            ax.plot(tailx, taily, marker='.', markersize=10, color="#f9e26b", alpha=0.7)
            ax.add_artist(e)
        plt.show()

        #f, axes = plotting(alldata, [0,1,2,3], filename)
        #plt.savefig(os.path.join(basedir,filetimestamp+'_random_orient.pdf'), dpi=600)
