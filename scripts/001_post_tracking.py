### imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox, filedialog

def get_distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0]) + (pt1[1]-pt2[1])*(pt1[1]-pt2[1]))

def get_disconts(data, thr_disc=80., thr_coh=0.001):
    data['flags'] = pd.Series(np.zeros(len(data.index)), index=data.index)
    data['last_index'] = pd.Series(np.zeros(len(data.index)), index=data.index)
    data['last_distance [px]'] = pd.Series(np.zeros(len(data.index)), index=data.index)
    flag = 0
    for index, row in data.iterrows():
        if row['distance [px]'] > thr_disc or flag == 1:
            if index == 0:
                print('First frame breaker!')
            elif flag == 0:
                data.loc[index, 'flags'] = 1
                flag = 1
                data.loc[index, 'last_index'] = index-1
            else:
                data.loc[index, 'last_index'] = data['last_index'].loc[index-1]
                this_last = data.loc[index, 'last_index']
                dist = get_distance((data.loc[index, 'body_x [px]'], data.loc[index, 'body_y [px]']), (data.loc[this_last, 'body_x [px]'], data.loc[this_last, 'body_y [px]']))
                data.loc[index, 'last_distance [px]'] = dist
                if dist < thr_disc or row['distance [px]'] < thr_coh:
                    data.loc[index, 'flags'] = 0
                    flag = 0
                else:
                    data.loc[index, 'flags'] = 1
    return data

def get_frame_skips(data, println=False):
    secs = data['elapsed_time [s]']
    frameskips = data['frame_dt [s]']
    total = frameskips.shape[0]
    strict_skips = np.sum(frameskips > (1/30)+(1/30))
    easy_skips = np.sum(frameskips > (1/30)+(1/3))
    if println:
        print('Skips of more than 1 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/30), strict_skips, 100*strict_skips/total))
        print('Skips of more than 10 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/3), easy_skips, 100*easy_skips/total))
    return strict_skips, easy_skips

def load_data(filename):
    ## test, whether header is provided as first row in file
    test_header = pd.read_csv(filename, sep="\s+", nrows=0)
    test_header = [col for col in test_header.columns]
    # if header contains 'NaN', it is probably data
    use_header = None
    renaming = ['datetime []', 'elapsed_time [s]', 'frame_dt [s]', 'body_x [px]', 'body_y [px]', 'angle [rad]', 'major [px]', 'minor [px]']
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

def translate_to(data, filename):
    filestart = np.loadtxt(allfiles['timestart'], dtype=bytes).astype(str)
    filestart = datetime.datetime.strptime(filestart[1][:19], '%Y-%m-%dT%H:%M:%S')
    ##filestart = filestart - datetime.timedelta(hours=1, seconds=5) ## needed for test/01
    mask = (data['datetime []'] > filestart)
    data = data.loc[mask]
    return data, data.index[0]


####PLOTTING test
def plotting(data, indices, vidfile):
    import imageio
    vid = imageio.get_reader(vidfile,  'ffmpeg')
    dur = 1800 ### = 1 minute
    start = first_frame
    vidframe = start
    f, axes = plt.subplots(15, 4, figsize=(15,45))
    for row in axes:
        for ax in row:
            if start == first_frame:
                print(data[0].loc[start:start+dur].head(1))
            image = vid.get_data(start+dur)
            secs = data[0].loc[start,'elapsed_time [s]'] - data[0].loc[first_frame,'elapsed_time [s]']
            ax.set_title('frame #{}, {:2.0f} - {:2.0f} mins'.format(start, secs/60, (secs/60) + 1), fontsize=10)
            ax.imshow(image)
            for index in indices:
                x0 = data[index].query('flags == 0')['body_x [px]']
                y0 = data[index].query('flags == 0')['body_y [px]']
                x1 = data[index].query('flags == 1')['body_x [px]']
                y1 = data[index].query('flags == 1')['body_y [px]']
                ax.plot(x0.loc[start:start+dur], y0.loc[start:start+dur], 'k-', lw=0.2)#, markersize=0.75)
                ax.plot(x1.loc[start:start+dur], y1.loc[start:start+dur], color='r', marker='.', markersize=1)
                try:
                    ax.plot(x0.loc[start], y0.loc[start], color='m', marker='.', markersize=1)
                except KeyError:
                    print("Start point is flagged")
            start += dur
    return f, axes

"""
Returns raw and video folder of the experiment
"""
def get_dir(my_system):
    ##openfilename = filedialog.askopenfilename()
    #openfilename = "/Users/degoldschmidt/Google Drive/PhD Project/Tracking Analysis/tracking test/data/02/cam01_2017-11-21T14_10_06.avi"
    if my_system == 'nt':
        experiment_folder = "E:/Dennis/Google Drive/PhD Project/Experiments/001-DifferentialDeprivation/"
    else:
        experiment_folder = "/Users/degoldschmidt/Google Drive/PhD Project/Experiments/001-DifferentialDeprivation/"
    raw_folder = os.path.join(experiment_folder, "data/raw/")
    video_folder = os.path.join(experiment_folder, "data/videos/")
    num_videos = len([eachavi for eachavi in os.listdir(video_folder) if eachavi.endswith("avi")])
    print("Start analysis for experiment: {}".format(os.path.basename(os.path.dirname(experiment_folder))))
    print("Found {} videos in experiment folder.".format(num_videos))
    return raw_folder, video_folder, num_videos

"""
Returns dictionary of all raw data files
"""
def get_files(timestampstring, basedir, video):
    return {
                    "data" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "fly" in eachfile and timestampstring in eachfile],
                    "food" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "food" in eachfile and timestampstring in eachfile],
                    "geometry" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "geometry" in eachfile and timestampstring in eachfile][0],
                    "timestart" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile and timestampstring in eachfile][0],
                    "video" : [os.path.join(video, eachfile) for eachfile in os.listdir(video) if timestampstring in eachfile][0],
    }


"""
Returns timestamp from session folder
"""
def get_time(session):
    from datetime import datetime
    any_file = os.listdir(session)[0]
    timestampstr = any_file.split('.')[0][-19:-3]
    dtstamp = datetime.strptime(timestampstr, "%Y-%m-%dT%H_%M")
    return dtstamp, timestampstr

"""
Print dictionary prettier
"""
def print_files(_dict, count=0):
    print("FILES:")
    count += 1
    for k, v in _dict.items():
        if type(v) is str or type(v) is int or type(v) is float:
            print("\t{}:\t{}".format(k,v))
        elif type(v) is dict:
            print("\t{}:\t{}".format(k,print_dict(v, count=count+1)))
        elif type(v) is list:
            print("\t{}:\t{}".format(k, v[0]))
            for elem in v[1:]:
                print("\t\t", elem)
    print("")

"""
Returns sessions start and end
"""
def sessions_from_args(args):
    if args.start_session is None:
        start = 1
    else:
        start = int(args.start_session)
    if args.end_session is None:
        end = n
    else:
        end = int(args.end_session)
    return start, end

if __name__ == '__main__':
    ### parsing arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-from', action='store', dest='start_session', help='Store a start session')
    parser.add_argument("-to", action='store', dest='end_session', help='Store a end session')
    parser.add_argument("-fly", action='store', dest='flies', help='Store a end session')
    args = parser.parse_args()

    ## define full directory path and filenames
    tk.Tk().withdraw()
    raw, video, n = get_dir(os.name)

    ### go through SESSIONS
    start, end = sessions_from_args(args)
    for each_session in range(start,end+1):
        ### get timestamp and all files from session folder
        session_folder = os.path.join(raw, "{:02d}".format(each_session))
        print("\nStart post-tracking analysis for session: {:02d}".format(each_session))
        dtstamp, timestampstr = get_time(session_folder)
        print("Timestamp:", dtstamp.strftime("%A, %d. %B %Y %H:%M"))
        allfiles = get_files(timestampstr, session_folder, video)

        ### these are the raw data files
        filenames = allfiles['data']

        ### getting metadata
        meta_dict = get_meta(timestampstr, dtstamp)

        alldata = []
        skip = 0
        for ix, eachfile in enumerate(filenames[skip:]):
            print(basedir, filetimestamp, eachfile)
            ### 1) Load raw data file
            raw_data = load_data(eachfile)

            ### 2) Check frame skips
            skips, bskips = get_frame_skips(raw_data)

            ### 3) Get start timestamp --> startpos
            raw_data, first_frame = translate_to(raw_data, allfiles['timestart'])

            ### 4) discontinuities
            frames = np.array(raw_data.index)
            dx = np.diff(raw_data['body_x [px]'])
            dy = np.diff(raw_data['body_y [px]'])
            diff = np.append(0., np.sqrt( dx*dx + dy*dy ))
            raw_data['distance [px]'] = pd.Series(diff, index=raw_data.index)
            diff = raw_data['distance [px]']

            ### 5) check orientation
            #mov_dir = np.append(0., np.arctan2(dy,dx))
            #orient = raw_data["angle [rad]"]
            #alignm = np.cos(mov_dir - orient)  ### positive if aligned
            #print("mean alignment: ", np.mean(alignm))
            #plt.plot(alignm, 'k-')
            #plt.show()


            raw_data = get_disconts(raw_data)
            for each in range(skip):
                alldata.append([])
            alldata.append(raw_data)

        #for ix, df in enumerate(alldata):
            #df.to_csv(os.path.join(basedir, filenames[ix].split('.')[0]+'_cleaned.csv'), sep=' ')

        for eachfile in filenames:
            filename = os.path.join(basedir, eachfile.split('.')[0]+'_cleaned.csv')
            alldata.append(pd.read_csv(filename, header=None))

        first_frame = alldata[0].index[0]
        print(first_frame)

        filename = allfiles['video']
        f, axes = plotting(alldata, [0,1,2,3], filename)
        plt.savefig(os.path.join(basedir,filetimestamp+'_trajs.pdf'), dpi=600)
        """
