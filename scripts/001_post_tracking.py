### imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox, filedialog
import warnings

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
Returns angle between to given points centered on pt1
"""
def get_angle(pt1, pt2):
    dx = pt2[0]-pt1[0]
    dy = pt2[1]-pt1[1]
    return np.arctan2(dy,dx)

"""
Returns arguments from CLI
"""
def get_args():
    ### parsing arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-from', action='store', dest='start_session', help='Store a start session')
    parser.add_argument("-to", action='store', dest='end_session', help='Store a end session')
    parser.add_argument("-fly", action='store', dest='flies', help='Store a end session')
    return parser.parse_args()

"""
Returns list of raw data for filenames
"""
def get_data(filenames):
    print("Loading raw data...", flush=True, end="")
    renaming = ['datetime []', 'elapsed_time [s]', 'frame_dt [s]', 'body_x [px]', 'body_y [px]', 'angle [rad]', 'major [px]', 'minor [px]']
    data = []
    for each in filenames:
        ## load data
        data.append(pd.read_csv(each, sep="\s+", skiprows=1))
        # renaming columns with standard header
        data[-1].columns = renaming
        # datetime strings to datetime objects
        data[-1]['datetime []'] =  pd.to_datetime(data[-1]['datetime []'])
    print("done.")
    return data

"""
Returns raw and video folder of the experiment
"""
def get_dir(my_system, exp):
    ##openfilename = filedialog.askopenfilename()
    #openfilename = "/Users/degoldschmidt/Google Drive/PhD Project/Tracking Analysis/tracking test/data/02/cam01_2017-11-21T14_10_06.avi"
    if my_system == 'nt':
        system_folder = "E:/Dennis/Google Drive/PhD Project/Experiments/"
    else:
        system_folder = "/Users/degoldschmidt/Google Drive/PhD Project/Experiments/"
    if exp == "DIFF":
        experiment_folder = os.path.join(system_folder, "001-DifferentialDeprivation/")

    raw_folder = os.path.join(experiment_folder, "data/raw/")
    manual_folder = os.path.join(experiment_folder, "data/manual/")
    if my_system == 'nt':
        video_folder = os.path.join(experiment_folder, "data/videos/")
        num_videos = len([eachavi for eachavi in os.listdir(video_folder) if eachavi.endswith("avi")])
    else:
        video_folder = ""
        count = 0
        for dirs in os.listdir(raw_folder):
            this_dir = os.path.join(raw_folder,dirs)
            if os.path.isdir(this_dir):
                count += (len(os.listdir(this_dir)) >= 10)
        num_videos = count
    print("Start analysis for experiment: {} [{}]".format(os.path.basename(os.path.dirname(experiment_folder)), exp))
    print("Found {} videos in experiment folder.".format(num_videos))
    return experiment_folder, raw_folder, video_folder, manual_folder, num_videos

"""
Returns distance between to given points
"""
def get_distance(pt1, pt2):
    dx = pt1[0]-pt2[0]
    dy = pt1[1]-pt2[1]
    return np.sqrt(dx**2 + dy**2)

"""
Returns dictionary of all raw data files
"""
def get_files(timestampstring, basedir, video, noVideo=False):
    if noVideo:
        return {
                        "data" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "fly" in eachfile and timestampstring in eachfile],
                        "food" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "food" in eachfile and timestampstring in eachfile],
                        "geometry" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "geometry" in eachfile and timestampstring in eachfile][0],
                        "timestart" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile and timestampstring in eachfile][0],
        }
    else:
        return {
                    "data" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "fly" in eachfile and timestampstring in eachfile],
                    "food" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "food" in eachfile and timestampstring in eachfile],
                    "geometry" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "geometry" in eachfile and timestampstring in eachfile][0],
                    "timestart" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile and timestampstring in eachfile][0],
                    "video" : [os.path.join(video, eachfile) for eachfile in os.listdir(video) if timestampstring in eachfile][0],
                    }

"""
Returns food spots data
"""
def get_food(filenames):
    data = []
    for each in filenames:
        eachdata = np.loadtxt(each)
        if len(eachdata.shape) == 1:
            eachdata = np.reshape(eachdata, (1, 2))
        ## load data
        data.append(eachdata)
    return data

"""
Returns frame dimensions as tuple (height, width, channels)
"""
def get_frame_dims(filename):
    warnings.filterwarnings("ignore")
    import skvideo.io
    videogen = skvideo.io.vreader(filename)
    for frame in videogen:
        dims = frame.shape
        break
    warnings.filterwarnings("default")
    return dims

"""
Returns list of raw data for filenames
"""
def get_geom(filename):
    print("Loading geometry data...", flush=True, end="")
    renaming = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'O1', 'L1', 'M1', 'O2', 'L2', 'M2', 'O3', 'L3', 'M3',  'O4', 'L4', 'M4']
    ## load data
    data = pd.read_csv(filename, sep="\s+")
    data.columns = renaming
    data['R1'] = 0.25*(data['L1']+data['M1'])  ### radius = half of mean of major and minor
    data['R2'] = 0.25*(data['L2']+data['M2'])  ### radius = half of mean of major and minor
    data['R3'] = 0.25*(data['L3']+data['M3'])  ### radius = half of mean of major and minor
    data['R4'] = 0.25*(data['L4']+data['M4'])  ### radius = half of mean of major and minor
    data = data.loc[:, ['X1', 'Y1', 'R1', 'X2', 'Y2', 'R2', 'X3', 'Y3', 'R3', 'X4', 'Y4', 'R4']]
    print("done.")
    return [(data.loc[len(data.index)-1, 'X'+str(ix+1)], data.loc[len(data.index)-1, 'Y'+str(ix+1)], data.loc[len(data.index)-1, 'R'+str(ix+1)]) for ix in range(4)]

"""
Returns metadata from session folder
"""
def get_meta(allfiles, dtstamp):
    print("Metadata:")
    print("")

"""
Returns new spots
"""
def get_new_spots(spots, mindist, left=0):
    new_spot = []
    for each_spot in spots:
        mat = rot(120, in_degrees=True)
        vec = np.array([each_spot[0], each_spot[1]])
        out = np.dot(mat, vec)
        x, y = out[0], out[1]
        dists = [get_distance((spot[0], spot[1]), (x,y)) for spot in spots]
        for each_distance in dists:
            if each_distance < mindist:
                mat = rot(-120, in_degrees=True)
                vec = np.array([each_spot[0], each_spot[1]])
                out = np.dot(mat, vec)
                x, y = out[0], out[1]
        new_spot.append([x, y])
        if left > 1:
            mat = rot(-120, in_degrees=True)
            vec = np.array([each_spot[0], each_spot[1]])
            out = np.dot(mat, vec)
            x, y = out[0], out[1]
            new_spot.append([x, y])
    print(new_spot)
    return new_spot

"""
Returns number of frames from raw data
"""
def get_num_frames(data):
    assert (all(len(x.index) == len(data[0].index) for x in data)), "All files should have same number of frames!"
    return len(data[0].index)

"""
Returns datetime for session start
"""
def get_session_start(filename):
    from datetime import datetime
    filestart = np.loadtxt(filename, dtype=bytes).astype(str)
    return datetime.strptime(filestart[1][:19], '%Y-%m-%dT%H:%M:%S')

"""
Returns timestamp from session folder
"""
def get_time(session):
    from datetime import datetime
    for each in os.listdir(session):
        if "timestart" in each:
            any_file = each
    timestampstr = any_file.split('.')[0][-19:]
    dtstamp = datetime.strptime(timestampstr, "%Y-%m-%dT%H_%M_%S")
    return dtstamp, timestampstr[:-3]

"""
Returns list of directories in given path d with full path
"""
def flistdir(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

"""
Returns rotation matrix for given angle in two dimensions
"""
def rot(angle, in_degrees=False):
    if in_degrees:
        rads = np.radians(angle)
        return np.array([[np.cos(rads), -np.sin(rads)],[np.sin(rads), np.cos(rads)]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

"""
Plot figs along program flow
"""
def plot_along(f, ax):
    warnings.filterwarnings("ignore")
    mng = plt.get_current_fig_manager()
    ### works on Ubuntu??? >> did NOT working on windows
# mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed') #works fine on Windows!
    f.show()
    try:
        f.canvas.start_event_loop(0)
    except tk.TclError:
        pass
    warnings.filterwarnings("default")

"""
Plot figs along program flow
"""
def plot_overlay(vidfile, arenas=[], spots=[]):
    import imageio
    from matplotlib.patches import Circle
    vid = imageio.get_reader(vidfile,  'ffmpeg')
    f, ax = plt.subplots()
    image = vid.get_data(0)
    ax.imshow(image)

    fix_outer = 260
    for each in arenas:
        x = each[0]
        y = each[1]
        r = each[2] + 30
        circ_outer = Circle((x, y), radius=fix_outer, alpha=0.25, color="#ff45cb")
        ax.add_artist(circ_outer)
        circ = Circle((x, y), radius=r, alpha=0.4, color="#0296a4")
        ax.add_artist(circ)
        for rad in [30., 120., 212.5]:
            circ = Circle((x, y), radius=rad, ec="k", fc='none', ls='dashed')
            ax.add_artist(circ)
        ax.plot(x, y, marker='+', markersize=10, color="#0296a4")
    for ix, each_arena in enumerate(spots):
        x0 = arenas[ix][0]
        y0 = arenas[ix][1]
        colors = ["#ff8d17", "#ff4500", "#009cff"]
        for each_spot in each_arena:
            x = each_spot[0] + x0
            y = each_spot[1] + y0
            spot_radius = 12.815
            patch = Circle((x, y), radius=spot_radius, alpha=0.4, color=colors[int(each_spot[2])])
            ax.add_artist(patch)
    return f, ax

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

"""
Validate food spots
"""
def validate_food(spots, geom):
    fix_outer = 260
    spot_radius = 12.815
    for ix, each_arena in enumerate(spots):
        print("\nArena:", ix)
        p0 = geom[ix][0]        # arena center
        f0 = p0 - fix_outer     # frame origin

        all_spots = []

        n_inner, n_outer = 3, 3 # counter for inner and outer spots found

        ### spots for inner and outer triangle
        inner, outer = [], []

        for each_spot in each_arena:
            spot = each_spot - p0 + f0  ### vector centered around arena center
            d = get_distance((0, 0), spot)
            a = get_angle((0, 0), spot)

            ### validate existing spots
            if d > 30. and d < 212.5:
                all_spots.append([spot[0], spot[1], 0])
                if d < 120.:        ### inner spots
                    n_inner -= 1
                    inner.append([spot[0], spot[1], a])
                    print('near:', spot[0], spot[1], d, np.degrees(a))
                else:               ### outer spots
                    n_outer -= 1
                    outer.append([spot[0], spot[1], a])
                    print('far:', spot[0], spot[1], d, np.degrees(a))
            ### removal
            else:
                print("Removed: ", spot[0], spot[1], d)

        ### Check whether existing inner spots are of right number
        if n_inner < 0:
            print("Too many inner spots. Removing {} spots.".format(-n_inner))
            min_dis = 260.
            ### remove as many spots as needed based on avg distance to all spots
            for each in range(-n_inner):
                for ispot, spot in enumerate(inner):
                    mean_dis = np.mean([get_distance((spot[0], spot[1]), (other[0], other[1])) for other in inner])
                    if mean_dis < min_dis:
                        min_dis = mean_dis
                        remove = ispot
                all_spots.remove([inner[ispot][0], inner[ispot][1], 0])
                del inner[ispot]

        ### Check for translation (needs to be after removal of extra spots)
        if len(inner) == 3:
            tx, ty = 0, 0
            for each_spot in inner:
                tx += each_spot[0]
                ty += each_spot[1]
            tx /= len(inner)
            ty /= len(inner)
        if len(inner) < 3:
            tx, ty = 0, 0
            for each_spot in inner:
                dr = get_distance((0, 0), (each_spot[0], each_spot[1])) - 85.43 #### This is 10 mm
                tx += dr * np.cos(get_angle((0, 0), (each_spot[0], each_spot[1])))
                ty += dr * np.sin(get_angle((0, 0), (each_spot[0], each_spot[1])))
            tx /= len(inner)
            ty /= len(inner)
        print("Translation detected: ({}, {})".format(tx, ty))
        ### Correcting for translation
        for spot in inner:
            spot[0] -= tx
            spot[1] -= ty
        for spot in outer:
            spot[0] -= tx
            spot[1] -= ty
        for spot in all_spots:
            spot[0] -= tx
            spot[1] -= ty

        if n_inner > 0:
            print("Too few inner spots. Missing {} spots.".format(n_inner))
            ### getting new spots by means of rotation
            near_new = get_new_spots(inner, 80, left=n_inner)
            ### overlapping spots get averaged to one
            if get_distance(near_new[0], near_new[1]) < 40.:
                near_new = [[0.5*(near_new[0][0] + near_new[1][0]), 0.5*(near_new[0][1] + near_new[1][1])]]
            ### add new one to list of all spots
            for spot in near_new:
                all_spots.append([spot[0], spot[1], 1])

        ### Check whether existing outer spots are of right number
        if n_outer < 0:
            print("Too many outer spots. Removing {} spots.".format(-n_outer))
            min_dis = 260.
            ### remove as many spots as needed based on avg distance to all spots
            for each in range(-n_outer):
                for ispot, spot in enumerate(outer):
                    mean_dis = np.mean([get_distance((spot[0], spot[1]), (other[0], other[1])) for other in inner])
                    if mean_dis < min_dis:
                        min_dis = mean_dis
                        remove = ispot
                all_spots.remove([inner[ispot][0], inner[ispot][1], 0])
                del inner[ispot]

        if n_outer > 0:
            print("Too few outer spots. Missing {} spots.".format(n_outer))
            far_new = get_new_spots(outer, 250, left=n_outer)
            ### overlapping spots get averaged to one
            if get_distance(far_new[0], far_new[1]) < 40.:
                far_new = [[0.5*(far_new[0][0] + far_new[1][0]), 0.5*(far_new[0][1] + far_new[1][1])]]
            ### add new one to list of all spots
            for spot in far_new:
                all_spots.append([spot[0], spot[1], 1])

        ### Adding sucrose by rotating yeast positions
        sucrose = []
        for spot in all_spots:
            mat = rot(60, in_degrees=True)
            vec = np.array([spot[0], spot[1]])
            out = np.dot(mat, vec)
            sucrose.append([out[0], out[1], 2])
        # add them all to list
        for each_spot in sucrose:
            all_spots.append(each_spot)

        ### return all spots for this arena into list
        spots[ix] = np.array(all_spots)
    print()
    return spots


"""
"""
"""
MAIN
"""
"""
"""
if __name__ == '__main__':
    args = get_args()
    ## define full directory path and filenames
    tk.Tk().withdraw()
    EXP_ID = "DIFF"
    exp, raw, video, manual, n = get_dir(os.name, EXP_ID)

    ### go through SESSIONS
    start, end = sessions_from_args(args)
    for each_session in range(start,end+1):
        ### get timestamp and all files from session folder
        session_folder = os.path.join(raw, "{:02d}".format(each_session))
        print("\nStart post-tracking analysis for session: {:02d}".format(each_session))
        dtstamp, timestampstr = get_time(session_folder)
        print("Timestamp:", dtstamp.strftime("%A, %d. %B %Y %H:%M"))
        allfiles = get_files(timestampstr, session_folder, video, noVideo = (os.name != "nt"))

        ### these are the raw data files
        filenames = allfiles['data']
        ### get data from files as list of DataFrames
        raw_data = get_data(filenames)
        geom_data = get_geom(allfiles['geometry'])
        food_data = get_food(allfiles['food'])
        food_data = validate_food(food_data, geom_data)
        f, ax = plot_overlay(allfiles['video'], arenas=geom_data, spots=food_data)
        plot_along(f, ax)



        ### getting metadata
        meta = {}
        meta['arena_geom'] = geom_data
        meta['datadir'] = os.path.join(exp.split('/')[-2], "each_session")
        meta['experiment'] = EXP_ID
        dims = get_frame_dims(allfiles["video"])
        meta['frame_height'] = dims[0]
        meta['frame_width'] = dims[1]
        meta['frame_channels'] = dims[2]
        meta['num_frames'] = get_num_frames(raw_data)
        meta['session_start'] = get_session_start(allfiles["timestart"])
        meta['video'] = os.path.basename(allfiles["video"])
        meta['video_start'] = dtstamp

        ### get conditions files
        meta["files"] = flistdir(manual)
        meta["conditions"] =  [os.path.basename(each).split('.')[0] for each in meta["files"]]
        meta["variables"] = []
        for ix, each in enumerate(meta["files"]):
            with open(each, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    meta["variables"].append(meta["conditions"][ix])
                else:
                    meta[meta["conditions"][ix]] = lines[0]

        #print(meta)
        #meta_dict = get_meta(allfiles, dtstamp)
        ### SESSION FILE NAME



        """ PLOTTING ALONG USE
        f, ax = plt.subplots()
        ax.plot(np.arange(40), 'r-')
        plot_along(f, ax)
        """

        for ix, eachfile in enumerate(filenames):
            flymeta = meta.copy()
            flyid = ix + (each_session-1)*4 + 1
            print("{}_{:03d}.csv".format(EXP_ID, flyid))
            flymeta['datafile'] = "{}_{:03d}.csv".format(EXP_ID, flyid)

        """
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
