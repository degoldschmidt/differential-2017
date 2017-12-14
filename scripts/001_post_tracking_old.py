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
from datetime import datetime, timedelta
from string import Template
import imageio
from pytrack_analysis import get_angle

"""
Returns raw and video folder of the experiment (DATAIO)
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
    out_folder = os.path.join(experiment_folder, "data/processed/post_tracking")
    if my_system == 'nt':
        video_folder = os.path.join(experiment_folder, "data/videos/")
        num_videos = len([eachavi for eachavi in os.listdir(video_folder) if eachavi.endswith("avi")])
    else:
        video_folder = "/Volumes/DATA_BACKUP/data/tracking/all_videos/"
        num_videos = len([eachavi for eachavi in os.listdir(video_folder) if eachavi.endswith("avi")])
    print("Start analysis for experiment: {} [{}]".format(os.path.basename(os.path.dirname(experiment_folder)), exp))
    print("Found {} videos in experiment folder.".format(num_videos))
    return experiment_folder, raw_folder, video_folder, manual_folder, out_folder, num_videos

"""
Returns discontinuities flags (TODO: old, write new function) (PROCESSING)
"""
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



"""
Returns food spots data (DATAIO)
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
Returns frame dimensions as tuple (height, width, channels) (DATAIO)
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
Returns list of raw data for filenames (DATAIO)
"""
def get_geom(filename):
    print("Loading geometry data...", flush=True, end="")
    renaming = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'O1', 'L1', 'M1', 'O2', 'L2', 'M2', 'O3', 'L3', 'M3',  'O4', 'L4', 'M4']
    ## load data
    data = pd.read_csv(filename, sep="\s+")
    data.columns = renaming
    data['R1'] = 0.25*(data['L1']+data['M1']) + 30 ### radius = half of mean of major and minor
    data['R2'] = 0.25*(data['L2']+data['M2']) + 30 ### radius = half of mean of major and minor
    data['R3'] = 0.25*(data['L3']+data['M3']) + 30 ### radius = half of mean of major and minor
    data['R4'] = 0.25*(data['L4']+data['M4']) + 30 ### radius = half of mean of major and minor
    data = data.loc[:, ['X1', 'Y1', 'R1', 'X2', 'Y2', 'R2', 'X3', 'Y3', 'R3', 'X4', 'Y4', 'R4']]
    print("done.")
    return [(data.loc[len(data.index)-1, 'X'+str(ix+1)], data.loc[len(data.index)-1, 'Y'+str(ix+1)], data.loc[len(data.index)-1, 'R'+str(ix+1)]) for ix in range(4)]

"""
Returns metadata from session folder (DATAIO)
"""
def get_meta(allfiles, dtstamp, conditions):
    meta = {}

    ### Arena geometry
    geom_data = get_geom(allfiles['geometry'])

    ### Food spots
    food_data = get_food(allfiles['food'])
    food_data = validate_food(food_data, geom_data)
    food_dict = {}
    labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
    for kx, each_arena in enumerate(food_data):
        food_dict[labels[kx]] = {}
        for ix, each_spot in enumerate(each_arena):
            food_dict[labels[kx]][str(ix)] = {}
            labs = ['x', 'y', 'substrate']
            substrate = ['10% yeast', '20 mM sucrose']
            for jx, each_pt in enumerate(each_spot):
                if jx == 2:
                    if int(each_pt) < 2:
                        food_dict[labels[kx]][str(ix)][labs[jx]] = substrate[0]
                    if int(each_pt) == 2:
                        food_dict[labels[kx]][str(ix)][labs[jx]] = substrate[1]
                else:
                    food_dict[labels[kx]][str(ix)][labs[jx]] = float(each_pt)

    meta['food_spots'] = food_dict
    ###
    dict_geom = {}
    for index, each_arena in enumerate(geom_data):
        dict_geom[labels[index]] = {}
        labels2 = ['x', 'y', 'r']
        for j, each_pt in enumerate(each_arena):
            dict_geom[labels[index]][labels2[j]] = float(each_pt)
    meta['arena_geom'] = dict_geom
    dims = get_frame_dims(allfiles["video"])
    meta['frame_height'] = dims[0]
    meta['frame_width'] = dims[1]
    meta['frame_channels'] = dims[2]
    meta['session_start'] = get_session_start(allfiles["timestart"])
    meta['video'] = os.path.basename(allfiles["video"])
    meta['video_start'] = dtstamp
    ### get conditions files
    meta["files"] = flistdir(conditions)
    meta["conditions"] =  [os.path.basename(each).split('.')[0] for each in meta["files"]]
    meta["variables"] = []
    for ix, each in enumerate(meta["files"]):
        with open(each, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                meta["variables"].append(meta["conditions"][ix])
            else:
                meta[meta["conditions"][ix]] = lines[0]
    return meta

"""
Returns new spots (DATAIO)
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
    return new_spot

"""
Returns number of frames from raw data (DATAIO)
"""
def get_num_frames(data):
    assert (all(len(x.index) == len(data[0].index) for x in data)), "All files should have same number of frames!"
    return len(data[0].index)

"""
Returns datetime for session start (DATAIO)
"""
def get_session_start(filename):
    from datetime import datetime
    filestart = np.loadtxt(filename, dtype=bytes).astype(str)
    return datetime.strptime(filestart[1][:19], '%Y-%m-%dT%H:%M:%S')

"""
Returns flags to flip the orientation (PROCESSING)
"""
def get_signs(data, vidfile, only=None, space=300):
    warnings.filterwarnings("ignore")
    vid = imageio.get_reader(vidfile)#,  'ffmpeg')
    signs = np.zeros(data['angle'].shape)
    x = data['body_x']
    y = data['body_y']
    orird = data['angle']
    major = data['major']
    next_one = 0
    percent = 0
    if only is None:
        end = 10000 #len(data.index)
        for jx in range(0, end, space):
            if jx > next_one:
                print("{} %, frame {}".format(percent, jx))
                percent += 20
                next_one += end/5
            frame = data.index[jx]
            image = vid.get_data(frame)
            signs[jx] = get_sign(image, frame, x, y, major, orird)
            if jx > 0 and space != 1:
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
    else:
        sumsign = 0
        if len(only) > 1:
            if space < only[1]-only[0]:
                next_one = space
            else:
                next_one = only[1]-only[0]
            for i in range(next_one):
                frame = data.index[0]
                image = vid.get_data(frame)
                sumsign += get_sign(image, frame, x, y, major, orird)
        else:
            next_one = space
            for i in range(next_one):
                frame = data.index[0]
                image = vid.get_data(frame)
                sumsign += get_sign(image, frame, x, y, major, orird)
        signs[0:only[0]-1] = np.sign(sumsign/space)
        for i,each in enumerate(only):
            if i < len(only)-1:
                if space < only[i+1]-only[i]:
                    next_one = space
                else:
                    next_one = only[i+1]-only[i]
            sumsign = 0
            for i in range(next_one):
                frame = data.index[each+1+i]
                image = vid.get_data(frame)
                sumsign += get_sign(image, frame, x, y, major, orird)
            if i < len(only)-1:
                signs[each:only[i+1]-1] = np.sign(sumsign/space)
            else:
                signs[each:] = np.sign(sumsign/space)
    vid.close()
    return signs

### Helper for get_signs function
def get_patch_average(image, pt, radius=1):
    pxls = []
    for x in range(-radius, radius+1):
        yr = radius-abs(x)
        for y in range(-yr, yr+1):
            pxls.append(image[int(pt[1])+y, int(pt[0])+x, 0])
    return np.mean(np.array(pxls))

### Helper for get_signs function
def get_sign(image, frame, x, y, major, orird):
    headx, heady = x.loc[frame]+0.5*major.loc[frame]*np.cos(orird.loc[frame]), y.loc[frame]+0.5*major.loc[frame]*np.sin(orird.loc[frame])
    tailx, taily = x.loc[frame]-0.5*major.loc[frame]*np.cos(orird.loc[frame]), y.loc[frame]-0.5*major.loc[frame]*np.sin(orird.loc[frame])
    headpx = get_patch_average(image, (headx, heady))
    tailpx = get_patch_average(image, (tailx, taily))
    pixeldiff = tailpx - headpx
    return np.sign(pixeldiff)





"""
Plot figs along program flow (VISUAL)
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
Plot figs overlay  (VISUAL)
"""
def plot_overlay(vidfile, arenas=[], spots=[]):
    import imageio
    from matplotlib.patches import Circle, Ellipse
    vid = imageio.get_reader(vidfile, 'ffmpeg')
    f, ax = plt.subplots()
    image = vid.get_data(0)
    ax.imshow(image)

    fix_outer = 260
    for each in arenas:
        x = each[0]
        y = each[1]
        r = each[2]
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
    vid.close()
    return f, ax

"""
Print dictionary prettier  (CLI)
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
Returns sessions start and end (CLI)
"""
def sessions_from_args(args, n):
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
Prints timedelta
"""
class DeltaTemplate(Template):
    delimiter = "%"
def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:06.3f}'.format(seconds + tdelta.microseconds/1000000)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)

"""
Returns translated data for given session start (PROCESSING)
"""
def translate_to(data, start, time=''):
    mask = (data[time] > start)
    data = data.loc[mask]
    return data, data.index[0]

"""
Validate food spots (DATAIO)
"""
def validate_food(spots, geom, VERBOSE=False):
    print("Loading food spots...", flush=True, end="")
    fix_outer = 260
    spot_radius = 12.815
    pxmm = 8.543
    for ix, each_arena in enumerate(spots):
        if VERBOSE: print("Loading food spots for arena {}...".format(ix), flush=True, end="")
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
                    if VERBOSE: print('inner:', spot[0], spot[1], d, np.degrees(a))
                else:               ### outer spots
                    n_outer -= 1
                    outer.append([spot[0], spot[1], a])
                    if VERBOSE: print('outer:', spot[0], spot[1], d, np.degrees(a))
            ### removal
            else:
                if VERBOSE: print("Removed: ", spot[0], spot[1], d)

        ### Check whether existing inner spots are of right number
        if n_inner < 0:
            #print("Too many inner spots. Removing {} spots.".format(-n_inner))
            min_dis = fix_outer
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
                dr = get_distance((0, 0), (each_spot[0], each_spot[1])) - 10.*pxmm #### This is 10 mm
                tx += dr * np.cos(get_angle((0, 0), (each_spot[0], each_spot[1])))
                ty += dr * np.sin(get_angle((0, 0), (each_spot[0], each_spot[1])))
            tx /= len(inner)
            ty /= len(inner)
        if VERBOSE: print("Translation detected: ({}, {})".format(tx, ty))
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
            if VERBOSE: print("Too few inner spots. Missing {} spots.".format(n_inner))
            ### getting new spots by means of rotation
            near_new = get_new_spots(inner, 80, left=n_inner)
            ### overlapping spots get averaged to one
            if get_distance(near_new[0], near_new[1]) < 40.:
                near_new = [[0.5*(near_new[0][0] + near_new[1][0]), 0.5*(near_new[0][1] + near_new[1][1])]]
            ### add new one to list of all spots
            for spot in near_new:
                inner.append([spot[0], spot[1], get_angle((0,0), spot)])
                all_spots.append([spot[0], spot[1], 1])

        ### Check whether existing outer spots are of right number
        if n_outer < 0:
            if VERBOSE: print("Too many outer spots. Removing {} spots.".format(-n_outer))
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
            if VERBOSE: print("Too few outer spots. Missing {} spots.".format(n_outer))
            if n_outer < 3:
                far_new = get_new_spots(outer, 250, left=n_outer)
                ### overlapping spots get averaged to one
                if get_distance(far_new[0], far_new[1]) < 40.:
                    far_new = [[0.5*(far_new[0][0] + far_new[1][0]), 0.5*(far_new[0][1] + far_new[1][1])]]
                ### add new one to list of all spots
                for spot in far_new:
                    all_spots.append([spot[0], spot[1], 1])
            else:
                for spot in inner:
                    x, y, a = spot[0], spot[1], spot[2]
                    nx = x + 10.*pxmm * np.cos(a)
                    ny = y + 10.*pxmm * np.sin(a)
                    mat = rot(90, in_degrees=True)
                    vec = np.array([nx, ny])
                    out = np.dot(mat, vec)
                    all_spots.append([out[0], out[1], 1])

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
        if VERBOSE: print("found", len(spots[ix]), "food spots.")
    print("found a total of {} spots.".format(sum([len(each) for each in spots])))
    return spots

"""
Writes dict into json file
"""
def write_yaml(_file, _dict):
    import io, yaml
    """ Writes a given dictionary '_dict' into file '_file' in YAML format. Uses UTF8 encoding and no default flow style. """
    with io.open(_file, 'w+', encoding='utf8') as outfile:
        yaml.dump(_dict, outfile, default_flow_style=False, allow_unicode=True)


"""

    MAIN FUNCTION

"""
def main(args):
    ## define full directory path and filenames
    #tk.Tk().withdraw()
    EXP_ID = "DIFF"
    exp, raw, video, manual, output, n = get_dir(os.name, EXP_ID)

    ### go through SESSIONS
    start, end = sessions_from_args(args, n)
    if args.not_this is not None:
        list_not = [int(each) for each in args.not_this.split(',')]
    else:
        list_not = []
    print("Not analyzing", list_not)
    for each_session in range(start,end+1):
        if each_session in list_not:
            continue
        ### get timestamp and all files from session folder
        session_folder = os.path.join(raw, "{:02d}".format(each_session))
        print("\nStart post-tracking analysis for session: {:02d}".format(each_session))
        dtstamp, timestampstr = get_time(session_folder)
        print("Timestamp:", dtstamp.strftime("%A, %d. %B %Y %H:%M"))
        allfiles = get_files(timestampstr, session_folder, video)

        ### these are the raw data files
        filenames = allfiles['data']
        ### get data from files as list of DataFrames
        raw_data, data_units = get_data(filenames)
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

        #print(meta)
        #meta_dict = get_meta(allfiles, dtstamp)
        ### SESSION FILE NAME


        for ix, eachfile in enumerate(filenames):
            flymeta = meta.copy()
            flyid = ix + (each_session-1)*4 + 1
            print("Condition:", conditions.loc[each_session, conditions.columns[ix]])
            for each_condition in meta['variables']:
                flymeta[each_condition] = conditions.loc[each_session, conditions.columns[ix]]
            #print("{}_{:03d}.csv".format(EXP_ID, flyid))
            flymeta['datafile'] = "{}_{:03d}.csv".format(EXP_ID, flyid)
            flymeta['metafile'] = "{}_{:03d}.yaml".format(EXP_ID, flyid)
            ### 2) Check frame skips
            skips, bskips, maxskip, max_skip_arg = get_frame_skips(raw_data[ix])
            flymeta['frameskips'] = int(skips)
            flymeta['maxskip'] = float(maxskip)
            flymeta['maxskip_index'] = int(max_skip_arg)
            labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
            flymeta['arena'] = labels[ix]
            flymeta['arena_geom'] = flymeta['arena_geom'][labels[ix]]
            flymeta['px_per_mm'] = 2.*flymeta['arena_geom']['r'] / 49.75
            flymeta['food_spots'] = flymeta['food_spots'][labels[ix]]
            flymeta['food_spot_radius'] = 12.815

            ### 3) Get start timestamp --> startpos
            raw_data[ix], first_frame = translate_to(raw_data[ix], flymeta['session_start'], time='datetime')
            flymeta['first_frame'] = int(first_frame)

            ### 4) discontinuities
            frames = np.array(raw_data[ix].index)
            dx = np.diff(raw_data[ix]['body_x'])
            dy = np.diff(raw_data[ix]['body_y'])
            diff = np.append(0., np.sqrt( dx*dx + dy*dy ))
            raw_data[ix]['distance'] = pd.Series(diff, index=raw_data[ix].index)
            diff = raw_data[ix]['distance']
            data_units['distance'] = 'px'

            raw_data[ix] = get_disconts(raw_data[ix])

            ### 5) detect jumps
            jumps = raw_data[ix]['distance'] > 8
            jump_ids = np.where(jumps)[0]
            if len(jump_ids) > 0:
                valid_jumps = [each for i, each in enumerate(jump_ids[:-1]) if np.diff(jump_ids)[i] > 1]
                valid_jumps.append(jump_ids[-1])

            ### 6) get pixel values for head & tail
            print("Start pixel intensity algorithm...")
            #signs = get_signs(raw_data[ix], allfiles['video'], only=valid_jumps, space=500)
            warnings.filterwarnings("default")
            print("Done.")
            #raw_data[ix].loc[signs==-1, 'angle'] += np.pi
            raw_data[ix]['head_x'] = raw_data[ix]['body_x'] + 0.5*raw_data[ix]['major'] * np.cos(raw_data[ix]['angle'])
            raw_data[ix]['head_y'] = raw_data[ix]['body_y'] + 0.5*raw_data[ix]['major'] * np.sin(raw_data[ix]['angle'])

            ### 7) center around arena center
            labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
            for each in ['head', 'body']:
                raw_data[ix][each+'_x'] = raw_data[ix][each+'_x'] - flymeta['arena_geom']['x']
                raw_data[ix][each+'_y'] = raw_data[ix][each+'_y'] - flymeta['arena_geom']['y']

            ### 8) Save data to file
            #raw_data[ix]['frame'] = raw_data[ix].index
            outdf = raw_data[ix].loc[:, ['frame_dt', 'body_x', 'body_y', 'head_x', 'head_y', 'angle', 'major', 'minor']]
            #print(outdf)
            outdf.to_csv(os.path.join(output, flymeta['datafile']), sep='\t')
            write_yaml(os.path.join(output, flymeta['metafile']), flymeta)


            #f, ax = plt.subplots(3, sharex=True)
            #colors = ["#ff7744", "#77ff77", "#7777ff", "#ffaa00"]
            #labels = ['topleft', 'topright', 'bottomleft', 'bottomright']

            """
            ax[0].plot(raw_data[ix].index, signs, '-', color=colors[ix], label="fly " + str(ix+1))
            ax[0].set_ylabel('Alignment r. avg')
            ax[0].set_xlim([first_frame,len(raw_data[ix].index)])
            ax[1].plot(raw_data[ix]['distance'], '-', color=colors[ix], label="fly " + labels[ix])
            ax[2].plot(jumps, '-', color=colors[ix], label="fly " + labels[ix])
            ax[1].set_ylabel('Displacement [px]')
            ax[1].set_xlabel('#frame')
            ax[1].set_xlim([first_frame,len(raw_data[ix].index)])
            """
            #plot_along(f, ax)
        """
        choices = np.random.choice(np.arange(first_frame, meta['num_frames']), 10)
        print(choices)
        from matplotlib.patches import Circle, Ellipse
        for each in choices:
            vid = imageio.get_reader(allfiles['video'])
            f, ax = plt.subplots()
            image = vid.get_data(each)
            ax.imshow(image)
            for fly in range(4):
                x = raw_data[fly].loc[each, "body_x"]
                y = raw_data[fly].loc[each, "body_y"]
                hx = raw_data[fly].loc[each, "head_x"]
                hy = raw_data[fly].loc[each, "head_y"]
                angle = raw_data[fly].loc[each, "angle"]
                major = raw_data[fly].loc[each, "major"]
                minor = raw_data[fly].loc[each, "minor"]
                e = Ellipse((x, y), major, minor, angle=np.degrees(angle), edgecolor="#6bf9b5", lw=1, facecolor='none', alpha=0.6)
                ax.add_artist(e)
                circ = Circle((hx, hy), radius=1, alpha=0.4, color=colors[fly])
                ax.add_artist(circ)
                ax.set_title("frame {}".format(each))
            vid.close()
        plt.show()
        """
        #for ix, df in enumerate(alldata):
            #df.to_csv(os.path.join(basedir, filenames[ix].split('.')[0]+'_cleaned.csv'), sep=' ')

        """
        for eachfile in filenames:
            filename = os.path.join(basedir, eachfile.split('.')[0]+'_cleaned.csv')
            alldata.append(pd.read_csv(filename, header=None))

        first_frame = alldata[0].index[0]
        print(first_frame)

        filename = allfiles['video']
        f, axes = plotting(alldata, [0,1,2,3], filename)
        plt.savefig(os.path.join(basedir,filetimestamp+'_trajs.pdf'), dpi=600)
        """


if __name__ == '__main__':
    startdt = datetime.now()
    args = get_args()
    main(args)
    print("Done. Runtime:", strfdelta(datetime.now() - startdt, "%H:%M:%S"))
