import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
#from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
#from pytrack_analysis import Statistics
from pytrack_analysis import Multibench
import logging

### plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Differential 2017', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    print("Process kinematics...", end="\t\t\t", flush=True)
    kinematics = Kinematics(db)
    for ix in range(1,288):
        outdir = get_out(profile)
        data = kinematics.run(db.session(ix))
        spots = db.sessions[ix].metadata['food_spots']
        spot_radius = db.sessions[ix].metadata['food_spot_radius']
        pxmm = 1/db.sessions[ix].metadata['px_per_mm']
        f, ax = plt.subplots()
        for each in spots:
            if each['substrate'] == '10% yeast':
                col = "#ff6600"
            if each['substrate'] == '20 mM sucrose':
                col = "#2222ff"
            circ = Circle((each['x'], each['y']), radius=spot_radius*pxmm, color=col, alpha=0.5)
            ax.add_artist(circ)
        #ax.plot(data['body_x'], data['body_y'], 'k-')
        ax.plot(data['head_x'], data['head_y'], 'k-', lw=0.75)
        ax.set_aspect('equal')
        ax.set_xlim([-260*pxmm,260*pxmm])
        ax.set_ylim([-260*pxmm,260*pxmm])
        ax.set_title('DIFF_'+"{:03d} ({})".format(ix, db.sessions[ix].metadata['metabolic']))
        plt.savefig(os.path.join(outdir, 'DIFF_'+"{:03d}".format(ix+1)+'.png'), dpi=600)

    #f, ax = plt.subplots()
    #ax.plot(data.index, data['smooth_head_speed'], 'k-')
    #ax.set_ylim([0,20])



    print("[DONE]")



    del kinematics
    del db
    #plt.show()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
