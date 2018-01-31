import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
#from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
#from pytrack_analysis import Statistics
from pytrack_analysis import Multibench
from pytrack_analysis.cli import colorprint, flprint, prn
import logging

### plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('DIFF', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    #print("Process kinematics...", end="\t\t\t", flush=True)
    #kinematics = Kinematics(db)
    #for ix in range(1,288):
        #outdir = get_out(profile)
        #data = kinematics.run(db.session(ix))

    #del kinematics
    #del db
    #plt.show()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
