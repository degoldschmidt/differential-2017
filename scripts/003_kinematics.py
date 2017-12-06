import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Statistics
from pytrack_analysis import Multibench
import logging

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Differential 2017', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)

    ### Example session "CANS_005" for Fig 1C,D
    """
    print("Process kinematics...", end="\t\t\t", flush=True)
    kinematics = Kinematics(db)
    kinematics.run(this_one, _ALL=True)
    figcd = fig_1cd(db.session(this_one).data, db.session(this_one))
    print("[DONE]")
    """
    log.close()
    log.show()

    #del kinematics
    del db

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
