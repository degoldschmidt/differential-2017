import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
#from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
#from pytrack_analysis import Statistics
from pytrack_analysis import Multibench
import logging

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Differential 2017', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file

    print("Process kinematics...", end="\t\t\t", flush=True)
    kinematics = Kinematics(db)
    kinematics.run_many(db.sessions[:1], _VERBOSE=True)
    print("[DONE]")
    del kinematics
    del db

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
