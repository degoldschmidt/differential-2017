import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('DIFF', 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db()) # database from file

    # initialize modules of the pipeline

    ### GO THROUGH SESSIONS
    for each in db.sessions[:1]:
        df, meta = each.load()
        kine = Kinematics(df, meta)
        print(kine.df.columns)
        print(each.name+'\t'+str(meta['datetime'])+'\t'+str(kine.df.shape))
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
