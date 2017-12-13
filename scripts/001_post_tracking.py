from pytrack_analysis import Multibench
from pytrack_analysis.cli import get_args
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile

def main():
    args = get_args(['exp', 'exp', 'Select experiment by four-letter ID'], ['u', 'user', 'Select user'], SILENT=False)
    profile = get_profile(args.exp, args.user, script=get_scriptname(__file__))
    #show_profile(profile)
    print(profile)

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
