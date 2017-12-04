import argparse
from xtcav.GenerateLasingOffReference import *
parser = argparse.ArgumentParser()
parser.add_argument(
    'ref_run', help='The run to use as the reference')
parser.add_argument(
    'valid_start', help='First run for which the reference will be valid')
parser.add_argument(
    'valid_end', help='Last run for which the reference will be valid')
args = parser.parse_args()

GLOC = GenerateLasingOffReference()
GLOC.experiment = 'amolr2516'
GLOC.runs = str(args.ref_run)
GLOC.maxshots = 2000
GLOC.nb = 1
# see confluence documentation for how to set this parameter
GLOC.islandsplitmethod = 'scipyLabel'
# see confluence documentation for how to set this parameter
GLOC.groupsize = 5
# delete second run number argument to have the validity range be open-ended ("end")
GLOC.SetValidityRange(args.valid_start, args.valid_end)
GLOC.Generate()
