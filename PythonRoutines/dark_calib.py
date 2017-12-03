import argparse
from xtcav.GenerateDarkBackground import *
parser = argparse.ArgumentParser()
parser.add_argument(
    'bg_run', help='The run to use as the background')
parser.add_argument(
    'valid_start', help='First run for which the reference will be valid')
parser.add_argument(
    'valid_end', help='Last run for which the reference will be valid')
args = parser.parse_args()

GDB = GenerateDarkBackground()
GDB.experiment = 'amolr2516'
GDB.runs = str(args.bg_run)
GDB.maxshots = 2000
# delete second run number argument to have the validity range be open-ended ("end")
if args.valid_end == -1:
    GDB.SetValidityRange(args.valid_start)
else:
    GDB.SetValidityRange(args.valid_start, args.valid_end)
GDB.Generate()
