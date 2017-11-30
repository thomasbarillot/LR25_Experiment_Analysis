# mpirun -n 2 python mpi_driver.py  exp=sxrh4315:run=174 -n 10
# in batch:
# bsub -q psanaq -n 2 -o %J.log -a mympi python mpi_driver.py exp=sxrh4315:run=174

#from master import runmaster
from XTCExporter import runclient

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#assert size>1, 'At least 2 MPI ranks required'
#numClients = size-1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exprun", help="psana experiment/run string (e.g. exp=amoj1516:run=43)")
parser.add_argument("-n","--noe",help="number of events, all events=0",default=-1, type=int)

args = parser.parse_args()


###############################################################################
# Variable input
# Detectors:
args.SHES   = 'OPAL3'
args.UXS    = 'OPAL1'
args.ITOF   = 'ACQ1'#'Acq01'#'ACQ1'
args.GMD    = 'FEEGasDetEnergy'
args.EBeam  = 'EBeam'
args.PRESS  = 'AMO:LMP:VG:21:PRESS'
args.XTCAV  = 'xtcav'

# GMD intensity threshold for shots to be considered for online plots:
#args.gmdThr = 0.1
# Prefered gmd value for online plots:
#args.prefgmd = 0

# Prefered  channel for online plots and raw traces of itof:
#args.prefch = 1

# Save files:
args.save = True
# Number of shots saved per file
args.nsave  = 2000
# number of shots transfered for online plots:
#args.nonline = 120

# cfd parameters (for itof):
#args.cfdBaseline  = 0
#args.cfdThreshold = [0.0075,0.02]
#args.cfdFraction  = 0.5
#args.cfdDeadtime  = 20

# Rebinning of raw MBES traces to save to files:
#args.rebin = False
#args.bins  = 2500
#args.crop =[0,0]

# Bins for pump and probe intensity out of XTCAV
#args.minIPump = 0
#args.maxIPump = 11
#args.minIProbe = 0
#args.maxIProbe = 11

# Pump and probe intensity histograms:
#args.Ihistmin = 0
#args.Ihistmax = 3
#args.Ihistbin = 0.1

# Delay histogram:
#args.Delayhistmin = 0
#args.Delayhistmax = 2
#args.Delayhistbin = 0.05

# Detection mode:
#args.Elmode = True
# Maximum electron kinetic energy in eV:
#args.maxEkin = 300
# Minimum electron kinetic energy in eV:
#args.minEkin = 0.2
# Bins in eV:
#args.ebins = 0.2
# Time-energy conversion factors [t0,conv,offs]:
#args.elconv = [1.24E-6,6.23E-12,0]
# Maximum ion m/z:
#args.maxM = 65
# Minimum ion m/z:
#args.minM = 1
# m/z bins:
#args.ibins = 0.2
# Time-m/z conversion factors [t0,conv]:
#args.iconv = [1.24E-6,6.23E-12]

# Make online plots:
#args.onlineplot = True


###############################################################################

#if rank==0:
#    runmaster(args,numClients)
#else:
runclient(args)

MPI.Finalize()
#if rank == 0:
print 'mpidriver done'
