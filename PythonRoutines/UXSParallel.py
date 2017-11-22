"""
Script to accumulate OPAL images from the UXS

Processes XTC-data using the MPI parallellization.

Runs a gaussian filter on the data, then a threshold and finally accumulates it.

Saves output to MPIsumimage1337.out # Where 1337 is runnr

Launch with:
bsub -n 64 -o myoutput%J.log -q psanaq mpirun python UXSParallel.py 

"""

# Core imports
from psana import *
import numpy as np
import scipy.ndimage as filter

# UXS Data processing
from UXSDataPreProcessing import UXSDataPreProcessing 

# Set the runnr here to also change the savefile
runnr = 30
ds = DataSource('exp=amolr2516:run={}:smd:dir=/reg/d/ffb/amo/amolr2516/xtc:live'.format(runnr))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# UXS Detector
opal_det = Detector('OPAL1')

# Create image for summation
summedimage = np.zeros((1024,1024))
#summedspectrum = np.zeros(1024)

for nevent,evt in enumerate(ds.events()):
    if nevent%size!=rank:
        # Different ranks look at different events
        continue 
    print "Rank {} is working on evt: {}".format(rank,nevent)
    opal_raw = opal_det.raw(evt)
    if opal_raw is None:
        print("Bad event")
        continue
    # Got the data
    #opal = opal_raw.copy()
    opal = opal_raw

    # Run a gaussian filter on the image
    opal = filter.gaussian_filter(opal,sigma=1,order=0)
    
    # Lets add a threshold
    # Everything below 100 becomes 0
    threshold = 100
    idx = opal[:,:] < threshold
    opal[idx] = 0

    # Project into spectrum
    #uxspre = UXSDataPreProcessing(opal)
    #uxspre.CalculateProjection()

    ## Accumulate the data by summing
    summedimage += opal
    #summedspectrum += uxspre.wf

# Sum the data from all nodes into the final image.
totalsum = np.empty_like(summedimage)
comm.Reduce(summedimage, totalsum)

print 'Rank',rank,'sum:',summedimage.sum()
if rank==0:
    print 'Total sum:',totalsum.sum()
    #plotimg = Image(0,"UXS Summed Image",totalsum)
    #publish.send('UXSIMAGESUM',plotimg) 
    print "Saving"
    np.savetxt('MPIsumimage{}.out'.format(runnr), totalsum)

MPI.Finalize()
