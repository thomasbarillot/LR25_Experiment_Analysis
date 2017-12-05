"""
Script that processes UXSData for calibration purposes.

Saves that as python pickle. See bottom.

Launch with:
bsub -n 64 -o myoutput%J.log -q psanaq mpirun python UXSParallel.py --run 1337

"""
# Core imports
import sys
import argparse
import pickle
from psana import *
import numpy as np
import scipy.ndimage as filter

# Helper function
# Find nearest index in numpyarray
def findnearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# UXS Data processing
from UXSDataPreProcessing import UXSDataPreProcessing 
from SHESPreProcessing import SHESPreProcessor

# Set the runnr here to also change the savefile
parser = argparse.ArgumentParser()
parser.add_argument("--run", help="Run number", required=True)
args = parser.parse_args()

runnr = args.run
ds = DataSource('exp=amolr2516:run={}:smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live'.format(runnr))
#ds = DataSource('exp=amolr2516:run={}:smd:dir=/reg/d/ffb/amo/amolr2516/xtc:live'.format(runnr))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

# UXS Detector
opal_det = Detector('OPAL1')

# Ebeam to get photon energy
ebeam = Detector('EBeam')

# Getting the FEE gas detector
feedet = Detector('FEEGasDetEnergy')

# Create image for summation
summedimage = np.zeros((1024,1024))
thissummedspectrum = np.arange(1024)

minenergy = 3200
maxenergy = 3500

l3bins = np.arange(minenergy,maxenergy,1)
nl3bins = len(l3bins)
shotsperl3 = np.zeros(nl3bins)
totalintensity = np.zeros(nl3bins)

# L3 Map
thisl3map = np.zeros((1024,nl3bins))

# L3 SHES
thisl3shes = np.zeros((714,nl3bins))

# SHES Map
minSHES = 0
maxSHES = 1024
nSHESbins = 1000
SHESBins = np.linspace(minSHES,maxSHES,nSHESbins)
thisSHESMap = np.zeros((1024, nSHESbins))
shotsperSHES = np.zeros(nSHESbins)

# SHESUXSMap
# Make a map of the SHES with UXS fit on other axis
minSHESUXS = 0
maxSHESUXS = 1024
nSHESUXSbins = 1024
SHESUXSBins = np.linspace(minSHESUXS,maxSHESUXS,nSHESUXSbins)
thisSHESUXSMap = np.zeros((714, nSHESUXSbins))
shotsperSHESUXS = np.zeros(nSHESUXSbins)

# UXSUXSMap
nUXSUXSbins = 1024
UXSUXSBins = np.arange(nUXSUXSbins)
thisUXSUXSMap = np.zeros((1024, nUXSUXSbins))
shotsperUXSUXS = np.zeros(nUXSUXSbins)

# Make a histogram with distance between both plots
thisPulseDistanceHistogram = np.zeros((1024,1024))
shotpershotSeparation = np.zeros(1024)

for nevent,evt in enumerate(ds.events()):
    if nevent%size!=rank:
        # Different ranks look at different events
        continue
    #if nevent > 10000:
    #    break
    print "Rank {}, {}, is working on evt: {}".format(rank, hostname, nevent)
    opal_raw = opal_det.raw(evt)
    ebeamdata = ebeam.get(evt)
    feedata = feedet.get(evt)
    if ebeamdata is None:
        continue
    if feedata is None:
        continue
    if opal_raw is None:
        continue

    # Get SHES data
    try:
        SHES = SHESPreProcessor()
        SHESdata = SHES.PreProcess(evt)
    except:
        continue
    
    # Got the data
    opal = opal_raw
    uxspre = UXSDataPreProcessing()
    opal = uxspre.FilterImage(opal_raw)
    #uxspre.MaskImage(xmin=0, xmax=1024, ymin=480, ymax=505)

    # Divide image by Gas Detectors
    feeenergy = feedata.f_11_ENRC()/2+feedata.f_12_ENRC()/2
    opal = opal/feeenergy
    [pos1, sigma1, int1, pos2, sigma2, int2], unfilteredwf, wf, energyscale = uxspre.StandardAnalysis(opal, returnmore=True)
    if np.isnan(pos1):
        continue
    if sigma1 > 50:
        continue
    if pos1<0 or pos1>1024:
        continue
    if not np.isnan(pos1) and not np.isnan(pos2):
        pos1pixel = findnearest(energyscale, pos1)
        pos2pixel = findnearest(energyscale, pos2)
        thisPulseDistanceHistogram[pos1,pos2] += 1
        separation = np.abs(pos1-pos2)
        separationpixel = findnearest(energyscale, separation)
        shotpershotSeparation[separationpixel] += 1

    spectrum = wf #unfilteredwf
    ## Accumulate the data by summing
    summedimage += opal
    thissummedspectrum += spectrum
    # Make L3Map
    l3energy = ebeamdata.ebeamL3Energy()
    binpos = findnearest(l3bins, l3energy)
    thisl3map[:,binpos] +=  spectrum/feeenergy
    totalintensity[binpos] += np.sum(opal)
    shotsperl3[binpos] += 1

    # Make L3Shes
    binpos = findnearest(l3bins, l3energy)
    thisl3shes[:,binpos] +=  SHESdata[2]/feeenergy

    # Make SHESmap
    SHESmoment1, SHESmoment2 = SHES.Moments(evt, inPix=True)
    binpos = findnearest(SHESBins, SHESmoment1)
    thisSHESMap[:,binpos] += spectrum
    shotsperSHES[binpos] += 1

    # Make SHESUXSMap
    binpos = findnearest(SHESUXSBins, pos1)
    thisSHESUXSMap[:,binpos] += SHESdata[2]/feeenergy
    shotsperSHESUXS[binpos] += 1

    # Make UXSUXSMap
    binpos = findnearest(UXSUXSBins, pos1)
    thisUXSUXSMap[:,binpos] += spectrum/feeenergy
    shotsperUXSUXS[binpos] += 1


print "Rank {}, {}, is done! Reducing!".format(rank, hostname)
# Sum the data from all nodes into the final image.
totalsum = np.zeros_like(summedimage)
comm.Reduce(summedimage, totalsum, root=0)

# all totalintensity
print "Reducing some small stuff"
totaltotalintensity = np.zeros_like(totalintensity)
comm.Reduce(totalintensity, totaltotalintensity, root=0)

totalshotsperl3 = np.zeros_like(shotsperl3)
comm.Reduce(shotsperl3, totalshotsperl3, root=0)

totalshotsperSHES = np.zeros_like(shotsperSHES)
comm.Reduce(shotsperSHES, totalshotsperSHES, root=0)

print "Reducing L3 map"
# all l3 map
alll3map = np.zeros_like(thisl3map)
comm.Reduce(thisl3map, alll3map, root=0)

print "Reducing L3 SHES"
# all l3 SHES
alll3shes = np.zeros_like(thisl3shes)
comm.Reduce(thisl3shes, alll3shes, root=0)

print "Reducing Spectrum"
# all total spectrum
allsummedspectrum = np.zeros_like(thissummedspectrum)
comm.Reduce(thissummedspectrum, allsummedspectrum, root=0)

print "Reducing SHES"
# all SHES map
allSHESMap = np.zeros_like(thisSHESMap)
comm.Reduce(thisSHESMap, allSHESMap, root=0)

print "Reducing SHESUXS"
# all SHESUXs map
allSHESUXSMap = np.zeros_like(thisSHESUXSMap)
comm.Reduce(thisSHESUXSMap, allSHESUXSMap, root=0)

print "Reducing UXSSUXS"
# all UXSSUXs map
allUXSUXSMap = np.zeros_like(thisUXSUXSMap)
comm.Reduce(thisUXSUXSMap, allUXSUXSMap, root=0)

print "Reducing Pulse distance histogram"
# All Pulse distance histogram
allPulseDistanceHistogram = np.zeros_like(thisPulseDistanceHistogram)
comm.Reduce(thisPulseDistanceHistogram, allPulseDistanceHistogram, root=0)

print "Reducing small stuff"
totalshotsperUXSUXS = np.zeros_like(shotsperUXSUXS)
comm.Reduce(shotsperUXSUXS, totalshotsperUXSUXS, root=0)

totalshotsperSHESUXS = np.zeros_like(shotsperSHESUXS)
comm.Reduce(shotsperSHESUXS, totalshotsperSHESUXS, root=0)

totalshotpershotSeparation = np.zeros_like(shotpershotSeparation)
comm.Reduce(shotpershotSeparation, totalshotpershotSeparation, root=0)

print 'Rank',rank,'sum:',summedimage.sum()
if rank==0:
    savedata = {'AccumulatedImage': totalsum,
                'AccumulatedSpectrum': allsummedspectrum,
                'L3Bins': l3bins,
                'ShotsPerL3': totalshotsperl3,
                'ShotsPerSHES': totalshotsperSHES,
                'ShotsPerSHESUXS': totalshotsperSHESUXS,
                'SHESUXSMap': allSHESUXSMap,
                'TotalYield': totaltotalintensity,
                'L3SHES': alll3shes, 
                'L3Map': alll3map, 
                'SHESMap': allSHESMap,
                'UXSUXSMap': allUXSUXSMap,
                'ShotsPerUXSUXS': totalshotsperUXSUXS,
                'PulsePosHistogram': allPulseDistanceHistogram,
                'ShotPerShotSeparationHistogram': totalshotpershotSeparation}
    print 'Total sum:',totalsum.sum()
    print "Saving"
    with open('MPIrun{}.p'.format(runnr), 'wb') as f:
        pickle.dump(savedata, f)

MPI.Finalize()
