""" 
Online monitoring of the UXS spectrometer at LR25
Data is accessed using shared memory
"""
import time
from psana import *
import numpy as np
import scipy.ndimage as filter

from UXSDataPreProcessing import UXSDataPreProcessing

ds = DataSource('shmem=psana.0:stop=no')
print "Connected to shmem"
print DetNames()
opal_det = Detector('OPAL1')

# Visualization
from psmon.plots import Image,XYPlot
from psmon import publish

# Accumulate
numframes = 2400 # number of frames to accumulate
# Keep numframes in a 3D matrix
images = np.zeros((1024, 1024, numframes))
spectra = np.zeros((1024, numframes))

frameidx = 0 # The nextidx to replace in the circular image buffer

summedimage = np.zeros((1024,1024))

start = time.time()
for nevt, evt in enumerate(ds.events()):
    evttime = evt.get(EventId).idxtime().time()
    opal_raw = opal_det.raw(evt)
    if opal_raw is None:
        print "Opal_raw was None"
        continue
    uxspre = UXSDataPreProcessing(opal_raw)
    uxspre.FilterImage()
    debug = False
    if debug:
        uxspre.AddFakeImageSignal(center=200, curvature=450)
        uxspre.AddFakeImageSignal(center=600, curvature=450)
        plotimg = Image(0, "UXS Monitor uncorrected {} {}".format(frameidx, str(evt.get(EventId))), uxspre.image)
        publish.send("UXSMonitorDebug", plotimg)
    uxspre.CorrectImageGeomerty()
    uxspre.CalculateProjection()
    if debug:
        plotimg = Image(0, "UXS Monitor image {} {}".format(frameidx, str(evt.get(EventId))), uxspre.image)
        publish.send("UXSMonitor", plotimg)
    # Make the accumulation # TODO make sure this we don't get errors with this method due to floats
    summedimage -= images[:,:,frameidx]
    images[:,:,frameidx] = uxspre.image
    summedimage += images[:,:,frameidx]
    #summedimage = np.sum(images,axis=2) # Alternate slower method
    
    # Project into a spectrum
    spectra[:,frameidx] = uxspre.wf
    # Sum all accumulated spectra
    summedspectrum = np.sum(spectra,axis=1)
   
    # Only publish every 10th frame
    if frameidx%10 == 0:
        # Calculate our processing speed
        speed = int(10/(time.time()-start))
        start = time.time()
        plotimg = Image(0, "UXS Monitor image {} {}Hz {}".format(frameidx, speed, str(evt.get(EventId))), summedimage)
        publish.send("UXSMonitor", plotimg)
        
        plotxy = XYPlot(0, "UXS Monitor Spectrum {}".format(str(evt.get(EventId))), range(1024), summedspectrum)
        publish.send("UXSMonitorSpectrum", plotxy)
 
        # Count rate evolution over the frames.
        #counts = np.sum(images, axis=(0,1))
        #counts = np.roll(counts, -frameidx)
        #plotxy = XYPlot(0, "UXS Counts {}".format(str(evt.get(EventId))), range(numframes), counts)
        #publish.send("UXSMonitorCounts", plotxy)
 
    # Iterate framenumber
    frameidx += 1
    frameidx = frameidx%numframes
        
