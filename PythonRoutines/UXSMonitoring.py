""" 
Online monitoring of the UXS spectrometer at LR25
Data is accessed using shared memory
"""
import os
import time
import thread

from psana import *
import numpy as np
import scipy.ndimage as filter

from UXSDataPreProcessing import UXSDataPreProcessing

# Visualization
from psmon.plots import Image,XYPlot, MultiPlot
from psmon import publish

# Some settings for saving data
saveIterator = 0 # Iterates the filename
savePath = "./UXSAlign/{}.out"

if not os.path.exists("./UXSAlign/"):
    raise Exception("Please create directory UXSAlign")

# Lets start a small thread for user interaction
# Save data by pressing enter
saveNow = False
def input_thread():
    global saveNow
    while True:
        raw_input()
        saveNow = True

thread.start_new_thread(input_thread, ())

ds = DataSource('shmem=psana.0:stop=no')
print "Connected to shmem"
# Testing with old data
##ds = DataSource('exp=amof6215:run=200')
print DetNames()

# UXS Camera
opal_det = Detector('OPAL1')

# Ebeam to get photon energy
ebeam = Detector('EBeam')

# Accumulate frames in a circularbuffer
numframes = 240

# Keep numframes in a 3D matrix
images = np.zeros((1024, 1024, numframes))
spectra = np.zeros((1024, numframes))

# Keep a circularbuffer of metadata
metadata = ["" for frame in range(numframes)]

# Keep index for circularbuffers
frameidx = 0

# Keep the accumulated image
accimage = np.zeros((1024,1024))

start = time.time()
print "Press enter to save data"
for nevt, evt in enumerate(ds.events()):
    # Iterate through all events
    evttime = evt.get(EventId).idxtime().time()
    
    # Reading the UXS Opal-1000
    opal_raw = opal_det.raw(evt)
    if opal_raw is None:
        print "Opal_raw was None"
        continue
    
    # Reading the EBeam data, only for help while calibrating
    ebeamdata = ebeam.get(evt)
    if ebeamdata is None:
        print "Ebeamdata was None"
        photonenergy = 0
    if ebeamdata is not None:
        # TODO Shall we change to ebeamdata.ebeamL3Energy() ?
        photonenergy = ebeamdata.ebeamPhotonEnergy()
    # Set the metadata
    metadata[frameidx] = "{}, PhEn: {}, {}".format(frameidx, photonenergy, str(evt.get(EventId)))
    
    #### Make sure orientation is correct as soon as we get first data
    opal_raw = np.rot90(opal_raw.copy())
    uxspre = UXSDataPreProcessing(opal_raw) ## TODO only copy here if needed
    #uxspre.FilterImage() # TODO change filtering
    uxspre.CalculateProjection()
    # Copy here instead of standard analysis because it cuts the .wf
    spectrum = uxspre.wf.copy()
    energyscale = uxspre.energyscale.copy()
    fitresults = uxspre.StandardAnalysis()
    debug = False
    if debug:
        # Add something to look at
        uxspre.AddFakeImageSignal(center=200, curvature=450)
        funcurve = 450+100*(frameidx/numframes)
        uxspre.AddFakeImageSignal(center=600, curvature=funcurve)
    
    
    # Make the accumulation
    # TODO make sure this don't accumulate the floating point error
    accimage -= images[:,:,frameidx]
    images[:,:,frameidx] = uxspre.image
    accimage += images[:,:,frameidx]
    #accimage = np.sum(images,axis=2) # Alternate slower method
    
    # Add spectrum to circular buffer
    spectra[:,frameidx] = spectrum
    # Sum all accumulated spectra
    accspectrum = np.sum(spectra,axis=1)
   
    # Only publish every nth frame
    npublish = 5
    if frameidx%npublish == 0:
        # Calculate our processing speed
        speed = int(npublish/(time.time()-start))
        start = time.time()
   
        # Do the standard analysis to get the double gaussian fit
        print fitresults

        # Lets also plot the fit!
        fity = np.zeros(1024)
        if len(fitresults) > 0:
            a1,c1,w1,a2,c2,w2 = fitresults
            x = np.arange(0,1024)
            fity = UXSDataPreProcessing.DoubleGaussianFunction(x,a1,c1,w1,a2,c2,w2)
            print fity

        # Send single plots
        #plotimglive = Image(0, "UXS Monitor Live image {} {}Hz {}".format(frameidx, speed, metadata[frameidx]), uxspre.image)
        #plotimgacc = Image(0, "UXS Monitor Accumulated image {} {}Hz {}".format(frameidx, speed, metadata[frameidx]), summedimage)
        #plotxylive = XYPlot(0, "UXS Monitor Live Spectrum {}".format(metadata[frameidx]), range(1024), uxspre.wf)
        #plotxyacc = XYPlot(0, "UXS Monitor Accumulated Spectrum {}".format(metadata[frameidx]), range(1024), summedspectrum)

        #publish.send("UXSMonitorLive", plotimglive)
        #publish.send("UXSMonitorAcc", plotimgacc)
        #publish.send("UXSMonitorLiveSpectrum", plotxylive)
        #publish.send("UXSMonitorAccSpectrum", plotxy)

        # Send a multiplot
        plotimglive = Image(0, "Live", uxspre.image)
        plotimgacc = Image(0, "Acc", accimage)
        plotxylive = XYPlot(0, "Live", energyscale, spectrum)
        plotxyacc = XYPlot(0, "Acc", energyscale, accspectrum)
        plotxyfit = XYPlot(0, "Fit", energyscale, fity)

        multi = MultiPlot(0, "UXSMonitor {} Hz {}".format(speed, metadata[frameidx]), ncols=2)
        multi.add(plotimglive)
        multi.add(plotimgacc)
        multi.add(plotxylive)
        multi.add(plotxyacc)
        multi.add(plotxyfit)
        publish.send("UXSMonitor", multi)

        # Count rate evolution over the frames.
        #counts = np.sum(images, axis=(0,1))
        #counts = np.roll(counts, -frameidx)
        #plotxy = XYPlot(0, "UXS Counts {}".format(str(evt.get(EventId))), range(numframes), counts)
        #publish.send("UXSMonitorCounts", plotxy)
 
    # Iterate framenumber
    frameidx += 1
    frameidx = frameidx%numframes

    # Save to file
    if saveNow:
        while os.path.exists(savePath.format(saveIterator)):
            # Make sure we iterate the filename as to not overwrite
            saveIterator += 1
        with open(savePath.format(saveIterator), 'w') as f:
            f.write('\n'.join(metadata))
        np.savetxt(savePath.format(saveIterator)+".accimage", accimage)
        np.savetxt(savePath.format(saveIterator)+".accspec", accspectrum)
        np.savetxt(savePath.format(saveIterator)+".liveimage", uxspre.image)
        np.savetxt(savePath.format(saveIterator)+".livespec", spectrum)
        print "{} Data saved with number: {}".format(time.strftime("%X"), saveIterator)
        saveIterator += 1
        saveNow = False
