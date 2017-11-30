""" 
Online monitoring of the UXS spectrometer at LR25
Data is accessed using shared memory
"""
import os
import time
import thread

import scipy.ndimage

from psana import *
import numpy as np
import scipy.ndimage as filter

from UXSDataPreProcessing import UXSDataPreProcessing

# Visualization
from psmon.plots import Image,XYPlot, MultiPlot
from psmon import publish

#publish.local = True

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
#ds = DataSource('exp=amof6215:run=158') # 2015 beamtime
#ds = DataSource('exp=amox23616:run=86') # Ghost imaging
print DetNames()

# UXS Camera
opal_det = Detector('OPAL1')

# Ebeam to get photon energy
ebeam = Detector('EBeam')

# Accumulate frames in a circularbuffer
numframes = 1000

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
        photonenergy = 0
    if ebeamdata is not None:
        # TODO Change to ebeamdata.ebeamL3Energy()
        photonenergy = ebeamdata.ebeamPhotonEnergy()
    # Set the metadata
    metadata[frameidx] = "{}, PhEn: {}, {}".format(frameidx, photonenergy, str(evt.get(EventId)))
    
    #opal_raw = np.rot90(opal_raw.copy()) # Do not rotate on LR25!
    uxspre = UXSDataPreProcessing()
    ## TODO only copy here if needed
    [pos1, sigma1, int1, pos2, sigma2, int2], spectrum, filtspec, cutenergyscale = uxspre.StandardAnalysis(opal_raw.copy(), True)
    energyscale = uxspre.energyscale
    
    # Make the accumulation
    # TODO make sure this don't accumulate the floating point error
    accimage -= images[:,:,frameidx]
    images[:,:,frameidx] = uxspre.image
    accimage += images[:,:,frameidx]
    
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
        # Create a mock array for showing the fitresults
        if sigma1 != np.nan and sigma2 != np.nan:
            # Double Gaussian
            fitresults = UXSDataPreProcessing.DoubleGaussian([int1, pos1, sigma1, int2, pos2, sigma2], cutenergyscale)
        elif sigma1 != np.nan:
            # Simple gaussian
            fitresults = UXSDataPreProcessing.Gaussian([int1, pos1, sigma1], cutenergyscale)
        else:
            fitresults = np.zeros(1024)
        livetitle = """<table><tr><td>Pos1:</td><td>{:.2f}</td><td>Sigma1:</td><td>{:.2f}</td></tr>
                              <tr><td>Pos2:</td><td>{:.2f}</td><td>Sigma2:</td><td>{:.2f}</td></tr>
                       </table>""".format(pos1,sigma1,pos2,sigma2)
        # Send a multiplot
        plotimglive = Image(0, "Live", uxspre.image)
        plotimgacc = Image(0, "Acc", accimage)
        plotxylive = XYPlot(0, livetitle, [energyscale, cutenergyscale, cutenergyscale], [spectrum, fitresults, filtspec])
        plotxyacc = XYPlot(0, "Acc", energyscale, accspectrum)
        multi = MultiPlot(0, "UXSMonitor {} Hz {}".format(speed, metadata[frameidx]), ncols=2)
        multi.add(plotimglive)
        multi.add(plotimgacc)
        multi.add(plotxylive)
        multi.add(plotxyacc)
        publish.send("UXSMonitor", multi)

        # Count rate evolution over the frames.
        #counts = np.sum(images, axis=(0,1))
        #counts = np.roll(counts, -frameidx)
        #counts = scipy.ndimage.gaussian_filter1d(counts,1)
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
