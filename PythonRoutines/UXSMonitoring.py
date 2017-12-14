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
from psmon.plots import Image, XYPlot, Hist, MultiPlot
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

#ds = DataSource('shmem=psana.0:stop=no')
#print "Connected to shmem"
# Testing with old data
#ds = DataSource('exp=amof6215:run=158') # 2015 beamtime
#ds = DataSource('exp=amox23616:run=74') # Ghost imaging
ds = DataSource('exp=amolr2516:run=203')
#print DetNames()

# UXS Camera
opal_det = Detector('OPAL1')

# Ebeam to get photon energy
ebeam = Detector('EBeam')

# Accumulate frames in a circularbuffer
numframes = 200
# Numframes in the history
numhistory = 2000

# Keep numframes in a 3D matrix
images = np.zeros((1024, 1024, numframes))
spectra = np.zeros((1024, numframes))
sigmamonitor = [np.zeros(numhistory), np.zeros(numhistory)]
posmonitor = [np.zeros(numhistory), np.zeros(numhistory)]
heightmonitor = [np.zeros(numhistory), np.zeros(numhistory)]
intensitymonitor = np.zeros(numhistory)
filtintensitymonitor = np.zeros(numhistory)

pulsemonitor = np.zeros(numhistory)

# Keep a circularbuffer of metadata
metadata = ["" for frame in range(numframes)]

# Keep index for circularbuffers
frameidx = 0
histidx = 0

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
    metadata[frameidx] = "Frame: {}, PhEn: {}, {}".format(frameidx, photonenergy, str(evt.get(EventId)))
    
    #opal_raw = np.rot90(opal_raw.copy()) # Do not rotate on LR25!
    uxspre = UXSDataPreProcessing()
    #opal_raw = uxspre.FilterImage(opal_raw)
    [pos1, sigma1, int1, pos2, sigma2, int2], spectrum, darkremovespec, filtspec, cutenergyscale = uxspre.StandardAnalysis(opal_raw, True)
    energyscale = uxspre.energyscale

    # Sort peaks on energy rather than intensity
    pulsemonitor[histidx] = 0 
    if not np.isnan(sigma1) and not np.isnan(sigma2):
        if int1 > 10000 and int2 > 10000:
            pulsemonitor[histidx] = 2
            #if pos2 > pos1:
            #    pos1, sigma1, int1, pos2, sigma2, int2 = pos2, sigma2, int2, pos1, sigma1, int1
    elif not np.isnan(sigma1):
        if int1 > 10000:
            pulsemonitor[histidx] = 1
    
    # Make the accumulation
    # TODO make sure this don't accumulate the floating point error
    accimage -= images[:,:,frameidx]
    images[:,:,frameidx] = uxspre.FilterImage(uxspre.image)
    #images[:,:,frameidx] = uxspre.image
    accimage += images[:,:,frameidx]
    
    # Add spectrum to circular buffer
    spectra[:,frameidx] = spectrum
    # Sum all accumulated spectra
    accspectrum = np.sum(spectra,axis=1)

    # Pos monitor
    posmonitor[0][histidx] = pos1
    posmonitor[1][histidx] = pos2

    # Sigma monitor
    sigmamonitor[0][histidx] = sigma1
    sigmamonitor[1][histidx] = sigma2

    # Heightmonitor
    heightmonitor[0][histidx] = int1
    heightmonitor[1][histidx] = int2

    # Intensity monitor
    intensitymonitor[histidx] = np.sum(spectrum)
    filtintensitymonitor[histidx] = np.sum(filtspec)

    # Only publish every nth frame
    npublish = 5
    if frameidx%npublish == 0:
        # Calculate our processing speed
        speed = int(npublish/(time.time()-start))
        start = time.time()
        # Create a mock array for showing the fitresults
        if not np.isnan(sigma1) and not np.isnan(sigma2):
            # Double Gaussian
            fitresults = UXSDataPreProcessing.DoubleGaussian([int1, pos1, sigma1, int2, pos2, sigma2], cutenergyscale)
        elif not np.isnan(sigma1):
            # Simple gaussian
            fitresults = UXSDataPreProcessing.Gaussian([int1, pos1, sigma1], cutenergyscale)
        else:
            fitresults = np.zeros(1024)
        livetitle = """<table><tr><td>Live</td><td>-----------</td><td>----------</td><td>-----------</td></tr>
                              <tr><td>Pos1:</td><td>{:.2f}</td><td>Sigma1:</td><td>{:.2f}</td></tr>
                              <tr><td>Pos2:</td><td>{:.2f}</td><td>Sigma2:</td><td>{:.2f}</td></tr>
                              <tr><td>"Int2/Int1":</td><td>{:.2f}</td><td></td><td></td></tr>
                       </table>""".format(pos1,sigma1,pos2,sigma2,(int2*sigma2)/(int1*sigma1))
        # Send a multiplot
        plotimglive = Image(0, "Live", uxspre.image)
        plotimgacc = Image(0, "Acc", accimage)
        plotxylive = XYPlot(0, livetitle, [energyscale, cutenergyscale, cutenergyscale], [spectrum, fitresults, darkremovespec])#filtspec])
        plotxyacc = XYPlot(0, "Acc", energyscale, accspectrum)
        multi = MultiPlot(0, "UXSMonitor {} Hz {}".format(speed, metadata[frameidx]), ncols=2)
        multi.add(plotimglive)
        multi.add(plotimgacc)
        multi.add(plotxylive)
        multi.add(plotxyacc)
        publish.send("UXSMonitor", multi)

        # 2nd UXSMonitor
        multi = MultiPlot(0, "UXSMonitor2 {} Hz {}".format(speed, metadata[frameidx]), ncols=3)
        plotsigma = XYPlot(0, "Sigma1: {:.2f}<br>Sigma2: {:.2f}".format(sigma1,sigma2), 2*[range(numhistory)], [np.roll(s, -histidx-1) for s in sigmamonitor], formats='.')
        plotheight = XYPlot(0, "Height1: {:.2f}<br>Height2: {:.2f}".format(int1,int2), 2*[range(numhistory)], [np.roll(h, -histidx-1) for h in heightmonitor], formats='.')
        plotpos = XYPlot(0, "Pos1: {:.2f}<br>Pos2:{:.2f}".format(pos1,pos2), 2*[range(numhistory)], [np.roll(p, -histidx-1) for p in posmonitor], formats='.')
        plotintensity = XYPlot(0, "Intensity", range(numhistory), np.roll(intensitymonitor, -histidx-1), formats='.')
        plotfiltintensity = XYPlot(0, "Filtered Intensity", range(numhistory), np.roll(filtintensitymonitor, -histidx-1), formats='.')
        pulsehistogram, pulsehistogramedges = np.histogram(pulsemonitor,3)
        plotpulsemonitor = Hist(0, "Pulse count histogram", pulsehistogramedges, pulsehistogram)
        multi.add(plotsigma)
        multi.add(plotheight)
        multi.add(plotpos)
        multi.add(plotintensity)
        multi.add(plotfiltintensity)
        multi.add(plotpulsemonitor)
        publish.send("UXSMonitor2", multi)
 
    # Iterate framenumber
    frameidx += 1
    frameidx = frameidx%numframes

    # Numhistory
    histidx +=1
    histidx = histidx%numhistory

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
