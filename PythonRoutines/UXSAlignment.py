import json

import numpy as np
import scipy

from UXSDataPreProcessing import UXSDataPreProcessing

# Visualize with psmon
from psmon import publish
from psmon.plots import Image, XYPlot, MultiPlot

"""
In this file we keep the routines that is needed during calibration and alignment of the spectrometer.

Produced calibrations will be:

xshifts per y-line to calibrate for curvature
An energyscale calculated from three peaks, and then polyfitted 2nd degree.

Inputs will be an image, that can be accumulated or just one shot.

We will hardcode the output in the UXSDataPreProcessing
"""
def CurvatureFinder(image, xmin=0, xmax=600, ymin=250, ymax=800, nslices=10):
    """
    Helper function to be used to find the xshifts for the
    curvature correction. Will only be used when calibrating.
    """
    # We begin by cutting the image in slices
    # Only look in the specified box
    # Please note 0,0 is top left corner in pyqtgraph
    slices = np.array_split(image[ymin:ymax, xmin:xmax], nslices, 0)
    curv_x = []
    curv_y = []
    for i,theslice in enumerate(slices):
        if len(theslice) == 0:
            continue # No data skipping this one
        theslice = np.average(theslice, axis=0) # Project the slice into 1D, TODO change to sum
        theslice = Smooth1D(theslice, 5)
        # Start with some guess
        p = GaussianFit(theslice, np.max(theslice), np.argmax(theslice), len(theslice/4), 200)
        x = p[1] # Got the gaussian center
        sliceymin = float(i)*(ymax-ymin)/nslices+ymin
        sliceymax = (float(i)+1)*(ymax-ymin)/nslices+ymin
        y = (sliceymin+sliceymax)/2
        if x > len(theslice) or x < 0:
            # x cannot possibly be outside the data region
            print "Warning, slicedata outside region. Skipping y={}, x={}".format(x,y)
            continue # Skipping this one
        curv_x.append(x)
        curv_y.append(y)
    return zip(curv_x, curv_y)
    
def CreateXshifts(zippedxy):
    """
    Fit the found curv_x and curv_y using polyfit
    """
    curv_x, curv_y = zip(*zippedxy)
    if False: # Change by terminal input
        print json.dumps(zip(curv_x, curv_y))
        try:
            userin = input("Please enter changed array [X,Y]:\n")
        except SyntaxError:
            print "Leaving it unchanged"
            userin = zip(curv_x, curv_y)
        curv_x, curv_y = zip(*userin)
    # Now we got the gaussian center of the slices
    # Lets fit them using a polynomial
    curv_poly = np.polyfit(curv_y, curv_x, 2) #2nd order
    # Find the center, then we will shift everything on the
    # The line will be straightened and placed on this line
    center = np.polyval(curv_poly, 512)
    xshifts = center-np.polyval(curv_poly, range(1024))
    # Make the shifts into integers so that they match the pixles
    xshifts = xshifts.astype(int)
    return xshifts

def Gaussian(p,x):
    """
    Gaussian
    """
    return p[0] * np.exp(-((x-p[1])/p[2])**2/2)

def GaussianFit(data, height, mu, sigma, maxfev=800):
    """
    Helper function to fit gaussian on slices and while aligning
    """

    def error(p,x,y):
        return Gaussian(p,x) - y
    x = np.arange(len(data))
    p0 = np.array([height, mu, sigma])
    p1, success = scipy.optimize.leastsq(error, p0, args=(x, data), maxfev=maxfev)
    return p1


def AddCurvLine(image, xshifts, intensity=1000):
    """
    For visualization we can add the curvature to the raw image
    """
    for y, x in zip(range(1024), xshifts):
        image[y,512-x] = intensity
    return image


def readImage(filename):
    image = np.loadtxt(filename)
    return image
 
 
def saveXshifts(filename, xshifts):
    """
    Save the xshifts to file so that it can later be copied to the UXSDataPreProcessing
    """
    print json.dumps(list(xshifts))
    with open(filename+"_CURV", 'w') as f:
        f.write(json.dumps(list(xshifts)))
    print "Curvature saved to "+filename+"_CURV"


def Smooth1D(data, width=5):
    """
    Smooth the using gaussian filter useful before fitting
    """
    return scipy.ndimage.gaussian_filter1d(data, width)

def Threshold1D(data, sfr=40):
    """
    Removes threshold from 1D data by using the mean of the sfr first bins
    """
    return data-np.mean(data[sfr])

def RemoveBaseline(data, sfr=40):
    """
    Removes 1d baseline by fitting 1st order poly to sfr parts of beginning and
    end of data
    """
    begin = np.mean(data[:sfr])
    end = np.mean(data[-sfr:])
    baselinepoly = np.polyfit([0,len(data)],[begin,end],1)
    cdata = data - np.polyval(baselinepoly, np.arange(0,len(data)))
    return cdata

def CurvatureMain():
    # We will probably do this in ipython notebook
    # Otherwise here we can run it
    # Read input file
    filename = "./UXSAlign/0.out.accimage"
    print "Reading {}".format(filename)
    image = readImage(filename)
    uxspre = UXSDataPreProcessing(image.copy())
    uxspre.CalculateProjection()
    uncorrspectrum = uxspre.wf
    # Produce curvaturecorrection
    zippdxy = CurvatureFinder(image)
    xshifts = CreateXshifts(zippdxy)
    # Monkey path the correction
    uxspre.xshifts = xshifts
    uxspre.CorrectImageGeometry()
    uxspre.CalculateProjection()
    # Plot difference with and without calibration
    multi = MultiPlot(0, "UXSCalibration {}".format(filename), ncols=2)
    image = AddCurvLine(image, xshifts)
    uncorrimage = Image(0, "Uncorrected", image)
    multi.add(uncorrimage)
    corrimage = Image(0, "Corrected", uxspre.image)
    multi.add(corrimage)
    uncorrspec = XYPlot(0, "Uncorrected", range(len(uncorrspectrum)), uncorrspectrum)
    multi.add(uncorrspec)
    corrspec = XYPlot(0, "Corrected", range(len(uxspre.wf)), uxspre.wf)
    multi.add(corrspec)
    for i in range(10):
        publish.send("UXSCalibration", multi)
    saveXshifts(filename, xshifts)
    print "fin"



if __name__=='__main__':
    CurvatureMain()
