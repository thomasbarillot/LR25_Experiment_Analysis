# Standard Python imports
from psana import *
import numpy as np
import cv2
import time
## Class for UXS
#from UXSDataPreProcessing import UXSDataPreProcessing
# Class for processing SHES data
from SHESPreProcessing import SHESPreProcessor
# Class to estimate photon energy from ebeam L3 energy
from L3EnergyProcessing import L3EnergyProcessor
# 2D histogram from old psana library
from skbeam.core.accumulators.histogram import Histogram
# Imports for plotting
from psmon.plots import XYPlot,Image,Hist,MultiPlot
from psmon import publish
publish.local=True # changeme

#
#ds=DataSource("exp=AMO/amolr2516:run=38:smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live")
ds=DataSource('shmem=psana.0:stop=no')

# Initialise detectors
procL3Energy=L3EnergyProcessor()
procSHES=SHESPreProcessor()

# Define parameters for L3 energy/SHES first moment histogram
numbins_L3_elec1mom_L3=100
minhistlim_L3_elec1mom_L3=3300
maxhistlim_L3_elec1mom_L3=3400

numbins_L3_elec1mom_elec1mom=100
minhistlim_L3_elec1mom_elec1mom=0
maxhistlim_L3_elec1mom_elec1mom=714

numbins_L3Energy=100
minhistlim_L3Energy=3300
maxhistlim_L3Energy=3400

numbins_L3EnergyWeighted=100
minhistlim_L3EnergyWeighted=3300
maxhistlim_L3EnergyWeighted=3400

hist_L3_elec1mom = Histogram((numbins_L3_elec1mom_L3,minhistlim_L3_elec1mom_L3,\
                           mmaxhistlim_L3_elec1mom_L3),(numbins_L3_elec1mom_elec1mom,\
                           minhistlim_L3_elec1mom_elec1mom,maxhistlim_L3_elec1mom_elec1mom))

hist_L3Energy=Histogram((numbins_L3Energy,minhistlim_L3Energy,maxhistlim_L3Energy))
hist_L3EnergyWeighted=Histogram((numbins_L3EnergyWeighted,minhistlim_L3EnergyWeighted,maxhistlim_L3EnergyWeighted))

for nevt, evt in enumerate(ds.events()):
    print nevt
    energyL3=procL3Energy.L3Energy(evt)
    moments, numhits=procSHES.CalibProcess(evt, inPix=True)
    mom1=moments[0]

    hist_L3Energy.fill(energyL3)
    hist_L3EnergyWeighted.fill([energyL3], weights=[numhits])

    if nevt>100:
        break

avShotsPerL3Energy=hist_L3EnergyWeighted/np.float(hist_L3Energy.values)

L3_weighted_plot = XYPlot(nevt, 'L3_weighted', hist_L3EnergyWeighted.centers[0], hist_L3EnergyWeighted.values)
publish.send('L3 weighted', L3_weighted_plot)



