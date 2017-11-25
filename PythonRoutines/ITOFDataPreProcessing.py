#### Useful commands
#
# mpirun -n 40 --host daq-amo-mon02,daq-amo-mon03,daq-amo-mon04,daq-amo-mon05,daq-amo-mon06 amon0816.sh
#
#
#
#
#

# Standard PYTHON modules
print "IMPORTING STANDARD PYTHON MODULES...",
import cv2
import numpy as np
import math
import collections
import random
from skbeam.core.accumulators.histogram import Histogram

# LCLS psana to read data
from psana import *
from xtcav.ShotToShotCharacterization import *

# For online plotting
from psmon import publish
from psmon.plots import XYPlot,Image


class ITOFDataPreProcessing():

    def __init___(self,wf):
	self.wf = -wf
	self.iyield = 0
	self.offset = 0	
	self.std = 0	

    def FilterWaveform(self,refrange = [0,500]):
	self.offset = np.median(self.wf[refrange[0]:refrange[1]])
	self.wf=self.wf-self.offset

    def CalculateYield(self,refrange = [0,500], threshold = 3 ):
	self.std = np.std(self.wf[refrange[0]:refrange[1]])
	self.wf[self.wf<(self.std*threshold)] = 0.0
	self.iyield = np.sum(self.wf)

    def StandardAnalysis(self):
	self.FilterWaveform([0,500])
	self.CalculateYield([0,500],3)
	return self.iyield
