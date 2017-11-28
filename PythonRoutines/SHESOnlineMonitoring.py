"""This code for online monitoring of the Scienta Hemispherical 
Electron Spectrometer"""

from psana import *
import numpy as np
import cv2
import time
# This class for processing SHES data
from SHESPreProcessing import SHESPreProcessor
# This for estimating photon energy from ebeam L3 energy
from L3EnergyProcessing import L3EnergyProcessor
from FEEGasProcessing import FEEGasProcessor
# Import double-ended queue
from collections import deque
# 2D histogram from old psana code
from skbeam.core.accumulators.histogram import Histogram
# Imports for plotting
from psmon.plots import XYPlot,Image,Hist
from psmon import publish
publish.local=True # changeme

# Set parameters
#ds=DataSource("exp=AMO/amon0816:run=228:smd:dir=/reg/d/psdm/amo/amon0816/xtc:live")
ds=DataSource("exp=AMO/amox23616:run=86:smd:dir=/reg/d/psdm/amo/amox23616/xtc:live")
# TODO change to access the shared memory
threshold=500 # for thresholding of raw OPAL image

# Define ('central') photon energy bandwidth for plotting projected spectrum
# Andre thought maybe 2 eV bandwidth
min_cent_pe=0. # in eV
max_cent_pe=9999. # in eV

# Define parameters for L3 photon energy histogram 
minhistlim_L3PhotEnergy=500 #in eV
maxhistlim_L3PhotEnergy=560 #in eV
numbins_L3PhotEnergy=20

# Define parameters for FEE Gas Energy histogram 
minhistlim_FeeGasEnergy=0 # in mJ
maxhistlim_FeeGasEnergy=2 # in mJ
numbins_FeeGasEnergy=10

# Other parameters
history_len=1000
history_len_counts=1000 # different history length for plotting the estimated counts

plot_every=10 #plot every n frames

#%% For arcing warning
arc_freeze_time=0.2 # how many seconds to freeze plotting after arcing warning?

# For FEE Gas detector
fee_gas_threshold=0.0 #in mJ

#%% Define some functions
def sendPlots(x_proj_sum, image_sum, counts_buff, opal_image, hist_L3PhotEnergy, \
              hist_FeeGasEnergy, nevt, numshotsforacc):
        # Define plots
        plotxproj = XYPlot(nevt,'Accumulated electron spectrum over past '+\
                    str(numshotsforacc)+' good shots', \
                    np.arange(x_proj_sum.shape[0]), x_proj_sum)
        plotcumimage = Image(nevt, 'Accumulated sum ('+str(numshotsforacc)+' good shots)', image_sum)
        plotcounts = XYPlot(nevt,'Estimated number of identified electron counts over past '+ \
                            str(len(counts_buff))+' good shots', np.arange(len(counts_buff)), \
                            np.array(counts_buff))
        plotshot = Image(nevt, 'Single shot', opal_image)
        plotL3PhotEnergy = Hist(nevt,'Histogram of L3 \'central\' photon energies (plotting for '+str(np.round(min_cent_pe, 2))+\
        '- '+str(np.round(min_cent_pe, 2))+')',  hist_L3PhotEnergy.edges[0], \
                           np.array(hist_L3PhotEnergy.values))
        plotFeeGasEnergy = Hist(nevt,'Histogram of FEE gas energy (plotting for above'+str(np.round(fee_gas_threshold, 2))+\
        ' only)',  hist_FeeGasEnergy.edges[0], np.array(hist_FeeGasEnergy.values))
        
        # Publish plots
        publish.send('AccElectronSpec', plotxproj)
        publish.send('OPALCameraAcc', plotcumimage)
        publish.send('ElectronCounts', plotcounts)
        publish.send('OPALCameraSingShot', plotshot)
        publish.send('L3Histogram', plotL3PhotEnergy)
        publish.send('FEEGasHistogram', plotFeeGasEnergy)

#%% Now run
quot,rem=divmod(history_len, plot_every)
if rem!=0:
    history_len=plot_every*quot+1
    print 'For efficient monitoring of acc sum require history_len divisible by \
    plot_every, set history_len to '+str(history_len)

# Initialise SHES processor
processor=SHESPreProcessor(threshold=threshold)
# Extract shape of arrays which SHES processor will return
_,j_len,i_len=processor.pers_trans_params #for specified x_len_param/y_len_param in SHESPreProcessor,
#PerspectiveTransform() returns array of shape (y_len_param,x_len_param)

# Initialise L3 ebeam energy processor
l3Proc=L3EnergyProcessor()
# Initialise FEE  Gas processor
feeGas=FEEGasProcessor()

# Other initialisation
image_sum_buff=deque(maxlen=1+history_len/plot_every)  # These keep the most recent one NOT to
x_proj_sum_buff=deque(maxlen=1+history_len/plot_every) # be plotted so that it can be taken away
                                                       # from the rolling sum
image_buff=np.zeros((plot_every, i_len, j_len)) # this gets reset to 0
x_proj_buff=np.zeros((plot_every, j_len)) # this gets reset to 0
counts_buff=deque(maxlen=history_len_counts) # this doesn't get reset to 0

hist_L3PhotEnergy = Histogram((numbins_L3PhotEnergy,minhistlim_L3PhotEnergy,maxhistlim_L3PhotEnergy))
hist_FeeGasEnergy = Histogram((numbins_FeeGasEnergy,minhistlim_FeeGasEnergy,maxhistlim_FeeGasEnergy))
 
image_sum=np.zeros((i_len, j_len))
x_proj_sum=np.zeros(j_len)

arcing_freeze=False
rolling_count=0
arc_time_ref=0.0 # will be set at certain number of seconds if required

# Now being looping over events
for nevt, evt in enumerate(ds.events()):

    fee_gas_energy=feeGas.ShotEnergy(evt)
    cent_pe=l3Proc.CentPE(evt)

    # Check data exists
    if fee_gas_energy is None:
        print 'No FEE gas energy, continuing to next event'        
        continue

    if cent_pe is None:
        print 'No L3 e beam energy, continuing to next event'        
        continue
    
    # If data exists, fill histograms
    hist_L3PhotEnergy.fill(cent_pe)
    hist_FeeGasEnergy.fill(fee_gas_energy)

    #Check data falls within thresholds
    if fee_gas_energy < fee_gas_threshold:
        print 'FEE gas energy = '+str(fee_gas_energy)+' mJ -> continuing to next event'
        continue

    if not (cent_pe < max_cent_pe and cent_pe > min_cent_pe):
        print '\'Central\' photon energy = '+str(np.round(cent_pe,2))+\
        '-> outside specified range, skipping event'

    opal_image, x_proj, count_estimate, arced=processor.OnlineProcess(evt)

    if opal_image is None:
        print 'No SHES image, continuing to next event'
        continue

    if arced:
        print '***WARNING - ARC DETECTED!!!***'

        cv2.putText(opal_image,'ARCING DETECTED!!!', 
        (50,int(i_len/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 10)

        image_sum_temp=np.copy(image_sum)
        cv2.putText(image_sum_temp,'ARCING DETECTED!!!', 
        (50,int(i_len/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 10)
        
        sendPlots(x_proj_sum, image_sum_temp, counts_buff, opal_image, hist_L3PhotEnergy, hist_FeeGasEnergy, nevt, numshotsforacc)
        arcing_freeze=True
        arc_time_ref=time.time()

        continue # don't accumulate data for the arced shot
        
    image_buff[rolling_count]=opal_image
    x_proj_buff[rolling_count]=x_proj
    counts_buff.append(count_estimate)

    rolling_count+=1 #increment here

    if rolling_count==plot_every:

        if arcing_freeze:
            if (time.time()-arc_time_ref)>arc_freeze_time: arcing_freeze=False

        # Sum the data across the last small slice
        image_sum_slice=image_buff.sum(axis=0)
        x_proj_sum_slice=x_proj_buff.sum(axis=0)
        
        # Put these sums in the deques for when they need to be
        # removed from the rolling sum
        image_sum_buff.append(image_sum_slice)
        x_proj_sum_buff.append(x_proj_sum_slice)
        
        # But for the minute, add to the rolling sum
        image_sum+=image_sum_slice
        x_proj_sum+=x_proj_sum_slice
        
        # Only take away from the rolling sum if we have had more than the max history
        # length of shots
        if len(x_proj_sum_buff)==x_proj_sum_buff.maxlen:
            #print 'hit max length' 
            image_sum-=image_sum_buff[0] # don't pop, let the deque with finite maxlen
            x_proj_sum-=x_proj_sum_buff[0] # take care of that itself
            numshotsforacc=(len(x_proj_sum_buff)-1)*plot_every
        else:
            numshotsforacc=len(x_proj_sum_buff)*plot_every
        
        if not arcing_freeze:
            sendPlots(x_proj_sum, image_sum, counts_buff, opal_image, hist_L3PhotEnergy, hist_FeeGasEnergy, nevt, numshotsforacc)
        
        # Reset
        rolling_count=0
        image_buff=np.zeros((plot_every, i_len, j_len))
        x_proj_buff=np.zeros((plot_every, j_len))
