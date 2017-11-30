"""This code for online monitoring of the Scienta Hemispherical 
Electron Spectrometer

mpirun --oversubscribe -n 10 --host daq-amo-mon02,daq-amo-mon03,daq-amo-mon04,daq-amo-mon05,daq-amo-mon06 python SHESOnlineMonitoring.py"""

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
# Class to read in & average FEE gas energy
from FEEGasProcessing import FEEGasProcessor
# Double-ended queue
from collections import deque
# 2D histogram from old psana library
from skbeam.core.accumulators.histogram import Histogram
# Imports for parallelisation
from mpi4py import MPI
# Imports for plotting
from psmon.plots import XYPlot,Image,Hist,MultiPlot
from psmon import publish
publish.local=True # changeme

#%% Set parameters
# This will be shared memory for online analysis
#ds=DataSource("exp=AMO/amox23616:run=86:smd:dir=/reg/d/psdm/amo/amox23616/xtc:live")
ds=DataSource('shmem=psana.0:stop=no')
#uxs_det=Detector('OPAL1')

opal_threshold=500 # for thresholding of raw OPAL image

# Define ('central') photon energy bandwidth, taken from ebeam L3 energy,
# for taking electron spectra for real-time plotting
# Andre thought maybe 2 eV bandwidth
min_cent_pe=0. # in eV
max_cent_pe=9999. # in eV

# Define lower and upper bounds of region of interest (ROI) to monitor counts in
# these limits are shifted onto the closest pixel
roi_lower=490 #in eV
roi_upper=495 #in eV

# Define FEE gas energy threshold for taking electron spectra for real-time
# plotting
fee_gas_threshold=0.1 #in mJ

# Define parameters for L3 photon energy histogram 
minhistlim_L3PhotEnergy=500 #in eV
maxhistlim_L3PhotEnergy=510 #in eV
numbins_L3PhotEnergy=20

# Define parameters for FEE Gas Energy histogram 
minhistlim_FeeGasEnergy=0 # in mJ
maxhistlim_FeeGasEnergy=0.4 # in mJ
numbins_FeeGasEnergy=10

# Define parameters for FEE Gas Energy/ROI Counts 2D histogram
numbins_FEE_CountsROI_FEE=10
minhistlim_FEE_CountsROI_FEE=0
maxhistlim_FEE_CountsROI_FEE=0.4
numbins_FEE_CountsROI_CountsROI=10
minhistlim_FEE_CountsROI_CountsROI=0
maxhistlim_FEE_CountsROI_CountsROI=10

# Other parameters
history_len=1000 # for the projected electron spectrum and accumulated electron image
history_len_counts=1000 # different history length for plotting the estimated counts
refresh_rate=10 #plot every n frames

#%% Now run
# Initialisation for each core
##Initialise UXS processor
#uxsProc=UXSDataPreProcessing()
# Initialise SHES processor
processor=SHESPreProcessor(threshold=opal_threshold)
# Initialise L3 ebeam energy processor
l3Proc=L3EnergyProcessor()
# Initialise FEE Gas processor
feeGas=FEEGasProcessor()
# For parallelisation
comm = MPI.COMM_WORLD # define parallelisation object
rank = comm.Get_rank() # which core is script being run on
size = comm.Get_size() # no. of CPUs being used
print 'Core #'+str(rank)+' reporting for duty'

# Extract shape of arrays which SHES processor will return
_,j_len,i_len=processor.pers_trans_params # for specified x_len_param/y_len_param in SHESPreProcessor,
                                          # PerspectiveTransform() returns array of shape (y_len_param,x_len_param)
# Other initialisation
image_buff=np.zeros((refresh_rate, i_len, j_len)) # this gets reset to 0 after each reduction
x_proj_buff=np.zeros((refresh_rate, j_len)) # this gets reset to 0 after each reduction

counts_buff=np.zeros((refresh_rate,2)) # this gets reset to 0 after each reduction
counts_buff_roi=np.zeros((refresh_rate,2)) # this gets reset to 0 after each reduction

hist_L3PhotEnergy = Histogram((numbins_L3PhotEnergy,minhistlim_L3PhotEnergy,maxhistlim_L3PhotEnergy))
hist_FeeGasEnergy = Histogram((numbins_FeeGasEnergy,minhistlim_FeeGasEnergy,maxhistlim_FeeGasEnergy))
hist_FeeGasEnergy_CountsROI = Histogram((numbins_FEE_CountsROI_FEE,minhistlim_FEE_CountsROI_FEE,\
                                        maxhistlim_FEE_CountsROI_FEE),(numbins_FEE_CountsROI_CountsROI,\
                                        minhistlim_FEE_CountsROI_CountsROI,maxhistlim_FEE_CountsROI_CountsROI))

# Initialise variables which are only used by root core
image_sum=np.zeros((i_len, j_len)) # accumulated sum of OPAL image over history_len shots
x_proj_sum=np.zeros(j_len) # accumulated sum of projected electron spectrum over history_len shots
good_shot_count_all=0

image_sum_slice=np.zeros((i_len, j_len)) # 'slice' refers to time slice, sum of images from each core is reduced
                                             # into this
x_proj_sum_slice=np.zeros(j_len) # 'slice' refers to time slice, sum of projected electron spectra from each 
                                     # core is reduced into this

counts_buff_all=deque(maxlen=history_len_counts)
counts_buff_roi_all=deque(maxlen=history_len_counts)

image_sum_buff=deque(maxlen=1+history_len/(refresh_rate*size))  # The first element of these is the most recent one NOT to
x_proj_sum_buff=deque(maxlen=1+history_len/(refresh_rate*size)) # be plotted so that it can be taken away
                                                                # from the rolling sum
good_shot_count_buff=deque(maxlen=history_len/(refresh_rate*size))

hist_L3PhotEnergy_all = np.zeros(numbins_L3PhotEnergy)#, dtype=int32) # array for the Histogram.values to be reduced in to
hist_FeeGasEnergy_all = np.zeros(numbins_FeeGasEnergy)#, dtype=int32) # array for the Histogram.values to be reduced in to
hist_FeeGasEnergy_CountsROI_all = np.zeros((numbins_FEE_CountsROI_FEE, numbins_FEE_CountsROI_CountsROI))#, dtype=int32) # array
# for the Histogram.values to be reduced in to

#%% Set some variables 
# Adjust history_len if needs be so that it is integer number of refresh_rate
quot_a,rem_a=divmod(history_len, refresh_rate*size)
if rem_a!=0:
    history_len=refresh_rate*size*quot_a

count_conv=processor.count_conv # conversion factor from integrated signal -> number electron counts estimate
calib_array=processor.calib_array # calibration array mapping pixels onto eV for SHES

# Find indices for monitoring region of interest
roi_idx_lower, roi_idx_upper = (np.abs(calib_array-roi_lower)).argmin(), \
(np.abs(calib_array-roi_upper)).argmin()
roi_lower_act, roi_upper_act=calib_array[roi_idx_lower], calib_array[roi_idx_upper]

if rank==0: # initilisation for the root core only
    publish.init() # for plotting
    ref_time=time.time() # for estimating rate of data acquisition
    print 'Monitoring counts in region between ' +str(np.round(roi_lower_act,2))+\
          ' eV - '+str(np.round(roi_upper_act,2))+' eV'
    if rem_a!=0:
        print 'For efficient monitoring of accumulated electron spectra require history_len divisible by \
        refresh_rate*size, set history_len to '+str(history_len)   

    #%% Define plotting function
    def definePlots(x_proj_sum, image_sum, counts_buff, counts_buff_roi, opal_image, (hist_L3PhotEnergy, \
                    hist_L3PhotEnergy_edges), (hist_FeeGasEnergy, hist_FeeGasEnergy_edges), 
                    (hist_FeeGasEnergy_CountsROI, hist_FeeGasEnergy_CountsROI_edges), nevt, numshotsforacc,\
                    good_shot_count_pcage):
            # Define plots
            plotxproj = XYPlot(nevt,'Accumulated electron spectrum from good shots ('+str(np.round(good_shot_count_pcage, 2))+\
                               '%) over past '+str(numshotsforacc)+' shots', calib_array, x_proj_sum)
            plotcumimage = Image(nevt, 'Accumulated sum of good shots ('+str(np.round(good_shot_count_pcage, 2))+\
                                 '%) over past '+str(numshotsforacc)+' shots', image_sum)
            plotcounts = XYPlot(nevt,'Estimated number of identified electron counts over past '+ \
                            str(len(counts_buff))+' shots', np.arange(len(counts_buff)), np.array(counts_buff))     
            plotcountsregint = XYPlot(nevt,'Estimated number of identified electron counts in region '+\
                                      str(np.round(roi_lower_act,2))+' eV - '+\
                                      str(np.round(roi_upper_act,2))+' eV (inclusive) over past '+ \
                                       str(len(counts_buff_roi))+' shots', \
                                       np.arange(len(counts_buff_roi)), np.array(counts_buff_roi))
            opal_image[:,roi_idx_lower-3:roi_idx_lower-1]=900; opal_image[:,roi_idx_upper+1:roi_idx_upper+3]=900
            plotshot = Image(nevt, 'Single shot, ROI marked from '+str(np.round(roi_lower_act,2))+' eV - '+\
                             str(np.round(roi_upper_act,2))+' eV', opal_image)

            plotL3PhotEnergy = Hist(nevt,'Histogram of L3 \'central\' photon energies (getting electron data for '+str(np.round(min_cent_pe, 2))+\
            ' eV - '+str(np.round(max_cent_pe, 2))+' eV)',  hist_L3PhotEnergy_edges[0], hist_L3PhotEnergy_all)
            plotFeeGasEnergy = Hist(nevt,'Histogram of FEE gas energy (getting electron data for above '+str(np.round(fee_gas_threshold, 2))+\
            ' mJ only)',  hist_FeeGasEnergy_edges[0], hist_FeeGasEnergy_all)
            plotFeeGasEnergy_CountsROI = Image(nevt,'Histogram of FEE gas energy vs ROI ('+str(np.round(roi_lower_act,2))+' eV - '+\
                                              str(np.round(roi_upper_act,2))+' eV) counts', hist_FeeGasEnergy_CountsROI, \
                                              hist_FeeGasEnergy_CountsROI_edges[1], hist_FeeGasEnergy_CountsROI_edges[0])

            return plotxproj, plotcumimage, plotcounts, plotcountsregint, plotshot, plotL3PhotEnergy, plotFeeGasEnergy, plotFeeGasEnergy_CountsROI

    def sendPlots(x_proj_sum, image_sum, counts_buff, counts_buff_roi, opal_image, (hist_L3PhotEnergy, \
                  hist_L3PhotEnergy_edges), (hist_FeeGasEnergy, hist_FeeGasEnergy_edges), 
                  (hist_FeeGasEnergy_CountsROI, hist_FeeGasEnergy_CountsROI_edges), nevt, numshotsforacc,\
                  good_shot_count_pcage, speed):
            plotxproj, plotcumimage, plotcounts, plotcountsregint, plotshot, plotL3PhotEnergy, plotFeeGasEnergy, \
            plotFeeGasEnergy_CountsROI=\
            definePlots(x_proj_sum, image_sum, counts_buff, counts_buff_roi, opal_image, (hist_L3PhotEnergy, \
                    hist_L3PhotEnergy_edges), (hist_FeeGasEnergy, hist_FeeGasEnergy_edges), 
                    (hist_FeeGasEnergy_CountsROI, hist_FeeGasEnergy_CountsROI_edges), nevt, numshotsforacc, \
                    good_shot_count_pcage)
            # Publish plots
            publish.send('AccElectronSpec', plotxproj)
            publish.send('OPALCameraAcc', plotcumimage)
            publish.send('ElectronCounts', plotcounts)
            publish.send('ElectronCountsRegInt', plotcountsregint)
            publish.send('OPALCameraSingShot', plotshot)
            publish.send('L3Histogram', plotL3PhotEnergy)
            publish.send('FEEGasHistogram', plotFeeGasEnergy)
            publish.send('FEEGasROICountsHistogram', plotFeeGasEnergy_CountsROI)

    def sendMultiPlot(x_proj_sum, image_sum, counts_buff, counts_buff_roi, opal_image, (hist_L3PhotEnergy, \
                    hist_L3PhotEnergy_edges), (hist_FeeGasEnergy, hist_FeeGasEnergy_edges), 
                    (hist_FeeGasEnergy_CountsROI, hist_FeeGasEnergy_CountsROI_edges), nevt, numshotsforacc,\
                     good_shot_count_pcage, speed):
            plotxproj, plotcumimage, plotcounts, plotcountsregint, plotshot, plotL3PhotEnergy, plotFeeGasEnergy, \
            plotFeeGasEnergy_CountsROI=\
            definePlots(x_proj_sum, image_sum, counts_buff, counts_buff_roi, opal_image, (hist_L3PhotEnergy, \
                    hist_L3PhotEnergy_edges), (hist_FeeGasEnergy, hist_FeeGasEnergy_edges), 
                    (hist_FeeGasEnergy_CountsROI, hist_FeeGasEnergy_CountsROI_edges), nevt, numshotsforacc, \
                    good_shot_count_pcage)
            # Define multiplot
            multi = MultiPlot(nevt, 'SHES Online Monitoring, running at ~'+str(np.round(speed, 1))+' Hz', ncols=3)
            # Publish plots
            multi.add(plotshot)
            multi.add(plotcounts)
            multi.add(plotL3PhotEnergy)
            multi.add(plotcumimage)
            multi.add(plotcountsregint)            
            multi.add(plotFeeGasEnergy)
            multi.add(plotxproj)
            multi.add(plotFeeGasEnergy_CountsROI)

            publish.send('SHES Online Monitoring', multi)

rolling_count=0
good_shot_count=0 # FEE gas energy and photon energy within range to take electron spectrum

# Now begin looping over events
for nevt, evt in enumerate(ds.events()):
    if nevt%size!=rank: continue # each core only processes its own runs
    gather_electrons=True
    fee_gas_energy=feeGas.ShotEnergy(evt)
    cent_pe=l3Proc.CentPE(evt)

    # Check data exists
    if fee_gas_energy is None:
        print 'FEE gas energy is None, continuing to next event'        
        continue

    if cent_pe is None:
        print 'L3 e-beam energy is None, continuing to next event'
        continue
    
    # If data exists, fill histograms
    hist_L3PhotEnergy.fill(cent_pe) 
    hist_FeeGasEnergy.fill(fee_gas_energy)

    opal_image, x_proj, arced=processor.OnlineProcess(evt)

    if opal_image is None:
        print 'SHES OPAL image is None, continuing to next event'
        continue

    if arced:
        print '***WARNING - ARC DETECTED!!!***'
        opal_image_copy=np.copy(opal_image)
        cv2.putText(opal_image_copy,'ARCING DETECTED!!!', \
        (50,int(i_len/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 10)
        # Just send the single shot so you can see
        plotshot = Image(nevt, 'Single shot', opal_image_copy)
        publish.send('OPALCameraSingShot', plotshot)

        continue # don't accumulate any data for the arced shot
    
    #uxs_img=uxs_det.raw(evt)
    #if uxs_img is None:
    #    print 'No UXS image, continuing to next event'
    #    continue

    #uxs_data,_=uxsProc.StandardAnalysis(uxs_img)
    #uxs_pos1, uxs_int1, uxs_pos2, uxs_int2=uxs_data[0], uxs_data[2], uxs_data[3], uxs_data[5]
    
    # Estimated count rate for all spectrum and ROI
    count_estimate=x_proj.sum()/float(count_conv) 
    counts_buff[rolling_count,0]=nevt; counts_buff[rolling_count,1]=count_estimate
    # The below ignores the fact that the MCP display doesn't fill the entire OPAL array, which 
    # artificially decreases the integrated signal to count rate conversion factor (it divides the 
    # integrated signal) compared to the case where the array you are integrating over is entirely filled
    # by the MCP image, which is most likely the case for the region of interest 
    count_estimate_roi=x_proj[roi_idx_lower:roi_idx_upper+1].sum()/float(count_conv) 
    counts_buff_roi[rolling_count,0]=nevt; counts_buff_roi[rolling_count,1]=count_estimate_roi
    
    hist_FeeGasEnergy_CountsROI.fill(fee_gas_energy, count_estimate_roi)

    rolling_count+=1 #increment here

    #Check data falls within thresholds to see whether to gather electron spectra
    if fee_gas_energy < fee_gas_threshold:
        print 'FEE gas energy = '+str(fee_gas_energy)+' mJ, not gathering electron data for this event'
        gather_electrons=False

    if not (cent_pe < max_cent_pe and cent_pe > min_cent_pe):
        print ('and ' if not gather_electrons else '')+'\'Central\' photon energy = '+str(np.round(cent_pe,2))+\
        '-> outside specified range, not gathering electron data for this event'
        gather_electrons=False

    if gather_electrons:
        # Only gather electron data if FEE gas energy and photon energy within specified range
        image_buff[rolling_count-1]=opal_image # rolling_count has been incremented already
        x_proj_buff[rolling_count-1]=x_proj    # so fill these with rolling_count-1
        good_shot_count+=1

    if rolling_count==refresh_rate:
        # Reduce all the sums to the sum on the root core. First histograms which are never reset.
        comm.Reduce(hist_L3PhotEnergy.values, hist_L3PhotEnergy_all, root=0) # array onto array
        comm.Reduce(hist_FeeGasEnergy.values, hist_FeeGasEnergy_all, root=0) # array onto array
        comm.Reduce(hist_FeeGasEnergy_CountsROI.values, hist_FeeGasEnergy_CountsROI_all, root=0)
        # array onto array

        #Now the ones which are going to be reset
        good_shot_count_all=comm.reduce(good_shot_count, root=0) # syntax is different for 
                                                                 # reduce() vs Reduce()     
        comm.Reduce(image_buff.sum(axis=0), image_sum_slice, root=0)
        comm.Reduce(x_proj_buff.sum(axis=0), x_proj_sum_slice, root=0)
        # gather() creates list of lists which will then be concatenated
        counts_buff_toappend=comm.gather(counts_buff, root=0)
        counts_buff_roi_toappend=comm.gather(counts_buff_roi, root=0)     

        # Now reset the variables which should be
        good_shot_count=0
        image_buff=np.zeros((refresh_rate, i_len, j_len))
        x_proj_buff=np.zeros((refresh_rate, j_len))

        counts_buff=np.zeros((refresh_rate,2)) # reset to 0
        counts_buff_roi=np.zeros((refresh_rate,2)) # reset to 0

        hist_L3PhotEnergy = Histogram((numbins_L3PhotEnergy,minhistlim_L3PhotEnergy,maxhistlim_L3PhotEnergy))
        hist_FeeGasEnergy = Histogram((numbins_FeeGasEnergy,minhistlim_FeeGasEnergy,maxhistlim_FeeGasEnergy))
        hist_FeeGasEnergy_CountsROI = Histogram((numbins_FEE_CountsROI_FEE,minhistlim_FEE_CountsROI_FEE,\
                                                maxhistlim_FEE_CountsROI_FEE),(numbins_FEE_CountsROI_CountsROI,\
                                                minhistlim_FEE_CountsROI_CountsROI,maxhistlim_FEE_CountsROI_CountsROI))

        rolling_count=0

        if rank==0:
            # Don't care so much that these are in order because we don't plot
            # them shot-to-shot
            image_sum_buff.append(image_sum_slice) 
            x_proj_sum_buff.append(x_proj_sum_slice)
            good_shot_count_buff.append(good_shot_count_all)
          
            image_sum+=image_sum_slice
            x_proj_sum+=x_proj_sum_slice
            good_shot_count_pcage=100*(sum(good_shot_count_buff)/float(len(good_shot_count_buff)*refresh_rate*size))

            # Only take away from the rolling sum if we have had more than the max history
            # length of shots
            if len(x_proj_sum_buff)==x_proj_sum_buff.maxlen:
                image_sum-=image_sum_buff[0] # don't pop, let the deque with finite maxlen
                x_proj_sum-=x_proj_sum_buff[0] # take care of that itself
                numshotsforacc=(len(x_proj_sum_buff)-1)*refresh_rate*size
            else:
                numshotsforacc=len(x_proj_sum_buff)*refresh_rate*size
            
            # Ensure that the counts buffer is filled in the correct order and fill it
            counts_buff_tosort=np.concatenate(counts_buff_toappend)
            counts_buff_all+=counts_buff_tosort[np.argsort(counts_buff_tosort[:,0])][:,1].tolist()

            counts_buff_roi_tosort=np.concatenate(counts_buff_roi_toappend)
            counts_buff_roi_all+=counts_buff_roi_tosort[np.argsort(counts_buff_roi_tosort[:,0])][:,1].tolist()
   
            ref_time_update=time.time()
            speed=refresh_rate*size/float(ref_time_update-ref_time) # estimate of rate of data acquisition
            ref_time=ref_time_update
             
            sendMultiPlot(x_proj_sum, image_sum, counts_buff_all, counts_buff_roi_all, opal_image, \
                          (hist_L3PhotEnergy_all, hist_L3PhotEnergy.edges), (hist_FeeGasEnergy_all, \
                          hist_FeeGasEnergy.edges), (hist_FeeGasEnergy_CountsROI_all, hist_FeeGasEnergy_CountsROI.edges), \
                          nevt, numshotsforacc, good_shot_count_pcage, speed)
            #print 'I just tried to plot'

        # Reset these to zero on all cores (to be sure...!)
        good_shot_count_all=0
        image_sum_slice=np.zeros((i_len, j_len))
        x_proj_sum_slice=np.zeros(j_len)



