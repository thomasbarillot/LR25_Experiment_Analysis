##### Useful commands
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

# custom algorithms
#from pypsalg import find_blobs
import find_blobs

# parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# custom functions
def moments(arr):
    if np.count_nonzero(arr) == 0:
        return 0,0
    nbins = len(arr)
    bins  = range(nbins)
    mean  = np.average(bins,weights=arr)
    var   = np.average((bins-mean)**2,weights=arr)
    sdev  = np.sqrt(var)
    return mean,sdev

def FWHM(arr):  #  ****************** IN PROGRESS *************************
    if np.count_nonzero(arr) == 0:
        return 0,0
    nbins = len(arr)
    bins  = range(nbins)
    mean  = np.average(bins,weights=arr)
    var   = np.average((bins-mean)**2,weights=arr)
    sdev  = np.sqrt(var)
    return mean,sdev

if rank==0:
    publish.init()


# Set up event counters
eventCounter = 0
evtGood = 0
evtBad = 0
evtUsed = 0

# Buffers for histories
history_len = 6
history_len_long = 100
opal_hit_buff = collections.deque(maxlen=history_len)
opal_hit_avg_buff = collections.deque(maxlen=history_len_long)
opal_circ_buff = collections.deque(maxlen=history_len)
xproj_int_buff = collections.deque(maxlen=history_len)
xproj_int_avg_buff = collections.deque(maxlen=history_len)
moments_buff = collections.deque(maxlen=history_len_long)
#moments_avg_buff = collections.deque(maxlen=history_len_long)
#xhistogram_buff = collections.deque(maxlen=history_len)
hitxprojhist_buff = collections.deque(maxlen=history_len)
hitxprojjitter_buff = collections.deque(maxlen=history_len)
hitxprojseeded_buff = collections.deque(maxlen=history_len)
delayhist2d_buff = collections.deque(maxlen=history_len)
minitof_volts_buff = collections.deque(maxlen=history_len)
hitrate_roi_buff = collections.deque(maxlen=history_len_long)


# Histograms
hithist = Histogram((100,0.,1023.))
hitjitter = Histogram((100,0.,1023.))
hitseeded = Histogram((100,0.,1023.))
delayhist2d = Histogram((100,0.,1023.),(20,-100.,100.))

# ion yield array for SXRSS scan
ion_yield = np.zeros(102) ##choose appropriate range

# For perspective transformation and warp
pts1 = np.float32([[96,248],[935,193],[96,762],[935,785]])
xLength = 839
yLength = 591
pts2 = np.float32([[0,0],[xLength,0],[0,yLength],[xLength,yLength]])
M = cv2.getPerspectiveTransform(pts1,pts2)

print "DONE"

###### --- Online analysis
ds = DataSource('shmem=psana.0:stop=no')
###### --- Offline analysis
#ds = DataSource("exp=AMO/amon0816:run=156:smd:dir=/reg/d/ffb/amo/amon0816/xtc:live")#"exp=AMO/amon0816:run=165:smd"

offline = 'shmem' not in ds.env().jobName()

if not offline:
    calibDir = '/reg/d/psdm/amo/amon0816/calib'
    setOption('psana.calib-dir', calibDir)

XTCAVRetrieval = ShotToShotCharacterization();
XTCAVRetrieval.SetEnv(ds.env())
opal_det = Detector('OPAL1')
#opal4_det = Detector('OPAL4')
minitof_det = Detector('ACQ1')
minitof_channel = 0
ebeam_det = Detector('EBeam')
eorbits_det = Detector('EOrbits')

for nevt,evt in enumerate(ds.events()):

        if offline:
            if nevt%size!=rank: continue
            
        ##############################  READING IN DETECTORS #############################
        ### Opal Detector
        opal_raw = opal_det.raw(evt)
        #opal4_raw = opal4_det.raw(evt)


        ### TOF
        minitof_volts_raw = minitof_det.waveform(evt)
        minitof_times_raw = minitof_det.wftime(evt)
        

        ### Ebeam
        #ebeam = ebeam_det.get(evt)

        ### Eorbits
        eorbits = eorbits_det.get(evt)
        
        # Check all detectors are read in
        eventCounter += 1
        if opal_raw is None or minitof_volts_raw is None or minitof_times_raw is None or eorbits is None:
            evtBad += 1
            print "Bad event"
            continue
        else:
            evtGood += 1

        opal = opal_raw.copy()
        #opal_screen = opal4_raw.copy()
        minitof_volts = minitof_volts_raw[minitof_channel]
        minitof_times = minitof_times_raw[minitof_channel]

        ##############################################################################



        ################################## OPAL ANALYSIS #############################
        # threshold the image
        threshold = 500
        opal_thresh = (opal>threshold)*opal

        # Perform Perspective Transformation
        opal_perspective = cv2.warpPerspective(opal_thresh,M,(xLength,yLength))

        # do two projections
        opal_thresh_xproj = np.sum(opal_thresh,axis=0)

        # sum up the projected image in bin range 100 to 200
        #integrated_projx = np.sum(opal_thresh_xproj[100:200])

        # do blob finding
        # find blobs takes two variables. an Opal image (first) and then a threshold value (second)
        c,w = find_blobs.find_blobs(opal,threshold)

        #find center of gravity. If there is a hit
        if len(c) != 0:
            x2,y2 = zip(*c)
            blob_centroid = np.sum(x2)/float(len(x2))
            #print blob_centroid

            shift = 512-int(blob_centroid)
            mu,sig = moments(opal_thresh_xproj)	
            index = int(blob_centroid/10)            
	
            # uncomment bellow for two pulse two color jitter correction only !!!!

            #bot = 0     # first edge of the first energy peak
            #top = 700   # second edge of the first energy peak
            #mu,sig = moments(opal_thresh_xproj[bot:top])	
            #shift = 250 - int(mu) - bot
            #index = int(mu/10)        
	
        else:
            shift = 0
            index = 0
            
        #print shift
        # Hit Histogram
        hithist.reset()
        hitjitter.reset()
        hitseeded.reset()
        delayhist2d.reset()
        for hit in c:
            hithist.fill(float(hit[1]+shift))
            hitjitter.fill(float(hit[1]))
                

        ###############################################################################


        ############################ MINI TOF ANALYSIS ################################

        # find yield of waveform over certain interval
        wf_yield = np.sum(minitof_volts[1:401]-minitof_volts[5300:5700]) #choose proper window later  
        # get maximum value of x-ray beam
        #max_hithist = np.amax(hithist.data,axis=0) # might have to change this to x2.data?
        max_hithist = np.amax(opal_thresh_xproj,axis=0) # might have to change this to x2.data?
        #print max_hithist

        if max_hithist > 25000: #change value later
            ion_yield[index] = ion_yield[index] + wf_yield
            for hit in c:
                hitseeded.fill(float(hit[0]))        

        minitof_volts_thresh = minitof_volts
        minitof_volts_thresh[minitof_volts_thresh > -0.01] = 0

        #print 'number of hits:',len(c)
        #find_blobs.draw_blobs(opal,c,w) # draw the blobs in the opal picture

#        if 'sig' in locals():
#            print sig
#            if sig < 10: #change width condition
#                ion_yield[index] = ion_yield[index] + wf_yield

        ###############################################################################

        ######################### XTCAV ANALYSIS #####################################
        if XTCAVRetrieval.SetCurrentEvent(evt):
            time,power,ok = XTCAVRetrieval.XRayPower()
            #print ok
            if ok:
                agreement,ok=XTCAVRetrieval.ReconstructionAgreement()
                #print ok
                if ok and agreement > 0.5:
                    times_p0 = np.asarray(time[0])
                    power_p0 = np.asarray(power[0])
                    #times_p1 = np.asarray(time[1])
                    #power_p1 = np.asarray(power[1])

                    mean_t_p0 = np.sum(times_p0*power_p0/np.sum(power_p0))
                    var_t_p0 = np.sum(times_p0**2*power_p0/np.sum(power_p0))
                    rms_p0 = np.sqrt(var_t_p0 - mean_t_p0**2)

                    #mean_t_p1 = np.sum(times_p1*power_p1/np.sum(power_p1))
                    #var_t_p1 = np.sum(times_p1**2*power_p1/np.sum(power_p1))
                    #rms_p1 = np.sqrt(var_t_p1 - mean_times_p1**2)
                    
                    pulse_separation = mean_t_p0#- mean_t_p1
                    #print pulse_separation
                    
                    #for hit in c:
                    #    delayhist2d.fill(hit[1],[pulse_separation])


        ###############################################################################
        ############################### FOR PARALLELIZATION ###########################
        ###############################################################################
        

        ############################# Histories of certain values #####################

        # x-projection histogram history
        hitxprojhist_buff.append(hithist.values)
        hitxprojhist_sum = sum(hitxprojhist_buff)

        hitxprojjitter_buff.append(hitjitter.values)
        hitxprojjitter_sum = sum(hitxprojjitter_buff)

        hitxprojseeded_buff.append(hitseeded.values)
        hitxprojseeded_sum = sum(hitxprojseeded_buff)

        delayhist2d_buff.append(delayhist2d.values)
        delayhist2d_sum = sum(delayhist2d_buff)

        # x-projection history
        #xproj_int_buff.append(integrated_projx)
        #xproj_int_sum = np.array([sum(xproj_int_buff)])#/len(xproj_int_buff)

        # TOF average
        minitof_volts_buff.append(minitof_volts_thresh)
        minitof_volts_sum = sum(minitof_volts_buff)



        # Opal hitcounter history
        opal_hit_buff.append(len(c))
        opal_hit_sum = np.array([sum(opal_hit_buff)])#/len(opal_hit_buff)        

        # Opal history
        opal_circ_buff.append(opal_thresh)
        opal_sum = sum(opal_circ_buff)
        

        # only update the plots and call comm.Reduce "once in a while"
        if evtGood%5 == 0:
            ### create empty arrays and dump for master
            #if not 'moments_sum_all' in locals():
            #    moments_sum_all = np.empty_like(moments_sum)
            #comm.Reduce(moments_sum,moments_sum_all)

            #if not 'xproj_in_sum_all' in locals():
            #    xproj_int_sum_all = np.empty_like(xproj_int_sum)
            #comm.Reduce(xproj_int_sum,xproj_int_sum_all)

            if not 'opal_hit_sum_all' in locals():
                opal_hit_sum_all = np.empty_like(opal_hit_sum)
            comm.Reduce(opal_hit_sum,opal_hit_sum_all)

            if not 'opal_sum_all' in locals():
                opal_sum_all = np.empty_like(opal_sum)
            comm.Reduce(opal_sum,opal_sum_all)

            if not 'hithist_sum_all' in locals():
                hithist_sum_all = np.empty_like(hithist.values)
            comm.Reduce(hithist.values,hithist_sum_all)

            if not 'hitxprojhist_sum_all' in locals():
                hitxprojhist_sum_all = np.empty_like(hitxprojhist_sum)
            comm.Reduce(hitxprojhist_sum,hitxprojhist_sum_all)

            if not 'hitjitter_sum_all' in locals():
                hitjitter_sum_all = np.empty_like(hitjitter.values)
            comm.Reduce(hitjitter.values,hitjitter_sum_all)

            if not 'hitxprojjitter_sum_all' in locals():
                hitxprojjitter_sum_all = np.empty_like(hitxprojjitter_sum)
            comm.Reduce(hitxprojjitter_sum,hitxprojjitter_sum_all)

            if not 'minitof_volts_sum_all' in locals():
                minitof_volts_sum_all = np.empty_like(minitof_volts_sum)
            comm.Reduce(minitof_volts_sum,minitof_volts_sum_all)

            if not 'delayhist2d_sum_all' in locals():
                delayhist2d_sum_all = np.empty_like(delayhist2d_sum)
            comm.Reduce(delayhist2d_sum,delayhist2d_sum_all)

            if not 'ion_yield_sum_all' in locals():
                ion_yield_sum_all = np.empty_like(ion_yield)
            comm.Reduce(ion_yield,ion_yield_sum_all)

            if not 'hitxprojseeded_sum_all' in locals():
                hitxprojseeded_sum_all = np.empty_like(hitxprojseeded_sum)
            comm.Reduce(hitxprojseeded_sum,hitxprojseeded_sum_all)



            if rank==0:
                ######################################### SOME ANALYSIS ON RANK 0 #####################################
                ###### calculating moments of the hithist.data
                m,s = moments(hitxprojjitter_sum_all) #changed from hitxprojhist_sum_all

                ###### History on master
                print eventCounter*size, 'total events processed.'
                opal_hit_avg_buff.append(float(opal_hit_sum_all[0])/float(len(opal_hit_buff)*size))
                #xproj_int_avg_buff.append(xproj_int_sum_all[0]/(len(xproj_int_buff)*size))
                #moments_avg_buff.append(moments_sum_all[0]/(len(moments_buff)*size))
                ### moments history
                moments_buff.append(s)
                #moments_sum = np.array([sum(moments_buff)])#/len(moments_buff)
                opal_pers_xproj = np.sum(opal_perspective,axis=0)
                hitrate_roi_buff.append(np.sum(hitxprojhist_sum_all[40:60])/(len(hitxprojhist_buff)*size))
                
                #####################################################################################################
                
                
                ###################################### EXQUISIT PLOTTING ############################################
                ax = range(0,len(opal_thresh_xproj))
                xproj_plot = XYPlot(evtGood, "X Projection", ax, opal_thresh_xproj,formats=".-") # make a 1D plot
                publish.send("XPROJ", xproj_plot) # send to the display

                # #
                opal_hit_avg_arr = np.array(list(opal_hit_avg_buff))
                # print opal_hit_avg_buff
                ax2 = range(0,len(opal_hit_avg_arr))
                hitrate_avg_plot = XYPlot(evtGood, "Hitrate history", ax2, opal_hit_avg_arr,formats=".-") # make a 1D plot
                # print 'publish',opal_hit_avg_arr
                publish.send("HITRATE", hitrate_avg_plot) # send to the display
                # #
                axroihit = range(0,len(hitrate_roi_buff))
                hitrate_avg_plot2 = XYPlot(evtGood, "Hitrate history ROI", axroihit, np.array(list(hitrate_roi_buff)),formats=".-") # make a 1D plot
                # print 'publish',opal_hit_avg_arr
                publish.send("ROIHITRATE", hitrate_avg_plot2) # send to the display


                #xproj_int_avg_arr = np.array(list(xproj_int_avg_buff))
                #ax3 = range(0,len(xproj_int_avg_arr))
                #xproj_int_plot = XYPlot(evtGood, "XProjection running avg history", ax3, xproj_int_avg_arr) # make a 1D plot
                #publish.send("XPROJRUNAVG", xproj_int_plot) # send to the display
                # #
                moments_avg_arr = np.array(list(moments_buff))
                ax4 = range(0,len(moments_avg_arr))
                moments_avg_plot = XYPlot(evtGood, "Standard Deviation of avg Histogram", ax4, moments_avg_arr,formats=".-") # make a 1D plot
                publish.send("MOMENTSRUNAVG", moments_avg_plot) # send to the display

                # average histogram of hits
                ax5 = range(0,hitxprojhist_sum_all.shape[0])
                hithist_plot = XYPlot(evtGood, "Hit Hist2", ax5, hitxprojhist_sum_all/(len(hitxprojhist_buff)*size),formats=".-") # make a 1D plot
                publish.send("HITHIST", hithist_plot) # send to the display

                # histogram of single shot
                ax6 = range(0,hithist.values.shape[0])
                hit1shot_plot = XYPlot(evtGood, "Hit Hist 1 shot", ax6, hithist.values,formats=".-") # make a 1D plot
                publish.send("HIT1SHOT", hit1shot_plot) # send to the display

                # #
                opal_plot = Image(evtGood, "Opal", opal) # make a 2D plot
                publish.send("OPAL", opal_plot) # send to the display

                # #
                #opalscreen_plot = Image(evtGood, "Opal screen", opal_screen) # make a 2D plot
                #publish.send("OPALSCREEN", opalscreen_plot) # send to the display

                # Perspective transformation
                opal_plot2 = Image(evtGood, "Opal perspective transformed", opal_perspective) # make a 2D plot
                publish.send("OPALPERS", opal_plot2) # send to the display

                # Perspective transformation avg
                opal_plot3 = Image(evtGood, "Opal average perspective transformed",cv2.warpPerspective(opal_sum_all,M,(xLength,yLength)) ) # make a 2D plot
                publish.send("OPALAVGPERS", opal_plot3) # send to the display

                # Perspective x-projection opal_pers_xproj
                opal_pers_xproj_plot = XYPlot(evtGood, "Opal perspective X-Projection", np.arange(len(opal_pers_xproj)),opal_pers_xproj,formats=".-") # make a 1D plot
                publish.send("OPALPERSXPROJ",opal_pers_xproj_plot) # send to the display

                # #
                opal_plot_avg = Image(evtGood, "Opal Average", opal_sum_all) # make a 2D plot
                publish.send("OPALAVG", opal_plot_avg) # send to the display

                # delayhist2d_sum_all
                delayhist2d_plot = Image(evtGood, "Histogram avg", delayhist2d_sum_all) # make a 2D plot
                publish.send("DELAYHIST", delayhist2d_plot) # send to the display
                
                # Exquisit plotting, acqiris
                wf_plot = XYPlot(evtGood, "MINI TOF - single traces", minitof_times,minitof_volts,formats=".-") # make a 1D plot
                publish.send("TOF", wf_plot) # send to the display

                # acqiris average threshold w/e minitof_volts_sum_all
                wf_plot2 = XYPlot(evtGood, "MINI TOF - average traces", minitof_times,minitof_volts_sum_all,formats=".-") # make a 1D plot
                publish.send("TOFAVG", wf_plot2) # send to the display

                # Exquisit plotting, ion yield
                ax_ion = range(0,len(ion_yield))
                ion_yield_plot = XYPlot(evtGood, "Ion Yield", ax_ion, ion_yield,formats=".-") # make a 1D plot
                publish.send("ION", ion_yield_plot) # send to the display

                # average of ion yield sum all
                ax_ion_all = range(0,ion_yield_sum_all.shape[0])
                ion_all_plot = XYPlot(evtGood, "Ion Yield All", ax_ion_all, ion_yield_sum_all,formats=".-") # make a 1D plot
                publish.send("IONALL", ion_all_plot) # send to the display

                # histogram of single shot, no jitter correction
                axj = range(0,hitjitter.values.shape[0])
                hitjitter_plot = XYPlot(evtGood, "Hit Hist No Jitter Correction", axj, hitjitter.values,formats=".-") # make a 1D plot
                publish.send("HITJITTER", hitjitter_plot) # send to the display

               # average histogram of hits, no jitter correction
                axja = range(0,hitxprojjitter_sum_all.shape[0])
                hitjitter_plot_avg = XYPlot(evtGood, "Hit Average, No Jitter Correction", axja, hitxprojjitter_sum_all/(len(hitxprojjitter_buff)*size),formats=".-") # make a 1D plot
                publish.send("HITJITAVG", hitjitter_plot_avg) # send to the display

                # histogram of single shot, seeded only
                axseed = range(0,hitseeded.values.shape[0])
                hitseeded_plot = XYPlot(evtGood, "Only Seeded", axseed, hitseeded.values,formats=".-") # make a 1D plot
                publish.send("HITSEED", hitseeded_plot) # send to the display

               # average histogram of hits, no jitter correction
                axseedavg = range(0,hitxprojseeded_sum_all.shape[0])
                hitseeded_plot_avg = XYPlot(evtGood, "Only Seeded Average", axja, hitxprojseeded_sum_all/(len(hitxprojseeded_buff)*size),formats=".-") # make a 1D plot
                publish.send("HITSEEDAVG", hitseeded_plot_avg) # send to the display
                
                # Eorbits X
                ax_eorbitsX = range(0,len(eorbits.fBPM_X()))
                eorbits_plotX = XYPlot(evtGood,"EOrbits X",ax_eorbitsX,eorbits.fBPM_X())
                publish.send("EORBITSX", eorbits_plotX) # send to the display

                # Eorbits X
                ax_eorbitsY = range(0,len(eorbits.fBPM_Y()))
                eorbits_plotY = XYPlot(evtGood,"EOrbits Y",ax_eorbitsY,eorbits.fBPM_Y())
                publish.send("EORBITSY", eorbits_plotY) # send to the display
