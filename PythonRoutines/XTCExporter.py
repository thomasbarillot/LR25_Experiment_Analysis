from psana import *
import numpy as np
import scipy.io
#import mpidata
import random
from xtcav.ShotToShotCharacterization import * 
import pypsalg
import os



#### Analysis modules
sys.path.append('/reg/neh/home4/tbarillo/amolr2516/LR25_Analysis/PythonRoutines/')

import ITOFDataPreProcessing
import UXSDataPreProcessing
import SHESPreProcessing
import XTCAV_Processing

#### Parallel processing

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###############################################################################

class XTCExporter(object):
    def __init__(self,args):
        self.args = args

###############################################################################

    def run(self):
        
        self.filenum = 0
        self.DetInit()
        self.endrun = False
        #self.md = mpidata()
        self.loop_idx = 0
        self.ArrInit(1024)
        for nevent,evt in enumerate(self.ds.events()):
            if nevent == self.args.noe : break
            if nevent%(size)!=rank: continue # different ranks look at different events
            if rank==0 and nevent%20==0:
                print nevent
            self.loop_idx += 1

            self.getevtdata(evt)
            
            
            #print self.nsave
            if hasattr(self,'nsave') and self.nsave == self.args.nsave:
                self.save()
                
        self.save()

        print 'Client', rank, 'done'
        
###############################################################################      
                
    def DetInit(self):
        self.ds = DataSource(self.args.exprun + \
                             ':smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live')
        
        #XTCAVRetrieval = ShotToShotCharacterization()
        #XTCAVRetrieval.SetEnv(self.ds.env())

        self.SHES  = SHESPreProcessing.SHESPreProcessor()
        self.UXS = Detector(self.args.UXS)
        self.UXS_Pre = UXSDataPreProcessing.UXSDataPreProcessing()
        self.ITOF = Detector(self.args.ITOF)    
        self.XTCAV = XTCAV_Processing.XTCavProcessor()
        self.EBeam = Detector(self.args.EBeam)
        self.GMD   = Detector(self.args.GMD)
        self.ENV = self.ds.env().configStore()
        self.PRESS = Detector('AMO:LMP:VG:21:PRESS')
    
###############################################################################

    def ArrInit(self,uxslength):
        ## SHES
        # Scienta Hemispherical array 2D: Energy projection; index of event
        shlength=len(self.SHES.calib_array)
        self.shEnergy = np.zeros(shlength)
        self.shProjArr = np.zeros((self.args.nsave,shlength))
        self.ehitsX = []
        self.ehitsY = []
        self.ehitsTS = []

        ## UXS
        # XRay spectro array 2D principal components: Ampl1,pos1,FWHM1,Ampl2,pos2,FWHM2 
        self.uxsPCArr = np.zeros((self.args.nsave,6))
        # Xray spectro array 2D: Energy projection
        self.uxsProjArr = np.zeros((self.args.nsave,uxslength))
    
        ## ITOF
        # micro tof: ion yield
        self.itofArr = np.zeros((self.args.nsave, 40000))
    
        ## GMD
        # Gas detector array (6 ebeam EL3 values)
        self.gmdArr   = np.zeros((self.args.nsave,6))

        ## ENV
        # Acq parameters: offset and fullscale for both channels low gain: 2, high gain 1
        self.envArr   = np.zeros((1,5))
            
        ## EBeam
        self.ebeamArr = np.zeros((self.args.nsave,21))

        ## EPICS
        # Sample pressure array
        self.sPressArr = np.zeros((self.args.nsave))
    
        # Time stamp array         
        self.TimeSt = np.zeros((self.args.nsave,))
       
        # initialize arrays to nan 
        self.shProjArr[:,:] = np.nan
        self.uxsProjArr[:,:] = np.nan
        self.uxsPCArr[:,:] = np.nan
        self.itofArr[:] = np.nan

        self.gmdArr[:,:]    = np.nan
        self.ebeamArr[:,:]  = np.nan
        self.envArr[:,:] = np.nan
        self.TimeSt[:]    = np.nan
        self.sPressArr[:] = np.nan
        self.nsave        = 0


###############################################################################
        
    def getevtdata(self,evt):
        print 'Loop iteration {0}, {1} events saved'.format(self.loop_idx, self.nsave)
        # Initialize the arrays
        if not hasattr(self,'shEnergy'):
            shEnergy = len(self.SHES.calib_array)
            if shEnergy is None:
                print 'shEnergy is None. Returning'
                return
        self.shEnergy = self.SHES.calib_array

        # Get the event ID and timestamp
        evtid = evt.get(EventId)
        evttime = evtid.idxtime()
        try:
            self.TimeSt[self.nsave] = evttime.time()
        except Exception as e:
            print 'getting event time failed with exception {0}'.format(e)
            return

        # Get the hemisperical analyser signal data
        lx,ly,proj,proj_raw = self.SHES.PreProcess(evt)
        self.ehitsX += lx #Xpos
        self.ehitsY += ly#Ypos
        self.ehitsTS += list(np.ones((len(lx)))*evttime.time()) #electron hits timestamps
        self.shProjArr[self.nsave,:]=proj
        if any((any(np.isnan([lx])), any(np.isnan([ly])), any(np.isnan([proj])), any(np.isnan([proj_raw])))):
            print 'Pre-processing of SHES data failed'

        # Get the Xray spectrometer analyser data
        XrayImg=self.UXS.raw(evt)
        if XrayImg is not None:
            self.uxsPCArr[self.nsave,:],self.uxsProjArr[self.nsave,:]=self.UXS_Pre.StandardAnalysis(XrayImg)
        else:
            print 'No UXS data ({0} events saved)'.format(self.nsave)
        
        # Get the ITOF data
        waveforms = self.ITOF.waveform(evt)
        if waveforms is not None:
            waveform = waveforms[1]
            tmp=ITOFDataPreProcessing.ITOFDataPreProcessing(waveform)
            tmp.StandardAnalysis()
            self.itofArr[self.nsave, :] = tmp.wf
        else:
            print 'No ITOF data ({0} events saved)'.format(self.nsave)

        # Get the XTCAV data
        #self.XTCAV.set_data_source(self.ds)
        #self.XTCAV.set_event(evt)
        #success=self.XTCAV.process()
        #if not success:
        #    print 'No XTCAV data'
  
        # Get the environnement data     
        envdata=self.ENV.get(psana.Acqiris.Config)
        gmddata  = self.GMD.get(evt)
        samplepressuredata = self.PRESS(evt)
    
        # Get the Ebeam data
        EBeamdata = self.EBeam.get(evt)
        if EBeamdata is not None:         
            # record ebeam parameters        
            ebeamlistpar=['damageMask','ebeamCharge','ebeamDumpCharge','ebeamEnergyBC1', \
                          'ebeamEnergyBC2','ebeamL3Energy','ebeamLTU250','ebeamLTU450', \
                          'ebeamLTUAngX','ebeamLTUAngY','ebeamLTUPosX','ebeamLTUPosY', \
                          'ebeamPhotonEnergy','ebeamPkCurrBC1','ebeamPkCurrBC2','ebeamUndAngX', \
                          'ebeamUndAngY','ebeamUndPosX','ebeamUndPosY','ebeamXTCAVAmpl','ebeamXTCAVPhase']
        
            for i,par in enumerate(ebeamlistpar):
                self.ebeamArr[self.nsave,i] = getattr(EBeamdata,par)()
        else:
            print 'No EBeam data ({0} events saved)'.format(self.nsave)
                
        
        if gmddata is not None:
            #upstream
            self.gmdArr[self.nsave,0] = gmddata.f_11_ENRC()
            self.gmdArr[self.nsave,1] = gmddata.f_12_ENRC()
            #downstream
            self.gmdArr[self.nsave,2] = gmddata.f_21_ENRC()
            self.gmdArr[self.nsave,3] = gmddata.f_22_ENRC()
            #set values
            self.gmdArr[self.nsave,4] = gmddata.f_63_ENRC()
            self.gmdArr[self.nsave,5] = gmddata.f_64_ENRC()
            
            gmd = self.gmdArr[self.nsave,self.args.prefgmd]
        else:
            print 'No Gas Detector data ({0} events saved)'.format(self.nsave)
            
        if envdata is not None and self.nsave==0:
            self.envArr[self.nsave,0] = envdata.vert()[0].offset()
            self.envArr[self.nsave,1] = envdata.vert()[0].fullScale()
            self.envArr[self.nsave,2] = envdata.vert()[2].offset()
            self.envArr[self.nsave,3] = envdata.vert()[2].fullScale()
            self.envArr[self.nsave,4] = envdata.horiz().sampInterval()
        else:
            print 'No Env data ({0} events saved)'.format(self.nsave)
            
        if samplepressuredata is not None:
            self.sPressArr[self.nsave] = samplepressuredata
        else:
            print 'No pressure data ({0} events saved)'.format(self.nsave)

        self.nsave += 1
        if self.nsave%100 == 0:
            print 'Processed {0} events in rank {1}'.format(self.nsave, rank)

###############################################################################        
            
    def save(self):
        if self.args.save == True:
             
            runnumber = int(self.args.exprun[18:])
            filename = 'amolr2516_r' + str(runnumber).zfill(4) + '_' + \
                        str(rank).zfill(3) + '_' + str(self.filenum).zfill(3)
                     
            directory_n='/reg/d/psdm/AMO/amolr2516/results/npzfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            directory_m='/reg/d/psdm/AMO/amolr2516/results/matfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            if not os.path.exists(directory_n):
                os.makedirs(directory_n)
            if not os.path.exists(directory_m):
                os.makedirs(directory_m)
            directory_n='/reg/d/psdm/AMO/amolr2516/results/npzfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            directory_m='/reg/d/psdm/AMO/amolr2516/results/matfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            if not os.path.exists(directory_n):
                os.makedirs(directory_n)
            if not os.path.exists(directory_m):
                os.makedirs(directory_m)
            
            if rank == 1:
                print 'rank 1 writing file...'
            
        # Save electrons hits in one array:
        SHESHits=np.zeros((len(self.ehitsX),3))
        SHESHits[:,0]=self.ehitsX
        SHESHits[:,1]=self.ehitsY
        SHESHits[:,2]=self.ehitsTS

        data={'EBeamParameters':self.ebeamArr,
              'SHESHits':SHESHits.astype(np.float64), \
              'SHESwf':self.shProjArr[0:self.nsave,:].astype(np.float16), \
              'UXSpc':self.uxsPCArr[0:self.nsave,:].astype(np.float16),\
              'UXSwf':self.uxsProjArr[0:self.nsave,:].astype(np.float16),\
              'ITOF':self.itofArr[0:self.nsave].astype(np.float16),\
              'Pressure':self.sPressArr[0:self.nsave],\
              'GasDetector':self.gmdArr[0:self.nsave,:], \
              'EnvVar':self.envArr[0:self.nsave,:], \
              'TimeStamp':self.TimeSt}
        scipy.io.savemat(directory_m+filename+'.mat',data)
                                       
        np.savez(directory_n+filename,**data)
        if rank == 1:
            print 'rank1 done with writing file.'
        self.filenum += 1    
        self.ArrInit(1024)

###############################################################################        
            
def runclient(args):
    client = XTCExporter(args)
    client.run()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.noe = -1
    args.SHES   = 'OPAL3'
    args.UXS    = 'OPAL1'
    args.ITOF   = 'ACQ1'#'Acq01'#'ACQ1'
    args.GMD    = 'FEEGasDetEnergy'
    args.EBeam  = 'EBeam'
    args.PRESS  = 'AMO:LMP:VG:21:PRESS'
    args.XTCAV  = 'xtcav'

    # GMD intensity threshold for shots to be considered for online plots:
    #args.gmdThr = 0.1
    # Prefered gmd value for online plots:
    args.prefgmd = 0

    # Prefered  channel for online plots and raw traces of itof:
    #args.prefch = 1

    # Save files:
    args.save = True
    # Number of shots saved per file
    args.nsave  = 2000
    args.exprun = 'exp=amolr2516:run=37'
    runclient(args)
