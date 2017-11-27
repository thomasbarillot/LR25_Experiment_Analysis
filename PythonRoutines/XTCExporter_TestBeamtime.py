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
        
        for nevent,evt in enumerate(self.ds.events()):
            if nevent == self.args.noe : break
            if nevent%(size)!=rank: continue # different ranks look at different events
            if rank==0 and nevent%20==0:
                print nevent

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
        
        XTCAVRetrieval = ShotToShotCharacterization()
        XTCAVRetrieval.SetEnv(self.ds.env())

        self.SHES  = Detector(self.args.SHES)
        self.UXS = Detector(self.args.UXS)
	self.ITOF = Detector(self.args.ITOF)	
	self.XTCAV = Detector(self.args.XTCAV)
        self.EBeam = Detector(self.args.EBeam)
        self.GMD   = Detector(self.args.GMD)
        self.ENV = self.ds.env().configStore()
	self.PRESS = Detector(self.args.PRESS)

###############################################################################

    def ArrInit(self,shlength,uxslength):

	## SHES
	# Scienta Hemispherical array 2D: Energy projection; index of event
	self.shenergy = np.zeros((shlength))
	self.shProjArr = np.zeros((self.args.nsave,shlength))
	self.ehits = [[],[],[]]

	## UXS
	# XRay spectro array 2D principal components: Ampl1,pos1,FWHM1,Ampl2,pos2,FWHM2 
	self.uxsPCArr = np.zeros((self.args.nsave,6))
	# Xray spectro array 2D: Energy projection
	self.uxsProjArr = np.zeros((self.args.nsave,uxslength))
	
	## ITOF
	# micro tof: ion yield
	self.itofArr = np.zeros((self.args.nsave))
	
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
	self.shProjArr[:,:,:] = np.nan
	self.uxsProjArr[:,:] = np.nan
	self.uxsPCArr[:,:] = np.nan
	self.itofArr[:] = np.nan

        self.gmdArr[:,:]    = np.nan
        self.ebeamArr[:,:]  = np.nan
        self.envArr[:,:] = np.nan
        self.TimeSt[:]    = np.nan
	self.sPressArr[:] = np.nan
        self.nsave        = np.nan


###############################################################################
        
    def getevtdata(self,evt):

	# Call the preprocessing modules here

	# Initialize the arrays
        if not hasattr(self,'shenergy'):
            shenergy = self.SHES.raw(evt).shape[0]
            if shenergy is None: return
                
            self.ArrInit(self.SHES.raw(evt).shape[0],self.UXS.raw(evt).shape[0])
            self.shenergy = mbtime[0,:]

        # Get the hemisperical analyser signal data
	self.ehits[0].append() #Xpos
        self.ehits[1].append() #Ypos
	self.ehits[2].append() #evt index
            
	# Get the Xray spectrometer analyser data
	XrayImg=self.UXS.raw(evt)
	if XrayImg is not None:
	    tmp=UXSDataPreProcessing(XrayImg)
	    uxsPCArr[self.nsave,:],uxsProjArr[self.nsave,:]=tmp.StandardAnalysis()
	# Get the ITOF data
        
	iwf = self.ITOF.waveform(evt)[0]
        if iwf is not None:
            tmp=ITOFDataPreProcessing(iwf)
            itofArr[self.nsave] = tmp.StandardAnalysis()
 	
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
                
        #EBeamEn = EBeamdata.ebeamL3Energy()
        
        evtid                   = evt.get(EventId)
        evttime                = evtid.idxtime()
        #get the event Id and timestamp
        self.TimeSt[self.nsave] = evttime.time()
        
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
            
        if envdata is not None and self.nsave==0:
            self.envArr[self.nsave,0] = envdata.vert()[0].offset()
            self.envArr[self.nsave,1] = envdata.vert()[0].fullScale()
            self.envArr[self.nsave,2] = envdata.vert()[2].offset()
            self.envArr[self.nsave,3] = envdata.vert()[2].fullScale()
            self.envArr[self.nsave,4] = envdata.horiz().sampInterval()
            
	if samplepressuredata is not None:
	    self.sPressArr[self.nsave] = samplepressuredata

        self.nsave += 1

###############################################################################        
            
    def save(self):
        if self.args.save == True:
             
            runnumber = int(self.args.exprun[17:])
            filename = 'amolr2516_r' + str(runnumber).zfill(4) + '_' + \
                        str(rank).zfill(3) + '_' + str(self.filenum).zfill(3)
                     
            directory_n='/reg/d/psdm/AMO/amolr2516/ftc/npzfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            directory_m='/reg/d/psdm/AMO/amolr2516/ftc/matfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            if not os.path.exists(directory_n):
                os.makedirs(directory_n)
            if not os.path.exists(directory_m):
                os.makedirs(directory_m)
            directory_n='/reg/d/psdm/AMO/amolr2516/ftc/npzfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            directory_m='/reg/d/psdm/AMO/amolr2516/ftc/matfiles/' + 'run' + \
                        str(runnumber).zfill(4) + '/'
            if not os.path.exists(directory_n):
                os.makedirs(directory_n)
            if not os.path.exists(directory_m):
                os.makedirs(directory_m)
            
            if rank == 1:
                print 'rank 1 writing file...'
                
            data={'EBeamParameters':self.ebeamArr,
                                       'SHESHits':self.[0:self.nsave,:].astype(np.float16), \
                                       'SHESwf':self.shProjArr[0:self.nsave,:].astype(np.float16), \
				       'UXSpc':,\
				       'UXSwf':,\
                                       'ITOF':self.itofArr[0:self.nsave].astype(np.float16)\
				       'Pressure':,\
				       'GasDetector':self.gmdArr[0:self.nsave,:], \
                                       'EnvVar':self.envArr[0:self.nsave,:], \
                                       'T':self.T, \
                                       'TimeStamp':self.TimeSt, \
                                       'Elmode':self.args.Elmode}
            scipy.io.savemat(directory_m+filename+'.mat',data)
                                       
            np.savez(directory_n+filename,**data)
                                       
                                       
            if rank == 1: print 'rank1 done with writing file.'
            self.filenum += 1
            
        self.ArrInit(len(self.shProjArr[0,:]),len(uxsProjArr[0,:]))

###############################################################################        
            
def runclient(args):
    client = Client(args)
    client.run()
