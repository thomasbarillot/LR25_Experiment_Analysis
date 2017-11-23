from psana import *
import numpy as np
import scipy.io
#import mpidata
import random
from xtcav.ShotToShotCharacterization import * 
import pypsalg
import os

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
        
#        XTCAVRetrieval = ShotToShotCharacterization()
#        XTCAVRetrieval.SetEnv(self.ds.env())

        self.SHES  = Detector(self.args.SHES)
        self.UXS = Detector(self.args.UXS)
	self.ITOF = Detector(self.args.ITOF)	
	self.XTCAV = Detector(self.args.XTCAV)
        self.EBeam = Detector(self.args.EBeam)
        self.GMD   = Detector(self.args.GMD)
        self.ENV = self.ds.env().configStore()
	self.EPICS = EpicsDetector(self.args.EPCIS,self.ds)

###############################################################################

    def ArrInit(self,shlength,uxslength):

	# Scienta Hemispherical array 2D: Energy projection; index of event
	self.shProjArr = np.zeros((shlength,self.args.nsave))
	
	# XRay spectro array 2D principal components: PE1,DE1,A1,PE2,DE2,A2 ; index of event
	self.uxsPCArr = np.zeros((6,self.args.nsave))
	# Xray spectro array 2D: Energy projection; index of event
	self.uxsProjArr = np.zeros((uxslength,self.args.nsave))
	
	# micro tof: ion yield, index of event 
	self.itofArr = np.zeros((self.args.nsave))

        # Gas detector array (6 ebeam EL3 values)
        self.gmdArr   = np.zeros((self.args.nsave,6))
        # Acq parameters: offset and fullscale for both channels low gain: 2, high gain 1
        self.envArr   = np.zeros((1,5))
        #
        self.ebeamArr = np.zeros((self.args.nsave,21))

	# Sample pressure array
	self.sPressArr = np.zeros((self.args.nsave))
	
        # Time stamp array         
        self.TimeSt = np.zeros((self.args.nsave,))
        
        #hits for blob finding: posX,posY, index of event
        self.hits  = [[],[],[]]
        
        
	self.shProjArr[:,:,:] = np.nan
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

	# Call the preprocessing modules here

	# Initialize the arrays
        if not hasattr(self,'shtime'):
            shtenergy = self.SHES.raw(evt).shape[0]
            if shenergy is None: return
                
            self.ArrInit(self.SHES.raw(evt).shape[0],self.UXS.raw(evt).shape[0])
            self.shenergy = mbtime[0,:]
            
	

        # Get the hemisperical analyser signal data
            
	# Get the Xray spectrometer analyser data
	
        
        # Get the environnement data 
        
        envdata=self.ENV.get(psana.Acqiris.Config)
        
        gmddata  = self.GMD.get(evt)

	samplepressuredata = self.EPICS.get(evt)
    
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
            #if self.args.rebin == True:
            #    self.rebin(0)
            #    self.MB1 = self.MBbin
            #    self.rebin(1)
            #    self.MB2 = self.MBbin
            #else:
            #    self.MB1 = self.mbArr[0,:,:]
            #    self.MB2 = self.mbArr[1,:,:]                                
            #    self.T = self.mbtime
             
		    
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
            
            if rank == 1:
                print 'rank 1 writing file...'
                
            data={'EBeamParameters':self.ebeamArr,
                                       'SHESHits':self.MB1[:,:,0:self.nsave].astype(np.float16), \
                                       'SHESwf':self.shProjArr[:,0:self.nsave].astype(np.float16), \
				       'UXSpc':self.,\
				       'UXSwf':self.,\
                                       'ITOF':self.,\
				       'Pressure':,\
				       'GasDetector':self.gmdArr[0:self.nsave,:], \
                                       'EnvVar':self.envArr[0:self.nsave,:], \
                                       'TimeStamp':self.TimeSt, \
                                       'Elmode':self.args.Elmode}
            scipy.io.savemat(directory_m+filename+'.mat',data)
                                       
            np.savez(directory_n+filename,**data)
                                       
                                       
            if rank == 1: print 'rank1 done with writing file.'
            self.filenum += 1
            
        self.ArrInit(len(self.mbtime))

###############################################################################
        
        #self.MBbin[:,:] = np.nan

        #var=len(self.mbtime)/self.args.bins
        #for i in np.arange(self.args.nsave):
            #self.mbArr[chan,i,:]=self.mbArr[chan,i,:]-np.mean(self.mbArr[chan,i,:700])
        #    if self.args.crop[1]!=0:
        #        self.MBbin[i,:]=self.mbArr[chan,i,self.args.crop[0]:-self.args.crop[1]].reshape(-1,var).sum(1)/var
        #    else:
        #        self.MBbin[i,:]=self.mbArr[chan,i,self.args.crop[0]:].reshape(-1,var).sum(1)/var
        #        
        #self.T=self.mbtime.reshape(-1,var).sum(1)/var
        
        

###############################################################################        
            
def runclient(args):
    client = Client(args)
    client.run()
