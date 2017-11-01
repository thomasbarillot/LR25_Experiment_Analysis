import numpy as np
import IPython
import scipy.io
import scipy.stats.mstats 
import warnings
import scipy.ndimage as im 
from scipy.optimize import curve_fit as cf
from math import exp, log, sqrt

import psana

convolv1d=scipy.ndimage.gaussian_filter1d
warnings.filterwarnings('ignore',category=UserWarning,module='UXS')

class UXSDataPreProcessing:

    def __init__(self,image):
        self.image=image
        self.wf=[]
        self.offset=0
        self.initparams=[]
        self.ok=0
        self.fitresults=[]
        self.fitcov=[]
        self.x=[]
        self.max=0
        self.rangelim=[]

    def DefineRange(self,rangelim):
        self.rangelim=rangelim
        if(len(self.rangelim)>0):
            self.wf=self.wf[self.rangelim[0]:self.rangelim[1]]

    def CalculateProjection(self):
        self.wf=np.sum(self.image,1)
        if(len(self.rangelim)==0):
            self.rangelim=[0,len(self.wf)]
        self.wf=self.wf[self.rangelim[0]:self.rangelim[1]]
        
    def RemoveOffset(self,sfr=40):
        self.offset=np.mean(self.wf[:sfr])
        self.wf=self.wf-self.offset
    
    def MedianFilter(self,points=3):
        self.wf = im.median_filter(self.wf,points)

    def NoiseThreshold(self,sfr=40,factor=10):
        thr = factor*np.std(self.wf[:sfr])
        self.wf = self.wf*(self.wf > thr)
 
    def GaussianFilter(self,width=12):
        self.wf = convolv1d(self.wf,width)


    def EstimateInitFitParam(self,convfactor=12):
        convwf=convolv1d(self.wf,convfactor)
        nzi = np.nonzero(convwf[1:-1])[0] - 1 #nzi= non-zero indices
        rdiff = convwf[1:] - convwf[:-1]
        peaks = np.array([p for p in nzi if rdiff[p] < 0 and rdiff[p-1] > 0]) 
        
        if len(peaks)==2:
            ampl0=self.wf[peaks[0]]        
            ampl1=self.wf[peaks[1]]
            self.initparams=[ampl0,peaks[0]+self.rangelim[0],10,ampl1,peaks[1]+self.rangelim[0],10]
            self.ok=1
        else: 
            self.initparams=[]
            warnings.warn_explicit('Discard, no two different optical peaks',UserWarning,'UXS',0)
            self.ok=0

    @staticmethod
    def DoubleGaussianFunction(x,a1,c1,w1,a2,c2,w2):
        return a1*np.exp(-(4*log(2))*((x-c1)**2)/(w1**2))+a2*np.exp(-(4*log(2))*((x-c2)**2)/(w2**2))

    def FitToDoubleGaussian(self):
        if self.ok==1:
            self.x=np.arange(0,len(self.wf))
            try:
                self.fitresults,self.fitcov=cf(UXSDataPreProcessing.DoubleGaussianFunction,self.x+self.rangelim[0],self.wf,self.initparams)
                #widths always positive
                self.fitresults[2::3]=abs(self.fitresults[2::3])
            except:
                warnings.warn_explicit('Fit failed',UserWarning,'UXS',0)

    def StandardAnalysis(self):
        self.CalculateProjection()
        self.RemoveOffset(10)
        #self.MedianFilter(4)
        #self.NoiseThreshold(3,1.5)
        self.GaussianFilter(20)
        self.DefineRange([0,400])
        self.EstimateInitFitParam(12)
        self.FitToDoubleGaussian()
        return self.fitresults
