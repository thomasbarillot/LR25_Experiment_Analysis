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
    """
    This class contains all the methods used for analysing the UXSData
    both for the online processing and for the preprocessing.
    """
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

    def FilterImage(self, sigma=1, order=0, threshold=100):
        """
        Gaussian filter and threshold for the image.
        """
        # Gaussian filter
        self.image = scipy.ndimage.gaussian_filter(self.image, sigma=sigma, order=order)
        
        # Threshold
        idx = self.image[:,:] < threshold
        self.image[idx] = 0
    
        # Remove border
        ##self.image[0,:] = 0
        ##self.image[:,0] = 0       

    def AddFakeImageSignal(self, center=200, curvature=200):
        """
        Add some fake signals to the image for testing
        the geometric correction and the projection
        """
        x = np.random.uniform(-1,1, 5000)
        y = np.random.normal(1024/2, 1024/8, 5000)
        x = x**3 * 1024/2 + center + curvature*(y/1024)**2
        x = x.astype(int)
        y = y.astype(int)
        mask = (y<1024) & (y > 0) & (x<1024) & (x>0)
        x,y = x[mask],y[mask]
        np.add.at(self.image, (1023-y,x), 1)


    def CorrectImageGeomerty(self):
        """
        Correct the curvature by using a known list of xshifts per yline
        """
        # TODO Actually produce a real list of xshifts
        xshifts = np.zeros(1024).astype(int) # No Shift!
        for y, shift in zip(range(1024), xshifts):
            # Do the shift for every y-line
            self.image[y,:] = np.roll(self.image[y,:], shift)
            # Remove pixels that has rolled outside the image
            # If needed we could add padding, but this should be sufficient
            if shift < 0:
                self.image[y, shift:] = 0
            if shift > 0:
                self.image[y, 0:np.abs(shift)] = 0

    def DefineRange(self,rangelim):
        """
        Define the range that will be included in the spectrum projection.
        Values outside will be cut.
        """
        self.rangelim=rangelim
        if(len(self.rangelim)>0):
            self.wf=self.wf[self.rangelim[0]:self.rangelim[1]]

    def CalculateProjection(self):
        """
        Project the image onto one axis
        """
        self.wf=np.sum(self.image,0) # TODO Make sure this axis is correct when we get the first data
        if(len(self.rangelim)==0):
            self.rangelim=[0,len(self.wf)]
        self.wf=self.wf[self.rangelim[0]:self.rangelim[1]]
        
    def RemoveOffset(self,sfr=40):
        """
        Removes offset from the calculated spectrum
        """
        self.offset=np.mean(self.wf[:sfr])
        self.wf=self.wf-self.offset
    
    def MedianFilter(self,points=3):
        """
        Runs a median filter on the calculated spectrum
        """
        self.wf = im.median_filter(self.wf,points)

    def NoiseThreshold(self,sfr=40,factor=10):
        """
        Noise threshold on the calculated spectrum
        """
        thr = factor*np.std(self.wf[:sfr])
        self.wf = self.wf*(self.wf > thr)
 
    def GaussianFilter(self,width=12):
        """
        Runs a gaussian filter on the calculated spectrum
        """
        self.wf = convolv1d(self.wf,width)


    def EstimateInitFitParam(self,convfactor=12):
        """
        Try to estimate the parameters needed to do a double gaussian fit.
        """
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
        """
        Helper function to be able to fit a double gaussian peak
        """
        return a1*np.exp(-(4*log(2))*((x-c1)**2)/(w1**2))+a2*np.exp(-(4*log(2))*((x-c2)**2)/(w2**2))

    def FitToDoubleGaussian(self):
        """
        Fits the projected spectrum with a double gaussian
        """
        if self.ok==1:
            # Ok if we managed to do a guess on the fit parameters
            self.x=np.arange(0,len(self.wf))
            try:
                self.fitresults,self.fitcov = cf(UXSDataPreProcessing.DoubleGaussianFunction,self.x+self.rangelim[0],self.wf,self.initparams)
                #widths always positive
                self.fitresults[2::3]=abs(self.fitresults[2::3])
            except:
                warnings.warn_explicit('Fit failed',UserWarning,'UXS',0)

    def StandardAnalysis(self):
        """
        This is the standard run that we do
        returns the fitresults as produced by FitToDoubleGaussian
        """
        self.CalculateProjection()
        self.RemoveOffset(10)
        #self.MedianFilter(4)
        #self.NoiseThreshold(3,1.5)
        self.GaussianFilter(20)
        self.DefineRange([0,400])
        self.EstimateInitFitParam(12)
        self.FitToDoubleGaussian()
        return self.fitresults
