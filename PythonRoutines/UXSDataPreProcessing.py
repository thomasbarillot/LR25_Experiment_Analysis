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

        # Hardcoded curvature correction
        self.xshifts = [0]*1024 # Start with no curvature
        #self.xshifts = [-328, -327, -326, -325, -324, -323, -322, -322, -321, -320, -319, -318, -317, -317, -316, -315, -314, -313, -312, -312, -311, -310, -309, -308, -307, -307, -306, -305, -304, -303, -302, -302, -301, -300, -299, -298, -297, -297, -296, -295, -294, -293, -293, -292, -291, -290, -289, -288, -288, -287, -286, -285, -284, -284, -283, -282, -281, -280, -280, -279, -278, -277, -276, -276, -275, -274, -273, -272, -272, -271, -270, -269, -268, -268, -267, -266, -265, -264, -264, -263, -262, -261, -261, -260, -259, -258, -257, -257, -256, -255, -254, -254, -253, -252, -251, -250, -250, -249, -248, -247, -247, -246, -245, -244, -244, -243, -242, -241, -240, -240, -239, -238, -237, -237, -236, -235, -234, -234, -233, -232, -231, -231, -230, -229, -228, -228, -227, -226, -225, -225, -224, -223, -222, -222, -221, -220, -219, -219, -218, -217, -216, -216, -215, -214, -214, -213, -212, -211, -211, -210, -209, -208, -208, -207, -206, -206, -205, -204, -203, -203, -202, -201, -200, -200, -199, -198, -198, -197, -196, -195, -195, -194, -193, -193, -192, -191, -191, -190, -189, -188, -188, -187, -186, -186, -185, -184, -183, -183, -182, -181, -181, -180, -179, -179, -178, -177, -177, -176, -175, -174, -174, -173, -172, -172, -171, -170, -170, -169, -168, -168, -167, -166, -166, -165, -164, -164, -163, -162, -162, -161, -160, -160, -159, -158, -158, -157, -156, -156, -155, -154, -154, -153, -152, -152, -151, -150, -150, -149, -148, -148, -147, -146, -146, -145, -144, -144, -143, -142, -142, -141, -140, -140, -139, -139, -138, -137, -137, -136, -135, -135, -134, -133, -133, -132, -132, -131, -130, -130, -129, -128, -128, -127, -126, -126, -125, -125, -124, -123, -123, -122, -121, -121, -120, -120, -119, -118, -118, -117, -117, -116, -115, -115, -114, -113, -113, -112, -112, -111, -110, -110, -109, -109, -108, -107, -107, -106, -106, -105, -104, -104, -103, -103, -102, -101, -101, -100, -100, -99, -98, -98, -97, -97, -96, -96, -95, -94, -94, -93, -93, -92, -91, -91, -90, -90, -89, -89, -88, -87, -87, -86, -86, -85, -85, -84, -83, -83, -82, -82, -81, -81, -80, -79, -79, -78, -78, -77, -77, -76, -76, -75, -74, -74, -73, -73, -72, -72, -71, -71, -70, -69, -69, -68, -68, -67, -67, -66, -66, -65, -65, -64, -63, -63, -62, -62, -61, -61, -60, -60, -59, -59, -58, -58, -57, -57, -56, -55, -55, -54, -54, -53, -53, -52, -52, -51, -51, -50, -50, -49, -49, -48, -48, -47, -47, -46, -46, -45, -45, -44, -44, -43, -43, -42, -42, -41, -41, -40, -40, -39, -39, -38, -38, -37, -37, -36, -36, -35, -35, -34, -34, -33, -33, -32, -32, -31, -31, -30, -30, -29, -29, -28, -28, -27, -27, -26, -26, -25, -25, -24, -24, -23, -23, -23, -22, -22, -21, -21, -20, -20, -19, -19, -18, -18, -17, -17, -16, -16, -16, -15, -15, -14, -14, -13, -13, -12, -12, -11, -11, -11, -10, -10, -9, -9, -8, -8, -7, -7, -6, -6, -6, -5, -5, -4, -4, -3, -3, -3, -2, -2, -1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 18, 19, 19, 20, 20, 20, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 25, 25, 25, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 68, 68, 68, 68, 69, 69, 69, 69, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 74, 74, 74, 74, 75, 75, 75, 75, 76, 76, 76, 76, 77, 77, 77, 77, 78, 78, 78, 78, 78, 79, 79, 79, 79, 80, 80, 80, 80, 81, 81, 81, 81, 81, 82, 82, 82, 82, 83, 83, 83, 83, 83, 84, 84, 84, 84, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 92, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112]

        # Hardcoded energy calibration
        # Defined as [Channel, Energy]
        ###energycalibrationpoints = np.array([[0, -500],
        ###                                    [512, 100],
        ###                                    [1024, 500]])
        # Create energy scale by polyfitting calibrationpoints
        ###energypoly = np.polyfit(energycalibrationpoints[:,0], energycalibrationpoints[:,1], 2)
        # Todo change to hardcoded
        ###self.energyscale = np.polyval(energypoly, np.arange(0,1024)) # TODO also cut the energy scale when defining range
        self.energyscale = np.arange(0,1024)


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
        self.image[0,:] = 0
        self.image[:,0] = 0       

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

    def CorrectImageGeometry(self):
        """
        Correct the curvature by using a known list of xshifts per yline
        """
        for y, shift in zip(range(1024), self.xshifts):
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
        self.CorrectImageGeometry()
        self.CalculateProjection()
        self.RemoveOffset(10)
        #self.MedianFilter(4)
        #self.NoiseThreshold(3,1.5)
        self.GaussianFilter(20)
        self.DefineRange([0,400])
        self.EstimateInitFitParam(12)
        self.FitToDoubleGaussian()
        return self.fitresults
