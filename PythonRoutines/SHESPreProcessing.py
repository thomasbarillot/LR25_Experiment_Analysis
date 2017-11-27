"""Pre-processing for the Scienta Hemispherical Analyser

Ended up copying all of the useful code from Andre into here because there were
too many processes being repeated in here and then in his find_blobs.py routines
"""
#%% Imports
from psana import *
import numpy as np
from scipy import ndimage
import cv2 # may be needed for the perspective transform that Andre does, don't
# know what is going on there yet

#%% These parameters hard-coded in

# Define detector
det_name='OPAL1' #TODO may need changing across beamtimes

# Define estimated conversion rate from integrated (after thresholding) signal
# to electron counts
count_conv=10765.3295101 # from all in "exp=AMO/amon0816:run=228:smd:dir=/reg/d/psdm/amo/amon0816/xtc:live", mean=10765.3295101 & stddev=1503.99298626

# Define perspective transform
pts1  = np.float32([[131,212],[845,162],[131,701],[845,750]])
x_len_param = 714
y_len_param = 489
pts2 = np.float32([[0,0],[x_len_param,0],[0,y_len_param],[x_len_param,y_len_param]])
M = cv2.getPerspectiveTransform(pts1,pts2)
# these are updated from Andre's numbers, taken 20171127

# For defining circles to check for arcing, and arcing threshold
innerR, outerR = 460, 540
xc, yc = 500, 460

arcThresh=3.2e6 #change me

# Potentially require parameters for polynomial fitting
poly_fit_params=None

#TODO DiscardBorder before or after perspective transform?

#%%

class SHESPreProcessor(object):

    def __init__(self, threshold=500., discard_border=1):

        self.threshold=threshold
        self.discard_border=discard_border

        # And then hardcoded parameters. These are attributes of each
        # instance in case they ever want to be changed
        self.opal_det=Detector(det_name) # N.B. requires a
        # psana.DataSource instance to exist
        self.count_conv=count_conv
        self.pers_trans_params=M, x_len_param, y_len_param #perspective transform parameters
        self.poly_fit_params=poly_fit_params
        self.arcThresh=arcThresh 
        self.arcMask=makeCircles((innerR, outerR), (xc, yc))
        
    def ArcCheck(self, opal_image):
        arc=np.sum(opal_image*self.arcMask)>self.arcThresh
        return arc

    def GetRawImg(self, event):
        raw_img=self.opal_det.raw(event)
        if raw_img is None:
            return raw_img
        return np.rot90(np.copy(raw_img),-1) #rotation added 20171127, needed for LR25 beamtime 
        # makes a copy because Detector.raw(evt) returns a read-only array for
        # obvious reasons

    def DiscardBorder(self, opal_image):
        
        opal_image_cp=np.copy(opal_image)
        opal_image_cp[ :self.discard_border,:] = 0
        opal_image_cp[-self.discard_border:,:] = 0
        opal_image_cp[:, :self.discard_border] = 0
        opal_image_cp[:,-self.discard_border:] = 0
        
        return opal_image_cp

    def Threshold(self, opal_image):
        'Take greater than or equal to'
        return opal_image*self.Binary(opal_image) #TODO does this change the array in-place,
        #I think not
        
    def PerspectiveTransform(self, opal_image):
        M, x_len_param, y_len_param = self.pers_trans_params
        return cv2.warpPerspective(opal_image, M, (x_len_param, y_len_param))
        # this returns an nd.array of shape (y_len_param, x_len_param), I don't
        # understand why #TODO understand! My best guess is that inside the cv2.warpPerspective
        #/cv2.getPerspectiveTransform functions, row becomes x-axis and col becomes y-axis

    def PolyFit(self, opal_image):
        return opal_image #TODO
 
    def XProj(self, opal_image):    
        return opal_image.sum(axis=0) #TODO check this is the correct axis

    def Binary(self, opal_image):
        return opal_image>self.threshold
    
    def FindComs(self, opal_image):
        'Find the center of all blobs above threshold'
        binary = self.Binary(opal_image)
    
        labelled, num_labels = ndimage.label(binary)
        if num_labels==0:
            return [(np.nan, np.nan)], np.zeros(opal_image.shape)
        centers = ndimage.measurements.center_of_mass(binary, labelled, range(1,num_labels+1))
        return centers, labelled

    def FindBlobs(self, opal_image):

        centers, labelled = self.FindComs(opal_image)
        if centers==[(np.nan, np.nan)]:
           return [(np.nan, np.nan)], [(np.nan, np.nan)]
        widths = []
        for i in range(len(centers)):
        
            c = centers[i]
            r_slice = labelled[int(c[0]),:]
            zx = np.where( np.abs(r_slice - np.roll(r_slice, 1)) == i+1 )[0]
        
            c_slice = labelled[:,int(c[1])]
            zy = np.where( np.abs(c_slice - np.roll(c_slice, 1)) == i+1 )[0]
        
            if not (len(zx) == 2) or not (len(zy) == 2):
                #print "WARNING: Peak algorithm confused about width of peak at", c
                #print "         Setting default peak width (5,5)"
                widths.append( (5.0, 5.0) )
            else:
                x_width = zx[1] - zx[0]
                y_width = zy[1] - zy[0]
                widths.append( (x_width, y_width) )
        
        return centers, widths

    def PreProcess(self, event):  
        'This is the standard pre-processing for the SHES OPAL arrays'
        opal_image=self.GetRawImg(event)
        if opal_image is None:
            return np.nan, np.nan, np.nan
        opal_image=self.PerspectiveTransform(opal_image)
     
        opal_image=self.DiscardBorder(opal_image)
        xs, ys=zip(*self.FindComs(opal_image)[0]) # FindComs() doesn't need thresholded array
        x_proj=self.XProj(self.Threshold(opal_image)) # So threshold here  

        return list(xs), list(ys), x_proj

    def OnlineProcess(self, event);
        #TODO include suitable behaviour (give warning) if the MCP is arcing
        'This is the standard online processing for the SHES OPAL arrays'
        opal_image=self.GetRawImg(event)
        if opal_image is None:
            return None, None, None # returns NoneType
        opal_image=self.PerspectiveTransform(opal_image)
        opal_image=self.Threshold(self.DiscardBorder(opal_image))

        count_estimate=opal_image.sum().sum()/float(self.count_conv)        
        x_proj=self.XProj(opal_image)
        
        return opal_image, x_proj, count_estimate

#%% some functions
def makeCircles((innerR, outerR)=(460, 540), (xc, yc)=(500, 460)):
    'Function adapted from Andre to define mask for arcing test'
    arcMask = np.zeros((1024, 1024), dtype=np.double) # arcing mask

    for xx in range(1024):
        for yy in range(1024):
            rad=(xx-xc)**2+(yy-yc)**2
            if rad >= innerR*innerR and rad <= outerR*outerR: # greater than
                # or equal to condition should exactly recover performance for 
                # Andre's previous version
                arcMask[xx, yy]=1
            
    return arcMask
