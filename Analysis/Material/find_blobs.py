#!/usr/bin/env python

"""
find_blobs.py

This script extracts OPAL images and does a basic peakfinding algorithm on them.

-- TJ Lane 9.4.41
"""


from glob import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import psana


def find_blobs(image, sigma_threshold=5.0, discard_border=1):
    """
    Find peaks, or `blobs`, in a 2D image.
    
    This algorithm works based on a simple threshold. It finds continuous
    regions of intensity that are greater than `sigma_threshold` standard
    deviations over the mean, and returns each of those regions as a single
    blob.
    
    Parameters
    ----------
    image : np.ndarray, two-dimensional
        An image to peakfind on.
        
    Returns
    -------
    centers : list of tuples of floats
        A list of the (x,y) positions of each peak, in pixels.
        
    widths : list of tuples of floats
        A list of the (x,y) size of each peak, in pixels.
        
    Optional Parameters
    -------------------
    sigma_threshold : float
        How many standard deviations above the mean to set the binary threshold.
    
    discard_border : int
        The size of a border region to ignore. In many images, the borders are
        noisy or systematically erroneous.
    
    Notes
    -----
    Tests indicate this algorithm takes ~200 ms to process a single image, so
    can run at ~5 Hz on a single processor.
    """
    
    if not len(image.shape) == 2:
        raise ValueError('Can only process 2-dimensional images')
    
    # discard the borders, which can be noisy...
    image[ :discard_border,:] = 0
    image[-discard_border:,:] = 0
    image[:, :discard_border] = 0
    image[:,-discard_border:] = 0
    
    # find the center of blobs above `sigma_threshold` STDs
    #binary = (image > (image.mean() + image.std() * sigma_threshold))
    binary = (image>sigma_threshold)
    labeled, num_labels = ndimage.label(binary)
    centers = ndimage.measurements.center_of_mass(binary, 
                                                  labeled,
                                                  range(1,num_labels+1))
                                                    
                                                  
    # for each peak, find it's x- & y-width
    #   we do this by measuring how many pixels are above 5-sigma in both the
    #   x and y direction at the center of each blob
    
    widths = []
    for i in range(num_labels):
        
        c = centers[i]
        r_slice = labeled[int(c[0]),:]
        zx = np.where( np.abs(r_slice - np.roll(r_slice, 1)) == i+1 )[0]
        
        c_slice = labeled[:,int(c[1])]
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
    
    
def draw_blobs(image, centers, widths):
    """
    Draw blobs (peaks) on an image.
    
    Parameters
    ----------
    image : np.ndarray, two-dimensional
        An image to render.
    
    centers : list of tuples of floats
        A list of the (x,y) positions of each peak, in pixels.
        
    widths : list of tuples of floats
        A list of the (x,y) size of each peak, in pixels.
    """
    
    plt.figure()
    #plt.imshow(image.T, interpolation='nearest')
    
    centers = np.array(centers)
    widths = np.array(widths)
    
    diagonal_widths = widths.copy()
    diagonal_widths[:,0] *= -1.0
    
    for i in range(len(centers)):
        
        pts = np.array([
               centers[i] + widths[i] / 2.0,
               centers[i] - diagonal_widths[i] / 2.0,
               centers[i] - widths[i] / 2.0,
               centers[i] + diagonal_widths[i] / 2.0,
               centers[i] + widths[i] / 2.0
              ])
        
        #plt.plot(pts[:,0], pts[:,1], color='orange', lw=3)
        
    plt.xlim([0, image.shape[0]])
    plt.ylim([0, image.shape[1]])
    plt.show()
    
    return pts[:,0], pts[:,1]


def parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze OPAL images')
    
    parser.add_argument('-r', '--run', type=int,
        default=-1, help='Which run to analyze, -1 for live stream')
    parser.add_argument('-n', '--num-max', type=int,
        default=0, help='Stop after this number of shots is reached')
    parser.add_argument('-v', '--view', action='store_true',
        default=False, help='View each OPAL image (good for debugging)')
    parser.add_argument('-s', '--sigma', type=float,
        default=6.0, help='The number of std above the mean to search for blobs')
    
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_args()
    
    if args.run == 0:
        print 'Analyzing data from shared memory...'
        try:
            ds = psana.DataSource('shmem=1_1_psana_XCS.0:stop=no')
        except:
            raise IOError('Cannot find shared memory stream.')
    else:
        print 'Analyzing run: %d' % args.run
        ds = psana.DataSource('exp=sxra8513:run=%d' % args.run) # CHANGE THIS FOR NEW EXPT
    
    # this may also need to change for the new expt
    opal_src = psana.Source('DetInfo(SxrEndstation.0:Opal1000.1)')

    # iterate over events and extract peaks
    for i,evt in enumerate(ds.events()):
        
        # gets an "opal" object
        opal = evt.get(psana.Camera.FrameV1, opal_src)
        if opal:
            image = opal.data16().copy() # this is a numpy array of the opal image
            centers, widths = find_blobs(image, sigma_threshold=args.sigma)
            n_blobs = len(centers)
            print 'Shot %d :: found %d blobs :: %s' % (i, n_blobs, str(centers))
            
            if args.view and (n_blobs > 0):
                draw_blobs(image, centers, widths)
                
        # if we reach the max number of shots, stop
        if i+1 == args.num_max:
            print 'Reached max requested number of shots (%d)' % (i+1,)
            break
        
    return


def test():
    
    for f in glob('*.npy'):
        print f
        img = np.load(f)
        b = find_blobs(img)
        if len(b[0]) > 0:
            draw_blobs(img, *b)
            
    return

    
if __name__ == '__main__':
    main()
