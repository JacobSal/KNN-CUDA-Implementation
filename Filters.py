# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:24:10 2020

@author: jsalm
"""
import numpy as np
import math
from scipy.ndimage import convolve
import skimage.filters
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

import cv2

def range_cont(set_min,set_max,incr,shift = 0,reverse = False):
    """
    generates a list of "thresholds" to be used for _contours.
    uses the max and min of the array as bounds and the number of layers as
    segmentation.
    array = np.array()
    layers = number of differentials (int)
    maxlayers = number of layers kept (int)
    shfit = a variable that shifts the sampling up and down based on maxlayers (int)
    reverse = start at the min or end at max of the array (True == end at max) (bool)
    """
    maxsteps = math.floor(set_max/incr)
    incr_i = incr
    maxincr = set_min+(incr*(maxsteps))
    if set_min+(incr*maxsteps) > set_max:
        maxincr = set_max    
    if reverse == True:
        thresh = [int(set_max-maxincr-shift*(incr*(maxsteps+1)))]
        arraystart = set_max-maxincr - shift*(incr*(maxsteps+1))
        maxincr = set_max - shift*(incr*(maxsteps+1))
        
    elif reverse == False:
        thresh = [int(set_min + shift*(incr*(maxsteps)))]
        arraystart = set_min + shift*(incr*(maxsteps))
        if shift > 0:
            maxincr = arraystart + shift*(incr*(maxsteps-1))
        'end if'
    try:
        while thresh[-1] < maxincr:
            thresh.append(np.int32(arraystart+incr_i))
            incr_i = incr_i + incr
        'end while'
        thresh.append(np.int32(set_max))
    except:
        raise ValueError("Max and Min == 0")
    'end try'
    return thresh
'end def'

def adaptive_threshold(image,sW,bins,sub_sampling=True):      
#this helps significantly, but les try adding sub sample
    nH,nW = image.shape
    new_image = np.zeros((nH,nW))
    sbxx = range_cont(0,nW,sW)
    sbxy = range_cont(0,nH,sW)
    if sub_sampling:
        for i in range(0,len(sbxy)-1):
            for j in range(0,len(sbxx)-1):
                try:
                    thresh = skimage.filters.threshold_otsu(image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]],nbins=256)
                except ValueError:
                    thresh = np.max(image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]])
                threshed = (image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]] > thresh)*image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]]
                new_image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]] = threshed
    else:
        thresh = skimage.filters.threshold_otsu(image,nbins=bins)
        new_image = image*(image>thresh)
    return new_image

def _create_circle(dia,fill=False):
    """
    creates a circle of radius
    to be used in curve detection
    dia = diameter; int (must be odd)

    """
    if dia%2 == 0:
        raise ValueError("Radius must be odd")
    'end if'
    circle = np.zeros((dia,dia))
    N = int(dia/2)
    x_val = list(range(0,dia))
    y_val = list(range(0,dia))
    for x in x_val:
        for y in y_val:
            circle[int(x),int(y)] = np.sqrt((x-N)**2+(y-N)**2)
        'end for'
    'end for'
    circle_bool = np.logical_not(np.add(circle>(dia/2),circle<(dia-2)/2))
    if fill == True:
        circle_bool = circle<(dia/2)
    'end if'
    return circle_bool
'end def'
 
def Hi_pass_filter(image,width):
     dft = cv2.dft(np.float32(image),flags=cv2.DFT_COMPLEX_OUTPUT)
     dft_shift = np.fft.fftshift(dft)
     
     rows,cols = image.shape
     crow,ccol = rows//2,cols//2
     
     # mask = np.zeros((rows,cols,2),np.uint8)
     # mask[crow-width:crow+width, ccol-width:ccol+width] = 1
     # mask = np.uint8(mask == 0)
     
     circle = _create_circle(width,True)
     mask = np.zeros((rows,cols,2),np.uint8)
     adjust = width//2
     mask[crow-adjust:crow+adjust+1, ccol-adjust:ccol+adjust+1,0] = circle
     mask[crow-adjust:crow+adjust+1, ccol-adjust:ccol+adjust+1,1] = circle
     mask = np.uint8(mask == 0)
     
     
     
     fshift = dft_shift*mask
     f_ishift = np.fft.ifftshift(fshift)
     img_back = cv2.idft(f_ishift)
     img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
     return img_back
"end def"    

def average_filter(image,width,C):
    kernel = np.ones((width,width),dtype=np.uint8)
    kernel = kernel*C
    image_out = convolve(image,kernel)
    return image_out
# def FFfilt(image,reduction_factor,width,Lo_pass=True):
#     if Lo_pass:
#         began = int(image.shape[0]/2)-int(width/2)
#         end = int(image.shape[0]/2)+int(width/2)
#         gfilt = np.zeros((image.shape[0],image.shape[1]))
#         gfft = np.fft.fftshift(np.fft.fft(image))           
#         gfft[:,:began] = reduction_factor*gfft[:,:began]
#         gfft[:,end:] = reduction_factor*gfft[:,end:]
#         gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
#         # greturn = gfilt
#         greturn = np.multiply(gfilt,image)
#     else:
#         began = int(image.shape[0]/2)-int(width/2)
#         end = int(image.shape[0]/2)+int(width/2)
#         gfilt = np.zeros((image.shape[0],image.shape[1]))
#         gfft = np.fft.fftshift(np.fft.fft(image))           
#         gfft[:,began:end] = reduction_factor*gfft[:,began:end]
#         gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
#         # greturn = gfilt
#         greturn = np.multiply(gfilt,image)
#     'end if'
#     return greturn
# 'end def'

def _d3gaussian(vector_width,multiplier_a,multiplier_d):
    """
    creates a 3D gaussian inlayed into matrix form.
    of the form: G(x,y) = a*e^(-((x-N)**2+(y-N)**2)/(2*d**2))
    """
    x = np.arange(0,vector_width,1)
    y = np.arange(0,vector_width,1)
    d2gau = np.zeros((vector_width,vector_width)).astype(float)
    N = int(vector_width/2)
    for i in x:
        d2gau[:,i] = multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
    'end for'
    return d2gau     
'end def'

def _onediv_d3gaussian(vector_width,multiplier_a,multiplier_d):
    """
    creates a 3D gaussian inlayed into matrix form.
    of the form: G(x,y) = a*e^(-((x-N)**2+(y-N)**2)/(2*d**2))
    """
    x = np.arange(0,vector_width,1)
    y = np.arange(0,vector_width,1)
    d2gau_x = np.zeros((vector_width,vector_width)).astype(float)
    d2gau_y = np.zeros((vector_width,vector_width)).astype(float)
    N = int(vector_width/2)
    for i in x:
        d2gau_x[:,i] = -((i-N)/multiplier_d**2)*multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
        d2gau_y[:,i] = -((y-N)/multiplier_d**2)*multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
    'end for'
    d2gau = np.add(d2gau_x**2,d2gau_y**2)
    return d2gau     
'end def'

def diffmat(image,theta,dim=(10,2)):
    if type(dim) != tuple:
        raise ValueError('dim must be tuple')
    if dim[1]%2 != 0:
        raise ValueError('n must be even')
    'end if'
    outarray = np.zeros((image.shape[0],image.shape[1]))
    dfmat = np.zeros((max(dim),max(dim)))
    dfmat[:,0:int(dim[1]/2)] = -1
    dfmat[:,int(dim[1]/2):dim[1]] = 1
    
    dmatx = dfmat
    dmaty = np.transpose(dfmat)
    for angle in theta:
        dmat = dmatx*np.cos(angle)+dmaty*np.sin(angle)
        dm = np.divide(convolve(image,dmat)**2,math.factorial(max(dim)))
        outarray = np.add(dm,outarray)
    'end for'
    return outarray
'end def'

def diagonal_map(image):\
    #variable filter that convolves different 
    pass
'end def'

def imshow_overlay(im, mask, savefig=False, title="default", genfig=True, alpha=0.3, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)
    if genfig == True:
        plt.figure('overlayed')  
        plt.imshow(im, **kwargs)
        plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
    'end if'
    
    if savefig == True:
        savedir = r"C:\Users\jsalm\Documents\Python Scripts\3DRecon\saved_Im"
        plt.savefig(os.path.join(savedir,"overlayed_"+str(title)+".tif"),dpi=600,quality=95,pad_inches=0)
    'end if'
'end def'