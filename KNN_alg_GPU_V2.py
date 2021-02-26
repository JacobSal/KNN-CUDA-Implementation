# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:50:07 2020

@author: jsalm
"""
#Required Modules
from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.ndimage import convolve
from numba import cuda, float32, cfunc, types, carray
from math import ceil
import os
from scipy.stats import mode

#temp modules
import ThreeD_Recon_V2
import Training_Gen_V2
import Filters
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import time

global BLOCK_DIM

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

def generate_train_sert_ID(boolim,image):
    if type(boolim[0,0]) != np.bool_:
        raise TypeError("args need to be type bool and tuple respectively")
    'end if'
    count = 0
    data = np.zeros((2,boolim.shape[0]*boolim.shape[1]))
    point_data = np.zeros((2,boolim.shape[0]*boolim.shape[1]))
    #generate list of points
    for i,row in enumerate(boolim):
        for j,col in enumerate(row):
            if col == True:
                data[0,count] = image[i,j]
                data[1,count] = 1
                point_data[0,count] = i
                point_data[1,count] = j
                count+=1
            else:
                data[0,count] = image[i,j]
                data[1,count] = 0
                point_data[0,count] = i
                point_data[1,count] = j
                count+=1
            'end if'
        'end for'
    'end for'
    return data,point_data
'end def'


def generate_test_sert_ID(boolim,image):
    if type(boolim[0,0]) != np.bool_:
        raise TypeError("args need to be type bool and tuple respectively")
    'end if'
    count =  0
    t_data = np.sum(boolim)
    data = np.zeros((2,t_data))
    point_data = np.zeros((2,t_data))
    for i,row in enumerate(boolim):
        for j,col in enumerate(row):
            if col == True:
                data[0,count] = image[i,j]
                data[1,count] = 0
                point_data[0,count] = i
                point_data[1,count] = j
                count+=1
    return data,point_data
'end def'

@cuda.jit
def extract_with_interpolation(nthreads,data,n_xy_coords,extracted_data,n_max_coord,channels,height,width):
    index = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    for i in range(index,nthreads,cuda.blockDim.x*cuda.gridDim.x):
        n = (index/n_max_coord)
        nd = n*n_max_coord*channels
        x = n_xy_coords[index*2]
        y = n_xy_coords[index*2+1]
        x0 = int(x)
        x1 = x0+1
        y0 = int(y)
        y1 = y0+1
        
        if x0 <= 0: x0 = 0
        elif x0 >= (width-1): x0 = (width-1)
        else:  x0 = x0
        
        if y0 <= 0: y0 = 0
        elif y0 >= (width-1): y0 = (width-1)
        else:  y0 = y0
        
        if x1 <= 0: x1 = 0
        elif x1 >= (width-1): x1 = (width-1)
        else:  x1 = x1
        
        if y1 <= 0: y1 = 0
        elif y1 >= (width-1): y1 = (width-1)
        else:  y1 = y1
        
        wx0 = x1 - x
        wx1 = x - x0
        wy0 = y1 - y
        wy1 = y - y0
        
        if x0 == x1:
            wx0 = 1
            wx1 = 0
        if y0 == y1:
            wy0 = 1
            wy1 = 0
        for c in range(channels):
            nc = (n*channels+c)*height
            extracted_data[nd + index%n_max_coord+n_max_coord*c] = wy0*wx0*data[(nc+y0)*width+x0]
            + wy1*wx0*data[(nc+y1)*width+x0]
            + wy0*wx1*data[(nc+y0)*width+x1]
            + wy1*wx1*data[(nc+y1)*width+x1]
@cuda.jit
def distance_kern(A,wA,B,wB,dim,dist_mem):
    """

    Parameters
    ----------
    A : float32[:,:]
        array containing all test points for KNN
    wA : int
        width of the array A
    B : float32[:,:]
        array contiaing all training points for KNN
    wB : int
        width of the array B
    dim : int
        the dimension of each array, how many variables there are to consider. 
        Typically  dim = 1
    dist_mem : float32[:,:]
        array of zero values 

    Returns
    -------
    None.

    """
    #establish global variable to be used for dediating shared memory
    BLOCK_DIM = 16
    #Declaration of the shared memory for submatrices of A and B: These will be where we
    #temporarily store our information from A and B for each block instance.
    shared_A = cuda.shared.array((BLOCK_DIM,BLOCK_DIM),float32)
    shared_B = cuda.shared.array((BLOCK_DIM,BLOCK_DIM),float32)
    
    #Thread Index: This is how we assign each thread in our block instance.
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    #other variables: temporary variables loaded for each thread
    tmp = 0
    ssd = 0
    
    #loop parameters: In order to ensure we are assigning the data to the shared
    #memory properly we need to take into acount what block we are in and 
    #how large the block is.
    begin_A = BLOCK_DIM*cuda.blockIdx.y
    begin_B = BLOCK_DIM*cuda.blockIdx.x
    step_A = BLOCK_DIM*wA
    step_B = BLOCK_DIM*wB
    end_A = begin_A + (dim-1)*wA
    
    #conditions: these determine the start and stop points for each block assginment.
    #these conditions tell the program when to stop assgining data to the shared memory.
    cond0 = (begin_A + tx < wA)
    cond1 = (begin_B + tx < wB)
    cond2 = (begin_A + ty < wA)
    
    #Loop over all the sub-matrices of A and B required to compute the block sub-matrix:
    #Now we assgin the data from A and B to our shared memory using our conditions.
    a = begin_A
    b = begin_B
    while a <= end_A:
        if (a/wA+ty < dim):
            if cond0: shared_A[ty][tx] = A[a+wA*ty+tx]
            else: shared_A[ty][tx] = 0
            if cond1: shared_B[ty][tx] = B[b+wB*ty+tx]
            else: shared_B[ty][tx] = 0
        else:
            shared_A[ty][tx] = 0
            shared_B[ty][tx] = 0
        a += step_A
        b += step_B
        'end if'
    'end while'
    #make sure to sync your threads when you need to perform and operation
    #on all the data in the shared memory at the same time, otherwise
    #your kernel will perform in asynchronously.
    cuda.syncthreads()
    
    #Now we calculate the distances between the two shared memory chunks. Ensuring
    #we are within the bounds of A and B
    if cond2 and cond1:
        for k in range(BLOCK_DIM):
            tmp = shared_A[k][ty] - shared_B[k][tx]
            ssd += math.sqrt(tmp**2)
        'end for'
    'end if'
    cuda.syncthreads()
    
    #store the data in another matrix, loaded onto the device (GLOBAL memory) that can be accessed later
    if cond2 and cond1:
        # dist_mem = cuda.carray(dist_mem,(wA*wB,1))
        dist_mem[(begin_A+ty)*wB + begin_B + tx] = ssd
    'end if'
'end def'



@cuda.jit #('void(float32[:],float32[:],int32,int32,int32)')
def sorting_kern(dist, ind, width, height, k):
    """
    This is a pretty advanced sorting algorithm that sorts items from the minimum item in the 
    array to the maximum item in the array. But we limit the amount of items it sorts only to the first
    k items as any further processing is unnecessary for the KNN.

    Parameters
    ----------
    dist : float32[:,:]
        array containing distances for all test points.
    ind : int32[:,:]
        array containing indices for all points in dist
    width : int
        width of query array, or test array.
    height : int
        height, or dimension, of test array. typically dim = 1
    k : int
        number of neighbors to consider,i.e. k-nearest neighbors.

    Returns
    -------
    None.

    """
    l,i,j = int(0),int(0),int(0)
    curr_dist, max_dist = float(0),float(0)
    curr_row, max_row = int(0),int(0)
    xIndex = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
    #Part 1: sort kth first elementz
    if xIndex < width:
        p_dist = dist[xIndex:]
        p_ind = ind[xIndex:]
        max_dist = p_dist[0]
        p_ind[0] = 1
        for l in range(1,k):
            curr_row = l*width
            curr_dist = p_dist[curr_row]
            if curr_dist < max_dist:
                i = l - 1
                for a in range(0,l-1):
                    if p_dist[a*width] > curr_dist:
                        i = a
                        break
                    'end if'
                'end for'
                for j in range(l,i,-1):
                    p_dist[j*width] = p_dist[(j-1)*width]
                    p_ind[j*width] = p_ind[(j-1)*width]
                'end for'
                p_dist[i*width] = curr_dist
                p_ind[i*width] = l+1
            else:
                p_ind[l*width] = l + 1
            'end if'
            max_dist = p_dist[curr_row]
        'end for'
    
        #Part 2: insert element in the k-th first lines
        max_row = (k-1)*width
        for l in range(k,height):
            curr_dist = p_dist[l*width]
            if curr_dist < max_dist:
                i = k-1
                for a in range(0,k-1):
                    if p_dist[a*width]>curr_dist:
                        i=a
                        break
                    'end if'
                'end for'
                for j in range(k-1,i,-1):
                    p_dist[j*width] = p_dist[(j-1)*width]
                    p_ind[j*width] = p_ind[(j-1)*width]
                'end for'
                p_dist[i*width] = curr_dist
                p_ind[i*width] = l+1
                max_dist = p_dist[max_row]
            'end if'
        'end for'
    'end if'
    
'end def'

@cuda.jit
def cuParrallelSqrt(dist,width,k):
    xIndex = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    yIndex = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y
    if xIndex < width and yIndex < k:
        dist[yIndex*width+xIndex] = math.sqrt(dist[yIndex*width+xIndex])
    'end if'
'end def'

def extract_cuda(activation, n_batch, n_channel, height, width, coords, n_max_coord, dim_coord, extracted_activation):
    """
    Parameters
    ----------
    activation : float32[:,:]
        reference feature map
    n_batch : int
        number of feature maps
    n_channel : int
        size of the feature dimension
    height : int
        height of the feature map
    width : int
        width of the feature map
    coords : float32[:,:]
        coordinates of hte points for extraction
    n_max_coord : int
        
    dim_coord : int
        DESCRIPTION.
    extracted_activation : float32[:,:]
        array containing extracted feature map

    Returns
    -------
    None.

    """
    size_of_float = 1
    
    #variables
    #cuda init
    #allocation of memory
    activation_device = cuda.to_device(activation[0])
    extracted_activation_device = cuda.to_device(np.zeros((n_batch*n_channel*n_max_coord*size_of_float)))
    
    coord_device = cuda.to_device(coords[0])
    
    
    #grids and threads
    g_size_r = [(n_batch*n_max_coord*dim_coord)/256,1,1]
    t_size_r = [256,1,1]
    
    if (n_batch*n_max_coord*dim_coord)%256 != 0:
        g_size_r[0] += 1
    'end if'
    
    g_size = [(n_batch*n_max_coord)/256,1,1]
    t_size = [256,1,1]
    if n_batch*n_max_coord%256 != 0:
        g_size[0] += 1
    'end if'
    
    extract_with_interpolation[g_size,t_size](n_batch*n_max_coord,activation_device,
                                              coord_device,extracted_activation_device,
                                              n_max_coord,n_channel,height,width)
    extracted_activation_device.copy_to_host(extracted_activation)
    
    #copy coords to host
    extracted_activation.copy_to_host(extracted_activation_device)
    
    

def knn_cuda(ref_host, ref_width, query_host, query_width, height, k, dist_host, ind_host):
    """

    Parameters
    ----------
    ref_host : float32[:,:]
        array containing all the training values for the KNN
    ref_width : int
        the number of elements/points in the training dataset
    query_host : float32[:,:]
        array containing all the test values for the KNN
    query_width : int
        the number of elements/points in the test dataset
    height : int
        dimension or number of variables in the test dataset
    k : int
        number of neighbors to consider
    dist_host : float32[:,:]
        storage array of zeros for query_width*ref_width points. e.g. np.zeros((query_width*ref_width)).astype(float32)
    ind_host : float32[:,:]
        storage array of zeros for query_width*k points. e.g. np.zeros((query_width*k)).astype(float32)

    Returns
    -------
    None.

    """
    size_of_float = 1
    size_of_int = 1
    
    #cuda init
    
    #Memory Allocation of query points and distances to device
    query_dev = cuda.to_device(np.zeros((query_width*height*size_of_float)).astype(np.float32))
    dist_dev = cuda.to_device(np.zeros((query_width*ref_width*size_of_float)).astype(np.float32))
    #Memory Allocation of index
    ind_dev = cuda.to_device(np.zeros((query_width*k*size_of_int)).astype(np.float32))
    #Memory Allocation of for ref
    ref_dev = cuda.to_device(ref_host[0])
    #copy of part of query actually being treated
    query_dev = cuda.to_device(query_host[0])
    
    #Grids and thread
    g_16x16 = [ceil(query_width/16), ceil(ref_width/16), 1]
    t_16x16 = [16,16,1]
    if query_width%16 != 0: 
        g_16x16[0] += 1
    if ref_width%16 != 0:
        g_16x16[1] += 1
        
    g_256x1 = [ceil(query_width/256),1,1]
    t_256x1 = [256,1,1]
    if query_width%256 != 0:
        g_256x1[0] += 1
    'end if'
    
    
    #compute all distances
    distance_kern[g_16x16,t_16x16](ref_dev, ref_width, query_dev, 
                                   query_width, height, dist_dev)
    
    #sort each column
    sorting_kern[g_256x1, t_256x1](dist_dev,ind_dev,query_width,
                                  ref_width,k)
    
    dist_dev[:(k*query_width)].copy_to_host(dist_host)
    ind_dev.copy_to_host(ind_host)
'end def'
        

def main(test,train,num_neighbors,incr):
    """

    Parameters
    ----------
    test : np.float32[:,:]
        test dataset: array of dim 2 where each row represents some variable and each column is a point in a dataset represented by those variables
    train : np.float32[:,:]
         train dataset: array of dim 2 where each row represents some variable and each column is a point in a dataset represented by those variables
    num_neighbors : int
        the amount of neighbors to consider when the KNN does predictions
    incr : int
        number of test points to consider at once. adjust this if you CuMemAloc error is triggered. i.e. you are trying to access more memory
        on the GPU than is allowed

    Returns
    -------
    dist_host_tot : np.float32[:,:]
        sorted list of the distances between each test point and all the training points, reduced to the number of num_neighbors.
    ind_host_tot : np.float32[:,:]
        sorted list of the indexes for each respective distance. i.e. dist_host_tot[0,0] has index of ind_host_tot[0,0] in the test set. think of pointer arrays from c++
    data : np.float32[:,:]
        classifications for each test point

    """
    
    #dividing image into point sets
    thr_test = range_cont(0,test.shape[1],incr)
    
    #storage arrays
    dist_host_tot = np.zeros((num_neighbors,test.shape[1]))
    ind_host_tot = np.zeros((num_neighbors,train.shape[1]))
    
    #loops through image array using knn and generates id's for points
    for i in range(len(thr_test)-1):
        start = time.time()
        #splice test and train data
        test_s = test[0:1,thr_test[i]:thr_test[i+1]]
        train_s = train[0:1,:]
        
        tr_shape = train_s.shape[1]
        te_shape = test_s.shape[1]
        
        dist_host = np.zeros((te_shape*num_neighbors)).astype(np.float32)
        ind_host = np.zeros((te_shape*num_neighbors)).astype(np.float32)
        
        #optimize this further by not bring dist_dev to host just do all looping inside knn_cuda kernel
        knn_cuda(train_s,tr_shape,test_s,te_shape,test_s.shape[0],num_neighbors,dist_host,ind_host)
        # extract_cuda()
        
        dist_host = dist_host.reshape(num_neighbors,te_shape)
        dist_host_tot[:,thr_test[i]:thr_test[i+1]] =  dist_host
        ind_host_tot[:,thr_test[i]:thr_test[i+1]] = ind_host.reshape(num_neighbors,(thr_test[i+1]-thr_test[i]))
        end = time.time()
        totim = float(end-start)
        print(f'time to run {incr} test points is {totim} seconds')
    'end for'
    
    data = np.zeros((2,ind_host_tot.shape[1]))
    for i in range(ind_host_tot.shape[1]-1):
        neighbors = train[:,ind_host_tot[:,i].astype(int)-1]
        mode_out,freq = mode(neighbors[1,:])
        mean_out = np.mean(neighbors[0,:])
        data[1,i] = mode_out[0]
        data[0,i] = mean_out
    'end for'
    return dist_host_tot, ind_host_tot, data
'end def'

def create_image(boolim,train_pred):
    newim = np.zeros((boolim.shape[0],boolim.shape[1]))
    count = 0
    for i,row in enumerate(boolim):
        for j,col in enumerate(row):
            if col == True:
                newim[i,j] = train_pred[1,count]
                count += 1
            'end if'
        'end for'
    'end for'
    return newim

def imshow_overlay(im, mask, savefig=False, title="default", genfig=True, alpha=0.5, color='red', **kwargs):
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


def intensity_error(data,test):
    err = np.zeros((1,data.shape[1]))
    for i in range(data.shape[1]):
        err[i] = abs(data[0,i] - test[0,i])/test[0,i]
    'end for'
    return err
        

if __name__ == "__main__":
    #Step 1: change directory to the folder containg the image files
    os.chdir(r"C:\Users\jsalm\Documents\Python Scripts\KNN\images_5HT")
    
    #Step 2: Load the first image, additionally need to remove all channels from the iamge other than the red
    #channel, channel 2, and convert the image over to a float format from UINT8.
    image1 = np.array(cv2.imread("dAIH_20x_sectioning_1_CH2.tif")[:,:,2]/255).astype(np.float32)
    #Step 3: Initiate our image object so we can quickly store and process the image data into a singular entity.
    obj1 = ThreeD_Recon_V2.Gen_Training(image1)
    #Step 4: Fourier filter, high pass filtering, which removes all the low frequency areas in the image
    ffimhi1 = Filters.FFfilt(image1,0,250,False)
    #Step 4 1/2: Gaussian blur performed on the high pass filtered image to generate a less specific feature map
    #This image will be used in identifying general areas that we want to consider for the KNN
    gauim1 = convolve(ffimhi1,Filters._d3gaussian(5,1,1))
    #Step 5: Generate bit maps as to where we think the serotonin is. The thresholds here were determined by simply 
    #looking at the image and thresholding at a value that was slightly above the intensities in the background
    #of the image
    boolim1_1 = ffimhi1 > 0.033
    boolim1_2 = gauim1 > 0.03
    #printing output of all the image processing
    plt.figure("1: original");plt.imshow(image1)
    plt.figure("1: High Pass Filtered");plt.imshow(ffimhi1)
    plt.figure("1: Gaussian Blurred Image");plt.imshow(gauim1)
    plt.figure("1: small domain bit map (butons and some appendages)");plt.imshow(boolim1_1)
    plt.figure("1: large domain bit map (features and general areas)");plt.imshow(boolim1_2)
    imshow_overlay(image1,boolim1_1,True,"training")
    
    #Step 2
    image2 = np.array(cv2.imread("dAIH_20x_sectioning_2_CH2.tif")[:,:,2]/255).astype(np.float32)
    #Step 3
    obj2 = ThreeD_Recon_V2.Gen_Training(image2)
    #Step 4
    ffimhi2 = Filters.FFfilt(image2,0,250,False)
    #Step 4 1/2
    gauim2 = convolve(ffimhi2,Filters._d3gaussian(5,1,1))
    #Step 5
    boolim2_1 = ffimhi2 > 0.04
    boolim2_2 = gauim2 > 0.03
    #Print
    plt.figure("2: original");plt.imshow(image2)
    plt.figure("2: High Pass Filtered");plt.imshow(ffimhi2)
    plt.figure("1: Gaussian Blurred Image");plt.imshow(gauim2)
    plt.figure("2: small domain bit map (butons and some appendages)");plt.imshow(boolim2_1)
    plt.figure("2: large domain bit map (features and general areas)");plt.imshow(boolim2_2)
    
    #Generate Test Data using gaussian bit map domain and high pass filtered image.
    #The output variable "Test" is a 2 dimensional array of size: 2 by image_size_x*image_size_y
    #where image_size is simply the x and y component of the image. In the case of our
    #example we use a limited domain to increase processing time, so the size will be
    #a 2 dimensional array of size: 2 by number of True values in the boolim2_2 bit map.
    test,point_te = generate_test_sert_ID(boolim2_2,ffimhi2)
    #We convert to float32, as compared to the default float64, as it reduces the amount of memory we need
    test = test.astype(np.float32)

    #Generate Training data using the fourier filter bit map domain and high pass filtered image
    #typically we use a more rigorous training dataset, but for this case we will be using 
    #a poorer representation just to get the point across. 
    train,point_tr = generate_train_sert_ID(boolim1_1,ffimhi1)
    train = train.astype(np.float32)
    
    #The "main" program contains the set up for the CUDA kernels that we use for the KNN as well as memory
    #managment and data generation.
    dist_host,ind_host,data = main(test,train,10,100)
    #The KNN data is convereted from its matrix orientation back to its original image format so we can view the results
    image_knn = create_image(boolim2_2,data)
    plt.figure('knn final');plt.imshow(image_knn)
    imshow_overlay(image2,image_knn,True,"Knn_out")
