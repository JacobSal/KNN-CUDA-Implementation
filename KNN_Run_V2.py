# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:51:28 2020

@author: jsalm
"""
import Training_Gen_V3
import ThreeD_Recon_V2
import KNN_alg_GPU_V2
import Filters
import os

#######
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

#06/10/2020: need to figure out how we want to sort through the directory. Do we want to do one image at a time or whole folders
#how would we implement this in our code? see: Training_Gen.py and ThreeD_Recon_V2.py

def save_im(image,title):
    savedir = r"C:\Users\jsalm\Documents\Python Scripts\3DRecon\saved_Im"
    plt.figure("image_save",frameon=False)
    plt.imshow(image)
    plt.savefig(os.path.join(savedir,"overlayed_"+str(title)+".tif"),dpi=600,quality=95,pad_inches=0)

#Training Generation
############################
foldername = "Training_Im"


im_dir = ThreeD_Recon_V2.DataMang(foldername)
im_list = [i for i in range(0,im_dir.dir_len)]
train_im = []
count = 0

for gen in im_dir.open_dir(2,im_list):
    image,nW,nH = gen
    Training_Gen_V3.main(image,im_list[count])
    train_im.append(Training_Gen_V3.reconstruct_train(count,nW,nH))
    count += 1
    break
"end for"
############################


#Testing Generation
############################
foldername = "Test_Im"


im_dir = ThreeD_Recon_V2.DataMang(foldername)
im_list = [i for i in range(0,im_dir.dir_len)]
test = np.array([[0],[0]])

for gen in im_dir.open_dir(2,im_list):
    image_te,nW,nH = gen
    ffimhi_te = Filters.FFfilt(image_te,0,250,False)
    gauim_te = convolve(ffimhi_te, Filters._d3gaussian(5,1,1))
    boolim_te = gauim_te > 0.03
    im_data,point_te = KNN_alg_GPU_V2.generate_test_sert_ID(boolim_te,ffimhi_te)
    test = np.concatenate((test,im_data.astype(np.float32)),axis=1)
    break
'end for'
#splicing to remove initialization step
test = test[:,1:]
###########################


#Training Reconstitution
###########################
#Generate Training data using the fourier filter bit map domain and high pass filtered image
#typically we use a more rigorous training dataset, but for this case we will be using 
#a poorer representation just to get the point across.
im_dir = ThreeD_Recon_V2.DataMang(foldername)
im_list = [i for i in range(0,im_dir.dir_len)]
train = np.array([[0],[0]])
count = 0

for gen in im_dir.open_dir(2,im_list):
    image,nW,nH = gen
    ffimhi = Filters.FFfilt(image,0,250,False)
    im_data,point_tr = KNN_alg_GPU_V2.generate_train_sert_ID(train_im[count],ffimhi)
    save_im(train_im[count],"bool_train"+str(count))
    train = np.concatenate((train,im_data.astype(np.float32)),axis=1)
    count+=1
    break
'end for'
#splicing to remove initialization step
train = train[:,1:]
#The "main" program contains the set up for the CUDA kernels that we use for the KNN as well as memory
#managment and data generation.
############################

#KNN Main
############################
print("starting KNN...")  
dist_host,ind_host,data = KNN_alg_GPU_V2.main(test,train,10,100)
#The KNN data is convereted from its matrix orientation back to its original image format so we can view the results
image_knn = KNN_alg_GPU_V2.create_image(boolim_te,data)
plt.figure('knn final');plt.imshow(image_knn)
KNN_alg_GPU_V2.imshow_overlay(image_te,image_knn,True)
############################

    
    