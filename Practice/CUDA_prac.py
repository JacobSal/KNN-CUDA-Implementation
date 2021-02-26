# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:45:19 2020

@author: jsalm
"""
from __future__ import division
from numba import cuda
import numpy as np
import math


print(cuda.gpus)

def train_loop_cuda_int(train,test_row):
    x,y = cuda.grid(1)
    distance = np.array(train.shape[0])
    temp = 0
    if x < train.shape[0] and y < train.shape[1]:
        temp += (train[x,y] - test_row[1,y])**2
        distance[x] = np.sqrt(np.add(temp))
    return distance
    
'end def'

train = 
test_row = 
threadsperblock = (16,16)
blockspergrid_x = math.ceil(an_array.shape[0]/threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1]/threadsperblock[1])
blockspergrid = (blockspergrid_x,blockspergrid_y)
increment_by_one[blockspergrid,threadsperblock](an_array)
print(an_array)
