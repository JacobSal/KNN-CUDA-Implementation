# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:36:35 2020

@author: jsalm
"""
import numpy as np
import matplotlib.pyplot as plt

def square_wave(width,intensity):
    x = np.arange(-(width+5),(width+5),0.01)
    y = np.zeros((x.shape[0]))
    w1 = int(x.shape[0]/2-width/0.02)
    w2 = int(x.shape[0]/2+width/0.02)
    y[w1:w2] = intensity
    return x,y
'end def'

def sine_wave(width,intensity):
    x = np.arange(-(width+5),(width+5),0.01)
    y = np.sin(x)*intensity
    return x,y

def quadratic(width,intensity):
    x = np.arange(-(width+5),(width+5),0.01)
    y = -intensity*x**2+100
    w1 = int(x.shape[0]/2-width/0.02)
    w2 = int(x.shape[0]/2+width/0.02)
    y[:w1] = 0
    y[w2:] = 0
    return x,y

def fourier_fun(function):
    x,y = function
    ffty = np.fft.fftshift(np.fft.fft(y))
    plt.figure('f(x)');plt.plot(x,y)
    plt.figure('fft(f(x))');plt.plot(x,ffty/np.max(y))
    return 0
'end def'

if __name__ == "__main__":
    fourier_fun(square_wave(20,1))
    fourier_fun(quadratic(20,1))
'end if'
