#!/usr/bin/env python3

from scipy import signal
from scipy.signal import max_len_seq
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft,  fftshift, ifftshift


#https://humble-ballcap-e09.notion.site/14-IQ-f05a7383ce384e51a13de2b04708bbc8

file_name = "output_det.txt"
ft = 100e3 # кб в секунду
fs = 900e6 # частота дискретезации
kr=max_len_seq(5)[0]
m=2*kr-1
ns = fs / ft # отчетов на символ
b = np.ones(int(ns)) #Коэффициенты фильтра интерполятора
ts1t =np.array([0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1])
ts1t = 2 * ts1t - 1
x_IQ = np.hstack((ts1t,m)) # формирование пакета 

data = np.loadtxt(file_name, dtype=np.complex64, delimiter=",")




def processing_data(xrec1):
    xrec = xrec1/np.mean(xrec1**2)
    
    fsr=2 * np.pi * ft/fs
    b2,a2 = signal.butter(10,fsr)
    y1 = signal.lfilter(b2,a2,xrec)
    #y2=signal.decimate(y1,ns)
    yf=np.convolve(np.abs(y1.real),b)
    y2=signal.decimate(yf,int(ns))
    y=np.correlate(y2, ts1t)
    #plt.figure(2)
    plt.plot(np.abs(y))
    ind = np.argmax(abs(y),axis=0)
    yy=y2[ind:ind+len(data)] 
    print(len(xrec1))
    plt.show()

processing_data(data)




