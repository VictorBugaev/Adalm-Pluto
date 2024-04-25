#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:30:07 2023

@author: plutosdr
"""



import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import max_len_seq
from scipy.fftpack import fft, ifft,  fftshift, ifftshift




CON_TX = 1
CON_RX = 1
CON_SINH = 2

sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = 1000000
sdr.tx_destroy_buffer()
 

sdr.rx_lo = 2000000000
sdr.tx_lo = 2000000000
sdr.tx_cyclic_buffer = True
#sdr.tx_cyclic_buffer = False
sdr.tx_hardwaregain_chan0 = -5
sdr.gain_control_mode_chan0 = "slow_attack"


fs = sdr.sample_rate
rs=100000
ns=fs//rs
 

data=max_len_seq(8)[0] 
data = np.concatenate((data,np.zeros(1)))
 
 
x_ = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])
b7=np.array([1,-1,1,1,1,-1,1])
ts1 =np.array([0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1])
ts2 =[0,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1]
ts3 =[1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,1]
ts4 =[1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0]
m=2*data-1
#ts1t=2*ts1-1
ts1t=b7
 

b = np.ones(int(ns))
 
#qpsk
filename_input = "input_qpsk_new_type.txt"
filename_output = "output_qosk_new_type.txt"
  

x=np.reshape(m,(2,128))
xi=x[0,:]
xq=x[1,:]
x_bb=(xi+1j*xq)/np.sqrt(2)
plt.figure(1, figsize=(10,10))
plt.subplot(2,2,1)
plt.title("Исходный")
plt.scatter(x_bb.real,x_bb.imag)

xiq=2**14*x_bb
 
n_frame= len(xiq)
sdr.tx(xiq)

#Сохранение в файл
np.savetxt(filename_input, xiq, delimiter=":")
sdr.rx_rf_bandwidth = 1000000
sdr.rx_destroy_buffer()
sdr.rx_hardwaregain_chan0 = -5
sdr.rx_buffer_size =2*n_frame*4
#Принятый сигнал
xrec1=sdr.rx()
#Отключение циклической передачи
sdr.tx_destroy_buffer()



#   Грубая частотная синхронизация
if CON_SINH == 1:
    #Сохранение в файл
    np.savetxt(filename_output, xrec1, delimiter=":")
    xrec = xrec1/np.mean(xrec1**2)
    
    plt.subplot(2,2,2)
    plt.title("Принятый")
    plt.scatter(xrec.real,xrec.imag)
    m = 4
    xrec = xrec ** m
    plt.subplot(2,2,3)
    plt.title("Возведение в степень: "+str(m))
    plt.scatter(xrec.real,xrec.imag, color = "r")
    sig_fft = abs(fft(xrec, 2048))
    plt.subplot(2,2,4)
    plt.title("модуль от сигнала преобразованного по FFT")
    plt.stem(sig_fft, "green")
    
    index_max_elem = np.argmax(sig_fft)
    print("index_max_elem =", index_max_elem)
    max_fft_sig = sig_fft[index_max_elem]
    print("max_fft_sig =",max_fft_sig)
    sig_fft_shift = fftshift(sig_fft)
    plt.figure(2, figsize=(10,10))
    plt.subplot(2,2,1)
    plt.title("Сдвиг спектра")
    plt.stem(sig_fft_shift, "black")
    w = np.linspace(-np.pi, np.pi, len(sig_fft_shift))
    index_max_elem = np.argmax(abs(sig_fft_shift))
    print("index_max_elem =", index_max_elem)
    max_fft_sig = sig_fft[index_max_elem]
    print("max_fft_sig =",max_fft_sig)
    print("w[index_max_fft_shift] =", w[index_max_elem])
    fax = w[index_max_elem] / m
    print("fax = ", fax)
    fax = abs(fax)
    t2 = np.exp( -1j * fax)
    #Исходный сигнал умножается на угол сдвига
    xrec2 = xrec1 * t2
    plt.subplot(2,2,2)
    plt.title("Сдвиг принятого сигнала")
    plt.scatter(xrec2.real,xrec2.imag, color = "g")
if 0:
    plt.subplot(2,2,2)
    plt.title("Принятый")
    plt.scatter(xrec1.real,xrec1.imag)
    size = len(xrec1)
    F = 2000
    a1 = 0.05#коэф фильтра
    exp_line = np.zeros(size)
    F_e = np.zeros(size)
    integrator_t = 0
    f_integrator_out = np.zeros(size)
    #Fe - ошибка
    F_e[0] = np.random.rand(1) * np.pi
    F_0 = np.zeros(size)
    #Fen -разность фаз
    for i in range(len(xrec1)):
        
        
        exp_line[i] = np.exp(1j * F_0[i])
        F_e[i] = np.angle( xrec1[i] * exp_line[i] ) * a1
        f_integrator_out[i] += F_e[i]
        
if CON_SINH == 2:
    plt.subplot(2,2,2)
    plt.title("Принятый")
    plt.scatter(xrec1.real,xrec1.imag)
    size = len(xrec1)
    F = 2000
    a1 = 0.05#коэф фильтра
    exp_line = np.zeros(size)
    F_e = np.zeros(size)
    integrator_t = 0
    f_integrator_out = np.zeros(size)
    #Fe - ошибка
    F_e[0] = np.random.rand(1) * np.pi
    F_0 = np.zeros(size)
    #Fen -разность фаз
    for i in range(len(xrec1)):
        
        
        exp_line[i] = np.exp(1j * F_0[i])
        F_e[i] = np.angle( xrec1[i] * exp_line[i] ) * a1
        f_integrator_out[i] += F_e[i]
        
        #exp_line[i] = 
        xrec1 *= F_e[i]
    
    plt.subplot(2,2,3)
    plt.title("Test")
    plt.scatter(xrec1.real,xrec1.imag, color = "r")























