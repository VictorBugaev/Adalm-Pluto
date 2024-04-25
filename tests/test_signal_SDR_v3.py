#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:46:23 2024

@author: plutosdr
"""


import os
import sys

import numpy as np
from scipy import signal
from scipy.signal import max_len_seq
from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import matplotlib.pyplot as plt
import subprocess
import time

import adi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import tx_rx.conf as conf

import sample.sample as sample

PLATFORM = "Linux"

sample_rate  = 1000000

def config_(sdr):
    
    F_n = 2900100011
    sdr.sample_rate = sample_rate
    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    #sdr.rx_lo = 1900100011
    #sdr.tx_lo = 1900100011
    sdr.rx_lo = F_n
    sdr.tx_lo = F_n
    sdr.tx_cyclic_buffer = True
    #sdr.tx_cyclic_buffer = False
    sdr.tx_hardwaregain_chan0 = 0
    sdr.rx_hardwaregain_chan0 = 20
    #sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.gain_control_mode_chan0 = "manual"
    


#rf_module = conf.RxTx(adi.Pluto('ip:192.168.2.1'))
#rf_module.print_parameters()

def sqrt_rc_imp(ns, alpha, m):
    n = np.arange(-m * ns, m * ns + 1)
    b = np.zeros(len(n))
    ns *= 1.0
    a = alpha
    for i in range(len(n)):
       #if abs(1 - 16 * a ** 2 * (n[i] / ns) ** 2) <= np.finfo(np.float32).eps/2:
        #   b[i] = 1/2.*((1+a)*np.sin((1+a)*np.pi/(4.*a))-(1-a)*np.cos((1-a)*np.pi/(4.*a))+(4*a)/np.pi*np.sin((1-a)*np.pi/(4.*a)))
       #else:
           b[i] = 4*a/(np.pi * (1 - 16 * a ** 2 * (n[i] / ns) ** 2))
           b[i] = b[i]*(np.cos((1+a) * np.pi * n[i] / ns) + np.sinc((1 - a) * n[i] / ns) * (1 - a) * np.pi / (4. * a))
    return b    

def OFDM(index):
    PI = 3.14
    T = 10 ** -4
    Nc = 4
    df = 10**4
    fs = 4 * df
    ts = 1/fs
    delta_f = 1 / T
    t = np.arange(0, 0.5, ts)
    s = np.array([1+3j, 1-5j, -1+3j, 4+5j])
    f1 = 1 
    f2 = 2 
    f3 = 14
    f4 = 10 
    dert = [np.zeros(len(t), dtype="complex") for i in range(len(s))]
    xt = np.zeros(len(t), dtype="complex")
    for i in range(len(t)):
        dert[0][i] = s[0] * np.exp(1j * PI * 2 * f1 * t[i]) 
        dert[1][i] = s[1]* np.exp(1j * PI * 2 * f2 * t[i])
        dert[2][i] = s[0] * np.exp(1j * PI * 2 * f3 * t[i]) 
        dert[3][i] = s[1]* np.exp(1j * PI * 2 * f4 * t[i])
    for i in range(len(s)):
        xt += dert[i]
    plt.figure(20, figsize = (20, 20))
    plt.subplot(2,2,1)
    plt.plot(t, xt.real)
    plt.plot(t, xt.imag)
    plt.subplot(2,2,2)
    for i in range(len(s)):
        plt.plot(dert[i])
    return dert[index]

dir_gen_packet = "../src/generate_packet/"
argvs = ['data.txt', 'gold_sequence.txt', 'gold_seq_end.txt', 'data_bin.txt' ]

for i in range(len(argvs)):
    argvs[i] = dir_gen_packet + argvs[i]

module_gen_header = "generate_packet"
if(PLATFORM == "Win"):
    module_gen_header += ".exe"

    
data = "e"

file = open(argvs[0], "w")
file.write(data)
file.close()
#sys.exit()
subprocess.call([dir_gen_packet + module_gen_header] + argvs)

time.sleep(1)

file = open(argvs[-1], "r")

data_bin = file.read()
data_bin = list(map(int, data_bin))
for i in data_bin:
    print(i, end="")

#data = np.random.randint(0, 2, 200)
#data_bin = data
plt.figure(1, figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Data")
plt.plot(data_bin)

N = 10
data_rep = data_bin
N_qam = 4
data_qpsk = np.array(sample.encode_QAM(data_rep, N_qam))
#data_qpsk = sample.encode_QPSK(data_rep, 4)
N2 = 10#длительность символа
#data_qpsk = sample.duplication_sample(data_qpsk, N2)
#Количество поднесущих
Nb = 64
#Защитный интервал
N_interval = 16

fs = sample_rate
rs=100000
ns=fs//rs

symbol_ofdm = 0.7 + 0.7j


data_qpsk = np.array(data_qpsk)
plt.subplot(2, 2, 2)
plt.title(f"QAM{N_qam}")
plt.scatter(data_qpsk.real, data_qpsk.imag)

    

def gardner_TED(data):
    error = 0
    tau = 2
    t1 = 1
    errors = [0 for i in range(len(data))]
    for i in range(1, len(data)):
        t1 = i
        t2 = t1 + tau
        errors[i] = (data.real[i-1]) % N
    
#Не преминяется для QAM16 
def TED_loop_filter(data): #ted loop filter 
    BnTs = 0.01 
    Nsps = 10
    C = np.sqrt(2)
    Kp = 1
    teta = ((BnTs)/(Nsps))/(C + 1/(4*C))
    K1 = (-4*C*teta)/((1+2*C*teta+teta**2)*Kp)
    K2 = (-4*teta**2)/((1+2*C*teta+teta**2)*Kp)
    print("K1 = ", K1)
    print("K2 = ", K2)
    #K1_2 = (1/Kp)*((((4*C)/(Nsps**2))*((BnTs/(C + (1/4*C)))**2))/(1 + ((2 * C)/Nsps)*(BnTs/(C + (1/(4*C))))+(BnTs/(Nsps*(C+(1/4*C))))**2))
    err = np.zeros(len(data)//10, dtype = "complex_")
    data = np.roll(data,-0)
    nsp = 10
    p1 = 0
    p2 = 0
    n = 0
    mass_cool_inex = []
    mass_id = []
    for ns in range(0,len(data)-(2*nsp),nsp):
        #real = (data.real[ns+n] - data.real[nsp+ns+n]) * data.real[n+(nsp)//2+ns]
        #imag = (data.imag[ns+n] - data.imag[nsp+ns+n]) * data.imag[n+(nsp)//2+ns]
        real = (data.real[nsp+ns+n] - data.real[ns+n]) * data.real[n + (nsp)//2+ns]
        imag = (data.imag[nsp+ns+n] - data.imag[ns+n] ) * data.imag[n + (nsp)//2+ns]
        err[ns//nsp] = np.mean(real + imag)
        #err[ns//nsp] = np.mean((np.conj(data[nsp+ns+n]) - np.conj(data[ns+n]))*(data[n + (nsp)//2+ns])) 
        error = err.real[ns//nsp]
        p1 = error * K1
        p2 = p2 + p1 + error * K2
        #print(ns ," p2 = ",p2)  
        while(p2 > 1):
            #print(ns ," p2 = ",p2)
            p2 = p2 - 1
        while(p2 < -1):
            #print(ns ," p2 = ",p2)
            p2 = p2 + 1
        
        n = round(p2*10)  
        n1 = n+ns+nsp   
        mass_cool_inex.append(n1)
        mass_id.append(n)

    #mass_cool_inex = [math.ceil(mass_cool_inex[i]) for i in range(len(mass_cool_inex))]
    mass1 = np.asarray(mass_cool_inex)
    mass = np.asarray(mass_id)
    plt.figure(10, figsize=(10, 10))
    plt.subplot(2,1,1)
    plt.plot(err) 
    plt.subplot(2,1,2)
    plt.plot(mass)   
    
    return mass1
def PLL(conv):
    mu = 1  
    theta = 1
    phase_error = np.zeros(len(conv))  
    output_signal = np.zeros(len(conv), dtype=np.complex128)

    for n in range(len(conv)):
        theta_hat = np.angle(conv[n]) 
        #print(theta_hat)
        phase_error[n] = theta_hat - theta  
        output_signal[n] = conv[n] * np.exp(-1j * theta)  
        theta = theta + mu * phase_error[n]  
    return output_signal


#gardner_TED(data_conv)
#rrc_filter(N, data_conv.real)

#sdr.rx_rf_bandwidth = 1000000
#sdr.rx_destroy_buffer()
#sdr.rx_hardwaregain_chan0 = -5

data_qpsk *= 2**14
plt.subplot(2, 2, 4)
plt.title(f"QAM{N_qam}")
plt.scatter(data_qpsk.real, data_qpsk.imag)


#sample.OFDM_modulator(data_qpsk, Nb)

#sys.exit()



IF_SDR = False
IF_SDR = True

if IF_SDR:
    sdr = adi.Pluto('ip:192.168.3.1')
    
    sdr2 = sdr
    #sdr2 = adi.Pluto('ip:192.168.3.1')

    
    config_(sdr)
    config_(sdr2)


    sdr.rx_buffer_size =2*len(data_qpsk) *40
    sdr2.rx_buffer_size =2*len(data_qpsk) *40

CON = 1

if CON == 1:
    #OFDM
    Nc = N_qam
    #xt2 = data_qpsk
    xt2 = sample.OFDM_modulator(data_qpsk, Nb, N_interval, symbol_ofdm)
    #xt2 = np.fft.ifft(data_qpsk, Nc)
    #sdr.tx(data_qpsk)
    plt.figure(12, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    #xt2 = np.concatenate([xt2, np.zeros(4000)])
    plt.title("OFDM")
    plt.plot(abs(xt2))
    #sys.exit()
    print("Отправлено:", len(xt2))
    if IF_SDR:
        sdr.rx_buffer_size =2*len(xt2) * 40
        sdr2.rx_buffer_size =2*len(xt2) * 40
        
        sdr.tx(xt2)
    
    
        data_read = sdr2.rx()
    else:
        data_read = xt2
    if IF_SDR:
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        sdr2.tx_destroy_buffer()
        sdr2.rx_destroy_buffer()
    data_read2 = data_read
    
    index1 = sample.correlat_ofdm(data_read2, N_interval, Nb)
    print("index1:", index1)
    
    arr = []
    start_data = -1
    if_start = 0.9
    for i in range(0, len(data_read2)):
        
        #a = np.vdot(data_read2[0:N_interval], data_read2[Nb:Nb + N_interval])
        a = sample.norm_corr(data_read2[0:N_interval], 
                             data_read2[Nb:Nb + N_interval])
        a = abs(a)
        if start_data == -1 and if_start <= a:
            start_data = i
        arr.append(a)
        data_read2 = np.roll(data_read2, -1)
    plt.subplot(2, 2, 2)
    plt.title("Correlation")
    plt.plot(arr)
    #start_data = 0
    
    print("Начало данных:", start_data)
    data = data_read[start_data:start_data + len(xt2)]
    data = sample.del_prefix_while(data, Nb, N_interval)
    data2 = data_read[index1:index1 + len(xt2)]
    data2 = sample.del_prefix_while(data2, Nb, N_interval)
    
    plt.figure(5, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("data_read")
    plt.scatter(data_read.real, data_read.imag)
    
    plt.subplot(2, 2, 2)
    plt.title("data")
    plt.scatter(data.real, data.imag)
    plt.scatter(data2.real, data2.imag)
    
    
    n = 3
    data_readF = np.fft.fft(data)
    data_readF2 = np.fft.fft(data2)
    
    #data_read = data_readF
    #data = np.convolve(np.ones(N2), data_readF)/1000
    plt.subplot(2, 2, 3)
    plt.title("data_readF")
    plt.scatter(data_readF.real, data_readF.imag)
    plt.scatter(data_readF2.real, data_readF2.imag)
    
    
    #sys.exit()
    
    #indexs = TED_loop_filter(data_read)
    
    #data_read = data_read[indexs]
    data_read1 = PLL(data)

    
    plt.figure(10, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Принятый сигнал")
    plt.scatter(data_read.real, data_read.imag)
    plt.subplot(2, 2, 2)
    plt.plot(data_read.real)
    plt.plot(data_read.imag)
    
    plt.subplot(2, 2, 3)
    plt.title("Обработанный сигнал")
    plt.scatter(data.real, data.imag)
    plt.subplot(2, 2, 4)
    plt.title("Обработанный сигнал")
    plt.scatter(data_read1.real, data_read1.imag)
    
    plt.show()

elif CON == 2:
    #OFDM
    Nc = N_qam
    xt2 = data_qpsk
    
    xt2 = np.fft.ifft(data_qpsk, Nc)
    #sdr.tx(data_qpsk)
    sdr.tx(xt2)
    
    
    data_read = sdr2.rx()
    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    sdr2.tx_destroy_buffer()
    sdr2.rx_destroy_buffer()
    n = 3
    data_readF = np.fft.fft(data_read, n)
    data_read = data_readF
    data_read = np.convolve(np.ones(N2), data_read)/1000
    indexs = TED_loop_filter(data_read)
    
    data_read = data_read[indexs]
    data_read1 = PLL(data_read)
    #data_read = data_read/np.mean(data_read**2)
    #data_read_1 = data_read
    
    plt.figure(5, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Принятый сигнал")
    plt.scatter(data_read.real, data_read.imag)
    plt.subplot(2, 2, 2)
    plt.plot(data_read.real)
    plt.plot(data_read.imag)
    
    plt.subplot(2, 2, 3)
    plt.title("Обработанный сигнал")
    plt.scatter(data_read1.real, data_read1.imag)
    plt.show()










