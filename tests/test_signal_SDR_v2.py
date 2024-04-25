#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:58:16 2024

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


sdr = adi.Pluto('ip:192.168.2.1')

#sdr2 = sdr
sdr2 = adi.Pluto('ip:192.168.3.1')



def config_(sdr):
    

    sdr.sample_rate = 1000000
    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    sdr.rx_lo = 1900100011
    sdr.tx_lo = 1900100011
    sdr.tx_cyclic_buffer = True
    #sdr.tx_cyclic_buffer = False
    sdr.tx_hardwaregain_chan0 = 0
    sdr.rx_hardwaregain_chan0 = 20
    #sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.gain_control_mode_chan0 = "manual"
    

config_(sdr)
config_(sdr2)


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

    
data = "tESTadfasfasdfsdf"

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

data = np.random.randint(0, 2, 200)
data_bin = data
plt.figure(1, figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Data")
plt.plot(data_bin)



N = 10
#data_rep = sample.duplication_sample(data_bin, N)

#data_rep = data_bin
data_rep = data_bin
N_qam = 4
data_qpsk = np.array(sample.encode_QAM(data_rep, N_qam))
#data_qpsk = sample.encode_QPSK(data_rep, 4)
N2 = 10#длительность символа
data_qpsk = sample.duplication_sample(data_qpsk, N2)

fs = sdr.sample_rate
rs=100000
ns=fs//rs

CON_GEN_DATA = 0

if CON_GEN_DATA == 1:
    data=max_len_seq(8)[0] 
    data = np.concatenate((data,np.zeros(1)))
    x_ = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])
    b7=np.array([1,-1,1,1,1,-1,1])
    m=2*data-1
    ts1t=b7
    b1 = sqrt_rc_imp(ns,0.35,4) 
    b = np.ones(int(ns))
    x=np.reshape(m,(2,128))
    xi=x[0,:]
    xq=x[1,:]
    x_bb=(xi+1j*xq)/np.sqrt(2)
    N_input = len(x_bb)
    xup = np.hstack((x_bb.reshape(N_input,1),np.zeros((N_input, int(ns-1)))))
elif CON_GEN_DATA == 2:
    N_input = len(data_qpsk)
    xup = np.hstack((data_qpsk.reshape(N_input,1),np.zeros((N_input, int(ns-1)))))
elif CON_GEN_DATA == 3:
    pass

if CON_GEN_DATA:
    xup= xup.flatten()
    b = np.ones(int(ns))
    x1 = signal.lfilter(b, 1,xup) 
    data_qpsk = x1
#data_qpsk += 7+7j

#data_qpsk = 
data_qpsk = np.array(data_qpsk)
plt.subplot(2, 2, 2)
plt.title(f"QAM{N_qam}")
plt.scatter(data_qpsk.real, data_qpsk.imag)

if 0:
    h1 = np.ones(N)
    # Noise
    pos_read = 0
    noise_coef = 0.0001
    data_noise = data_qpsk[0:] 
    noise = np.random.normal(0, noise_coef, len(data_noise))
    #data_noise += noise
    
    plt.subplot(2, 2, 4)
    plt.title(f"QAM{N_qam} + noise({noise_coef})")
    plt.scatter(data_noise.real, data_noise.imag)
    
    data_conv = np.convolve(h1,data_noise,'full')
    
    plt.figure(3, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("data convolve")
    plt.plot(data_conv)
    
    #eye diagram
    data_conv_real = data_conv.real
    plt.subplot(2, 2, 2)
    for i in range(0,len(data_conv_real), N):
         plt.plot(data_conv_real.real[0:N*2])
         data_conv_real1 = np.roll(data_conv_real,-1* N)
         data_conv_real= data_conv_real1
    

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
sdr.rx_buffer_size =2*len(data_qpsk) *4
sdr2.rx_buffer_size =2*len(data_qpsk) *4

data_qpsk *= 2**14
plt.subplot(2, 2, 4)
plt.title(f"QAM{N_qam}")
plt.scatter(data_qpsk.real, data_qpsk.imag)


if 0:
    #OFDM
    Nc = N_qam
    xt2 = data_qpsk
    #xt2 = np.fft.ifft(data_qpsk, Nc)
    #sdr.tx(data_qpsk)
    sdr.tx(xt2)
    
    
    data_read = sdr2.rx()
    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    sdr2.tx_destroy_buffer()
    sdr2.rx_destroy_buffer()
    n = 3
    #data_readF = np.fft.fft(data_read, n)
    
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

else:
    xrec1=sdr.rx()
    #Отключение циклической передачи
    sdr.tx_destroy_buffer()
    plt.subplot(2,2,2)
    plt.title("Принятый")
    plt.scatter(xrec1.real,xrec1.imag)
    #qpsk
    filename_input = "input_qpsk_new_type.txt"
    filename_output = "output_qosk_new_type.txt"
      
    #Сохранение в файл
    np.savetxt(filename_output, xrec1, delimiter=":")
    xrec = xrec1/np.mean(xrec1**2)

    #plt.subplot(2,2,2)
    #plt.title("Принятый")
    #plt.scatter(xrec.real,xrec.imag)

    #   Грубая частотная синхронизация
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








