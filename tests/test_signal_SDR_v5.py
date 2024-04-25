#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:17:56 2024

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

import platform

def EXIT(show = 1):
    #plt.ion()
    #plt.pause(100)
    if(show):
        plt.show()
    sys.exit()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import tx_rx.conf as conf

import sample.sample as sample



PLATFORM = platform.system()#"Linux"

sample_rate  = 1000000

IF_SDR = False
READ_FILE = 0#Если True, то игнорируются блоки: 1, 2, 3, 4, сигнал считывается из файла

CON = 1# метод обработки сигнала (1, 2)

filename_output = "input_qpsk.txt" #файл записи принятого сигнала

print("Platform:", PLATFORM)

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

## PARAMETRS
if 1:
    N2 = 10#длительность символа
    #data_qpsk = sample.duplication_sample(data_qpsk, N2)
    #Количество поднесущих
    Nb = 32
    #Защитный интервал
    N_interval = 4
    RS = 4 # < Nb
    Nz = 6 # < Nb
    
    pilot = 100009.7 + 100009.7j


if READ_FILE == 0:
    ## BLOCK 1 - формирование сигнала (11 - 20)
    print("BLOCK 1")
    dir_gen_packet = "../src/generate_packet/"
    argvs = ['data.txt', 'gold_sequence.txt', 'gold_seq_end.txt', 'data_bin.txt' ]

    for i in range(len(argvs)):
        argvs[i] = dir_gen_packet + argvs[i]

    module_gen_header = "generate_packet"
    if(PLATFORM == "Windows"):
        module_gen_header += ".exe"

    data = "Andrey Karpenko"
    data = "ert"
    file = open(argvs[0], "w")
    file.write(data)
    file.close()
    #EXIT()
    subprocess.call([dir_gen_packet + module_gen_header] + argvs)

    time.sleep(1)

    file = open(argvs[-1], "r")

    data_bin = file.read()
    data_bin = list(map(int, data_bin))
    for i in data_bin:
        print(i, end="")

    #data = np.random.randint(0, 2, 200)
    #data_bin = data
    plt.figure(11, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Data")
    plt.plot(data_bin)

    N = 10
    data_rep = data_bin
    N_qam = 4
    data_qpsk = np.array(sample.encode_QAM(data_rep, N_qam))
    #data_qpsk = sample.encode_QPSK(data_rep, 4)
    

    fs = sample_rate
    rs=100000
    ns=fs//rs


    data_qpsk = np.array(data_qpsk)
    plt.subplot(2, 2, 2)
    plt.title(f"QAM{N_qam}")
    plt.scatter(data_qpsk.real, data_qpsk.imag)

        



    #gardner_TED(data_conv)
    #rrc_filter(N, data_conv.real)

    #sdr.rx_rf_bandwidth = 1000000
    #sdr.rx_destroy_buffer()
    #sdr.rx_hardwaregain_chan0 = -5

    data_qpsk *= 2**14
    plt.subplot(2, 2, 4)
    plt.title(f"QAM{N_qam} ** 14")
    plt.scatter(data_qpsk.real, data_qpsk.imag)


    #sample.OFDM_modulator(data_qpsk, Nb)

        #OFDM
    Nc = N_qam
    ofdm_argv = sample.OFDM_modulator(data_qpsk, Nb, N_interval, pilot, RS, Nz)
    ofdm_tx = ofdm_argv[0]
    count_ofdm = ofdm_argv[1]
    #ofdm_indexes = ofdm_argv[2]
    plt.figure(12, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("OFDM, count " + str(ofdm_argv[1]))
    plt.plot(abs(np.fft.fft(ofdm_tx,int(1e6))))
    len_data_tx = len(ofdm_tx)
    len_one_ofdm = int(len_data_tx / count_ofdm)
    print("len_data_tx = ", len_data_tx)
    
    
    
    if IF_SDR:
        ## BLOCK 2 - настройка SDR (21-30)
        print("BLOCK 2")
        import adi
        sdr = adi.Pluto('ip:192.168.3.1')
        
        sdr2 = sdr
        #sdr2 = adi.Pluto('ip:192.168.3.1')
        config_(sdr)
        config_(sdr2)
        sdr.rx_buffer_size =2*len(data_qpsk) *40
        sdr2.rx_buffer_size =2*len(data_qpsk) *40

    ## BLOCK 3 - отправка сигнала (31-40)
    print("BLOCK 3")
    if IF_SDR:
        sdr.rx_buffer_size =2*len(ofdm_tx) * 40
        sdr2.rx_buffer_size =2*len(ofdm_tx) * 40
        sdr.tx(ofdm_tx)
        print("Отправлено:", len(ofdm_tx))
    else:
        #Добавление шума
        print("Add noise")
        plt.figure(31, figsize = (10, 10))
        
        e = 50
        e_no_data = 1000
        np.random.normal
        start_t_data = 40
        len_end = 50
        noise = np.random.normal(loc=-e, scale=e, size= start_t_data + len(ofdm_tx) + len_end)
        noise = noise + noise * 1j
        
        plt.subplot(2, 2, 1)
        plt.title("Noise")
        plt.plot(noise)
        ofdm_rx = ofdm_tx #/ 50
        noise_s = np.random.normal(loc=-e, scale=e, size= start_t_data + len(ofdm_tx) + len_end)
        noise = noise + noise * 1j
        if 1:
            t_s = np.random.normal(loc=-e_no_data, scale=e_no_data, size= start_t_data)
            t_s = t_s + t_s * 1j
            t_e = np.random.normal(loc=-e_no_data, scale=e_no_data, size= len_end)
            t_e = t_e + t_e * 1j
        else:
            t_s = np.zeros(start_t_data, dtype="complex_")
            t_e = np.zeros(len_end, dtype="complex_")
        ofdm_rx = np.concatenate([t_s, ofdm_rx, t_e])
        if 1:
            ofdm_rx += noise
        plt.subplot(2, 2, 2)
        plt.title("Data + Noise")
        plt.plot(ofdm_rx)
        plt.subplot(2, 2, 3)
        plt.title("Data + Noise")
        plt.plot(abs(np.fft.fft(ofdm_rx,int(1e6))))


    ## BLOCK 4 - принятие сигнала  (41-50)
    if IF_SDR:
        print("BLOCK 4")
        ofdm_rx = sdr2.rx()
        
        np.savetxt(filename_output, ofdm_rx, delimiter=":")
        if IF_SDR:
            sdr.tx_destroy_buffer()
            sdr.rx_destroy_buffer()
            sdr2.tx_destroy_buffer()
            sdr2.rx_destroy_buffer()
else:
##    (41-50)
    ofdm_rx = np.loadtxt(filename_output, delimiter=":", dtype=complex)


## BLOCK 5 - обработка сигнала (51-60)
print("BLOCK 5")

if CON == 1:

    ofdm_rx_2 = ofdm_rx
    index1 = sample.correlat_ofdm(ofdm_rx_2, N_interval, Nb)
    print("index1:", index1)
    
    arr = []
    start_data = -1
    if_start = 0.9 + 0.9j
    # print("0 :", N_interval)
    # print(N_interval, ":", Nb)
    print("0 :", N_interval)
    print(len_one_ofdm - N_interval, ":", len_one_ofdm)
    for i in range(0, len(ofdm_rx_2) - Nb):
        
        #a = np.vdot(data_read2[0:N_interval], data_read2[Nb:Nb + N_interval])
       
        a = sample.norm_corr(ofdm_rx_2[0:N_interval], 
                             ofdm_rx_2[len_one_ofdm - N_interval: len_one_ofdm])
        #a = abs(a)
        #print("a = ", abs(a.real), abs(a.imag), end = " | ")
        
        if start_data == -1 and a.real >= abs(if_start.real) and a.imag > (if_start.imag):
        #if start_data == -1 and a >= if_start:
            start_data = i

        # if start_data == -1 and -if_start <= a and a <= if_start:
        #     start_data = i
        arr.append(a)
        ofdm_rx_2 = np.roll(ofdm_rx_2, -1)
    plt.figure(51, figsize = (10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Correlation")
    plt.plot(arr)
    if(start_data == -1):
        print("не удалось определить начало")
        EXIT()
    print("Начало данных:", start_data)
    #start_data = 19
    #EXIT(1)
    data_rx = ofdm_rx[start_data:start_data + len_data_tx]
    plt.figure(52, figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("data_rx")
    plt.scatter(ofdm_rx.real, ofdm_rx.imag)

    #EXIT()
    print("OFDM demodulator")
    data_decode = sample.OFDM_demodulator(data_rx, ofdm_argv[1:], Nb, N_interval, pilot, RS, Nz)
    print("len data_decode = ", len(data_decode))
    plt.figure(52, figsize=(10, 10))
    plt.subplot(2, 2, 2)
    plt.title("data")
    plt.scatter(data_decode.real, data_decode.imag)
    plt.subplot(2, 2, 3)
    plt.title("data, No FR")
    data_decode_nfr = sample.OFDM_demodulator_NO_FR(data_rx, ofdm_argv[1:], Nb, N_interval, pilot, RS, Nz)
    print("len data_decode_nfr = ", len(data_decode_nfr))
    plt.scatter(data_decode_nfr.real, data_decode_nfr.imag)
    EXIT()


elif CON == 2:
    #Не доработано или удалить
    data_read = data_rx

    sdr2.rx_destroy_buffer()
    n = 3
    data_readF = np.fft.fft(data_read, n)
    data_read = data_readF
    data_read = np.convolve(np.ones(N2), data_read)/1000
    indexs = sample.TED_loop_filter(data_read)
    
    data_read = data_read[indexs]
    data_read1 = sample.PLL(data_read)
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

