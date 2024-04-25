#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:55:24 2024

@author: plutosdr
"""

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import max_len_seq
from scipy import fftpack

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

#import tx_rx.conf as conf
#import sample.sample as sample

def OFDM_gen(data, Nc):
    sc_matr = np.zeros((Nc, len(data)), dtype = complex)
    sd = np.zeros((1,Nc))




PI = 3.14

T = 10 ** -4 #длинна символа

#Nc = 4
df = 1/T# частотный интервал между поднесущими

Nc = 16 # количество поднесущих

ts = T/Nc #интервал дискретезации

k = 10#индекс поднесущей

t = ts * np.arange(0, Nc)

#формирование поднесущей
s = 1 / np.sqrt(T) *np.exp(1j * 2 * np.pi * k * df * t)

plt.figure(1, figsize = (10, 10))
plt.subplot(2, 2, 1)
plt.plot(t, s.real)

plt.figure(2, figsize = (10, 10))

sc_matr = np.zeros((Nc, len(t)), dtype = complex)

sd = np.zeros((1,Nc))


for k in range(Nc):
    sk_k = 1/np.sqrt(T) * np.exp(1j * 2 * np.pi * k * df * t)
    sc_matr[k, :] = sk_k


    
 
#ОДПФ - алгоритм формирования ofdm сигнала
sd = np.sign(np.random.rand(1, Nc) - 0.5) + 1j * np.sign(np.random.rand(1, Nc) - 0.5)
plt.subplot(2, 2, 1)
plt.scatter(sd.real, sd.imag)
sd = sd.reshape(Nc)

xt = np.zeros((1,len(t)), dtype=complex)
 
# формирование суммы модулированных поднесущих
for k in range(Nc):
    sc = sc_matr[k,:]
    xt=xt + sd[k] * sc
xt = xt.reshape(Nc)
plt.subplot(2, 2, 2)
plt.plot(t, xt.real)

xt = xt[-4:]

xt2 = np.fft.ifft(sd, Nc)
plt.subplot(2, 2, 3)
plt.plot(t, xt.real)

n = 3
sr = ts * np.sum(xt * np.conjugate(sc_matr[n,:]))

print("sd[0] =",sd[n], "sr =", sr)

sr2 = np.sqrt(T) / Nc * np.fft.fft(xt)

plt.subplot(2, 2, 4)
plt.scatter(sr2.real, sr2.imag)
    
#Для OFDM нужно использовать защитный интервал - Gurad interval
