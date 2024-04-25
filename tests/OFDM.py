#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:52:47 2024

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






PI = 3.14

T = 10 ** -4

Nc = 4
df = 10**4
fs = 4 * df

ts = 1/fs

delta_f = 1 / T

step = 0.5

t = np.arange(0, step, ts)
s = np.array([1+3j, 1-5j, -1+3j, 4+5j])
#s = np.array([1, 1, -1, 4])

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
    
    #xt[i] = dert[]
    #xt[i] += xt[i] * 1j
    
for i in range(len(s)):
    xt += dert[i]
plt.figure(1, figsize = (20, 20))
plt.subplot(2,2,1)
plt.plot(t, xt.real)
#plt.plot(t, xt.imag)


#S = (S1(t) + S2(t) ...)
res = xt * dert[0]
#plt.plot(t, res.real)
sum1 = (np.sum(res )* step * ts) / np.sqrt(step)
print("sum1 = ", sum1)
plt.subplot(2,2,2)
for i in range(len(s)):
    plt.plot(dert[i])


