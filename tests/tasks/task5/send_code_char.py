#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:43:11 2023

@author: plutosdr
"""

#from scipy.fftpack import fft, ifft,  fftshift, ifftshift
#from scipy import signal
#from scipy.signal import kaiserord, lfilter, firwin, freqz
import numpy as np
import matplotlib.pyplot as plt
import adi 
#import time





fm = int(2000e6 + 2e6 * 4)
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = 1e6
sdr.rx_buffer_size = 1000
sdr.rx_lo = fm
sdr.tx_lo = fm

sdr.tx_cyclic_buffer = False


data = []
apl = 2**9
data += [apl + 1j * apl for i in range(0, 300)]
data += [1 + 1j*1 for i in range(0, 700)]
#print(data)
#plt.plot(data)
#plt.show()
data1 = np.array([])
len_sample = 300

sample_bit_1 = [apl + 1j * apl for i in range(0, len_sample)]
sample_bit_0 = [(apl/4) + (apl/4) * 1j for i in range(0, len_sample)]


def char_to_bit(char):
    return ''.join(format(ord(i), '08b') for i in char)

def bit_to_sample(byte_str):
    sample_code = []
    
    for i in range(len(byte_str)):
        if(byte_str[i] == '0'):
            sample_code += sample_bit_0
        elif( byte_str[i] == '1'):
            sample_code += sample_bit_1
    return sample_code


#data1 = []


string = 'w'

string_bit = char_to_bit(string)
print(f"{string} = {string_bit} ")
data = bit_to_sample(string_bit)


sr_start = 0
sr_end = len_sample
sr_step = len_sample
step_count = 0
for i in range(0, 1000):
    if (i > 100 and i % 100 == 0):
        if(step_count < len(data) / sr_step):
            print(sr_start, sr_end)
            sdr.tx(data[sr_start:sr_end])
            sr_start += sr_step
            sr_end += sr_step
            step_count += 1
            
        
    # if(i == 200):
    #     sdr.tx(data)
            
    d = sdr.rx()
    data1 = np.concatenate([data1, abs(d)])


plt.plot(data1)
plt.ylim(-3000, 3000)
plt.show()




















