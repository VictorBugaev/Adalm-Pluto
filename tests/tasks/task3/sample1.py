# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:27:25 2023

@author: Andrey
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import fftpack
 



#Ts1=0.5e-3# 1/ 2000 отсчетов
#Ts1=1e-3#1 / 1000 отсчетов
Ts1=5e-3#1 / 500 отсчетов


#freq_analog = 100 #аналоговая частота
norm_freq = 0.4 * np.pi 
freq_analog = (norm_freq * (1 / Ts1)) / (2 * np.pi)
print("analog freq = ",freq_analog, "Hz")
#t = np.arange(-20, 41)*Ts1
t = np.arange(-20, 180)*Ts1
#tR = np.arange(-20, 41) * Ts1
print("len t = ", len(t))
s =  np.cos(freq_analog*t*(2*np.pi))
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t, s )
plt.subplot(3,1,2)
plt.stem(t, s, 'r')



# Ts2=2e-3
# fs2=1/Ts2
# t = np.arange(-5, 11)*Ts2
# s =  np.cos(freq_analog *t*(2*np.pi))
# plt.subplot(3,1,3)
# plt.stem(t, s, 'g' )

#plt.show()
#plt.figure(2)
#plt.magnitude_spectrum(s,Fs=fs2,sides='twosided')

plt.figure(2)
 

sp = fftpack.fft(s)
#freqs = fftpack.fftfreq(len(s)) * fs2 
#freqs=np.arange(0,fs2,fs2/len(s))
freqs=np.arange(0,1/Ts1,(1/Ts1)/len(s))

print("len = ", len(freqs))
print("len = ", len(sp))

#fig, ax = plt.subplots()
plt.stem(freqs, np.abs(sp))
plt.xlabel('Частота в герцах [Hz]')
plt.ylabel('Модуль спектра')