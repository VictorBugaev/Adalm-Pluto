#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:59:22 2023

@author: plutosdr
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import adi 
import time

from scipy.fftpack import fft, ifft,  fftshift, ifftshift

fm = int(2400e6 + 2e6 * 4)
sdr = adi.Pluto("ip:192.168.3.1")
sdr.sample_rate = 1e6
sdr.rx_buffer_size = 10000
sdr.rx_lo = fm
sdr.tx_lo = fm
sdr.tx_cyclic_buffer = False
#sdr.rx_cyclic_buffer = True

t = np.arange(0, 1, 1/sdr.sample_rate)

fc = 50

i = np.cos(t * 2 *np.pi * fc) * 2 ** 14
q = np.sin(t * 2 *np.pi * fc) * 2 ** 14

sample = i + 1j * q  #данные для отправки


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
def DrawData(e):
    rx_data = sdr.rx()
    ax1.clear()
    plt.ylim(-2000, 2000)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("Полученные данные")
    #plt.plot( rx_data)
    plt.scatter(rx_data.real, rx_data.imag)

#Непрерывное получение сигнала и его анализ (анимированный сигнал)
ani = animation.FuncAnimation(fig, DrawData, interval=100)
plt.show()
#Отправка
#for i in range(300):
#    sdr.tx(sample)
    #time.sleep(1)
    






