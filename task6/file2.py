#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:24 2023

@author: plutosdr
"""

import numpy as np
import matplotlib.pyplot as plt
import adi 


fm = int(2400e6 + 2e6 * 4)
sdr = adi.Pluto("ip:192.168.3.1")
sdr.sample_rate = 1e6
sdr.rx_buffer_size = 1000000
sdr.rx_lo = fm
sdr.tx_lo = fm


#t = [ i for i in range(100)]
t = np.arange(0, 1, 1/sdr.sample_rate)

fc = 50
#t = np.arange(0, fc, 1/fc)

i = np.sin(t * 2 *np.pi * fc) * 2 ** 14
q = np.sin(t * 2 *np.pi * fc) * 2 ** 14

sample = i + 1j * q  #данные для отправки

plt.figure(1, figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.xlabel("time")
plt.ylabel("amplitude")
plt.title("Отправленные данные")
plt.plot(sample)

sdr.tx(sample)#отправка данных
data = sdr.rx()#данные которые приняли
plt.subplot(2, 2, 2)
plt.ylim(-3000, 3000)
plt.xlabel("time")
plt.ylabel("amplitude")
plt.title("Принятые данные")

plt.plot(data)






