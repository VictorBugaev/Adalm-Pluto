import time

import adi
import matplotlib.pyplot as plt
import numpy as np


# Create radio
sdr = adi.Pluto("ip:192.168.2.1")

# Configure properties
sdr.rx_lo = 2412000000
 
# Collect data
for r in range(30):
    rx = sdr.rx()
    plt.clf()
    
    wait = 0

    for i in rx.imag:
        if i > 0:
            wait += i
    wait /= len(rx.imag)
    plt.xlabel("time")
    plt.ylabel("ampl " + str(wait) )
    plt.plot(rx.real)
    plt.plot(rx.imag)
    
    plt.draw()

    plt.pause(0.05)


    if 200 < wait:
        time.sleep(3) 
            
        
    
    time.sleep(0.5)
plt.show()