from scipy.fftpack import fft, ifft,  fftshift, ifftshift
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import numpy as np
import matplotlib.pyplot as plt
import adi 
import time
fm = int(2000e6 + 2e6 * 4)
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = 1e6
sdr.rx_buffer_size = 1000
sdr.rx_lo = fm
sdr.tx_lo = fm

sdr.tx_cyclic_buffer = False


data = []
apl = 2**14
data += [apl + 1j * apl for i in range(0, 300)]
data += [1 + 1j*1 for i in range(0, 700)]
#print(data)
plt.plot(data)
plt.show()
data1 = np.array([])






#data1 = []
for i in range(0, 1000):
    if (i == 200):
        sdr.tx(data)
    d = sdr.rx()
    data1 = np.concatenate([data1, d])


plt.plot(data1)
plt.ylim(-3000, 3000)
plt.show()