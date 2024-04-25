import os
import sys

import numpy as np
from scipy import signal
from scipy.signal import max_len_seq
from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import matplotlib.pyplot as plt
import subprocess
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import tx_rx.conf as conf

import sample.sample as sample

PLATFORM = "Linux"

rf_module = conf.RxTx()
rf_module.print_parameters()


dir_gen_packet = "../src/generate_packet/"
argvs = ['data.txt', 'gold_sequence.txt', 'gold_seq_end.txt', 'data_bin.txt' ]

for i in range(len(argvs)):
    argvs[i] = dir_gen_packet + argvs[i]

module_gen_header = "generate_packet"
if(PLATFORM == "Win"):
    module_gen_header += ".exe"

    
data = "Test dataadfa fasdfasdf"

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

plt.figure(1, figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Data")
plt.plot(data_bin)



N = 10
data_rep = sample.duplication_sample(data_bin, N)
#data_qpsk = sample.encode_QPSK(data_rep, 4)
#data_rep = data_bin
data_rep = data_bin
data_qpsk = np.array(sample.encode_QAM(data_rep, 16))

plt.subplot(2, 2, 2)
plt.title("Data duplicate")
plt.plot(data_rep)

plt.subplot(2, 2, 3)
plt.title("BPSK")
plt.scatter(data_qpsk.real, data_qpsk.imag)


h1 = np.ones(N)
# Noise
pos_read = 0
noise_coef = 0.0001
data_noise = data_qpsk[0:] 
noise = np.random.normal(0, noise_coef, len(data_noise))
#data_noise += noise

plt.subplot(2, 2, 4)
plt.title(f"BPSK + noise({noise_coef})")
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
    



#gardner_TED(data_conv)
#rrc_filter(N, data_conv.real)







