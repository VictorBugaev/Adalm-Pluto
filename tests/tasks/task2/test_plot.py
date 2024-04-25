

import numpy as np
import matplotlib.pyplot as plt
import random
import time

 


n = 40
# time vector
t = np.linspace(0, 1, n, endpoint=True)
# sine wave
x = np.sin(np.pi*t) + np.sin(2*np.pi*t) + np.sin(3*np.pi*t) + np.sin(5*np.pi*t)

fig = plt.figure(figsize=(16, 5), dpi=100, layout='constrained')
axs = fig.subplot_mosaic([["analog", "analog"],
                          ["discrete", "quantum"],
                          ])

axs["analog"].plot(t, x, 'r')
axs["analog"].set_title("Аналоговый")
axs["analog"].grid(True)
axs["discrete"].stem(t, x, 'g')
axs["discrete"].set_title("Дискретный")
axs["discrete"].grid(True)
axs["quantum"].step(t, x, 'b')
axs["quantum"].set_title("Квантовый")
axs["quantum"].grid(True)

plt.show()

