import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sample.sample import test1
#from tx_rx.conf import *

import tx_rx.conf as conf


test1()


conf.print_test1()

asd = conf.RxTx()

asd.print_test()


buf = [1, 2, 3]

asd.send(buf)

print(asd.recv())

asd.print_parameters()

print("End")

