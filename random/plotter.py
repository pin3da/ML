#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

X = m_load('x.mio')
for i in xrange(1, 25):
    Y = m_load('y' + str(i) + '.mio')
    plt.plot(X,Y, 'rx')

plt.hold()
plt.show()
