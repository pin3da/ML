#! /usr/bin/python

import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep=' ')

X  = m_load('x.mio')
Y  = m_load('y.mio')
Xo = m_load('xo.mio')
Yo = m_load('yo.mio')

plt.plot(X,Y)
plt.plot(Xo,Yo,'xr')
plt.hold()
plt.show()


