#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

x_sin = arange(0, 8, 0.01)
y_sin = sin(x_sin)
plt.plot(x_sin, y_sin, 'g')

x  = m_load('x.mio')
y  = m_load('y.mio')
plt.plot(x, y, 'xb')

inval = m_load('new_y.mio')
p     = m_load('outval.mio')
plt.plot(inval, p, 'xr')

plt.hold()
plt.show()
