#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep=' ')

X = m_load('x.mio')
Y = m_load('y.mio')
plt.plot(X,Y, 'rx')
plt.hold()
plt.show()
