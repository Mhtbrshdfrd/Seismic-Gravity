import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
import numpy as np
import matplotlib as mpl


def fillline(startp,endp,pts):
    m=(endp[1]-startp[1])/(endp[0]-stratp[0])
    if m==float('inf'):
        xx[0:pts]=startp[0]
        yy[0:pts]=np.linspace(startp[1],endp[1],pts)
    
    