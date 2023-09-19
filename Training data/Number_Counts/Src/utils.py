#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:08:07 2017

@author: herranz
"""

import numpy as np
from scipy.interpolate import interp1d

fwhm2sigma   = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))
sigma2fwhm   = 1.0/fwhm2sigma

def interpolate_between_arrays(z0,z,x,y):

    """
    Interpola entre dos pares de arrays
          Z1:  {x1[0],x1[1],...,x1[n]}, {y1[0],y1[1],...,y1[n]}
          Z2:  {x2[0],x2[1],...,x2[m]}, {y2[0],y2[1],...,y2[m]}
    para un punto intermedio Z0

a    """

    xmin = np.max((x[0].min(),x[1].min()))
    xmax = np.min((x[0].max(),x[1].max()))
    nx1  = np.count_nonzero((x[0]>=xmin)&(x[0]<=xmax))
    nx2  = np.count_nonzero((x[1]>=xmin)&(x[1]<=xmax))
    nx   = np.max((nx1,nx2))

    xout  = np.linspace(xmin,xmax,nx)

    f1   = interp1d(x[0],y[0])
    f2   = interp1d(x[1],y[1])

    y1   = f1(xout)
    y2   = f2(xout)

    dz   = z[1]-z[0]
    dy   = y2-y1

    yout = y1+(z0-z[0])*dy/dz

    return xout,yout

def positions_around(x,array):
    snu = array-x
    m1  = snu<0
    m2  = snu>0
    i1  = int(np.where(snu==snu[m1].max())[0])
    i2  = int(np.where(snu==snu[m2].min())[0])
    return i1,i2

