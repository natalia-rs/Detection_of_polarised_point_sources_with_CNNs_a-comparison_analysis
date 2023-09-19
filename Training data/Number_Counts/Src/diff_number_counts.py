#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:22:37 2018

@author: herranz
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
os.chdir('.')
from utils           import interpolate_between_arrays,positions_around,fwhm2sigma
from astropy.table   import Table,Column
from astropy.io      import ascii
from scipy.integrate import simps


fdata = '../Data/'

# -------------------------------------------------------------------
###         DIFFERENTIAL COUNTS FROM INPUT
# -------------------------------------------------------------------

def make_diff_counts(area,fluxarray,lim1,lim2,nbin,equalsize=False):

    if equalsize:
        nxbin = fluxarray.size//nbin
        lims  = np.zeros(nbin+1)
        S     = fluxarray.copy()
        S.sort()
        for i in range(nbin):
            lims[i]   = np.log10(S[i*nxbin])
        lims[nbin] = np.log10(S.max())
        logS = (lims[1:]+lims[:nbin])/2
        
    else:
        lims = np.linspace(np.log10(lim1),np.log10(lim2),nbin+1)
        logS = (lims[1:]+lims[:nbin])/2
        dS   = lims[1:]-lims[:nbin]
    l10S = np.log10(fluxarray)

    ct   = []
    ut   = []
    lt   = []

    for i in range(nbin):
        mask1 = l10S>=lims[i]
        mask1 = mask1 & (l10S<lims[i+1])
        n     = np.count_nonzero(mask1)
        s     = np.sqrt(1.0*n)
        n     = n/(area*dS[i])
        s     = s/(area*dS[i])
        if n>0:
            ct.append(np.log10(n))
            ut.append(np.log10(n+s))
        else:
            ct.append(-1)
            ut.append(-1)
        l = np.log10(n-s)
        if (np.isreal(l) & np.isfinite(l)):
            l=l
        else:
            l=-1
        lt.append(l)

    ct = np.array(ct)
    ut = np.array(ut)
    lt = np.array(lt)
    yerr = [ct-lt,ut-ct]

    return logS,ct,yerr

# -------------------------------------------------------------------
###         RADIO SOURCE MODELS
# -------------------------------------------------------------------

freqs_dezotti = [8.4,11,15,20,30]
cols_dezotti  = ['log(S) [Jy]','flat-spectrum','steep-spectrum','total']
unit_dezotti  = '[log(S^{5/2} (dN/dS)), S in Jy]'
fnam_dezotti  = [fdata+'cr{0}dezotti.dat'.format(x) for x in freqs_dezotti]
fnam_dezotti[4] = fdata+'cr30dezotti_pol.dat'

def dezotti_reader(i):
    if i<4:
        t = ascii.read(fnam_dezotti[i],comment='\s*%')
        for k in range(4):
            t.rename_column('col{0}'.format(k+1),cols_dezotti[k])
    elif i==4:
        t = ascii.read(fnam_dezotti[i],comment='\s*#')
        for k in range(4):
            t.rename_column('col{0}'.format(k+1),cols_dezotti[k])
        t.rename_column('col5','total pol')
    else:
        print(' --- Warning: not valid table index in DEZOTTI_READER ')
        t = []
    return t

def interpola_dezotti(freq):
    nu = np.array(freqs_dezotti)
    if freq in freqs_dezotti:
        i = int(np.where(nu==freq)[0])
        t = dezotti_reader(i)
    else:
        i1,i2 = positions_around(freq,nu)
        t1  = dezotti_reader(i1)
        t2  = dezotti_reader(i2)
        z   = [nu[i1],nu[i2]]
        x   = [t1['log(S) [Jy]'],t2['log(S) [Jy]']]
        for ncol in range(1,len(cols_dezotti)):
            col   = cols_dezotti[ncol]
            y     = [t1[col],t2[col]]
            xa,ya = interpolate_between_arrays(freq,z,x,y)
            if ncol==1:
                col1 = Column(name='log(S) [Jy]',data=xa)
                t    = Table([col1])
            t.add_column(Column(name=col,data=ya))
    return t

def transform_dezotti_to_logdNdS(t):
    tabla = t.copy()
    logS = tabla['log(S) [Jy]']
    S    = 10**logS
    fact = np.power(S,5/2)
    for col in cols_dezotti[1:]:
        logx = tabla[col]
        x    = 10**logx
        y    = x/fact
        tabla[col] = np.log10(y)
    return tabla

def transform_dezotti_to_logdNdlogS(t):
    tabla = t.copy()
    logS = tabla['log(S) [Jy]']
    S    = 10**logS
    fact = np.power(S,5/2)
    for col in cols_dezotti[1:]:
        logx = tabla[col]
        x    = 10**logx
        y    = x/fact
        z    = y*S*np.log(10)
        tabla[col] = np.log10(z)
    return tabla


freqs_C2Ex    = [60,69,150,220,353,550,900]
cols_C2Ex     = ['flux density [Jy]','total','steep','flat',
                 'inverted','FSRQ','BLLac']
unit_C2Ex     = '[Jy^-1 sr^-1]'
fnam_C2Ex     = [fdata+'ns{0}_C2Ex.dat'.format(x) for x in freqs_C2Ex]

def C2Ex_reader(i):
    t = Table.read(fnam_C2Ex[i],format='ascii')
    for k in range(7):
        t.rename_column('col{0}'.format(k+1),cols_C2Ex[k])
    return t

def interpola_C2Ex(freq):
    nu = np.array(freqs_C2Ex)
    if freq in freqs_C2Ex:
        i = int(np.where(nu==freq)[0])
        t = C2Ex_reader(i)
    else:
        i1,i2 = positions_around(freq,nu)
        t1  = C2Ex_reader(i1)
        t2  = C2Ex_reader(i2)
        z   = [nu[i1],nu[i2]]
        x   = [np.log10(t1['flux density [Jy]']),
                        np.log10(t2['flux density [Jy]'])]
        for ncol in range(1,len(cols_C2Ex)):
            col   = cols_C2Ex[ncol]
            y     = [np.log10(t1[col]+1.e-8),np.log10(t2[col]+1.e-8)]
            xa,ya = interpolate_between_arrays(freq,z,x,y)
            if ncol==1:
                col1 = Column(name='flux density [Jy]',data=10**xa)
                t    = Table([col1])
            t.add_column(Column(name=col,data=10**ya))
    return t


def transform_C2Ex_to_logdNdlogS(t):
    tabla = t.copy()
    S    = tabla['flux density [Jy]']
    fact = 1.0
    for col in cols_C2Ex[1:]:
        x    = tabla[col]
        y    = x/fact
        z    = y*S*np.log(10)
        tabla[col] = np.log10(z)
    tabla['flux density [Jy]'] = np.log10(tabla['flux density [Jy]'])
    tabla.rename_column('flux density [Jy]','log(S) [Jy]')
    return tabla


def logdNdlogS_total(freq):

    if freq < np.min(freqs_dezotti):
        print(' --- Warning: frequency below the de Zotti models limit')

    elif freq <= np.max(freqs_dezotti):
        t0 = interpola_dezotti(freq)
        t1 = transform_dezotti_to_logdNdlogS(t0)
        logS       = t1['log(S) [Jy]']
        logdNdlogS = t1['total']

    elif freq < np.min(freqs_C2Ex):

        f1 = np.max(freqs_dezotti)
        t0 = interpola_dezotti(f1)
        t1 = transform_dezotti_to_logdNdlogS(t0)
        logSa       = t1['log(S) [Jy]']
        logdNdlogSa = t1['total']

        f2 = np.min(freqs_C2Ex)
        t0 = interpola_C2Ex(f2)
        t1 = transform_C2Ex_to_logdNdlogS(t0)
        logSb       = t1['log(S) [Jy]']
        logdNdlogSb = t1['total']

        z = [f1,f2]
        x = [logSa,logSb]
        y = [logdNdlogSa,logdNdlogSb]

        logS,logdNdlogS = interpolate_between_arrays(freq,z,x,y)

    elif freq <= np.max(freqs_C2Ex):
        t0 = interpola_C2Ex(freq)
        t1 = transform_C2Ex_to_logdNdlogS(t0)
        logS       = t1['log(S) [Jy]']
        logdNdlogS = t1['total']

    else:
        print(' --- Warning: frequency above the de C2Ex models limit')

    return logS,logdNdlogS


def convert_logdNdlogS_to_dNdS(logS,logdNdlogS):
    dNdlogS = np.power(10,logdNdlogS)
    S       = np.power(10,logS)
    dNdS    = dNdlogS/(np.log(10)*S)
    return S,dNdS

def dNdS_total(freq):
    logS,logdNdlogS = logdNdlogS_total(freq)
    S,dNdS          = convert_logdNdlogS_to_dNdS(logS,logdNdlogS)
    return S,dNdS


def plot_euclidean(freq,newplot=True,c=1):
    S,dNdS = dNdS_total(freq)
    if newplot:
        plt.figure()
    plt.plot(np.log10(S),np.log10(c*np.power(S,5/2)*dNdS))
    plt.xlabel('log$_{10}$S [Jy]')
    plt.ylabel(r'log$_{10}($S$^{5/2}$ N(S)) [Jy$^{3/2}$ sr$^{-1}$]')
    plt.title('{0} GHz'.format(freq))
    plt.grid()


def number_above(Slim,freq,area=4*np.pi*u.sr):
    S,dNdS = dNdS_total(freq)
    a,b = positions_around(Slim,S)
    x = S[a:]
    y = dNdS[a:]
    r = simps(y,x=x,even='first')/u.sr
    return (r*area).si


# -------------------------------------------------------------------
###        POWER SPECTRUM OF POINT SOURCES
# -------------------------------------------------------------------


def power_level(Smax,freq):
    S,dNdS = dNdS_total(freq)
    a,b = positions_around(Smax,S)
    x = S[:b+1]
    y = dNdS[:b+1]
    y = x*x*y
    return simps(y,x=x,even='last')*u.Jy*u.Jy/(u.sr)

def plot_Cls_MJysr(Smax,freq):
    powlev = power_level(Smax,freq)/(u.sr)
    powlev = powlev.to(u.MJy*u.MJy/(u.sr*u.sr))
    l      = np.arange(0,2000)
    y      = l*(l+1)*powlev/(2*np.pi)
    y      = np.sqrt(y)
    y      = y.value
    plt.loglog(l,y,label='{0} Jy'.format(Smax))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\sqrt{\ell (\ell+1) C_{\ell}/2\pi}$  [MJy/sr]')
    plt.grid()
    plt.title('{0} GHz'.format(freq))

def plot_Cls_uK(Smax,freq,polfrac=1,squared=False):
    powlev = power_level(Smax,freq)/(u.sr)
    powlev = powlev.to(u.MJy*u.MJy/(u.sr*u.sr))
    l      = np.arange(0,2000)
    y      = l*(l+1)*powlev/(2*np.pi)
    if polfrac<1:
        y = y/2
    y      = np.sqrt(y)
    y      = y.to(u.uK,equivalencies=u.brightness_temperature(freq*u.GHz))
    y      = y.value
    y      = y*polfrac
    if not squared:
        y  = y*y
    plt.loglog(l,y,label='{0} Jy'.format(Smax))
    plt.xlabel(r'$\ell$')
    if squared:
        plt.ylabel(r'$\sqrt{\ell (\ell+1) C_{\ell}/2\pi}$  [$\mu$K]')
    else:
        plt.ylabel(r'$\ell (\ell+1) C_{\ell}/2\pi$  [$\mu$K$^2$]')
    plt.grid()
    plt.title('{0} GHz'.format(freq))
    plt.axis([2,1500,2.e-7,100])


def noise_mJy(freq,fwhm,Smax):
    Cl = power_level(Smax,freq)
    s2 = np.pi*(fwhm2sigma*fwhm)**2 * Cl
    s  = np.sqrt(s2)
    s  = s.to(u.mJy)
    return s


def equivalent_polarized_noise_power(freq,Smax,polfrac=1):

    C      = power_level(Smax,freq)/(u.sr)
    if polfrac < 1:
        C = C/2  # the 1/2 factor comes from the uniform distribution of
                 # polarization angles, see Tucci et al. (2003)
        C = C*(polfrac**2)
    sC = np.sqrt(C)
    sC = sC.to(u.uK,equivalencies=u.brightness_temperature(freq*u.GHz))

    return 10800*sC*u.arcmin/np.pi



# -------------------------------------------------------------------
###         TESTS
# -------------------------------------------------------------------


def tests(polfrac=0.0283):

    plt.close('all')

    plt.figure(figsize=(16,16))

    plt.subplot(221)
    freq = 30
    plot_Cls_uK(0.1/polfrac,freq,polfrac=polfrac)
    plot_Cls_uK(0.005/polfrac,freq,polfrac=polfrac)
    plt.title('{0} GHz'.format(freq))
    plt.grid()

    plt.subplot(222)
    freq = 70
    plot_Cls_uK(0.1/polfrac,freq,polfrac=polfrac)
    plot_Cls_uK(0.005/polfrac,freq,polfrac=polfrac)
    plt.title('{0} GHz'.format(freq))
    plt.grid()

    plt.subplot(223)
    freq = 100
    plot_Cls_uK(0.1/polfrac,freq,polfrac=polfrac)
    plot_Cls_uK(0.005/polfrac,freq,polfrac=polfrac)
    plt.title('{0} GHz'.format(freq))
    plt.grid()

    plt.subplot(224)
    freq = 217
    plot_Cls_uK(0.1/polfrac,freq,polfrac=polfrac)
    plot_Cls_uK(0.005/polfrac,freq,polfrac=polfrac)
    plt.title('{0} GHz'.format(freq))
    plt.grid()


def test_plot():
    plt.close('all')
    freqses = [20,25,30,35,40,45,50,55,60,65,70,75,100,200]
    for f in freqses:
        x,y = logdNdlogS_total(f)
        plt.plot(x,y,label='{0} GHz'.format(f))
    plt.xlabel('Log(s) [Jy]')
    plt.ylabel('log(N/logS)')
    plt.legend()

def test_plot2():
    plt.close('all')
    freqses = [10,20,25,30,35,40,45,50,55,60,65,70,75,100,200]
    for f in freqses:
        x,y = dNdS_total(f)
        plt.loglog(x,y,label='{0} GHz'.format(f))
    plt.xlabel('S [Jy]')
    plt.ylabel(r'dN/dS [sr$^{-1}$]')
    plt.legend()