# -*- coding: utf-8 -*-
"""
Created on originally on Tue Oct 25 10:27:43 2016

@author: lesur
subsequently modified by: fournier

Given a set of L(L+2) Gauss coefficients i.e. g10, g11, h11, g20, g21, h21,...
Computes the X (North), Y (East), Z (Down) components of the magnetic field
for fields of internal and external origins.

Parameters
----------
li  : number
    Maximum degree in SH expansion for internal fields
le  : number
    Maximum degree in SH expansion for external fields
np.pi  : number (good old pi)
"""

import numpy as np

# Maximum SH degree of expansion
li = 8  # for the internal field
le = 1  # for the external field 
# Reference radius
#ra = 6371.2 # mean radius of Earth in km

def mk_cnd(tt, pp, rr):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the three
    #  components of the magnetic field generated inside and outside the
    #  sphere of radius ra at a point (theta,phi,r)
    #
    # ATTENTION : dd = 0. if abs(dd) < 1.e-14
    #
    # CALLED: SHB_Xi, SHB_Yi, SHB_Zi, SHB_Xe, SHB_Ye, SHB_Ze
    # DEFINED:  np.pi 3.14159...
    #           ra reference radius
    #
    try:
        aax = SHB_Xi(tt, pp, rr)
        aax = np.hstack((aax, SHB_Xe(tt, pp, rr)))
        aay = SHB_Yi(tt, pp, rr)
        aay = np.hstack((aay, SHB_Ye(tt, pp, rr)))
        aaz = SHB_Zi(tt, pp, rr)
        aaz = np.hstack((aaz, SHB_Ze(tt, pp, rr)))
        aa = np.vstack((aax, aay, aaz))
        aa[abs(aa) < 1.e-14] = 0.
    #
        return aa
    except Exception as msg:
        print('**** ERROR:')
        print(msg)
    else:
        print('mk_cnd: completed')


def WW(nd, eps):
    #
    #   nd (number data), eps (variance data)
    #
    #   Computes Data Weight Matrix (Inverse of Ce)
    #   data units in nT (minimum acceptable sigma 1 pT)
    #
    WW = (1./eps) * (1./eps) * np.eye(nd, nd)
    return WW


def DD(nm, rr, ra=6371.2):
    #
    #   nm (number parameters), rr (radius application constraint)
    #   Computes Damnp.ping Matrix (Inverse of C_m)
    #
    dd = np.zeros((nm), dtype=float)
    rw = ra/rr
    # im = 0
    for il in range(1, li+1, 1):
        k = il*il - 1
        dd[k] = float(il+1)*np.power(rw, float(2*il+4))
    #  im != 0
    for im in range(1, li+1, 1):
        for il in range(im, li+1, 1):
            k = il*il+2*im-2
            dd[k] = float(il+1)*np.power(rw, float(2*il+4))
            dd[k+1] = float(il+1)*np.power(rw, float(2*il+4))
    dd = np.diag(dd)
    return dd


def SHB_Xi(tt, pp, rr, ra=6371.2):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  North component of the magnetic field generated inside the sphere of
    #  radius ra at a point (theta,phi,r)
    #
    # CALLED: mk_dlf
    # DEFINED:  li maximum degreee of SH expansion
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        aax = np.zeros(li*(li+2), dtype=float)
        rw = ra/rr
        # im = 0
        im = 0
        plm = mk_dlf(im, li, tt)
        for il in range(1, li+1, 1):
            k = il*il - 1
            aax[k] = plm[il]*np.power(rw, float(il+2))
        #  im != 0
        for im in range(1, li+1, 1):
            plm = mk_dlf(im, li, tt)
            dc = np.cos(float(im)*pp*dtr)
            ds = np.sin(float(im)*pp*dtr)
            for il in range(im, li+1, 1):
                k = il*il+2*im-2
                ww = plm[il]*np.power(rw, float(il+2))
                aax[k] = ww*dc
                aax[k+1] = ww*ds
        return aax
    except Exception as msg:
        print('**** ERROR:SHB_Xi')
        print(msg)


def SHB_Yi(tt, pp, rr, ra=6371.2):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  East component of the magnetic field generated inside the sphere of
    #  radius ra at a point (theta,phi,r)
    #
    # ATTENTION : for theta=0 then sin(theta) set to 1.e-10
    #
    # CALLED: mk_lf
    # DEFINED:  li maximum degreee of SH expansion
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        st = np.sin(tt*dtr)
        if st == 0:
            st = 1.e-10
        aay = np.zeros(li*(li+2), dtype=float)
        rw = ra/rr
        #  im != 0
        for im in range(1, li+1, 1):
            plm = mk_lf(im, li, tt)
            dc = -float(im)*np.sin(float(im)*pp*dtr)
            ds = float(im)*np.cos(float(im)*pp*dtr)
            for il in range(im, li+1, 1):
                k = il*il+2*im-2
                ww = plm[il]*np.power(rw, float(il+2))/st
                aay[k] = -ww*dc
                aay[k+1] = -ww*ds
        return aay
    except Exception as msg:
        print('**** ERROR:SHB_Yi')
        print(msg)


def SHB_Zi(tt, pp, rr, ra=6371.2):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  Down component of the magnetic field generated inside the sphere of
    #  radius ra at a point (theta,phi,r)
    #
    # CALLED: mk_lf
    # DEFINED:  li maximum degreee of SH expension
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        aaz = np.zeros(li*(li+2), dtype=float)
        rw = ra/rr
        # im = 0
        im = 0
        plm = mk_lf(im, li, tt)
        for il in range(1, li+1, 1):
            k = il*il - 1
            aaz[k] = -float(il+1)*plm[il]*np.power(rw, float(il+2))
        #  im != 0
        for im in range(1, li+1, 1):
            plm = mk_lf(im, li, tt)
            dc = np.cos(float(im)*pp*dtr)
            ds = np.sin(float(im)*pp*dtr)
            for il in range(im, li+1, 1):
                k = il*il+2*im-2
                ww = -float(il+1)*plm[il]*np.power(rw, float(il+2))
                aaz[k] = ww*dc
                aaz[k+1] = ww*ds
        return aaz
    except Exception as msg:
        print('**** ERROR:SHB_Zi')
        print(msg)


def SHB_Xe(tt, pp, rr, ra=6371.2):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  North component of the magnetic field generated ouside the sphere
    #  of radius ra, at a point (theta,phi,r)
    #
    # CALLED: mk_dlf
    # DEFINED:  le maximum degreee of SH expension
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        aax = np.zeros(le*(le+2), dtype=float)
        rw = rr/ra
        # im = 0
        im = 0
        plm = mk_dlf(im, le, tt)
        for il in range(1, le+1, 1):
            k = il*il - 1
            aax[k] = plm[il]*np.power(rw, float(il-1))
        #  im != 0
        for im in range(1, le+1, 1):
            plm = mk_dlf(im, le, tt)
            dc = np.cos(float(im)*pp*dtr)
            ds = np.sin(float(im)*pp*dtr)
            for il in range(im, le+1, 1):
                k = il*il+2*im-2
                ww = plm[il]*np.power(rw, float(il-1))
                aax[k] = ww*dc
                aax[k+1] = ww*ds
        return aax
    except Exception as msg:
        print('**** ERROR:SHB_Xe')
        print(msg)


def SHB_Ye(tt, pp, rr, ra=6371.2):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  East component of the magnetic field generated ouside the sphere
    #  of radius ra, at a point (theta,phi,r)
    #
    # ATTENTION : for theta=0 then sin(theta) set to 1.e-10
    #
    # CALLED: mk_lf
    # DEFINED:  le maximum degreee of SH expension
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        st = np.sin(tt*dtr)
        if st == 0:
            st = 1.e-10
        aay = np.zeros(le*(le+2), dtype=float)
        rw = rr/ra
        #  im != 0
        for im in range(1, le+1, 1):
            plm = mk_lf(im, le, tt)
            dc = -float(im)*np.sin(float(im)*pp*dtr)
            ds = float(im)*np.cos(float(im)*pp*dtr)
            for il in range(im, le+1, 1):
                k = il*il+2*im-2
                ww = plm[il]*np.power(rw, float(il-1))/st
                aay[k] = -ww*dc
                aay[k+1] = -ww*ds
        return aay
    except Exception as msg:
        print('**** ERROR:SHB_Ye')
        print(msg)


def SHB_Ze(tt, pp, rr, ra=6371.2):
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  Down component of the magnetic field generated ouside the sphere
    #  of radius ra, at a point (theta,phi,r)
    #
    # CALLED: mk_lf
    # DEFINED:  le maximum degreee of SH expension
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        aaz = np.zeros(le*(le+2), dtype=float)
        rw = rr/ra
        # im = 0
        im = 0
        plm = mk_lf(im, le, tt)
        for il in range(1, le+1, 1):
            k = il*il - 1
            aaz[k] = float(il)*plm[il]*np.power(rw, float(il-1))
        #  im != 0
        for im in range(1, le+1, 1):
            plm = mk_lf(im, le, tt)
            dc = np.cos(float(im)*pp*dtr)
            ds = np.sin(float(im)*pp*dtr)
            for il in range(im, le+1, 1):
                k = il*il+2*im-2
                ww = float(il)*plm[il]*np.power(rw, float(il-1))
                aaz[k] = ww*dc
                aaz[k+1] = ww*ds
        return aaz
    except Exception as msg:
        print('**** ERROR:SHB_Ze')
        print(msg)


def mk_dlf(im, ll, tt):
    #
    # im integer, tt float, ll maximum degreee of SH expension
    #
    # Computes the derivative along theta of all legendre functions for
    # a given order up to degree ll at a given position : theta (in degree)
    #
    # CALLED: mk_lf
    # DEFINED:  np.pi 3.14159...
    #
    dlf = np.zeros(ll+1, dtype=float)
    #
    dtr = np.pi / 180.
    dc = np.cos(tt*dtr)
    ds = np.sin(tt*dtr)
    if ds == 0.:
        if im == 1:
            dlf[1] = -1.
            dlf[2] = -np.sqrt(3.)
            for ii in range(3, ll+1, 1):
                d1 = float(2*ii-1)/np.sqrt(float(ii*ii-1))
                d2 = np.sqrt(float(ii*(ii-2))/float(ii*ii-1))
                dlf[ii] = d1*dlf[ii-1]-d2*dlf[ii-2]
    else:
        dlf = mk_lf(im, ll, tt)
        for ii in range(ll, im, -1):
            d1 = np.sqrt(float((ii-im)*(ii+im)))
            d2 = float(ii)
            dlf[ii] = (d2*dc*dlf[ii]-d1*dlf[ii-1])/ds
        dlf[im] = float(im)*dc*dlf[im]/ds
    #
    return dlf


def mk_lf(im, ll, tt):
    #
    # im integer, tt float, ll maximum degreee of SH expension
    #
    # Computes all legendre functions for a given order up to degree ll
    # at a given position : theta (in degree)
    # recurrence 3.7.28 Fundations of geomagetism, Backus 1996
    #
    # CALLED: logfac
    # DEFINED:  np.pi 3.14159...
    #
    lf = np.zeros(ll+1, dtype=float)
    #
    dtr = np.pi / 180.
    dc = np.cos(tt*dtr)
    ds = np.sin(tt*dtr)
    #
    dm = float(im)
    d1 = logfac(2.*dm)
    d2 = logfac(dm)
    #
    d2 = 0.5*d1 - d2
    d2 = np.exp(d2 - dm*np.log(2.0))
    if im != 0:
        d2 = d2*np.sqrt(2.0)
    #
    d1 = ds
    if d1 != 0.:
        d1 = np.power(d1, im)
    elif im == 0:
        d1 = 1.
    #
    # p(m,m), p(m+1,m)
    lf[im] = d1*d2
    if im != ll:
        lf[im+1] = lf[im]*dc*np.sqrt(2.*dm+1.)
    #
    # p(m+2,m), p(m+3,m).....
    for ii in range(2, ll-im+1, 1):
        d1 = float((ii-1)*(ii+2*im-1))
        d2 = float((ii)*(ii+2*im))
        db = np.sqrt(d1/d2)
        d1 = float(2*(ii+im)-1)
        da = d1/np.sqrt(d2)
        #
        lf[im+ii] = da*lf[im+ii-1]*dc-db*lf[im+ii-2]
    #
    return lf


def logfac(dd):
    #
    # Calculate log(dd!)
    # dd is a float, but is truncated to an integer before calculation
    #
    id = int(dd)
    lgfc = np.sum(np.log(range(1, id+1, 1)))
    return lgfc
