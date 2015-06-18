# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:43:56 2015
Code for Spinor Solution from Arne
This is will propgate a Hermition matrix with eigenvalues between -1 and 1
@author: zag
"""
import numpy as np
from scipy.special import jv
import copy
from hamiltonian import hamiltonian_c
from numba import jit, autojit

def dbesj(x, alpha, n):
    """sequence of bessel functions where n is the number of elements of a
    bessel function of J_(alpha + k -1)(x) where k = 1...n
    """
    bessel_j = np.zeros(n+1)
    i = 0
    for k in range(1,n+1):
        bessel_j[i] = jv(alpha + k -1, x)
        i = i + 1
    y = len([num for num in bessel_j if abs(num) < 1e-20])
    return bessel_j, y
 
@autojit
def low_filter(vec):
    for i in range(len(vec)):
        if abs(vec[i]) < 1e-255:
            vec[i] = 0
        elif abs(vec[i]) > 1e155:
            vec[i] = 1e154
    return vec


def chebyshev_propagator(time_step, psi, n_tot, e, d,file):
    """propogate function"""
    epsilon = 1e-15
    #estimate upper bound with asymptotic form
    chebyshev_order = int(time_step) + 5
    Y = 0.5 * np.e* time_step
    X = (Y/chebyshev_order)**chebyshev_order/np.sqrt(2*np.pi*chebyshev_order)
    while X > epsilon:
        chebyshev_order = chebyshev_order + 10
        X = (Y/chebyshev_order)**chebyshev_order / np.sqrt(2*np.pi*chebyshev_order)
    #now compute the Bessel function
    bessel_j, nz = dbesj(time_step,0,chebyshev_order+1)
    #get the order of the polynomial, find index of first element less than epsilon
    order = next(i[0] for i in enumerate(bessel_j) if abs(i[1]) < epsilon)
    #print('Number of Chebyshev polynomials', order)
    
    psi_minus1 = copy.deepcopy(psi) #make sure to copy over
    file.write('{:<3} minus {:<25.15e}, {:<25.15e}\n'.format(0,psi_minus1[-1].real,psi_minus1[-1].imag))
    psi_0 = hamiltonian_c(n_tot,psi_minus1,e,d)
    phase = np.complex(0,-1)
    cx = 2 * bessel_j[1] * phase
    psi = np.multiply(psi,np.complex(bessel_j[0],0))
    psi = np.add(np.multiply(cx,psi_0),psi)
    file.write('{:<3} zero {:<25.15e}, {:<25.15e}\n'.format(1,psi_0[-1].real,psi_0[-1].imag))
    for i in range(2,order):
        phase = phase * np.complex(0,-1)
        cx = 2 * bessel_j[i] * phase
        file.write('{:<3} zero {:<25.15e}, {:<25.15e}\n'.format(i,psi_0[-1].real,psi_0[-1].imag))
        psi_plus1 = hamiltonian_c(n_tot,psi_0,e,d)
        file.write('{:<3} plus {:<25.15e}, {:<25.15e}\n'.format(i,psi_plus1[-1].real,psi_plus1[-1].imag))
        psi_plus1 = np.multiply(psi_plus1, np.complex(2,0))
        psi_plus1 = low_filter(psi_plus1)
        psi_plus1 = np.subtract(np.multiply(2,psi_plus1), psi_minus1)
     
        psi = np.add(psi, np.multiply(cx, psi_plus1))
        psi_minus1 = copy.deepcopy(psi_0)
        psi_0 = copy.deepcopy(psi_plus1)
        #psi_0 = copy.deepcopy(psi_minus1)
        #psi_plus1 = copy.deepcopy(psi_0)
    return psi
    
