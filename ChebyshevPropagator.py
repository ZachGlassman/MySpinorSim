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
#requested accuracy
from hamiltonian import hamiltonian_c

def dbesj(x, alpha, n):
    """sequence of bessel functions where n is the number of elements of a 
    bessel function of J_(alpha + k -1)/x where k = 1...n
    """
    bessel_j = np.zeros(n+1)
    i = 0
    for k in range(1,n+1):
        bessel_j[i] = jv(alpha + k -1, x)
        i = i + 1
    y = sum([num for num in bessel_j if num== 0])
    return bessel_j, y
    


def chebyshev_propagator(time_step, psi, n_tot, e, d):
    """propogate function"""
    epsilon = 1e-15
    chebyshev_order = int(time_step) + 5
    Y = 0.5 * time_step
    X = (Y/chebyshev_order)**chebyshev_order/np.sqrt(2*np.pi*chebyshev_order)
    while X > epsilon:
      
        chebyshev_order = chebyshev_order + 10
        X = (Y/chebyshev_order)**chebyshev_order / np.sqrt(2*np.pi*chebyshev_order) 
    #now compute the Bessel function
    bessel_j, nz = dbesj(time_step,0,chebyshev_order+1)   
    if nz != 0:
        print('Bessel function has zeros in expansion')
    
    #get the order of the polynomial, find index of first element less than epsilon
    order = next(i[0] for i in enumerate(bessel_j) if i[1] < epsilon) + 1
    
    #print('Number of Chebyshev polynomials', order)
    psi_minus1 = copy.deepcopy(psi) #make sure to copy over
    
    psi_0 = hamiltonian_c(n_tot,psi_minus1,e,d)
    phase = np.complex(0,-1)
    cx = 2 * bessel_j[0] * phase
    psi = psi_minus1 * np.complex(bessel_j[0],0)
    psi = bessel_j[0]*psi_minus1+cx*psi_0
    for i in range(1,order):
        phase = phase * np.complex(0,-1)
        cx = 2 * bessel_j[i] * phase
        psi_plus1 = hamiltonian_c(n_tot,psi_0,e,d)
        psi_plus1 = psi_plus1 * np.complex(2,0)
        psi_plus1 = 2 * psi_plus1 - psi_minus1
        psi = psi + cx * psi_plus1
        psi_minus1 = copy.deepcopy(psi_0)
        psi_0 = copy.deepcopy(psi_plus1)
        
    return psi
