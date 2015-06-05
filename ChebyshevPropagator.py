# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:43:56 2015
Code for Spinor Solution from Arne
This is will propgate a Hermition matrix with eigenvalues between -1 and 1
@author: zag
"""
import numpy as np
from scipy.special import jv
#requested accuracy

def hamiltonian_c(in_w, out_w, nmaxlocal):
    """function to compute tridiagonal Hamiltonian to a real vector"""
    pass

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
    


def chebyshev_propgator(time_step, Psi, ntot):
    epsilon = 1e-15
    chebyshev_order = int(time_step) + 5
    Y = 0.5 * time_step
    X = (Y/chebyshev_order)**chebyshev_order + np.sqrt(2*np.pi*chebyshev_order)
    while X > epsilon:
        chebyshev_order = chebyshev_order + 10
        X = (Y/chebyshev_order)**chebyshev_order + np.sqrt(2*np.pi*chebyshev_order)
        
    #now compute the Bessel function
    bessel_j, nz = dbesj(time_step,0,chebyshev_order+1)   
    if nz != 0:
        print('Bessel function has zeros in expansion')
    
    #get the order of the polynomial, find index of first element less than epsilon
    order = next(i[0] for i in enumerate(bessel_j) if i[1] < epsilon) + 1
    
    print('Number of Chebyshev polynomials', order)
    
    
