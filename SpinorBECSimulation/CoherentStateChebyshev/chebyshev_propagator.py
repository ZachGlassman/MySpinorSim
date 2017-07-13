# -*- coding: utf-8 -*-
"""
@author: zag
"""
import numpy as np
from scipy.special import jv

#we will try to import with numba, if not fall back on numpy versions
try:
    import numba
    from .chebyshev_functions_numba import hamiltonian_c
except ImportError:
    from .chebyshev_functions import hamiltonian_c

def dbesj(x, alpha, n):
    """sequence of bessel functions where n is the number of elements of a bessel function of
    
    .. math::
        J_{\\alpha + k -1}(x)
    
    where k = 1...n
    
    Parameters
    ----------
    x : np.array(float)
        array of numbers
    alpha : int
        order of function
    n : int
        class of Bessel Function
    Returns
    -------
    bessel_j : np.array(float)
        array of bessel functions
    """
    k = np.arange(0,n+1,1) + alpha
    bessel_j = jv(k,x)
    #y = len([num for num in bessel_j if abs(num) < 1e-20])
    return bessel_j

def chebyshev_propagator(time_step, psi, n_tot, e, d):
    """propogate function with Chebyshev Propgation
    
    Parameters
    ----------
    time_step : float
        simulation time step
    psi : np.array(complex)
        wavefunction
    n_tot : int
        total number of atoms
    d : np.array(complex)
        diagonal elements of Hamiltonian
    e : np.array(complex)
        off diagonal elements of Hamiltonian
    
    Returns
    -------
    psi : np.array(complex)
        propogated wavefunction
    """
    epsilon = 1e-15
    #estimate upper bound with asymptotic form
    chebyshev_order = int(time_step) + 5
    Y = 0.5 * np.e* time_step
    X = (Y/chebyshev_order)**chebyshev_order/np.sqrt(2*np.pi*chebyshev_order)
    while X > epsilon:
        chebyshev_order = chebyshev_order + 10
        X = (Y/chebyshev_order)**chebyshev_order / np.sqrt(2*np.pi*chebyshev_order)

    #now compute the Bessel function
    bessel_j = dbesj(time_step,0,chebyshev_order+1)
    #get the order of the polynomial, find index of first element less than epsilon
    for i in range(chebyshev_order-1,0,-1):
        if bessel_j[i] >= epsilon:
            break
    order = i + 2
    #print('Number of Chebyshev polynomials', order)

    psi_minus1 = psi #make sure to copy over
    psi_0 = hamiltonian_c(n_tot,psi_minus1,e,d)
    phase = np.complex(0,-1)
    cx = 2 * bessel_j[1] * phase
    psi = np.multiply(psi,np.complex(bessel_j[0],0))
    psi = np.add(np.multiply(cx,psi_0),psi)
    for i in range(2,order):
        phase = phase * np.complex(0,-1)
        cx = 2 * bessel_j[i] * phase
        psi_plus1 = hamiltonian_c(n_tot,psi_0,e,d)
        psi_plus1 = np.subtract(np.multiply(psi_plus1, np.complex(2,0)), psi_minus1)
        psi = np.add(psi, np.multiply(cx, psi_plus1))
        psi_minus1 = psi_0
        psi_0 = psi_plus1

    return psi
