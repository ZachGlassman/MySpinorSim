# -*- coding: utf-8 -*-
"""
Spyder Editor
Hamiltonian functions
"""
import numpy as np
from numba import jit, autojit

@autojit
def setup_scaled_H(q, c, n, m, nmaxfinal):
    """function to setup tridigonal Hamiltonian if first, return d,e"""
    first_n0 = np.mod(n-abs(m), 2)
    n0 = np.mod((n-abs(m)), 2)
    nmax = int((n-abs(m)-n0)/2 + 1)
 
    #create arrays
    e = np.zeros(int(nmax)-1)
    d = np.zeros(int(nmax))
  
    c_local = c/n
    
    #matrix elements of hamiltonian
    nm = (n - n0 - m)/2
    npp = (n - n0 + m)/2
    for j in range(int(nmax)):
        d[j] = (n-n0)*(q+0.5*c_local*(2*n0-1))
        if j < (nmax-1):
            e[j] = c_local*np.sqrt(nm*npp*(n0+2)*(n0+1))
        
        nm = nm - 1
        npp = npp - 1
        n0 = n0 + 2
    
    #estimate based on Gershgorin's circle theorem
    radius = abs(e[0])
    e_min = d[0] - radius
    e_max = d[0] + radius

    for j in range(2,int(nmax)-1):
        radius = abs(e[j-2]) + abs(e[j-1])
        e_min = min(e_min, d[j-1] - radius)
        e_max = max(e_max, d[j-1] + radius)
    radius = abs(e[nmax-2])
    e_min = min(e_min, d[nmax-1] - radius)
    e_max = max(e_max, d[nmax-1] + radius)
    
    radius = (e_max + e_min)/2
    
    for i in range(int(nmax)):
        d[i] = d[i] - radius
        
    radius = 2/(e_max-e_min)
    d = np.multiply(radius,d)
    e = np.multiply(radius,e)
    return e_min, e_max ,d ,e, first_n0
 
@autojit
def hamiltonian_c(n_max, in_w, e, d):
    """apply tridiagonal real Hamiltonian matrix to a complex vector
    different from Arne in that we pass in e,d"""
   
    out_w = np.zeros(len(in_w), dtype = complex)
    
    for i in range(n_max):
        out_w[i] = in_w[i]*d[i]
    
    j = 1
    for i in range(int(n_max)-1):
        out_w[i] = out_w[i] + e[i]*in_w[j]
        j = j + 1

    i = 0
    for j in range(1,int(n_max)):
        out_w[j] = out_w[j] + e[i] * in_w[i]
        i = i + 1
        
    return out_w

@autojit
def moments(wave,n):
    """take a complex wave and return mean and variance
    for each element take mag squared and compute mean
    where n is first_n0_element"""
    #epsilon  = 1e-200
    x =0
    x2 = 0
    for i in range(len(wave)):
        Y = abs(wave[i])**2
        x = x + Y * n
        x2 = x2 + Y * n * n
        n = n + 2
    return x, x2
    

    
    
    
