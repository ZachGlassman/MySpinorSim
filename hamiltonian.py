# -*- coding: utf-8 -*-
"""
Spyder Editor
Hamiltonian functions
"""
import numpy as np

def setup_scaled_H(q,c, total_atom_number, magnetization, nmaxfinal):
    """function to setup tridigonal Hamiltonian"""
    n = total_atom_number
    m = magnetization
    
    n0 = (n-abs(m))%2
    nmax = (n-abs(m)-n0)/2 + 1
    nmaxfinal = nmax
    
    #create arrays
    d = np.zeros(nmax)
    e = np.zeros(nmax-1)
    c_local = c/n
    
    #matrix elements of hamiltonian
    nm = (n - n0 - m)/2
    np = (n - n0 + m)/2
    for j in range(nmax):
        d[j] = (n-n0)*(q+0.5*c_local*(2*n0-1))
        if j <= (nmax-1):
            e[j] = c_local*np.sqrt(nm*np*(n0+2)*(n0+1))
        
        nm = nm - 1
        np = np - 1
        n0 = n0 + 2
    
    #estimate based on Gershgorin's circle theorem
    radius = abs(e[0])
    e_min = d[0] - radius
    e_max = d[0] + radius
    for j in range(1,nmax-1):
        radius = abs(e[j-1]) + abs(e[j])
        e_min = min(e_min, d[j] - radius)
        e_max = max(e_max, d[j] + radius)
    
    radius = abs(e[nmax-1])
    e_min = min(e_min, d[nmax] - radius)
    e_max = max(e_max, d[nmax] + radius)
    
    radius = (e_max + e_min)/2
    
    for i in range(nmax):
        d[i] = d[i] - radius
        
    radius = 2/(e_max-e_min)
    d = radius * d
    e = radius * e
    
    return e_min, e_max
