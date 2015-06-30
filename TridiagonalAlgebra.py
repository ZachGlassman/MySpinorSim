# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:03:24 2015
Try to sybolically solve the hopping problem
@author: zag
"""
import numpy as np
#from sympy import sqrt
from numpy import sqrt
from numba import autojit
import time
def qd(a,b):
    if a==b:
        return 1
    else:
        return 0
        
@autojit       
def H(k,kp,N):
    a = k*(2*(N-2*k)-1) * qd(k,kp)
    b = (kp+1)*sqrt((N-2*kp)*(N-2*kp-1))*qd(k,kp+1)
    c = kp*sqrt((N-2*kp+1)*(N-2*kp+1))*qd(k,kp-1)
    return 2*(a+b+c)


def make_matrix(Ntot):
    dim = int(Ntot/2 + 1)
    mat = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            mat[i,j] = H(i,j,Ntot)
    return mat


if __name__ == '__main__':
    s = time.time()
    Ntot = 40000
    mat = make_matrix(Ntot)
    e = time.time()        
    print(mat)
    print(e-s)