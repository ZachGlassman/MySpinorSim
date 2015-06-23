# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:37:35 2015
Fock State Simulation, based on Christopher Hamley, Chapman group
thesis
use fourth order Runge-Kutta to integrate equations.
@author: zag
"""
import numpy as np
import matplotlib.pyplot as plt
##########################
#Define Runge-Kutta method
##########################

    
def ynplus1(func, yn,t,dt):
    """evolve Runge kutta with function func which takes two input arguments
    yn and t
    :param yn: value at the previous iteration
    :param t: the time at current iteration
    :param dt: time step
    """
    
    k1 = func(yn,t)
    k2 =  func(yn+dt/2*k1,t+dt/2)
    k3 = func(yn+dt/2*k2,t+dt/2)
    k4 = func(yn+dt+k3,t+dt)
    a = k1+ 2 * k2 + 2 * k3 + k4
    return yn + dt*a/6

#########################
#Calculate magnetic field
#########################
def heaviside(x):
    """heaviside step function"""
    return 0.5 * (np.sign(x) + 1)
    
def bcon(t):
    a = (2-0.210) * heaviside(2e-3 -t)
    b = (2-0.210)/2e-3 * heaviside(t)*heaviside(2e-3-t)*t
    return a-b + 0.210
    
def bzdot(Bz,t,tauB):
    return 1/tauB*(bcon(t)-Bz)

def calculate_magnetic_field(total_time,dt,tauB):
    num_iter = int(total_time/dt)
    #prop_func = bzdot(yn,t,tauB)
    #create list of magnetic fields
    Blist = np.zeros(num_iter)
    Blist[0] = bcon(0)
    #now iterate in time
    for i in range(1,num_iter,1):
        Blist[i] = ynplus1(bcon,Blist[i-1],i*dt,dt)
    return Blist
  
#######################################################################
#FOCK STATE
#Vector is k[i] where [n-1,n0,n1] = [k-m1,N-2k+ml,k], k in (0,(N+ml)/2)
#######################################################################

if __name__ == '__main__':
    x = np.linspace(0,10e-3,1000)
    plt.plot(x,bcon(x))
    plt.show()