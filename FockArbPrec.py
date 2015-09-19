# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:37:35 2015
Fock State Simulation, based on Christopher Hamley, Chapman group
thesis
use fourth order Runge-Kutta to integrate equations.

Uses mpmath for arbitrary precision
@author: zag
"""
import matplotlib.pyplot as plt
from numba import autojit
import sys
import time
import mpmath as mpm
import numpy as np

mpm.mp.dps = 15
##########################
#Define Runge-Kutta method
##########################
def ynplus1(func, yn,t,dt,**kwargs):
    """evolve Runge kutta with function func which takes two input arguments
    yn and t and possibly extra arguments
    :param yn: value at the previous iteration
    :param t: the time at current iteration
    :param dt: time step
    """
    k1 = func(yn,t,**kwargs)
    k2 =  func(yn+dt/2*k1,t+dt/2,**kwargs)
    k3 = func(yn+dt/2*k2,t+dt/2,**kwargs)
    k4 = func(yn+dt*k3,t+dt,**kwargs)
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
    #define function to iterate
    def func(yn,t):
        return bzdot(yn,t,tauB)
    #now iterate in time
    for i in range(1,num_iter,1):
        Blist[i] = ynplus1(func,Blist[i-1],i*dt,dt)
    return Blist
    
#######################################################################
#FOCK STATE
#Vector is k[i] where [n-1,n0,n1] = [k-m1,N-2k+ml,k], k in (0,(N+ml)/2)
#######################################################################
def tri_ham(c,bfield,psi,n_atoms):
    '''compute the tridiagonal hamiltonian for fock state'''
    ans = np.arange(len(psi))
    nn = mpm.mpc(0,0)
    ans = ans * nn
    
    #first for diagonal interation
    for i in range(len(psi)):
        ans[i] = (i*(2*(n_atoms-2*i))-1)* c/n_atoms*psi[i] + bfield * i*psi[i]
    #now for ineraction with kp = i-1
    for i in range(1,len(psi)):
        ans[i] += i * np.sqrt((n_atoms - 2 * (i-1) - 1)*(n_atoms - 2*(i-1)))*psi[i-1]* c/n_atoms
    #now for kp = i +1
    for i in range(len(psi)-1):
        ans[i] += (i+1)*np.sqrt((n_atoms-2*(i+1)+1)*(n_atoms-2*(i+1)+2))*psi[i+1]* c/n_atoms
    return ans

#may need higher precision integration
def func_to_integrate(yn,t,bfield,c,n_atoms):
    com =  tri_ham(c,bfield,yn,n_atoms)
    return np.complex(0,-1)*com
    
def set_up_simulation(total_time,dt,tauB,mag_time,c,n_atoms):
    num_steps = int(total_time/dt)
    #calculate B field
    b_field = calculate_magnetic_field(mag_time,dt,tauB)
    params = {
    'c':c,
    'n_atoms':n_atoms,
    'bfield':b_field[0]
    }
    b_steps = int(mag_time/dt)
    return params, num_steps,b_steps,b_field
    
def create_init_state(n_atoms):
    nn = mpm.mpc(0,0)
    state = np.zeros(int(n_atoms/2)+1)
    state = state * nn
    state[0]= mpm.mpc(1,0)
    return state

def get_bfield(bfield,b_steps,step):
    if step < b_steps:
        ans =  bfield[step]
    else:
        ans =  0.21
    return 2*np.pi * 276.8 * ans**2
  
        
#fancy writeout
def write_progress(step,total):
    #write out fancy
    perc_done = step/(total) * 100
    #50 character string always
    num_marks = int(.5 * perc_done)
    out = ''.join('#' for i in range(num_marks))
    out = out + ''.join(' ' for i in range(50 - num_marks))
    sys.stdout.write('\r[{0}]{1:>2.0f}%'.format(out,perc_done))
    sys.stdout.flush()

################################################
#Calculate Expectation Values
################################################
def calc_n0_vals(psi,num_atoms):
    n0 = 0
    n0sqr = 0
    for k in range(len(psi)):
        n0 += (num_atoms-2*k) * mpm.fabs(psi[k])**2
        n0sqr += (num_atoms-2*k)**2 * mpm.fabs(psi[k])**2
    n0var = n0sqr - n0**2
    return n0, n0sqr , n0var

def calc_sx_sqr(psi,n):
    ans = 0
    #where i indexes k
    for i in range(len(psi)):
        ans += (-4*i*i+2*i*n-i+n)*np.abs(psi[i]*psi[i])
    for i in range(len(psi)-1):
        ans += i*np.sqrt((n-2*i+1)*(n-2*i+2))*np.abs(psi[i]*psi[i+1])
    for i in range(1,len(psi)):
        ans += (i+1)*np.sqrt((n-2*i)*(n-2*i-1))*np.abs(psi[i]*psi[i-1])
    return ans
    
def calc_qyz_sqr(psi,n):
    ans = 0
    #here i indexes k
    for i in range(len(psi)):
        ans += (-4*i*i+2*i*n-i+n)*np.abs(psi[i]*psi[i])
    for i in range(len(psi)-1):
        ans += -i*np.sqrt((n-2*i+1)*(n-2*i+2))*np.abs(psi[i]*psi[i+1])
    for i in range(1,len(psi)):
        ans += -(i+1)*np.sqrt((n-2*i)*(n-2*i-1))*np.abs(psi[i]*psi[i-1])
    return ans
###############################################
# main routine
###############################################
def main(total_time,dt,mag_time,tauB,n_atoms,c):
    params,num_steps,b_steps,b_field = set_up_simulation(total_time,
                                                dt,tauB,mag_time,c,n_atoms)
                                                                             
    psi = create_init_state(n_atoms) # create initial state
    
    n0 = np.zeros(num_steps)
    n0sqr = np.zeros(num_steps)
    n0var = np.zeros(num_steps)
    sxsqr = np.zeros(num_steps)
    qyzsqr = np.zeros(num_steps)
    #bf = 276.8* .37**2 * 2*np.pi
    bf = 144 * 2 *np.pi * .21**2
    #now evolve in time
    write_progress(0,num_steps)
    for i in range(num_steps):
        n0[i],n0sqr[i],n0var[i]=calc_n0_vals(psi,n_atoms)
        #sxsqr[i] = calc_sx_sqr(psi,n_atoms)
        #qyzsqr[i] = calc_qyz_sqr(psi,n_atoms)
        params['bfield'] = bf
        psi = ynplus1(func_to_integrate,psi,i*dt,dt,**params)
        write_progress(i + 1,num_steps)
            
    step_size = 5 #don't plot all data
    time = np.asarray([i * dt for i in range(0,num_steps,step_size)] )
    fig, ax = plt.subplots(3,1)
    ax[0].fill_between(time,n0[::step_size]-np.sqrt(n0var[::step_size]),
        n0[::step_size]-np.sqrt(n0var[::step_size]),facecolor='green',alpha=0.5)
    #ax[1].plot(time,sxsqr[::step_size],label = r'$S_x^2$')
    #ax[2].plot(time,qyzsqr[::step_size], label = r'$Q_{yz}^2$')
    ax[1].set_title(r'$\ln\langle S_x^2\rangle$')
    ax[2].set_title(r'$\ln\langle Q_{yz}^2\rangle$')
    ax[0].set_title(r'$N_0$')
    ax[2].set_xlabel('$t(s)$')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    plt.tight_layout()
    plt.show()
        
           
#############################################
# Simulation setup and program execution
#############################################
if __name__ == '__main__':
    simulation_params = {
    'total_time': .3, #simulated time (s),
    'mag_time':0.015,
    'dt':0.01e-3, #simulation time step,
    'tauB' : 1e-3,
    #'c':36*2*np.pi,
    'c':-7.5*2*np.pi,
    'n_atoms':45000,
    }
    s = time.time()
    main(**simulation_params)
    e = time.time()
    print('\n')
    print('Simulation time: {:5.2f}'.format(e-s))