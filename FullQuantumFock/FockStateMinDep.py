# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:37:35 2015
Fock State Simulation, based on Christopher Hamley, Chapman group
thesis
use fourth order Runge-Kutta to integrate equations.
Has minimal dependencies, hopefully used on cluster.
@author: Zachary Glassman
"""

import numpy as np
from scipy.integrate import ode
import argparse
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

def tri_ham_np(t,y,c,bfield,n_atoms):
    '''compute the tridiagonal hamiltonian for fock state
    so we can only use numpy, create arrays of the indices'''
    ans = np.empty(len(y), dtype = complex)
    i = np.mgrid[0:len(y)]
    ans = (i*(2*(n_atoms-2*i))-1)* c/n_atoms*y + 2 * bfield * i*y
    i = np.mgrid[1:len(y)]
    ans[1:] += i * np.sqrt((n_atoms - 2 * (i-1) - 1)*(n_atoms - 2*(i-1)))*y[:-1]* c/n_atoms
    i = np.mgrid[:(len(y)-1)]
    ans[:-1]+=(i+1)*np.sqrt((n_atoms-2*(i+1)+1)*(n_atoms-2*(i+1)+2))*y[1:]* c/n_atoms

    return list(np.complex(0,-1)*ans)

def tri_ham(c,bfield,y,n_atoms):
    '''compute the tridiagonal hamiltonian for fock state
    so we can only use numpy, create arrays of the indices'''
    ans = np.empty(len(y), dtype = complex)
    i = np.mgrid[0:len(y)]
    ans = (i*(2*(n_atoms-2*i))-1)* c/n_atoms*y + 2 * bfield * i*y
    i = np.mgrid[1:len(y)]
    ans[1:] += i * np.sqrt((n_atoms - 2 * (i-1) - 1)*(n_atoms - 2*(i-1)))*y[:-1]* c/n_atoms
    i = np.mgrid[:(len(y)-1)]
    ans[:-1]+=(i+1)*np.sqrt((n_atoms-2*(i+1)+1)*(n_atoms-2*(i+1)+2))*y[1:]* c/n_atoms

    return ans


def func_to_integrate(yn,t,bfield,c,n_atoms):
    com =  tri_ham(c,bfield,yn,n_atoms)
    return np.complex(0,-1)*com


def set_up_simulation(total_time,dt,tauB,mag_time,c,n_atoms):
    num_steps = int(total_time/dt)
    b_field = 0
    params = {
    'c':c,
    'n_atoms':n_atoms,
    'bfield':b_field
    }
    b_steps = int(mag_time/dt)
    return params, num_steps,b_steps,b_field

def create_init_state(n_atoms):
    state = np.zeros(int(n_atoms/2)+1,dtype = complex)
    state[0]= np.complex(1,0)
    return state

################################################
#Calculate Expectation Values
################################################
def calc_n0_vals(psi,num_atoms):
    k = np.mgrid[:len(psi)]
    n0 = ((num_atoms-2*k) * abs(psi)**2).sum()
    n0sqr = ((num_atoms-2*k)**2 * abs(psi)**2).sum()
    n0var = n0sqr - n0**2
    return n0, n0sqr , n0var


def calc_sx_sqr(psi,n):
    i = np.mgrid[0:len(psi)]
    ans = ((-4*i*i+2*i*n-i+n)*np.abs(psi*psi)).sum()
    i = np.mgrid[:(len(psi)-1)]
    ans += (i*np.sqrt((n-2*i+1)*(n-2*i+2))*np.abs(psi[:-1]*psi[1:])).sum()
    i = np.mgrid[1:len(psi)]
    ans += ((i+1)*np.sqrt((n-2*i)*(n-2*i-1))*np.abs(psi[1:len(psi)]*psi[:-1])).sum()
    return ans

def calc_qyz_sqr(psi,n):
    #here i indexes k
    i = np.mgrid[0:len(psi)]
    ans = ((-4*i*i+2*i*n-i+n)*np.abs(psi[i]*psi[i])).sum()
    i = np.mgrid[:(len(psi)-1)]
    ans += (-i*np.sqrt((n-2*i+1)*(n-2*i+2))*np.abs(psi[:-1]*psi[1:])).sum()
    i = np.mgrid[1:len(psi)]
    ans += (-(i+1)*np.sqrt((n-2*i)*(n-2*i-1))*np.abs(psi[1:len(psi)]*psi[:-1])).sum()
    return ans

###############################################
# main routine
###############################################
def fock_sim(total_time,dt,mag_time,tauB,n_atoms,c, bf):
    params,num_steps,b_steps,b_field = set_up_simulation(total_time,
                                                dt,tauB,mag_time,c,n_atoms)

    psi = create_init_state(n_atoms) # create initial state

    n0 = np.zeros(num_steps)
    n0sqr = np.zeros(num_steps)
    n0var = np.zeros(num_steps)
    sxsqr = np.zeros(num_steps)
    qyzsqr = np.zeros(num_steps)
    bf = 277* bf**2 #q
    params['bfield'] = bf
    #now evolve in time
    for i in range(num_steps):
        n0[i],n0sqr[i],n0var[i] = calc_n0_vals(psi,n_atoms)
        sxsqr[i] = calc_sx_sqr(psi,n_atoms)
        qyzsqr[i] = calc_qyz_sqr(psi,n_atoms)
        psi = ynplus1(func_to_integrate,psi,i*dt,dt,**params)

    step_size = 30 #don't plot all data
    time = np.asarray([i * dt for i in range(0,num_steps,step_size)] )
    return time, n0[::step_size], n0var[::step_size]


def fock_sim_fast(total_time,dt,mag_time,tauB,n_atoms,c, bf):
    params,num_steps,b_steps,b_field = set_up_simulation(total_time,
                                                dt,tauB,mag_time,c,n_atoms)

    psi = create_init_state(n_atoms) # create initial state
    params['bfield'] = bf
    n0 = []
    n0sqr = []
    n0var = []
    sxsqr = []
    qyzsqr = []
    t = []
    bf = 277* bf**2 #q
    #now evolve in time
    integrator = ode(tri_ham_np).set_integrator('zvode')
    integrator.set_f_params(c,bf,n_atoms)
    integrator.set_initial_value(list(psi), 0)

    while integrator.successful() and integrator.t < total_time:
        t.append(integrator.t)
        n0_t, n0sqr_t, n0var_t = calc_n0_vals(integrator.y,n_atoms)
        sxsqr.append(calc_sx_sqr(integrator.y,n_atoms))
        qyzsqr.append(calc_qyz_sqr(integrator.y,n_atoms))
        n0.append(n0_t)
        n0sqr.append(n0sqr_t)
        n0var.append(n0var_t)
        integrator.integrate(total_time, step = True)

    step_size = 30 #don't plot all data
    return t[::step_size], n0[::step_size], n0var[::step_size]



#############################################
# Simulation setup and program execution
#############################################
if __name__ == '__main__':
    simulation_params = {
    'total_time': .03, #simulated time (s),
    'mag_time':0.015,
    'dt':0.001e-4, #simulation time step,
    'tauB' : 1e-3,
    'c':36*2*np.pi,
    'n_atoms':40000,
    'bf':.37
    }
    fock_sim(**simulation_params)
