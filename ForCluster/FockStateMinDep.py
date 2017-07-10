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
import argparse
from numpy.lib import scimath
from tqdm import trange
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
    for i in trange(num_steps):
        n0[i],n0sqr[i],n0var[i] = calc_n0_vals(psi,n_atoms)
        sxsqr[i] = calc_sx_sqr(psi,n_atoms)
        qyzsqr[i] = calc_qyz_sqr(psi,n_atoms)
        psi = ynplus1(func_to_integrate,psi,i*dt,dt,**params)

    step_size = 30 #don't plot all data
    time = np.asarray([i * dt for i in range(0,num_steps,step_size)] )
    return time, n0[::step_size], n0var[::step_size]

def q_to_bf(q):
    return scimath.sqrt(q/277*(2*np.pi)**3)/(2*np.pi)

def write_out(name,sim_pars,q,t,n0,std):
    params = '{0}:{1}\n'
    data = '{0:<20.8f}{1:<20.8f}{2:<20.8f}\n'
    with open(name + '_results.txt','w') as fp:
        for i in ['total_time', 'dt','c','n_atoms']:
            fp.write(params.format(i,sim_pars[i]))
        fp.write(params.format('q',q))

        fp.write('{0:20} {1:20} {2:20}\n'.format('time','rho_0','std'))
        for i in range(len(t)):
            fp.write(data.format(t[i],n0[i],std[i]))


def main(args):
    simulation_params = {
    'total_time': args.total_time,
    'mag_time':0.015,
    'dt':args.dt,
    'tauB' : 1e-3,
    'c':args.c*2*np.pi,
    'n_atoms':args.num_atoms,
    'bf':q_to_bf(args.q)
    }
    t,n0,n0var = fock_sim(**simulation_params)
    write_out(args.name,simulation_params,args.q,t,n0,np.sqrt(n0var))

#############################################
# Simulation setup and program execution
#############################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', action='store',
                        dest='name',
                        default='name',
                        type=str,
                        help = 'path and name of simulation')
    parser.add_argument('-n',action='store',
                        dest= 'num_atoms',
                        default=40000,
                        type = int,
                        help = "number of atoms")
    parser.add_argument('-t', action = 'store',
                        dest='total_time',
                        default=0.2,
                        type=float,
                        help = 'total_time')
    parser.add_argument('-c', action = 'store',
                        dest='c',
                        default=30,
                        type=float,
                        help = 'c')
    parser.add_argument('-q', action = 'store',
                        dest='q',
                        default=-5,
                        type=float,
                        help = 'q')
    parser.add_argument('-dt', action = 'store',
                        dest='dt',
                        default=.004e-4,
                        type=float,
                        help = 'time step')
    results = parser.parse_args()
    main(results)
