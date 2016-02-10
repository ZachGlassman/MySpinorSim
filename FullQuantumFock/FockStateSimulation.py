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
from numba import autojit
import sys
import time
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

@autojit
def tri_ham(c,bfield,psi,n_atoms):
    '''compute the tridiagonal hamiltonian for fock state'''
    ans = np.empty(len(psi), dtype = complex)
    #first for diagonal interation
    for i in range(len(psi)):
        ans[i] = (i*(2*(n_atoms-2*i))-1)* c/n_atoms*psi[i] + 2 * bfield * i*psi[i]
    #now for ineraction with kp = i-1
    for i in range(1,len(psi)):
        ans[i] += i * np.sqrt((n_atoms - 2 * (i-1) - 1)*(n_atoms - 2*(i-1)))*psi[i-1]* c/n_atoms
    #now for kp = i +1
    for i in range(len(psi)-1):
        ans[i] += (i+1)*np.sqrt((n_atoms-2*(i+1)+1)*(n_atoms-2*(i+1)+2))*psi[i+1]* c/n_atoms

    return ans

def tri_ham2(c,bfield,psi,n_atoms,m):
    ans = np.empty(len(psi), dtype = complex)

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

def create_init_state(n_atoms,pairs = 0):
    state = np.zeros(int(n_atoms/2)+1,dtype = complex)
    state[pairs]= np.complex(1,0)
    return state

def get_bfield(bfield,b_steps,step):
    if step < b_steps:
        ans =  bfield[step]
    else:
        ans =  0.21
    return 2*np.pi * 276.8 * ans**2*2

################################################
#Calculate Expectation Values
################################################
@autojit
def calc_n0_vals(psi,num_atoms):
    n0 = 0
    n0sqr = 0
    for k in range(len(psi)):
        n0 += (num_atoms-2*k) * abs(psi[k])**2
        n0sqr += (num_atoms-2*k)**2 * abs(psi[k])**2
    n0var = n0sqr - n0**2
    return n0, n0sqr , n0var

@autojit
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

@autojit
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
def fock_sim(total_time,dt,mag_time,tauB,n_atoms,c, bf,npairs):
    try:
        """bf,c,total_time,dt are a list"""
        #calculate B field
        num_steps = [int(total_time[i]/dt[i]) for i in range(len(dt))]
        psi = create_init_state(n_atoms,npairs)
        psi_init = create_init_state(n_atoms,npairs)
        n0_ret = np.array([])
        n0sqr_ret = np.zeros([])
        n0var_ret = np.zeros([])
        sxsqr_ret = np.zeros([])
        qyzsqr_ret = np.zeros([])
        time_ret = np.zeros([])
        init_norm_ret = np.zeros([])
        t_sim = 0
        for step in trange(len(dt),desc='outer_loop',leave=True):
            params = {
            'c':c[step],
            'n_atoms':n_atoms,
            'bfield':bf[step]**2*277
            }
            n0 = np.zeros(num_steps[step])
            n0sqr = np.zeros(num_steps[step])
            n0var = np.zeros(num_steps[step])
            sxsqr = np.zeros(num_steps[step])
            qyzsqr = np.zeros(num_steps[step])
            time = np.zeros(num_steps[step])
            init_norm = np.zeros(num_steps[step])
            for i in trange(num_steps[step],desc = 'inner_loop',leave=True, nested=True):
                n0[i],n0sqr[i],n0var[i] = calc_n0_vals(psi,n_atoms)
                init_norm[i] = np.linalg.norm(psi-psi_init)
                sxsqr[i] = calc_sx_sqr(psi,n_atoms)
                qyzsqr[i] = calc_qyz_sqr(psi,n_atoms)
                time[i] = t_sim
                t_sim += dt[step]
                psi = ynplus1(func_to_integrate,psi,t_sim,dt[step],**params)

            n0_ret = np.hstack((n0_ret,n0))
            n0sqr_ret = np.hstack((n0sqr_ret,n0sqr))
            n0var_ret = np.hstack((n0var_ret,n0var))
            sxsqr_ret = np.hstack((sxsqr_ret,sxsqr))
            qyzsqr_ret = np.hstack((qyzsqr_ret,qyzsqr))
            time_ret = np.hstack((time_ret,time))
            init_norm_ret = np.hstack((init_norm_ret,init_norm))

        n0 = n0_ret
        n0sqr = n0sqr_ret
        n0var = n0var_ret
        sxsqr = sxsqr_ret
        qyzsqr = qyzsqr_ret
        time = time_ret
        init_norm = init_norm_ret

    except:
        """bf is just number"""
        params,num_steps,b_steps,b_field = set_up_simulation(total_time,
                                                    dt,tauB,mag_time,c,n_atoms)


        psi = create_init_state(n_atoms, npairs) # create initial state

        n0 = np.zeros(num_steps)
        n0sqr = np.zeros(num_steps)
        n0var = np.zeros(num_steps)
        sxsqr = np.zeros(num_steps)
        qyzsqr = np.zeros(num_steps)
        bf = 277* bf**2 #q
        #now evolve in time
        for i in trange(num_steps):
            n0[i],n0sqr[i],n0var[i] = calc_n0_vals(psi,n_atoms)
            sxsqr[i] = calc_sx_sqr(psi,n_atoms)
            qyzsqr[i] = calc_qyz_sqr(psi,n_atoms)
            params['bfield'] = bf
            psi = ynplus1(func_to_integrate,psi,i*dt,dt,**params)
        time = np.asarray([i * dt for i in range(num_steps)] )

    step_size = 30 #don't plot all data
    try:
        return time[::step_size], n0[::step_size], n0var[::step_size], init_norm[::step_size]
    except:
        return time[::step_size], n0[::step_size], n0var[::step_size], None



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
    s = time.time()
    fock_sim(**simulation_params)
    e = time.time()
    print('\n')
    print('Simulation time: {:5.2f}'.format(e-s))
