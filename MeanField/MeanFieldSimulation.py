# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:01:45 2015
Mean field simulation
Solves the mean field equations of motion for a spinor condensate

Two types of pulses:
Microwave - Changes effective q
RF - induces population transfer, evolve under mean field equations of spin flips
Model only valid when pulse duration small (<.1 ms)
@author: zachg
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import  ode
from numpy.lib import scimath
import sys
from tqdm import tqdm

def msqr(x):
    return np.conj(x)*x

#first define the system dy/dt = f(y,t)
def f(t,y,B,p1,p0,pm1,qu1,qu0,qum1,q1,q0,qm1,c):
    """system of ode we want to solve"""
    z1i = y[0]
    z0i = y[1]
    zmi = y[2]
    #now define equations
    f0 = ((p1*B+q1*B*B+qu1)*z1i + c*((msqr(z1i) + msqr(z0i) - msqr(zmi))*z1i + z0i*z0i*np.conj(zmi)))*np.complex(0,-1)
    f1 = ((p0*B+q0*B*B+qu0)*z0i + c*((msqr(z1i) + msqr(zmi))*z0i + 2*z1i*zmi*np.conj(z0i)))*np.complex(0,-1)
    f2 = ((pm1*B+qm1*B*B+qum1)*zmi + c*((msqr(zmi) + msqr(z0i) - msqr(z1i))*zmi + z0i*z0i*np.conj(z1i)))*np.complex(0,-1)
    return [f0,f1,f2]

def rf(t,y,sigma,dm,dp):
    """rf ode
    We will use the optical Bloch equations
    dp/dm = omega_b - Delta pm delta
    sigma is rabi frequency
    where omega_b is field frequency frequency
    Delta = 1/2 mu_b B_z
    delta = detuning

    sigma = Rabi frequency"""

    c1 = y[0]
    c0 = y[1]
    cm1 = y[2]
    s = 1j*sigma/np.sqrt(2)
    #now define equations
    f0 = s * c0 * np.exp(-1j*dm*t)
    f1 = s*cm1*np.exp(-1j*dp*t)+s*c1*np.exp(1j*dm*t)
    f2 = s*c0*np.exp(1j*dp*t)
    return [f0,f1,f2]


class Operator(object):
    """class of operator for 3x3 matrices in this problem"""
    def __init__(self, mat,rep):
        self.rep = rep
        self.mat = mat

    def apply(self,ele):
        return np.dot(np.conj(ele), np.dot(self.mat,ele.T)).real

def find_phase(ele):
    """find the phase theta_1+theta_-1-2theta_0"""
    ans =  np.angle(ele)
    return ans[0] + ans[2] - 2 * ans[1]

def generate_states(N1,N0,Nm1,theta,s):
    """generate quasi probability distribution"""
    N = N1 + N0 + Nm1
    r0 = N0/N
    r1 = N1/N
    rm1 = Nm1/N
    #need to divide by N
    a = np.sqrt(2*r0*r1)
    b = np.sqrt(2*r0*rm1)

    sx_mean = np.cos(theta/2)*(a+b)
    sy_mean = np.sin(theta/2)*(b-a)
    nyz_mean = -np.sin(theta/2)*(a+b)
    nxz_mean = np.cos(theta/2)*(a-b)

    var_one = N*np.sqrt(r0**2-2*r0*np.sqrt(r1)*np.sqrt(rm1)*np.cos(theta)-r0*r1-r0*rm1+r1**(3/2)*np.sqrt(rm1)*np.cos(theta)+np.sqrt(r1)*rm1**(3/2)*np.cos(theta)+0.25*r1**2+0.5*r1*rm1*np.cos(2*theta) + r1*rm1 + 0.25*rm1**2)
    var_two = N*np.sqrt(r0**2+2*r0*np.sqrt(r1)*np.sqrt(rm1)*np.cos(theta)-r0*r1-r0*rm1-r1**(3/2)*np.sqrt(rm1)*np.cos(theta)-np.sqrt(r1)*rm1**(3/2)*np.cos(theta)+0.25*r1**2+0.5*r1*rm1*np.cos(2*theta) + r1*rm1 + 0.25*rm1**2)


    sx = np.random.normal(loc = sx_mean, scale = 1/np.sqrt(var_one), size = s)
    sy = np.random.normal(loc = sy_mean, scale = 1/np.sqrt(var_two), size =s)
    nyz = np.random.normal(loc = nyz_mean, scale = 1/np.sqrt(var_one), size = s)
    nxz = np.random.normal(loc = nxz_mean, scale = 1/np.sqrt(var_two), size = s)

    txip = np.where((sx+nxz)>0,np.arctan(-(sy + nyz)/(sx+ nxz)),np.arctan(-(sy + nyz)/(sx+ nxz))+np.pi)
    txim = np.where((sx-nxz)>0,np.arctan((sy-nyz)/(sx-nxz)),np.arctan((sy-nyz)/(sx-nxz))+np.pi)


    a = (sx+nxz)**2/(np.cos(txip))**2
    b = (sx-nxz)**2/(np.cos(txim))**2

    rho_0 = 1/2 + scimath.sqrt(1/4-(a+b)/8)
    m = 1/rho_0*(a-b)/8

    states = np.zeros((len(m),3),dtype = complex)

    states[:,0] = scimath.sqrt((1-rho_0+m)/2) * np.exp(txip*1j)
    states[:,1] = scimath.sqrt(rho_0)
    states[:,2] = scimath.sqrt((1-rho_0-m)/2) * np.exp(txim*1j)

    return states

def get_q(qt,val,qu1,qu0,qum1):
    if qt == '0':
        return qu1,val,qum1
    elif qt == '-1':
        return qu1,qu0,val
    elif qt == '1':
        return val,qu0,qum1
    else:
        print('Improper Pulse specification, need component for micrwave pulse as string')


def solve_system(y0, B, p1, p0, pm1,qu1,qu0,qum1,q1,q0,qm1,c,pulses,tfinal):
    """solve the system by going through time and integrating the proper
    equations using the correct equation
    going in steps, for each step do proper integration"""
    #now go through the pulse sequence and create timing information
    #again assume they are ordered
    p_start = [i[0] for i in pulses]
    p_end = [i[1]+i[0] for i in pulses]
    p_type = [i[2] for i in pulses]
    p_args = [i[3:] for i in pulses]
    #now build up step arrays, excpet for first and last time, all times are both start and end times
    te = sorted(p_end + p_start+ [tfinal])
    ts = sorted([0] + p_start + p_end)
    #now interleave elements in each array
    #now loop through the arrays, assume order is spinor,pulse,spinor....
    #store data in list, don't know how to get arround this for adaptive integration
    ans = []
    pulse = False
    pulse_num = 0
    #only care about end since we will start from the end of the previous integration
    #we will create new integrator for each time (necessary since it calls old fortran)
    for  start, end in zip(ts,te):
        #check if there is a pulse and if so, get information, then increment pulse_num
        #then perform action
        if pulse:
            kind = p_type[pulse_num]
            if kind == 'RF':
                integrator = ode(rf).set_integrator('zvode')
                sigma,dm,dp = p_args[pulse_nu]
                integrator.set_f_params(sigma,dm,dp)
            else:
                #must be a microwave pulse so change relevent parameters in integrator
                # set initial parameters for spinor evolution
                qt, val = p_args[pulse_num]
                qu1n, qu0n, qum1n = get_q(qt,val,qu1,qu0,qum1)

                integrator = ode(f).set_integrator('zvode')
                integrator.set_f_params(B,p1,p0,pm1,qu1n,qu0n,qum1n,q1,q0,qm1,c)
            pulse_num +=1
            pulse = False
        else:
            #reset initial parameters for spinor evolution
            integrator = ode(f).set_integrator('zvode')
            integrator.set_f_params(B,p1,p0,pm1,qu1,qu0,qum1,q1,q0,qm1,c)
            pulse = True

        #now actually integrate with adaptive timestep
        #use new list for each evolution and only append at end for memory reasons
        t = []
        sol = []
        integrator.set_initial_value(y0, start)
        while integrator.successful() and integrator.t < end:
            integrator.integrate(end, step = True)
            #now only append if still less than what we want
            if integrator.t < end:
                t.append(integrator.t)
                sol.append(integrator.y)
        #reset intial values to last values for next iteration
        y0 = integrator.y

        #append new solution to old one (this way we can keep track of pulses)
        t = np.asarray([t])
        sol = np.asarray(sol)
        ans.append(np.concatenate((t.T,sol), axis = 1))
        #delete integrator so we don't pollute global variable namespace
        del integrator
    return [np.vstack(ans)]


###################################
# Operator Definitions
###################################
S_x = Operator(np.array([[0,1,0],[1,0,1],[0,1,0]])*1/np.sqrt(2),r'$S_x$')
N_yz = Operator(1j/np.sqrt(2)* np.array([[0,-1,0],[1,0,1],[0,-1,0]]),r'$N_{yz}$')
rho_0 = Operator(np.array([[0,0,0],[0,1,0],[0,0,0]]),r'$\rho_0$')


def get_exp_values(ans,step_size):
    """function to compute expectation values"""
    ans = ans[0]
    sol = ans[:,1:]
    r_0 = np.asarray([rho_0.apply(i) for i in sol[::step_size]]).real
    sx_calc = np.asarray([S_x.apply(i) for i in sol[::step_size]]).real
    nyz_calc = np.asarray([N_yz.apply(i) for i in sol[::step_size]]).real
    phase = np.asarray([find_phase(i) for i in sol[::step_size]]).real
    return np.asarray([ans[:,0][::step_size].real,r_0, sx_calc, nyz_calc, phase]), len(ans[:,0][::step_size])



def single_simulation(N1,N0,Nm1,theta,nsamps,c,tfinal,B,pulses,qu0=0):
    """main routine for integration
    problem is setup for arbitrary RF pulses
    pulse_dict
    structure pulsees
        [start,dur,type,*params]

    for microwave pulse, just change q
    so we can just write that into the arrays
    Pulse arrays must be ordered in time and not overlap!
    """

    #set default paraameters assume they remain the same unless the pulses change them
    pars = {}
    pars['B'] = B
    pars['p1'] = 0
    pars['p0'] = 0
    pars['pm1'] = 0
    pars['qu1'] = 0
    pars['qu0'] = qu0
    pars['qum1'] = 0
    pars['q1'] = 277
    pars['q0'] = 0
    pars['qm1'] = 277
    pars['c'] = c
    pars['pulses'] = pulses
    pars['tfinal'] = tfinal
    #now start calculation
    states = generate_states(N1,N0,Nm1,theta,nsamps)
    step_size = 1
    ans_1 = []
    #do calculation
    ll = len(states)
    ## we will also find the maximum number of timesteps
    t_max = 0
    t_index = 0
    with tqdm(total = len(states)) as pbar:
        for i, state in enumerate(states):
            vals, t_num = get_exp_values(solve_system(state,**pars),step_size)
            ans_1.append(vals)
            if t_num > t_max:
                t_max = t_num
                t_index = i

            pbar.update(1)

    #now we needto loop through the states again and inerpolate
    t_interp = ans_1[t_index][0]
    #alot array for interpolation
    ans = np.zeros((len(states),4,len(t_interp)))
    for i, vals in enumerate(ans_1):
        for j, row in enumerate(vals[1:]):
            ans[i,j] = np.interp(t_interp,vals[0],row)


    #output routine
    m = np.mean(ans[:,0],axis = 0)
    s = np.std(ans[:,0],axis = 0)

    return t_interp,m,s,np.mean(ans[:,3],axis=0)

if __name__ == '__main__':
    """main function for command line utility, won't usually be used
    in this way

    mostly for testing purposes
    """
    pulses1 = [[.04,.001,'MW','0',-426*np.pi]]
    nop = []
    N=10000
    tf = .1
    data= single_simulation(N,100,24*np.pi*4,tf,.1,pulses1)
