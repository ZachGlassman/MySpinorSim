# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:01:45 2015
Mean field simulation
Solves the mean field equations of motion for a spinor condensate
@author: zachg
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode
from numpy.lib import scimath
import sys
import time

#first define the system dy/dt = f(y,t)
def msqr(x):
    return np.conj(x) * x

def f(t,y,*args):
    """system of ode we want to solve"""
    z1i = y[0]
    z0i = y[1]
    zmi = y[2]
    #now define equations
    f0 = ((p1*B+q1*B**2+qu1)*z1i + c*((msqr(z1i) + msqr(z0i) - msqr(zmi))*z1i + z0i*z0i*np.conj(zmi)))*np.complex(0,-1)
    f1 = ((p0*B+q0*B**2+qu0)*z0i + c*((msqr(z1i) + msqr(zmi))*z0i + 2*z1i*zmi*np.conj(z0i)))*np.complex(0,-1)
    f2 = ((pm1*B+qm1*B**2+qum1)*zmi + c*((msqr(zmi) + msqr(z0i) - msqr(z1i))*zmi + z0i*z0i*np.conj(z1i)))*np.complex(0,-1)
    return [f0,f1,f2]
    
def rf(t,y,*args):
    """rf ode
    dp/dm = omega_b - Delta pm delta
    where omega_b is field frequency
    Delta = 1/2 mu_b B_z
    delta = detuning """
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
        
        
def generate_states(N,s):
    """generate quasi probability distribution from Chapman paper"""
    sx = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = s)
    sy = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size =s)
    nyz = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = s)
    nxz = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = s)
    txipi = -(sy + nyz)/(sx + nxz)
    tximi = (sy-nyz)/(sx-nxz)
    def pos(x):
        if x < 0:
            return x + np.pi
        else:
            return x
    txip = np.asarray([pos(i) for i in txipi])
    txim = np.asarray([pos(i) for i in tximi])
    
   
    a = (sx+nxz)**2/(np.cos(np.arctan(txip)))**2
    b = (sx-nxz)**2/(np.cos(np.arctan(txim)))**2
   

    rho_0 = 1/2 + scimath.sqrt(1/4-1/8*(a+b))
    m = 1/(8 * rho_0)*(a-b)

            
    states = np.zeros((len(m),3),dtype = complex)
    
    states[:,0] = scimath.sqrt((1-rho_0+m)/2) * np.exp(np.arctan(txip)*1j)
    states[:,1] = scimath.sqrt(rho_0)
    states[:,2] = scimath.sqrt((1-rho_0-m)/2) * np.exp(np.arctan(txim)*1j)
    
    return states


def solve_system(y0, B_arr, p1_arr, p0_arr, pm1_arr,qu1_arr,qu0_arr,qum1_arr,
                 q1_arr,q0_arr,qm1_arr,c_arr,tfinal,dt):
    r = complex_ode(f)
    r.set_initial_value(y0,0)
    #r.set_integrator('dopri5')
    ans = np.zeros((len(B_arr),3),dtype = complex)
    step  = 0
    while r.successful() and r.t < tfinal:
        #update the parameters
        B = B_arr[step]
        p1 = p1_arr[step]
        p0 = p0_arr[step]
        pm1 = pm1_arr[step]
        qu1 = qu1_arr[step]
        qu0 = qu0_arr[step]
        qum1 = qum1_arr[step]
        q1 = q1_arr[step]
        q0 = q0_arr[step]
        qm1 = qm1_arr[step]
        c = c_arr[step]
        ans[step] = np.asarray(r.integrate(r.t + dt))
        step += 1
    return ans
    
    
def validate(par,t):
    """function to validate arrays"""
    lt = len(t)
    if isinstance(par,float) or isinstance(par,int) or isinstance(par,np.complex):
        return np.asarray([par for i in t])
    elif len(par)==lt:
        return par
    else:
        print('check array dimensions')

#fancy writeout
def write_progress(step,total,string = None):
    """write the progress out to the window"""
    perc_done = step/(total) * 100
    #50 character string always
    num_marks = int(.5 * perc_done)
    out = ''.join('#' for i in range(num_marks))
    out = out + ''.join(' ' for i in range(50 - num_marks))
    sys.stdout.write('\r[{0}]{1:>2.0f}% {2}'.format(out,perc_done,string))
    sys.stdout.flush()
    
###################################
# Operator Definitions
###################################
S_x = Operator(np.array([[0,1,0],[1,0,1],[0,1,0]])*1/np.sqrt(2),r'$S_x$')
N_yz = Operator(1j/np.sqrt(2)* np.array([[0,-1,0],[1,0,1],[0,-1,0]]),r'$N_{yz}#')
rho_0 = Operator(np.array([[0,0,0],[0,1,0],[0,0,0]]),r'$\rho_0$')

def get_exp_values(sol,step_size):
    """function to compute expectation values"""
    r_0 = np.asarray([rho_0.apply(i) for i in sol[::step_size]])
    sx_calc = np.asarray([S_x.apply(i) for i in sol[::step_size]])
    nyz_calc = np.asarray([N_yz.apply(i) for i in sol[::step_size]])
    return np.asarray([r_0, sx_calc, nyz_calc])
    
    
if __name__ == '__main__':
    #first define arrays
    #define problem parameters
    pars = {}
    pars['dt'] = .6e-4
    tfinal = .3

    t = np.linspace(0, tfinal , int( tfinal/pars['dt'] ))
    pars['tfinal'] = t[-1] - pars['dt']
   
   
    c =2*np.pi*7.5 #2*np.pi*36
    B = 0.21#0.37  #Gauss
    p1 = 0
    p0 = 0
    pm1 = 0
    qu1 = 0
    qu0 = 0
    qum1= 0
    q1 = 2*np.pi * 72
    q0 = 0
    qm1= 2*np.pi * 72
    
    
    #generate array
    pars['B_arr'] = validate(B,t)
    pars['p1_arr'] = validate(p1,t)
    pars['p0_arr'] = validate(p0,t)
    pars['pm1_arr'] = validate(pm1,t)
    pars['qu1_arr'] = validate(qu1,t)
    pars['qu0_arr'] = validate(qu0,t)
    pars['qum1_arr'] = validate(qum1,t)
    pars['q1_arr'] = validate(q1,t)
    pars['q0_arr'] = validate(q0,t)
    pars['qm1_arr'] = validate(qm1,t)
    pars['c_arr'] = validate(c,t)
    
    #now start calculation
    N = 5000
    start = time.time()
    states = generate_states(N,400)
    step_size = 50
    ans = np.zeros((len(states),3,len(t[::step_size])))
    
    #do calculation
    ll = len(states)
    for i,state in enumerate(states):
        write_progress(i+1,ll)
        ans[i] = get_exp_values(solve_system(state,**pars),step_size)
    end = time.time()
    print('\nCalculation Finished in time:','{:<.2f}'.format(end-start))
    print('Now Plotting')
    #plot it
    fig, ax = plt.subplots(3,1)
    for i in range(3):
        m = np.mean(ans[:,i],axis = 0)
        s = np.std(ans[:,i],axis = 0)
        ax[i].plot(t[::step_size],m)
        ax[i].fill_between(t[::step_size,],m-s,m+s,facecolor='green',alpha=0.2)
    plt.tight_layout()
    plt.show()