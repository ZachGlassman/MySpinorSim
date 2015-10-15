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
from scipy.integrate import ode
from numpy.lib import scimath
import sys
import time

#first define the system dy/dt = f(y,t)
def msqr(x):
    return np.conj(x) * x

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
    dp/dm = omega_b - Delta pm delta
    where omega_b is field frequency
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
        
def generate_states(N,s):
    """generate quasi probability distribution from Chapman paper"""
    sx = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = s)
    sy = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size =s)
    nyz = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = s)
    nxz = np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = s)

    txip = np.arctan(-(sy + nyz)/(sx+ nxz))
    txim = np.arctan((sy-nyz)/(sx-nxz))
    a = (sx+nxz)**2/(np.cos(txip))**2
    b = (sx-nxz)**2/(np.cos(txim))**2
   
    rho_0 = 1/2 + scimath.sqrt(1/4-(a+b)/8)
    m = 1/rho_0*(a-b)/8

    states = np.zeros((len(m),3),dtype = complex)
    
    states[:,0] = scimath.sqrt((1-rho_0+m)/2) * np.exp(txip*1j)
    states[:,1] = scimath.sqrt(rho_0)
    states[:,2] = scimath.sqrt((1-rho_0-m)/2) * np.exp(txim*1j)
    
    return states


def solve_system(y0, B, p1, p0, pm1,qu1,qu0,qum1,q1,q0,qm1,c,t_arr,pulses):
    """solve the system by going through time and integrating the proper
    equations using the correct equation
    going in steps, for each step do proper integration"""
    r = ode(f).set_integrator('zvode')
    e = ode(rf).set_integrator('zvode')
    ans = np.zeros((len(B),3),dtype = complex)
    ans[0] = y0
    step = 1
    for per, t in enumerate(t_arr):
        y0 = ans[step-1]
        if pulses[per]['spinor']:
            r.set_initial_value(y0,t[0])
            for tstep in range(len(t)-1):
                #update the parameters
                #NEED TO MAKE IT WITH ADAPTIVE TIME STEP
                r.set_f_params(B[step],
                               p1[step],
                               p0[step],
                               pm1[step],
                               qu1[step],
                               qu0[step],
                               qum1[step],
                               q1[step],
                               q0[step],
                               qm1[step],
                               c[step])
                ans[step] = np.asarray(r.integrate(t[tstep+1]))
                step = step + 1
            
        else:
            e.set_initial_value(y0,t[0])
            e.set_f_params(1,1,1)
            for tstep in range(len(t)-1):
                #update the parameters
                ans[step] = np.asarray(e.integrate(t[tstep+1]))
                step = step + 1
       
        
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

def validate_q(par,t_arr, pulses):
    """validate the q array for microwave pulses
    assume q is set as scalar"""
    ans = []
    if len(pulses) == 1:
        #list is empty, no pulses
        return validate(par,np.hstack(t_arr))
    else:
        for i, t in enumerate(t_arr):
            if pulses[i]['type'] == 'MW':
                q = pulses[i]['other'][0]
                ans.append(np.asarray([q for i in t]))
            else:
                ans.append(np.asarray([par for i in t]))
        return np.hstack(ans)

#fancy writeout
def write_progress(step,total,string = 'Evolve'):
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
    phase = np.asarray([find_phase(i) for i in sol[::step_size]])
    return np.asarray([r_0, sx_calc, nyz_calc, phase])

###################################
# Microwave Pulses
###################################
def create_pulse_arrays(tf,dt,ps,pd,pt,pa):
    """function to create time array from pulse durations
       returns another array with boolean values for integrate spinor
    """
    if ps == None:
        ans = {}
        ans['spinor'] = True
        ans['type'] = 'Free'
        return [np.linspace(0, tf , int(tf/dt))],[ans]
    elif len(ps) == 0:
        ans = {}
        ans['spinor'] = True
        ans['type'] = 'Free'
        return [np.linspace(0, tf , int(tf/dt))],[ans]
    else:
        #we have some pulses check if they are at beginning
        #don't assume they are sorted, so sort with durations
        ps, pd = (list(t) for t in zip(*sorted(zip(ps, pd))))
        #build list of linspaces, then merge
        t = []
        #We will assume no pulse at beginnging (state is inialized)
        #thus we need to alternate between spinor evolution and pulse
        t.append(np.linspace(0,ps[0],int(ps[0]/dt),endpoint=False))
        for i in range(len(ps)):
            #apply pulse
            t.append(np.linspace(ps[i],ps[i]+pd[i],
                                 int(pd[i]/(dt*.01)),
                                 endpoint=False))
            #now apply spinor evolution
            start = ps[i]+pd[i]
            if (i+1) < len(ps):
                end = ps[i+1]
            else:
                end = tf
            t.append(np.linspace(start,end,int((end-start)/dt),endpoint=False))
        #now go through and build pulse dictionary
        pulse_info = []
        spinor = True
        pc = 0
        for i in range(len(t)):
            ans = {}
            if spinor == True:
                ans['spinor'] = True
                ans['type'] = 'Free'
                pulse_info.append(ans)
                spinor = False
            else:
                ans['spinor'] = False
                ans['type'] = pt[pc]
                if ans['type'] != 'RF':
                    ans['spinor'] = True
                ans['other'] = pa[pc]
                pulse_info.append(ans)
                spinor = True
                pc +=1
        return t, pulse_info
            
         
        
        
def single_simulation(N,nsamps,c,tfinal,dt,B,pulses,plot=True,qu1=0):
    """main routine for integration
    problem is setup for arbitrary RF pulses
    pulse_dict
    structure pulsees
        [start,dur,type,*params]
        
    for microwave pulse, just change q
    so we can just write that into the arrays
    """
    #define problem parameters
    
    pulse_start = [i[0] for i in pulses]
    pulse_dur = [i[1] for i in pulses]
    pulse_type = [i[2] for i in pulses]
    pulse_args = [i[3:] for i in pulses]
    if pulse_start != None:
        if len(pulse_start) != len(pulse_dur):
            print('Improper Pulse specification')
            
    t_arr,pulse_seq = create_pulse_arrays(tfinal,
                                     dt,
                                     pulse_start,
                                     pulse_dur,
                                     pulse_type,
                                     pulse_args)
                                     
    
    B = B  #Gauss
    c = c
    p1 = 0
    p0 = 0
    pm1 = 0
    #qu1 = 0
    qu0 = 0
    qum1= qu1
    q1 = 276.8
    q0 = 0
    qm1= q1

    #generate array
    t = np.hstack(t_arr)
    pars = {}
    pars['B'] = validate(B,t)
    pars['p1'] = validate(p1,t)
    pars['p0'] = validate(p0,t)
    pars['pm1'] = validate(pm1,t)
    pars['qu1'] = validate(qu1,t)
    pars['qu0'] = validate_q(qu0,t_arr,pulse_seq)
    pars['qum1'] = validate(qum1,t)
    pars['q1'] = validate(q1,t)
    pars['q0'] = validate(q0,t)
    pars['qm1'] = validate(qm1,t)
    pars['c'] = validate(c,t)
    pars['t_arr'] = t_arr
    pars['pulses'] = pulse_seq
    #now start calculation
    start = time.time()
    states = generate_states(N,nsamps)
    step_size = 2
    ans = np.zeros((len(states),4,len(t[::step_size])))
    
    #do calculation
    ll = len(states)
    for i,state in enumerate(states):
        write_progress(i+1,ll)
        ans[i] = get_exp_values(solve_system(state,**pars),step_size)
    end = time.time()
    print('\nCalculation Finished in time:','{:<.2f}'.format(end-start))

    #output routine
    m = np.mean(ans[:,0],axis = 0)
    s = np.std(ans[:,0],axis = 0)
    np.savetxt('meanout.txt',np.vstack((t[::step_size],np.mean(ans[:,0],axis = 0))))
    
    
    if plot:
        fig, ax = plt.subplots(3,1)
        ax[0].plot(t[::step_size],m)
        ax[0].fill_between(t[::step_size],m-s,m+s,facecolor='green',alpha=0.2)
        sxval = np.mean(ans[:,1],axis =0)
        qyzval = np.mean(ans[:,2],axis =0)
        ax[1].plot(t[::step_size],sxval)
        ax[2].plot(t[::step_size],qyzval)
        ax[1].set_yscale('log')
        ax[2].set_yscale('log')
        plt.tight_layout()
        plt.show()
    
    return np.vstack((t[::step_size],m,s,np.mean(ans[:,3],axis=0))), pars['qu0']
    
if __name__ == '__main__':
    """main function for command line utility, won't usually be used
    in this way
    """
    print('functionality not enabled at this time')