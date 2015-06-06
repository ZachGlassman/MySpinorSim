# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:24:21 2015
This is the python version of spinorf so I can understand whats going on
@author: zag
"""
import numpy as np
import math
from hamiltonian import setup_scaled_H, moments
from ChebyshevPropagator import chebyshev_propagator
#first we have initization variables

def find_nmax(tot,m):
    first = np.mod(tot - abs(m),2)
    return (tot-abs(m)-first)/2+1
    
def alpha_help(a,n):
    """helper function, here a is alpha_ and n is n_"""
    if a == np.complex(0,0):
        if n == 0:
            ln = np.complex(0,0)
        else:
            ln = np.complex(-1e200,0)
            
    elif n >= 7:
        """stirlings approximation"""
        ln = n *np.log(a)- (n*np.log(n)-n + np.log(2*np.pi*n)/2)/2
    else:
        ln = n * np.log(a) - np.log(math.factorial(int(n)))/2
    return ln
def find_norm(z):
    """find complex norm^2 of a vector of complex numbers"""
    return sum([i.real**2 + i.imag**2 for i in z])
    
    
init_state_solver = 'coherent_state'
propogate = 'Chebychev'
species = 'Na'
b_field = 0.0           #BField
n_tot = 300            #TotalAtomNumber
mag = 0                 #Magnetization
mag_range = 7          #MagRange
atom_range =10        #AtomRange
spinor_phase = 0.0      #SpinorPhase
n_0 = 380               #N0 in Hz
c_init = 30             #C_init in Hz

eqz = 0.007189 * b_field**2
ndiv = 1
delta_t= [0.04,0.001,0.04]
c = [24,24,24]
emw = [-2.5,-426,-2.5]
n_step = [50,10,50]

#now we want to allocate numpy array
sum_of_means = np.zeros(sum(n_step)) #one for each time step
sum_of_meansq = np.zeros(sum(n_step))
norm = np.zeros(sum(n_step))
time = np.zeros(sum(n_step))

density = np.zeros(sum(n_step) * int(n_tot)+atom_range+1)

if n_0 < 1e-20:
    alpha_zero =  np.complex(0,0)
else:
    alpha_zero = np.sqrt(n_0)*np.exp(np.complex(0,spinor_phase/2))
    
if (n_tot - n_0 + mag) < 1e-20:
    alpha_plus = np.complex(0,0)
else:
    alpha_plus = np.complex(np.sqrt(mag+(n_tot-n_0-mag)/2),0)

if (n_tot - n_0 - mag) < 1e-20:
    alpha_minus = np.complex(0,0)
else:
    alpha_minus = np.complex(np.sqrt((n_tot-n_0-mag)/2),0)
    
#calculate normalization factor
norm_factor = (abs(alpha_minus)**2 + abs(alpha_zero)**2 + abs(alpha_plus)**2)/2

#now loop over magnetizations to initialize
steps_to_count = 0
for m in range(mag-mag_range,mag+mag_range):
    print('step {0} out of {1}'.format(steps_to_count,2*mag_range))
    norm_for_m = 0
    for atom_n in range(n_tot - atom_range, n_tot + atom_range):
        if atom_n >= abs(m):
            n_max = find_nmax(atom_n,m)
            #call setup_scaled_h with first = True to get d,e
            e_min,e_max,d,e = setup_scaled_H(eqz + emw[0],c[0],atom_n,m,n_max, first = True)
            state = np.zeros(n_max, dtype = complex)
            sum_coef = 0
            
            #sensible bounds
            j_low = max(1,int((n_0-4*np.sqrt(n_0))/2))
            j_high = min(n_max, int((n_0+4*np.sqrt(n_0))/2))
            
            #now loop over j
            for j in range(int(n_max)):
                n_zero_min = np.mod(atom_n - abs(m),2)
                n_zero = n_zero_min + 2*j
                n_minus = (atom_n - n_zero - m)/2
                n_plus = m + n_minus
                if n_minus >= 0 and n_plus >=0 and n_zero >=0:
                    ln_minus = alpha_help(alpha_minus,n_minus)
                    ln_plus = alpha_help(alpha_plus, n_plus)
                    ln_zero = alpha_help(alpha_zero,n_zero)
                    
                    sum_ln = ln_minus + ln_plus + ln_zero
                    
                    ln_coef = sum_ln - norm_factor
                    coef = np.exp(ln_coef)
                    state[j] = coef
                else:
                    state[j] = np.complex(0,0)
                    
            #now do timestep loop
            t = 0
            t_step = 0
            mean, mean_sq = moments(state,n_max)
            sum_of_meansq[t_step] += mean_sq
            sum_of_means[t_step] += mean
            sum_coef = find_norm(state)
            norm_for_m += sum_coef
            norm[t_step] += sum_coef
            time[t_step] = t
            t_step = t_step + 1
            for interval in range(ndiv):
                q = eqz + emw[interval]
                e_min,e_max=setup_scaled_H(q,c[interval], atom_n, m,n_max)
                dt = delta_t[interval]/(n_step[interval]) #time step
                scaled_dt = 2*np.pi * (e_max - e_min)*dt/2
                t_local_scaled = 0
                for i in range(n_step[interval]):
                    t = t + dt
                    t_local_scaled += scaled_dt
                    state = chebyshev_propagator(scaled_dt,state,n_max,e,d) 
                    mean, mean_sq = moments(state,n_max)
                    sum_of_meansq[t_step] += mean_sq
                    sum_of_means[t_step] += mean
                    sum_coef = find_norm(state)
                    norm_for_m += sum_coef
                    norm[t_step] += sum_coef
                    time[t_step] = t
                    t_step += 1
    steps_to_count +=1     

print(norm)
outstring = '{:<15}{:<15}{:<15}\n'
infostring = '{0}={:>15}\n'
with open('results.txt', 'w') as fp:   
    fp.write(infostring.format('Species','23Na'))
    fp.write(infostring.format('B Field (muT)',b_field))
    fp.write(infostring.format('N_0', n_0))
    fp.write(infostring.format('C_init',c_init))
    fp.write(outstring.format('t(s)','mean','stddev'))
    for time_step in range(len(sum_of_means)):
        t = time[time_step]
        mean = sum_of_means[time_step]/norm[time_step]
        print(mean)
        meansq = sum_of_meansq[time_step]/norm[time_step]
        fp.write(outstring.format(t,np.sqrt(meansq-mean*mean),norm[time_step]))
            
            
