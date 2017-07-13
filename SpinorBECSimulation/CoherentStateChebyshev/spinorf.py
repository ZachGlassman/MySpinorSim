# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:24:21 2015
This is the python version of spinorf so I can understand whats going on
@author: zag
"""
import time as timemod
import numpy as np
import math
try:
    import numba
    from .chebyshev_functions_numba import setup_scaled_H, moments, find_norm
except ImportError:
    from .chebyshev_functions import setup_scaled_H, moments, find_norm

from .chebyshev_propagator import chebyshev_propagator
import sys
from tqdm import trange

def find_nmax(tot,m):
    first = np.mod(tot - abs(m),2)
    return (tot-abs(m)-first)/2+1

def alpha_help(a,n):
    """function to compute some approximations
    
    Parameters
    ----------
    a : complex
        number
    n : int
        number
    
    Returns
    ln : complex
        approximation
    """
    if a.real == 0 and a.imag == 0:
        if n == 0:
            ln = np.complex(0,0)
        else:
            ln = np.complex(-1e200,0)

    elif n >= 300:
        ln = n *np.log(a)- (n*np.log(n)-n + np.log(2*np.pi*n)/2)/2
    else:
        ln = n * np.log(a) - math.log(math.factorial(int(n)))/2
    return ln
    

def write_out(filename, b_field, n_0, c_init, n_tot, mag, mag_range, atom_range,
              spinor_phase, init_state_solver, propogate, delta_t, emw, eqz,
              c, n_step, sum_of_means, sum_of_meansq, norm, time):
    """Write out the simulation data file"""
    outstring1 = '{:<15}{:<15}{:<15}{:<15}\n'
    outstring = '{:<15.6e}{:<15.6e}{:<15.6e}{:<15.6e}\n'
    infostring = '{:<20} = {:<15}\n'
    with open(filename, 'w') as fp:
        #write out parameters
        fp.write(infostring.format('Species','23Na'))
        fp.write(infostring.format('B Field (muT)',b_field))
        fp.write(infostring.format('N_0', n_0))
        fp.write(infostring.format('C_init',c_init))
        fp.write(infostring.format('Total Atom Number', n_tot))
        fp.write(infostring.format('Magnetization', mag))
        fp.write(infostring.format('Mag Range', mag_range))
        fp.write(infostring.format('Atom Range', atom_range))
        fp.write(infostring.format('Spinor Phase', spinor_phase))
        fp.write(infostring.format('Initial State Solver', init_state_solver))
        fp.write(infostring.format('Propogator', propogate)+'\n')
        #write out the arrays
        fp.write('{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format('Delta t (s)',
                                                 'Emw(Hz)',
                                                 'q(Hz)',
                                                 'C(Hz)',
                                                 'num steps'))
        for i in range(len(delta_t)):
            fp.write('{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format(delta_t[i],
                                                      emw[i],
                                                      eqz + emw[i],
                                                      c[i],
                                                      n_step[i]))
        fp.write('\n')
        fp.write(outstring1.format('t(s)','mean','stddev','norm'))

        for time_step in range(len(sum_of_means)):
            t = time[time_step]
            mean = sum_of_means[time_step]/norm[time_step]
            meansq = sum_of_meansq[time_step]/norm[time_step]
            fp.write(outstring.format(t,mean,np.sqrt(meansq-mean*mean),norm[time_step]))

def solve_system(b_field, n_tot,mag, mag_range, atom_range,spinor_phase, n_0,
                 ndiv, delta_t,c, emw, n_step):
    """Solve coherent state"""
    eqz = np.real(0.0277 * b_field**2)
    sum_of_means = np.zeros(sum(n_step)+1) #one for each time step
    sum_of_meansq = np.zeros(sum(n_step)+1)
    norm = np.zeros(sum(n_step)+1)
    time = np.zeros(sum(n_step)+1)
    if n_0 < 1e-20:
        alpha_zero =  np.complex(0,0)
    else:
        alpha_zero = np.sqrt(n_0)*np.exp(np.complex(0,spinor_phase/2))

    if (n_tot - n_0 + mag) < 1e-20:
        alpha_plus = np.complex(0,0)
    else:
        alpha_plus = np.complex(np.sqrt(mag+(n_tot-n_0-mag)/2), 0)

    if (n_tot - n_0 - mag) < 1e-20:
        alpha_minus = np.complex(0,0)
    else:
        alpha_minus = np.complex(np.sqrt((n_tot-n_0-mag)/2),0)

    #calculate normalization factor
    norm_factor = (abs(alpha_minus)**2 + abs(alpha_zero)**2 + abs(alpha_plus)**2)/2

    #now loop over magnetizations to initialize
    for m in trange(mag-mag_range,mag+mag_range+1, desc = 'mag_loop', leave=True):
        norm_for_m = 0
        for atom_n in trange(n_tot - atom_range, n_tot + atom_range +1, desc = 'atom_loop',leave=True):
            if atom_n >= abs(m):
                n_max = find_nmax(atom_n,m)
                e_min,e_max,d,e,first_n0 = setup_scaled_H(eqz + emw[0],c[0],atom_n,m,n_max)
                state = np.zeros(int(n_max), dtype = complex)
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
                        state[j] = np.exp(ln_coef)
                    else:
                        state[j] = np.complex(0,0)

                #now do timestep loop
                t = 0
                t_step = 0
                mean, mean_sq = moments(state,first_n0)
                sum_of_meansq[t_step] += mean_sq
                sum_of_means[t_step] += mean
                sum_coef = find_norm(state)
                norm_for_m += sum_coef
                norm[t_step] += sum_coef
                time[t_step] = t
                t_step = t_step + 1
                for interval in range(ndiv):
                    q = eqz + emw[interval]
                    e_min,e_max,d,e, first_n0 =setup_scaled_H(q,c[interval], atom_n, m,n_max)
                    dt = delta_t[interval]/(n_step[interval]) #time step
                    scaled_dt = 2*np.pi * (e_max - e_min)*dt/2
                    t_local_scaled = 0

                    for i in trange(n_step[interval],desc = 'time_loop',leave=True):
                        t = t + dt
                        t_local_scaled += scaled_dt
                        state = chebyshev_propagator(scaled_dt,state,n_max,e,d)

                        mean, mean_sq = moments(state,first_n0)
                        sum_of_meansq[t_step] += mean_sq
                        sum_of_means[t_step] += mean
                        sum_coef = find_norm(state)
                        norm_for_m += sum_coef
                        norm[t_step] += sum_coef
                        time[t_step] = t
                        t_step += 1

    return sum_of_means, sum_of_meansq, norm, time


##############
# calling directly for testing
##############
if __name__ == '__main__':
    init_state_solver = 'coherent_state'
    propogate = 'Chebychev'
    species = 'Na'
    b_field = 0          #BField in microtesla
    n_tot = 2000            #TotalAtomNumber
    mag = 0                 #Magnetization
    mag_range = 2           #MagRange
    atom_range = 2        #AtomRange
    spinor_phase =0      #SpinorPhase
    n_0 = n_tot-2          #N_0 numbers tarting in m=0
    c_init = 24           #C_init in Hz
    filename = 'results.txt'


    ndiv = 3
    delta_t= [0.04,0.001,0.06]
    c = [c_init,c_init,c_init]
    emw = [-2.5,-426,-2.5]
    n_step = [30,6,30]
    start = timemod.time()
    sum_of_means, sum_of_meansq, norm, time = solve_system(b_field,
        n_tot,mag, mag_range, atom_range,spinor_phase, n_0,ndiv, delta_t,c, emw, n_step)
    write_out(filename, b_field, n_0, c_init, n_tot, mag, mag_range, atom_range,
                  spinor_phase, init_state_solver, propogate, delta_t, emw, eqz,
                  c, n_step, sum_of_means, sum_of_meansq, norm, time)
    end = timemod.time()
    print('Calculation Complete')
    print('Norm recovered', np.average(norm))
    print('Time for Calculation:', end-start)
    print('File written to:',filename)
