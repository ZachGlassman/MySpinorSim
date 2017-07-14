# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:24:21 2015
This is the python version of spinorf with multicore processing
@author: zag
"""
import time as timemod
import numpy as np
import math
from itertools import product
try:
    import numba
    from .chebyshev_functions_numba import setup_scaled_H, moments, find_norm
except ImportError:
    from .chebyshev_functions import setup_scaled_H, moments, find_norm

from .chebyshev_functions import alpha_help, find_nmax
from .chebyshev_propagator import chebyshev_propagator


from dask import compute, delayed
import dask.multiprocessing
try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm

def single_calc(m, atom_n, params):
    n_tot = params['n_tot']
    atom_range = params['atom_range']
    emw = params['emw']
    eqz = params['eqz']
    c = params['c']
    n_step = params['n_step']
    delta_t = params['delta_t']
    ndiv = params['ndiv']
    alpha_minus = params['alpha_minus']
    alpha_plus = params['alpha_plus']
    alpha_zero = params['alpha_zero']
    norm_factor = params['norm_factor']
    
    sum_of_means = np.zeros(sum(n_step) + 1)  # one for each time step
    sum_of_meansq = np.zeros(sum(n_step) + 1)
    norm = np.zeros(sum(n_step) + 1)
    time = np.zeros(sum(n_step) + 1)
    norm_for_m = 0

    if atom_n >= abs(m):
        n_max = find_nmax(atom_n, m)
      
        e_min, e_max, d, e, first_n0 = setup_scaled_H(eqz + emw[0], c[0], atom_n, m, n_max)

        state = np.zeros(int(n_max), dtype=complex)
        sum_coef = 0

        # now loop over j
        for j in range(int(n_max)):
            n_zero_min = np.mod(atom_n - abs(m), 2)
            n_zero = n_zero_min + 2 * j
            n_minus = (atom_n - n_zero - m) / 2
            n_plus = m + n_minus
            if n_minus >= 0 and n_plus >= 0 and n_zero >= 0:
                ln_minus = alpha_help(alpha_minus, n_minus)
                ln_plus = alpha_help(alpha_plus, n_plus)
                ln_zero = alpha_help(alpha_zero, n_zero)

                sum_ln = ln_minus + ln_plus + ln_zero

                ln_coef = sum_ln - norm_factor
                state[j] = np.exp(ln_coef)
            else:
                state[j] = np.complex(0, 0)

        # now do timestep loop
        t = 0
        t_step = 0
        mean, mean_sq = moments(state, first_n0)
        sum_of_meansq[t_step] += mean_sq
        sum_of_means[t_step] += mean
        sum_coef = find_norm(state)
        norm_for_m += sum_coef
        norm[t_step] += sum_coef
        time[t_step] = t
        t_step = t_step + 1
        for interval in range(ndiv):
            q = eqz + emw[interval]
            e_min, e_max, d, e, first_n0 = setup_scaled_H(
                q, c[interval], atom_n, m, n_max)
            dt = delta_t[interval] / (n_step[interval])  # time step
            scaled_dt = 2 * np.pi * (e_max - e_min) * dt / 2
            t_local_scaled = 0
            for i in range(n_step[interval]):
                t = t + dt
                t_local_scaled += scaled_dt
                state = chebyshev_propagator(scaled_dt, state, n_max, e, d)
                mean, mean_sq = moments(state, first_n0)
                sum_of_meansq[t_step] += mean_sq
                sum_of_means[t_step] += mean
                sum_coef = find_norm(state)
                norm_for_m += sum_coef
                norm[t_step] += sum_coef
                time[t_step] = t
                t_step += 1
    return time, sum_of_means, sum_of_meansq, norm
  
    
def solve_system(b_field, n_tot, mag, mag_range, atom_range, spinor_phase, n_0,
                 ndiv, delta_t, c, emw, n_step):

    eqz = np.real(0.0277 * b_field**2)

 
    if n_0 < 1e-20:
        alpha_zero = np.complex(0, 0)
    else:
        alpha_zero = np.sqrt(n_0) * np.exp(np.complex(0, spinor_phase / 2))

    if (n_tot - n_0 + mag) < 1e-20:
        alpha_plus = np.complex(0, 0)
    else:
        alpha_plus = np.complex(np.sqrt(mag + (n_tot - n_0 - mag) / 2), 0)

    if (n_tot - n_0 - mag) < 1e-20:
        alpha_minus = np.complex(0, 0)
    else:
        alpha_minus = np.complex(np.sqrt((n_tot - n_0 - mag) / 2), 0)

    # calculate normalization factor
    norm_factor = (abs(alpha_minus)**2 + abs(alpha_zero)
                   ** 2 + abs(alpha_plus)**2) / 2

    params = {
        'n_tot': n_tot,
        'atom_range': atom_range,
        'n_0': n_0,
        'eqz': eqz,
        'ndiv': ndiv,
        'delta_t': delta_t,
        'c': c,
        'emw': emw,
        'n_step': n_step,
        'alpha_minus': alpha_minus,
        'alpha_plus': alpha_plus,
        'alpha_zero': alpha_zero,
        'norm_factor': norm_factor
    }
   
    # generate m, atom_n pairs
    inputs = list(product([m for m in range(mag-mag_range, mag+mag_range+1)],
       [atom_n for atom_n in range(n_tot - atom_range, n_tot + atom_range + 1)]))
     
    values =  [delayed(single_calc)(i[0],i[1], params) for i in inputs]
    
    results = compute(*values, get=dask.multiprocessing.get)
    
    nr = np.array(results)
    time = np.mean(nr, axis=0)[0]
    sums = np.sum(nr, axis=0)
  
    return sums[1], sums[2], sums[3], time