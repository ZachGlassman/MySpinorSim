# -*- coding: utf-8 -*-
"""
Simulation Runner - This is a module containing a unified interface
into the 3 simulation types written.  It should be the primary way to interact
with the simulation codes.  It will allow easy comparison of results

Three types of codes and requirements
Mean Field:
    Number of atoms
    number of samples (truncated wigner)
    c
    tfinal (simulation time)
    pulses
    qu0 (optional)
    magnetic field
Fock State Full Quantum:
   Number of atoms
   c
   mag_time(for magnetic field) not currently implemented
   tauB(magnetic field decay) not currently implemented
   dt (timestep)
   magnetic field
Coherent State full Quantum:
    Number of Atoms
    magnetic field
    magnetization
    magnetic_field range loop
    atom_range loop
    spinor_phase
    n_0
    c (list)
    delta_t (list)
    ndiv (list)
    emw (list)
    n_step (list)

It will provide a unified plotting interface and variable interface
@author: Zachary Glassman
"""
import colorama
from MeanField.MeanFieldSimulation import single_simulation as mean_sim
from FullQuantumFock.FockStateSimulation import fock_sim
from CoherentStateChebyshev.spinorf import solve_system as cheby_sim
import numpy as np
import matplotlib.pyplot as plt
import time as time_mod
import seaborn

def color_text(text, color):
    """color text"""
    try:
        return getattr(colorama.Fore,color) + text + colorama.Style.RESET_ALL
    except:
        return text

class SimulationResult(object):
    """class to hold results so we can parse different simulation into
    equivalent results"""
    def __init__(self,time,rho_0,std,color, name):
        self.t = time
        self.rho = rho_0
        self.std = std
        self.name = name
        self.col = color

    def plot(self, ax):
        """plot given axis ax"""
        ax.fill_between(self.t,self.rho-self.std,self.rho+self.std, color = self.col, alpha = .2)
        ax.plot(self.t, self.rho, label = self.name, color = self.col)

    def print_information(self):
        print(self.t)
        print(self.rho)


class Simulation(object):
    """Simulation Class is a simulation for a certain set of parameters
    Will automatically use correct factors to compare to real vales"""
    def __init__(self, name, pulses = [],number = False):
        """Inititalize name and all possible parameters set to reasonable values"""
        self.name = name
        self.params = {
            'N':5000,
            'c':24,
            'n_samps':200,
            'magnetic_field':27,
            'atom_range': 20,
            'mag_range': 20,
            'spinor_phase':0,
            'n_0':4990,
            'time_step': 0.001e-3,
            'tauB':1e-3,
            'total_time':.01,
            'mag_time':0.015,
            'mag':0,
        }
        self.pulses = pulses
        self.number = number

        self.fock = False
        self.mean = False
        self.cheby = False

    def run_fock(self, verbose = True):
        """run a fock simulation with the current parameters"""
        if verbose:
            print(color_text('Running Fock State Simulation', 'CYAN'))
            ts = time_mod.time()
        time, n0, n0var = fock_sim(self.params['total_time'],
                self.params['time_step'],
                self.params['mag_time'],
                self.params['tauB'],
                self.params['N'],
                self.params['c']*4*np.pi,
                self.params['magnetic_field'])
        std = np.sqrt(n0var)
        if not self.number:
            n0 = n0/self.params['N']
            std= std/self.params['N']
        self.fock_res = SimulationResult(time, n0, std, 'red','Fock')
        self.fock = True
        if verbose:
            te = time_mod.time()
            print(color_text('Finished Fock State Simulation', 'RED'))
            print('Execution Time: {0:>4.2f}'.format(te-ts))


    def run_mean(self, verbose = True):
        """run a mean field simulation with the current parameters"""
        if verbose:
            print(color_text('Running Mean Field Simulation', 'YELLOW'))
            ts = time_mod.time()
        time, mean, std, mw = self.mean_res = mean_sim(self.params['N'],
                 self.params['n_samps'],
                 self.params['c'] * 4 * np.pi,
                 self.params['total_time'],
                 self.params['magnetic_field'],
                 self.pulses,qu0=0)
        if self.number:
            mean = mean * self.params['N']
            std = std * self.params['N']
        self.mean_res = SimulationResult(time, mean, std, 'blue','Mean')
        self.mean = True
        if verbose:
            te = time_mod.time()
            print(color_text('Finished Mean Field Simulation', 'RED'))
            print('Execution Time: {0:>4.2f}'.format(te-ts))

    def run_cheby(self, verbose = True, save = False):
        """run a chebyshev simulation with the current paramters"""
        if verbose:
            print(color_text('Running Coherent Simulation', 'MAGENTA'))
            ts = time_mod.time()
        if self.pulses == []:
            dt = .005
            c = [self.params['c']/(4*np.pi)]
            emw = [0]
            n_step = [int(self.params['total_time']/dt)]
            ndiv = 1
            delta_t = [self.params['total_time']]

        sum_of_means, sum_of_meansq, norm, time = cheby_sim(self.params['magnetic_field']*100,
                  self.params['N'],
                  self.params['mag'],
                  self.params['mag_range'],
                  self.params['atom_range'],
                  self.params['spinor_phase'],
                  self.params['n_0'],
                  ndiv,
                  delta_t,
                  c,
                  emw,
                  n_step)
        mean = sum_of_means/norm
        meansq = sum_of_meansq/norm
        std = np.sqrt(meansq - mean*mean)
        self.cheby = True
        self.cheby_res = SimulationResult(time, mean,std, 'green', 'Coherent')

        if verbose:
            te = time_mod.time()
            print('\n',color_text('Finished Coherent Simulation', 'RED'))
            print('Execution Time: {0:>4.2f}'.format(te-ts))


    def plot(self):
        if not self._has_result:
            print('Cannot plot with no simulation')
        else:
            fig, ax = plt.subplots()
            if self.fock:
                self.fock_res.plot(ax)
            if self.mean:
                self.mean_res.plot(ax)
            if self.cheby:
                self.cheby_res.plot(ax)
            ax.set_xlabel('t (s)')
            if self.number:
                ax.set_ylabel(r'$N_{m_F=0}$')
            else:
                ax.set_ylabel(r'$\rho_0$')
            ax.legend()
            print('Saving Figure')
            plt.savefig('testing.pdf')


    def _has_result(self):
        if self.fock or self.mean or self.cheby:
            return True
        else:
            return False

    def reset(self):
        self.cheby = False
        self.mean = False
        self.fock = False


if __name__ == '__main__':
    s = Simulation('ha')
    s.params['total_time'] = .015
    s.params['atom_range'] = 2
    s.params['mag_range'] = 2
    s.params['magnetic_field'] = 0#.3
    s.params['n_samps']= 5000
    s.number = True
    s.run_cheby()
    s.run_fock()
    s.run_mean()
    s.plot()
