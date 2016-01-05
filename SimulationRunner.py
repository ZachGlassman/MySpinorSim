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

def color_text(text, color):
    """color text"""
    try:
        return getattr(colorama.Fore,color) + text + colorama.Style.RESET_ALL
    except:
        return text


class Simulation(object):
    """Simulation Class is a simulation for a certain set of parameters
    Will automatically use correct factors to compare to real vales"""
    def __init__(self, name):
        """Inititalize name and all possible parameters set to reasonable values"""
        self.name = name
        self.params = {
            'N':5000,
            'c':30,
            'n_samps':200,
            'magnetic_field':27,
            'atom_range': 20,
            'mag_range': 20,
            'spinor_phase':0,
            'n_0':4998,
            'time_step': 0.001e-4,
            'tauB':1e-3,
            'total_time':.01,
            'mag_time':0.015,
        }
        self.pulses = []
        self.fock = False
        self.mean = False
        self.cheby = False

    def run_fock(self):
        """run a fock simulation with the current parameters"""
        print(color_text('Running Fock State Simulation', 'CYAN'))
        fock_sim(self.params['total_time'],
                self.params['time_step'],
                self.params['mag_time'],
                self.params['tauB'],
                self.params['N'],
                self.params['c'],
                self.params['magnetic_field']/100)
        self.fock = True
        print(color_text('Finished Fock State Simulation', 'RED'))

    def run_mean(self):
        """run a mean field simulation with the current parameters"""
        print(color_text('Running Mean Field Simulation', 'YELLOW'))
        c = self.params['c'] * 4 * np.pi
        B = self.params['magnetic_field']
        mean_sim(self.params['N'],
                 self.params['n_samps'],
                 c,
                 self.params['total_time'],
                 B,
                 self.pulses,qu0=0)
        self.mean = True
        print(color_text('Finished Mean Field Simulation', 'RED'))

    def run_cheby(self):
        """run a chebyshev simulation with the current paramters"""
        print(color_text('Running Coherent Simulation', 'MAGENTA'))
        self.cheby = True
        print(color_text('Finished Coherent Simulation', 'RED'))

    def plot(self):
        if not self.fock and not self.mean and not self.cheby:
            print('Cannot plot with no simulation')
        else:
            fig, ax = plt.subplots()
            if self.fock:
                print('Did fock')
            if self.mean:
                print('Did mean')
            if self.cheby:
                print('did cheby')

if __name__ == '__main__':
    s = Simulation('ha')
    s.params['total_time'] = .001
    s.run_fock()
    s.run_mean()
    s.run_cheby()
    s.plot()
