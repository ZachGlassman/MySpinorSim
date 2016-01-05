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
from MeanField.MeanFieldSimulation import single_simulation as mean_sim
from FullQuantumFock.FockStateSimulation import fock_sim
from CoherentStateChebyshev.spinorf import solve_system as cheby_sim

class Simulation(object):
    """Simulation Class is a simulation for a certain set of parameters"""
    def __init__(self, name):
        """Inititalize name and all possible parameters set to reasonable values"""
        self.name = name
        self.params = {
            'N':40000,
            'c':30,
            'n_samps':200,
            'magnetic_field':27
        }
    def run_fock(self):
        pass
    def run_mean(self):
        pass
    def run_cheby(self):
        pass

if __name__ == '__main__':
    print('cool stuff homie')
