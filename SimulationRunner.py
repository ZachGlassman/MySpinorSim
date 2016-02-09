# -*- coding: utf-8 -*-
"""
Simulation Runner - This is a module containing a unified interface
into the 3 simulation types written.  It should be the primary way to interact
with the simulation codes.  It will allow easy comparison of results.

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
#uncomment if using only terminal access
#import matplotlib
#matplotlib.use('Agg')
from MeanField.MeanFieldSimulation import single_simulation as mean_sim
from FullQuantumFock.FockStateSimulation import fock_sim
from CoherentStateChebyshev.spinorf import solve_system as cheby_sim_s
from CoherentStateChebyshev.spinorf_multicore import solve_system as cheby_sim_p
import numpy as np
import matplotlib.pyplot as plt
import time as time_mod
import configparser
import argparse
from numpy.lib import scimath
import seaborn

#parallel or serial
cheby_sim = cheby_sim_s

def color_text(text, color):
    """Function color text
    :param data: text to color
    :type data: string
    :param color: color
    :type color: string
    """
    try:
        return getattr(colorama.Fore,color) + text + colorama.Style.RESET_ALL
    except:
        return text


class SimulationResult(object):
    """class to hold results so we can parse different simulation into
    equivalent results"""
    def __init__(self,time,rho_0,std,color, name, init_norm = None):
        self.t = time
        self.rho = rho_0
        self.std = std
        self.name = name
        self.col = color
        self.q = False
        self.init_norm = init_norm

    def plot(self, ax):
        """plot given axis ax"""
        ax.fill_between(self.t,self.rho-self.std,self.rho+self.std, color = self.col, alpha = .2)
        ax.plot(self.t, self.rho, label = self.name, color = self.col)

    def plot_ryan(self, ax):
        ax.plot(self.t, self.rho, label = self.name, color = self.col)
        ax.plot(self.t, self.std, label = self.name + 'std', color = self.col,linestyle = '--')

    def plot_no_color(self, ax, col):
        ax.plot(self.t, self.rho, label = self.name, color = col)
        ax.plot(self.t, self.std, label = self.name + 'std', color = col,linestyle = '--')

    def save(self, name):
        """function to save results to file"""
        try:
            with open('{0}_{1}_results.txt'.format(name,self.name),'w') as f:
                f.write('{0:10}{1:10}{2:10}{3:10}\n'.format('Time','Mean','STD','NORM'))
                for i, time in enumerate(self.t):
                    f.write('{:<20.8f}{:<20.8f}{:<20.8f}{:<20.8f}\n'.format(time,self.rho[i],self.std[i],self.init_norm[i]))
        except:
            with open('{0}_{1}_results.txt'.format(name,self.name),'w') as f:
                f.write('{0:10}{1:10}{2:10}\n'.format('Time','Mean','STD'))
                for i, time in enumerate(self.t):
                    f.write('{:<20.8f}{:<20.8f}{:<20.8f}\n'.format(time,self.rho[i],self.std[i]))

    def print_information(self):
        print(self.t)
        print(self.rho)

def q_to_b(q):
    return scimath.sqrt(q/277*(2*np.pi)**3)/(2*np.pi)

class Simulation(object):
    """Simulation Class is a simulation for a certain set of parameters
    Will automatically use correct factors to compare to real vales"""
    def __init__(self, name, pulses = [], number = False):
        """Inititalize name and all possible parameters set to reasonable values"""
        self.name = name
        self.params = {
            'n': 5000,
            'c': 24,
            'n_samps': 200,
            'magnetic_field': 27,
            'atom_range': 20,
            'mag_range': 20,
            'spinor_phase': 0,
            'n0':4998,
            'n1':0,
            'nm1':0,
            'time_step': 0.001e-3,
            'tauB': 1e-3,
            'total_time': .01,
            'mag_time': 0.015,
            'mag': 0,
        }
        self.pulses = pulses
        self.number = number

        self.fock = False
        self.mean = False
        self.cheby = False
        self.verbose = False

    def transform_q(self):
        self.params['magnetic_field'] = q_to_b(self.params['q'])

    def run_fock(self):
        """run a fock simulation with the current parameters"""
        if self.verbose:
            print(color_text('Running Fock State Simulation', 'CYAN'))
            ts = time_mod.time()
        npairs = self.params['n1']/2

        time, n0, n0var, init_norm = fock_sim(self.params['total_time'],
                self.params['time_step'],
                self.params['mag_time'],
                self.params['tauB'],
                int(self.params['n']),
                self.params['c']*2*np.pi,
                self.params['magnetic_field'],
                npairs)


        std = np.sqrt(n0var)
        if not self.number:
            n0 = n0/self.params['n']
            std= std/self.params['n']
        self.fock_res = SimulationResult(time, n0, std, 'red','Fock',init_norm=init_norm)
        self.fock = True
        if self.verbose:
            te = time_mod.time()
            print(color_text('Finished Fock State Simulation', 'RED'))
            print('Execution Time: {0:>4.2f}'.format(te-ts))


    def run_mean(self):
        """run a mean field simulation with the current parameters"""
        if self.verbose:
            print(color_text('Running Mean Field Simulation', 'YELLOW'))
            ts = time_mod.time()
        
        time, mean, std, mw = mean_sim(int(self.params['n1']),
                 int(self.params['n0']),
                 int(self.params['nm1']),
                 self.params['spinor_phase'],
                 int(self.params['n_samps']),
                 self.params['c']*2*np.pi,
                 self.params['total_time']+.05*self.params['total_time'],
                 self.params['magnetic_field'],
                 self.pulses,
                 qu0=0)

        if self.number:
            N = self.params['n0'] + self.params['n1'] + self.params['nm1']
            mean = mean * N
            std = std * N
        self.mean_res = SimulationResult(time, mean, std, 'blue','Mean')
        self.mean = True
        if self.verbose:
            te = time_mod.time()
            print(color_text('Finished Mean Field Simulation', 'RED'))
            print('Execution Time: {0:>4.2f}'.format(te-ts))

    def run_cheby(self,save = False):
        """run a chebyshev simulation with the current paramters"""
        if self.verbose:
            print(color_text('Running Coherent Simulation', 'MAGENTA'))
            ts = time_mod.time()
        if self.pulses == []:
            dt = .001
            c = [self.params['c']]
            emw = [0]
            mag_field = self.params['magnetic_field']*100/np.sqrt(2*np.pi)
            n_step = [int(self.params['total_time']/dt)]
            ndiv = 1
            delta_t = [self.params['total_time']]
        else:
            dt= [.001,.0001,.001]
            c= self.params['c']
            ndiv = len(c)
            emw = self.params['q']
            mag_field = 0
            n_step = [int(self.params['total_time'][i]/dt[i]) for i in range(len(dt))]
            delta_t = [i for i in self.params['total_time']]

        sum_of_means, sum_of_meansq, norm, time = cheby_sim(mag_field,
                  int(self.params['n']),
                  int(self.params['mag']),
                  int(self.params['mag_range']),
                  int(self.params['atom_range']),
                  self.params['spinor_phase'],
                  int(self.params['n0']),
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
        if self.verbose:
            te = time_mod.time()
            print('\n',color_text('Finished Coherent Simulation', 'RED'))
            print('Execution Time: {0:>4.2f}'.format(te-ts))


    def plot(self, col = False, region=False):

        if not self._has_result:
            print('Cannot plot with no simulation')
        if col != False:
            ax = plt.gca()
            if self.fock:
                self.fock_res.plot_no_color(ax,col=col)
            if self.mean:
                self.mean_res.plot_no_color(ax,col=col)
            if self.cheby:
                self.cheby_res.plot_no_color(ax,col=col)
            ax.set_xlabel('t (s)')
            if self.number:
                ax.set_ylabel(r'$N_{m_F=0}$')
            else:
                ax.set_ylabel(r'$\rho_0$')
            ax.legend()
        elif region == True:
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
        else:
            fig, ax = plt.subplots()
            if self.fock:
                self.fock_res.plot_ryan(ax)
            if self.mean:
                self.mean_res.plot_ryan(ax)
            if self.cheby:
                self.cheby_res.plot_ryan(ax)
            ax.set_xlabel('t (s)')
            if self.number:
                ax.set_ylabel(r'$N_{m_F=0}$')
            else:
                ax.set_ylabel(r'$\rho_0$')
            ax.legend()


    def _has_result(self):
        if self.fock or self.mean or self.cheby:
            return True
        else:
            return False

    def _reset(self):
        self.cheby = False
        self.mean = False
        self.fock = False



def single_simulation(config, args):
    #keys for configuarion file
    sims = 'Simulation Settings'
    gsp = 'Global Simulation Parameters'
    tw = 'TW Parameters'
    fsp = 'Fock Simulation Parameters'
    cscp = 'Coherent State Chebyshev Parameters'
    #create simulation objects
    name = config[sims].get('Name','sim')
    s = Simulation(name)
    if args.verbose == True:
        s.verbose = True
    #loop through each one
    print('Parameter Settings:')
    for con in [config[gsp],config[tw],config[fsp],config[cscp]]:
        for key in con:
            s.params[key] = float(con[key])
            if args.verbose == True:
                print('  {0:<15} set to {1}'.format(key,con[key]))

    #now check for q or magnetic field
    if s.params['q']:
        s.q = True
        #now mock mock the magnetic field such that we get q
        s.transform_q()
        print(s.params['magnetic_field'])
    #now run simulations
    if args.verbose == True:
        print(''.join('#' for i in range(20)))
        print('Simulations Set Up - Starting Numerics')
    ts = time_mod.time()
    s.number = True
    if config[sims].getboolean('run_coherent', False):
        s.run_cheby()
        if config[sims].getboolean('save', False):
            s.cheby_res.save(name)
    if config[sims].getboolean('run_fock', False):
        s.run_fock()
        if config[sims].getboolean('save', False):
            s.fock_res.save(name)
    if config[sims].getboolean('run_tw', False):
        s.run_mean()
        if config[sims].getboolean('save', False):
            s.mean_res.save(name)
    te = time_mod.time()
    if args.verbose == True:
        mins, secs = divmod(te-ts, 60)
        hours, mins = divmod(mins, 60)
        print(''.join('#' for i in range(20)))
        out_form = 'Total Sim Time {0:02.0f}h:{1:02.0f}m:{2:02.2f}s'
        print(out_form.format(hours,mins,secs))

    if config[sims].getboolean('plot',True):
        s.plot()
        print('Saving Figure','{0}_plot.pdf'.format(s.name))
        plt.savefig('{0}_plot.pdf'.format(s.name))
        plt.show()

    if args.verbose == True:
        print(''.join('#' for i in range(20)))
    print('Simulation Complete')

def main(config,args):
    single_simulation(config,args)

#Should write function where input not from connfig file


if __name__ == '__main__':
    #add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                       dest='verbose',
                       action='store',
                       default = True,
                       help='verbose output (default True)')
    parser.add_argument('-c',
                        dest ='config',
                        action = 'store',
                        help = 'Path to config file',
                        required = True)
    args = parser.parse_args()
    #get configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    main(config, args)
