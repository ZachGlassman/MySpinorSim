# Spinor Bec Simulation

## Overview
This repository contains several codes for simulation of Spinor Bose Einstein Condenstates.  They draw heavily on codes written by the Chapman group (Georgia Tech), Schwettman group (Oklahoma), and Eite Tsienga (NIST).

### CoherentStateChebyshev
This is a full quantum evolution of a coherent state with a Chebyshev time propogator.

### FockStateSimulation
Full quantum evolution of Fock state with fourth order Runge-Kutta integration

### Mean Field
Evolution of mean field equations with quasi-probability distribution to recover quantum statistics.  Multiple forms of ODE integration supported.  ODE integration uses an adaptive time-step to improve speed, decent quantum statistics for the normal experimental times we are concerned with (40k atoms, millisecond timescales) can be recovered in seconds (ensembles in the thousands).

## Usage
These codes are written in pure python with some just-in-time compilation schemes which should work on Windows, Linux, and MacOS.  The recommended python distribution is Anaconda or Miniconda, which can be obtained at [Anaconda](https://www.continuum.io/downloads) and [Miniconda](http://conda.pydata.org/miniconda.html) respectively.  The only difference in these packages is that Anaconda comes pre-installed with many common packages and miniconda has non, requiring the user to specify the packages.  Miniconda is the preferred method as it is much smaller.    Please download the latest Python 3 version (3.5 at the time of this writing).  The code will not run on python 2.

During installation make sure to allow the installer to change the path variable.  Open a terminal or command prompt and type python.  If you enter a python terminal with the correct python version, you are all good to go.

The code depends on a many python packages:

* numpy (basic numerics)
* scipy (ode integration)
* matplotlib (plotting)
* seaborn (fancy plotting)
* numba (JIT compiler)
* tqdm (progress indicator)
* colorama (fancy colors)

To install these use the following commands

```
conda install numpy scipy matplotlib seaborn numba colorama

pip install tqdm
```

To aquire the code you can either download the code from this site, or if you have git, you can use the command
```
git clone https://github.com/ZachGlassman/SpinorBECSimulation.git
```

Now you are ready to run the simulations.  The entire simulation information is stored in a configuration file.  An example file with the necessary fields is:
```
[Simulation Settings]
Name = 1-14-16-SpinorSim
run_fock = False
run_coherent = False
run_tw = True
plot = True

[Global Simulation Parameters]
N= 5000
c= 24
magnetic_field= 0
total_time = 0.015

[TW Parameters]
n_samps = 5000

[Fock Simulation Parameters]
time_step= 1e-6

[Coherent State Chebyshev Parameters]
atom_range= 4
mag_range= 4
spinor_phase= 0
n_0 = 4996
mag= 0
```

Remember **All** fields must be included in order to get a proper simulation.  The program will not necessarily crash, however, it may use default values instead.  Save this config file in the same directory as the `SimulationRunner.py` program.

To start a simulation, open a terminal or command prompt, navigate to the folder containing `SimulationRunner.py` and call
```
python SimulationRunner.py -c "configpath"
```
where "configpath" is the path to your configuration file.  If you saved it in the same directory then it is just the name of the file.

The program also has another flag `-v` which can be set to false `-v False` if you want to disable verbose output.




# To DO
* Easier support for small seeds
* Add output support
* Make magnetic field into q