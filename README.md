# Spinor Bec Simulation

This repository contains several codes for simulation of Spinor Bose Einstein Condenstates.  They draw heavily on codes written by the Chapman group (Georgia Tech), Schwettman group (Oklahoma), and Eite Tsienga (NIST).

## CoherentStateChebyshev
This is a full quantum evolution of a coherent state with a Chebyshev time propogator.

## FockStateSimulation
Full quantum evolution of Fock state with fourth order Runge-Kutta integration

## Mean Field
Evolution of mean field equations with quasi-probability distribution to recover quantum statistics.  Multiple forms of ODE integration supported.  ODE integration uses an adaptive time-step to improve speed, decent quantum statistics for the normal experimental times we are concerned with (40k atoms, millisecond timescales) can be recovered in seconds (ensembles in the thousands).

## Comparisons
The Mean Field and Fock State simulations are set up in the same scheme as one another.  However, the CoherentState propogation has some definition changes.  They are summed up in the following table

Parameter | Mean Field | Coherent
---|---|---
c| 4 pi c |  c
q | pi q | q
