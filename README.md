# MySpinorSim
SpinorBecSimulation

This repository contains several codes for simulation of Spinor Bose Einstein Condenstates.  They draw heavily on codes written by the Chapman group (Georgia Tech), Schwettman group (Oklahoma), and Eite Tsienga (NIST).

## CoherentStateChebyshev
This is a full quantum evolution of a coherent state with a chebyshev time propogator.

## FockStateSimulation
Full quantum evolution of Fock state with fourth order Runge-Kutta integration

## Mean Field
Evolution of mean field equations with quasi-probability distribution to recover quantum statistics.  Multiple forms of ODE integration supported.

$$\begin{align}
    i \dot{\zeta}_1&= (p_1 B + q_1 B^2 + q\mu_1(t))\zeta_1+c\left[(\zeta_1^*\zeta_1+\zeta_0^*\zeta_0-\zeta_{-1}^*\zeta_{-1})\zeta_1+\zeta_0\zeta_0\zeta_{-1}^*)\right]\\
    i\dot{\zeta}_0&=(p_0 B + q_0 B^2 + q\mu_0(t))\zeta_0+c\left[(\zeta_1^*\zeta_1 + \zeta_{-1}^*\zeta_{-1})\zeta_0 + 2\zeta_1\zeta_{-1}\zeta_0^*\right]\\
    i \dot{\zeta}_{-1}&= (p_{-1} B + q_{-1} B^2 + q\mu_{-1}(t))\zeta_{-1}+c\left[(\zeta_{-1}^*\zeta_{-1}+\zeta_0^*\zeta_0-\zeta_{1}^*\zeta_{1})\zeta_{-1}+\zeta_0\zeta_0\zeta_{1}^*)\right]\\
\end{align}
$$

## Comparisons
The Mean Field and Fock State simulations are set up in the same scheme as one another.  However, the CoherentState propogation has some definition changes.  They are summed up in the following table

Parameter | Mean Field | Coherent
---|---|---
$c$ | $4 \pi c$ |  $c$
$q$ | $\pi q$ | $q$