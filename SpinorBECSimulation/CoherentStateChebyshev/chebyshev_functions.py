import numpy as np
import math

def find_nmax(tot, m):
    first = np.mod(tot - abs(m), 2)
    return (tot - abs(m) - first) / 2 + 1

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

def find_norm(z):
    """find complex norm^2 of a vector of complex numbers"""
    k = 0
    for i in z:
        k = k + (i * np.conj(i)).real
    return k
    
    
def setup_scaled_H(q, c, n, m, nmaxfinal):
    """function to setup tridigonal Hamiltonian if first, return d,e
    
    Parameters
    ----------
    q : float
        quadratic zeeman shift
    c : float
        c_2n, spinor interaction rate
    n : int
        number of particles
    m : int
        magnetization
    nmaxfinal : int
        deprecated
        
    Returns
    -------
    e_min : float
        minimum eigenvalue
    e_max : float
        maximum eigenvalue
    d : np.array(complex)
        diagonal elements of Hamiltonian
    e : np.array(complex)
        off diagonal elements of Hamiltonian
    first_n0 : int
        n-|m| % 2
    
    """
    first_n0 = np.mod(n-abs(m), 2)
    n0 = np.mod((n-abs(m)), 2)
    nmax = int((n-abs(m)-n0)/2 + 1)
 
    #create arrays
    e = np.zeros(int(nmax)-1)
    d = np.zeros(int(nmax))
  
    c_local = c/n
    
    #matrix elements of hamiltonian
    nm = (n - n0 - m)/2
    npp = (n - n0 + m)/2
    for j in range(int(nmax)):
        d[j] = (n-n0)*(q+0.5*c_local*(2*n0-1))
        if j < (nmax-1):
            e[j] = c_local*np.sqrt(nm*npp*(n0+2)*(n0+1))
        
        nm = nm - 1
        npp = npp - 1
        n0 = n0 + 2
    
    #estimate based on Gershgorin's circle theorem
    radius = abs(e[0])
    e_min = d[0] - radius
    e_max = d[0] + radius

    for j in range(2,int(nmax)-1):
        radius = abs(e[j-2]) + abs(e[j-1])
        e_min = min(e_min, d[j-1] - radius)
        e_max = max(e_max, d[j-1] + radius)
    radius = abs(e[nmax-2])
    e_min = min(e_min, d[nmax-1] - radius)
    e_max = max(e_max, d[nmax-1] + radius)
    
    radius = (e_max + e_min)/2
    
    for i in range(int(nmax)):
        d[i] = d[i] - radius
        
    radius = 2/(e_max-e_min)
    d = np.multiply(radius,d)
    e = np.multiply(radius,e)
    return e_min, e_max ,d ,e, first_n0
 

def hamiltonian_c(n_max, in_w, e, d):
    """apply tridiagonal real Hamiltonian matrix to a complex vector
    
    Parameters
    ----------
    n_max : int
        maximum n for cutoff
    in_w : np.array(complex)
        state in
    d : np.array(complex)
        diagonal elements of Hamiltonian
    e : np.array(complex)
        off diagonal elements of Hamiltonian
    
    Returns
    -------
    out_w : np.array(complex)
        application of Hamiltonian to vector
    """
    n_max = int(n_max)
    out_w = in_w[:n_max]*d[:n_max]
    out_w[:(n_max-1)] += e[:(n_max-1)]*in_w[1:n_max]
    out_w[1:n_max] += e[:n_max-1] * in_w[:n_max-1]
        
    return out_w

def moments(wave, n):
    """mean and variance of wavefunction
    
    Parameters
    ----------
    wave : np.array(complex)
        wavefunction
    n : int
        number of atoms
    
    Returns
    -------
    x : float
        mean of wavefunction
    x2 : float
        variance of wavefunction
    """
    nn = np.arange(n, n+2*len(wave), 2)
    Y = (wave * np.conj(wave)).real
    x = np.sum(Y * nn)
    x2 = np.sum(Y * nn * nn)
    return x, x2