from scipy.linalg import eigh as eigs
from numpy.lib import scimath
from scipy.stats import poisson, norm
import numba
import numpy as np
from multiprocessing import Process, Queue

def generate_hamiltonian(n,m,n0,q,c):
    n_states = int((n-m)/2)
    ham = np.zeros((n_states,n_states),dtype = complex)
    for i in range(n_states):
        n0 = (n-m-2*i)
        #first for diagonal
        ham[i,i] = c/2*(m**2 + n + n0 + 2*n*n0-2*n0**2)+q*(n-n0)
        #now for off diagonal- use try statements to catch edge case
        try:
            ham[i,i+1] = c/2*scimath.sqrt(n0*(n0-1)*(n+m-n0+2)*(n-m-n0+2))
        except:
            pass
        try:
            ham[i,i-1] = c/2*scimath.sqrt((n0+1)*(n0+2)*(n+m-n0)*(n-m-n0))
        except:
            pass
    return ham

@numba.jit
def compute_evolution(psi_init,e,v,t_final,t_step):
    c = np.zeros(len(v),dtype=complex)
    for i in range(len(v)):
        c[i] = np.dot(np.conj(v[:,i]),psi_init)

    time_array = np.arange(0,t_final,t_step)
    psi = np.zeros((len(time_array),len(v)),dtype=complex)
    for t, time in enumerate(time_array):
        for i in range(len(v)):
            psi[t] += np.exp(0-1j*e[i]*time)*c[i]*v[:,i]
    return time_array , psi

def get_initial_state(n,m,n0):
    """num of pairs is (n-m)/2 - n0"""
    num = int((n-m)/2)
    psi = np.zeros(num)
    psi[int((n-m-n0)/2)]=1
    return psi

@numba.jit
def calc_n0_vals(psi,num_atoms,m):
    n0 = 0
    n0sqr = 0
    for k in range(len(psi)):
        n0 += (num_atoms-2*k-m) * abs(psi[k])**2
        n0sqr += (num_atoms-2*k-m)**2 * abs(psi[k])**2
    n0var = n0sqr - n0**2
    return n0,  np.sqrt(n0var)


def calc_state(n,m,n0,q,c,t_final,t_step):
    #compute hamiltonian and diagonalize
    ham = generate_hamiltonian(n,m,n0,2*np.pi*q,2*np.pi*c/n)
    e,v = eigs(ham)
    psi_init = get_initial_state(n,m,n0)
    t,psi = compute_evolution(psi_init,e,v,t_final,t_step)
    mean = np.zeros(len(t))
    std = np.zeros(len(t))
    for i, state in enumerate(psi):
        mean[i], std[i] = calc_n0_vals(state,n,m)
    return t, mean/n,std/n

def calc_state_with_prob(queue,n,m,n0,q,c,t_final,t_step,prob):
    #compute hamiltonian and diagonalize
    ham = generate_hamiltonian(n,m,n0,2*np.pi*q,2*np.pi*c/n)
    e,v = eigs(ham)
    psi_init = get_initial_state(n,m,n0)
    t,psi = compute_evolution(psi_init,e,v,t_final,t_step)
    mean = np.zeros(len(t))
    std = np.zeros(len(t))
    for i, state in enumerate(psi):
        mean[i], std[i] = calc_n0_vals(state,n,m)
    queue.put([t, mean/n*prob,std/n*prob])


def calc_prob(n,m,nn,mm):
    #return poisson.pmf(np.abs(nn-mm),mu=np.abs(n-m))
    a = poisson.pmf(nn,mu=n)
    mm = np.abs(m)
    if m != 0:
        b = norm.pdf(mm,loc=m,scale=np.sqrt(m))
    else:
        b = norm.pdf(mm,loc=m,scale=1)
    return a * b

def compute_coherent_state(n,m,n0,q,c,t_final,t_step,a_range,mag_range):
    """for a coherent state, we will asume that the magnetization and seed are whats conserved, NOT n0
    the probability is the multiplication of the probability for m and n"""
    #compute all the elements that need to be computed
    n_ele = np.arange(n-a_range, n + a_range+1,1)
    m_ele = np.arange(m-mag_range, m + mag_range+1,1)
    nm = [[nn,mm] for nn in n_ele for mm in m_ele]
    #loop through and compute details, save to multidimensional arrays
    mean = np.zeros((len(nm),int(t_final/t_step)))
    std =  np.zeros((len(nm),int(t_final/t_step)))
    total_prob = 0
    pairs = n-n0
    for i, state in enumerate(nm):
        prob = calc_prob(n,m,state[0],state[1])
        t, temp_m, temp_s = calc_state(int(state[0]),int(state[1]),int(state[0]-pairs-state[1]),q,c,t_final,t_step)
        mean[i] = temp_m * prob
        std[i] = temp_s * prob
        total_prob += prob
    return t, np.sum(mean,axis=0)/total_prob, np.sum(std,axis=0)/total_prob

def compute_coherent_state_multi(n,m,n0,q,c,t_final,t_step,a_range,mag_range):
    """for a coherent state, we will asume that the magnetization and seed are whats conserved, NOT n0
    the probability is the multiplication of the probability for m and n"""
    #compute all the elements that need to be computed
    n_ele = np.arange(n-a_range, n + a_range+1,1)
    m_ele = np.arange(m-mag_range, m + mag_range+1,1)
    nm = [[nn,mm] for nn in n_ele for mm in m_ele]
    pairs = n-n0
    #loop through and compute details, save to multidimensional arrays
    mean = np.zeros((len(nm),int(t_final/t_step)))
    std =  np.zeros((len(nm),int(t_final/t_step)))
    total_prob = 0
    #set up multiprocessing
    queue = Queue(5)
    procs = {}
    for i, state in enumerate(nm):
        prob = calc_prob(n,m,state[0],state[1])
        total_prob += prob
        procs[i] = Process(target = calc_state_with_prob,
          args = (queue, int(state[0]),int(state[1]),int(state[0]-pairs-state[1]),q,c,t_final,t_step,prob))
        procs[i].start()
    #get answers
    for i in range(len(nm)):
        t, mean[i], std[i] = queue.get()

    return t, np.sum(mean,axis=0)/total_prob, np.sum(std,axis=0)/total_prob




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    N = 100
    a_range = 10
    m_range = 10
    t1,m1,s1 = calc_state(N,0,N,-5,30,1,.005)
    print('fock calculated')
    ts = time.time()
    t,m,s = compute_coherent_state(N,0,N,-5,30,1,.005,a_range,m_range)
    tm = time.time()
    print('serial calculated')
    t2,m2,s2 = compute_coherent_state_multi(N,0,N,-5,30,1,.005,a_range,m_range)
    te = time.time()
    print('serial',tm-ts)
    print('parallel',te-tm)
    print('parallel calculated')
    plt.plot(t,m,label='coherent')
    plt.plot(t1,m1,label='fock')
    plt.plot(t2,m2,label='multi')
    plt.legend()
    plt.show()
