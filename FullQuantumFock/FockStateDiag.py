from scipy.linalg import eigh as eigs
#from scipy.sparse.linalg import eigs
def generate_hamiltonian(n,m,n0,q,c):
    n_states = int((n-m)/2)
    ham = np.zeros((n_states,n_states))
    for i in range(n_states):
        n0 = (n-m-2*i)
        #first for diagonal
        ham[i,i] = c/2*(m**2 + n + n0 + 2*n*n0-2*n0**2)+q*(n-n0)
        #now for off diagonal- use try statements to catch edge case
        try:
            ham[i,i+1] = c/2*np.sqrt(n0*(n0-1)*(n+m-n0+2)*(n-m-n0+2))
        except:
            pass
        try:
            ham[i,i-1] = c/2*np.sqrt((n0+1)*(n0+2)*(n+m-n0)*(n-m-n0))
        except:
            pass
    return ham

def compute_evolution(psi_init,e,v,t_final,t_step):
    c = np.zeros(len(v))
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
