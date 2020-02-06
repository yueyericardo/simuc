import time
import scipy
import scipy.linalg
import numpy as np
import sympy as sp
from sympy import oo
from sympy import diff


r, r1, r2, zeta = sp.symbols("r, r1, r2, zeta")
n = sp.Symbol('n', integer=True)


# --------- PART 1 Define Slator Type Orbital ---------

def STO(zeta, n, r=r):
    """
    Define a Slator Type Orbital function using sympy.

    INPUT:
    zeta: zeta for the STO.
    n: principle quantum number for the STO.
    """
    f = r ** (n - 1) * sp.exp(-zeta * r)
    N = sp.sqrt(1 / sp.integrate(4 * sp.pi * f * f * r * r, (r, 0, +oo)))
    return N * f


# --------- PART 2 Compute integrals between STO functions ---------

def S_int(fr, fs):
    """
    Compute overlap integral between two STO functions.
    """
    return sp.integrate(fr * fs * 4 * sp.pi * r * r, (r, 0, +oo))


def H_int(fr, fs, Z):
    """
    Compute H_core integral between two STO functions.
    H_core = electron kinetics energy + electron nuclear potential energy

    INPUT:
    Z: Nuclear charge
    """
    T = - ((1 / 2) * (1 / r) * diff(diff(r * fs, r), r))
    V = - (Z / r) * fs
    return sp.integrate(fr * (T + V) * 4 * sp.pi * r * r, (r, 0, +oo))


def R_int(four_bfs):
    """
    Compute electron-electron repulsion integral.

    INPUT:
    four_bfs: an array contain 4 basis functions
    """
    f1, f2, f3, f4 = four_bfs

    f1 = f1.subs(r, r1)
    f2 = f2.subs(r, r1)
    f3 = f3.subs(r, r2)
    f4 = f4.subs(r, r2)

    B = (1 / r1) * sp.integrate(f3 * f4 * 4 * sp.pi * r2 * r2, (r2, 0, r1)) + sp.integrate((1 / r2) * f3 * f4 * 4 * sp.pi * r2 * r2, (r2, r1, +oo))
    return sp.integrate(f1 * f2 * 4 * sp.pi * r1 * r1 * B, (r1, 0, +oo))


# --------- PART 3 Build matrix ---------

def S_matrix(bfs):
    """
    Compute overlap matrix S.

    INPUT:
        fs: basis functions
    OUTPUT:
        S: Overlap matrix
    """
    num_bfs = len(bfs)
    S = np.zeros((num_bfs, num_bfs))

    for r in range(num_bfs):
        for s in range(num_bfs):
            S[r, s] = S_int(bfs[r], bfs[s])

    return S


def H_matrix(bfs, Z):
    """
    Compute the core hamiltonian matrix H.
    H_core = electron kinetics energy + electron nuclear potential energy

    INPUT:
        bfs: basis functions
        Z: nuclear charge
    OUTPUT:
        H: core hamiltonian matrix
    """
    num_bfs = len(bfs)
    H = np.zeros((num_bfs, num_bfs))

    for r in range(num_bfs):
        for s in range(num_bfs):
            H[r, s] = H_int(bfs[r], bfs[s], Z)

    return H


def R_matrix(bfs):
    """
    Compute the electron repulsion integral matrix R.

    INPUT:
        fs: basis functions
    OUTPUT:
        R: repulsion matrix
    """
    start = time.time()
    num_bfs = len(bfs)
    R = np.zeros((num_bfs, num_bfs, num_bfs, num_bfs))

    for r in range(num_bfs):
        for s in range(num_bfs):
            for t in range(num_bfs):
                for u in range(num_bfs):
                    R[r, s, t, u] = R_int([bfs[r], bfs[s], bfs[t], bfs[u]])

    stop = time.time()
    print('time Repu: {:.1f} s'.format(stop-start))
    return R


def P_matrix(Co, N):
    """
    Compute density matrix P.

    INPUT:
        Co: coefficents matrix
        N: num of electrons
    OUTPUT:
        P: repulsion matrix
    """
    P = np.zeros([Co.shape[0], Co.shape[0]])

    for t in range(Co.shape[0]):
        for u in range(Co.shape[0]):
            for j in range(int(N/2)):
                P[t, u] += 2 * Co[t, j] * Co[u, j]
    return P


def G_matrix(P, R):
    """
    Compute G matrix.
    G =  coulombic repulsion energy + exchange energy

    INPUT:
        P: density matrix
        R: electron repulsion matrix
    OUTPUT:
        G: repulsion matrix
    """
    num_bfs = P.shape[0]
    G = np.zeros((num_bfs, num_bfs))

    for r in range(num_bfs):
        for s in range(num_bfs):
            g = 0
            for t in range(num_bfs):
                for u in range(num_bfs):
                    int1 = R[r, s, t, u]
                    int2 = R[r, u, t, s]
                    g += P[t, u] * (int1 - 0.5 * int2)
            G[r, s] = g

    return G


def F_matrix(H, G):
    """
    Compute fock matrix F.
    F =  H_core + G
    """
    return H + G


# --------- PART 4 Other Equations ---------

def secular_eqn(F, S):
    """
    Slove secular equation, return the MO energies (eigenvalue) and improved coeffients (eigenvector)

    INPUT:
        F: fock matrix or h_core matrix
        S: overlap integral
    OUTPUT:
        ei: eigenvalue
        C: eigenvector
    """
    ei, C = scipy.linalg.eigh(F, S)
    return ei, C


def energy_tot(e, N, P, H, Vnn=0):
    """
    Compute the total energy.

    INPUT:
    e: MO energies
    N: num of electrons
    P: density matrix
    H: h_core matrix
    Vnn: nuclear nuclear repulsion energy, for atom is 0
    """
    e_tot = 0
    for i in range(int(N/2)):
        e_tot += e[i].real
    e_tot = e_tot + 0.5 * (P * H).sum() + Vnn
    return e_tot


# --------- PART 5 Utils ---------

def print_info(S, H, e, Co, P, hf_e, start, stop, delta_e=0, verbose=False):
    """
    Print information while doing SCF interations.
    """
    if(verbose):
        # overlap
        print('Overlap:')
        print(S)

        # hamiltonian
        print('Core hamiltonian:')
        print(H)

        # Co
        print('Coefficients:')
        print(Co)

        # density
        print('Density matrix:')
        print(P)

        # MOs
        print('MO energies:')
        message = ', '
        m_list = ['e{} = {:0.3f}'.format(i+1, x) for i, x in enumerate(e)]
        message = message.join(m_list)
        print(message)

    print('HF energy: {:0.5f} (hartree) = {:0.5f} (eV)'.format(hf_e, hf_e*27.211))
    if delta_e != 0:
        print('dE       : {:.2e}'.format(delta_e))
    print('time used: {:.1f} s'.format(stop-start))


def compare(cal, ref, tol=1.0e-4):
    """
    Compare calculated result with reference data.
    """
    delta = np.abs(ref - cal)
    if delta < tol:
        message = '\33[32m' + 'PASSED' + '\x1b[0m'
    else:
        message = '\033[91m' + 'FAILED' + '\033[0m'
    print('-' * 32, message, '-' * 33)
    print('cal: {:.7f}, ref: {:.7f}\n\n'.format(cal, ref))
