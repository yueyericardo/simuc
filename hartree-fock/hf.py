import sympy as sp
from sympy import oo
import numpy as np
from itertools import product
from scipy.linalg import eigh
from sympy import diff
import time

r, r1, r2, zeta, zeta1, zeta2 = sp.symbols("r, r1, r2, zeta, zeta1, zeta2")
n = sp.Symbol('n', integer=True)


# Define STO function
def STO(zeta, n, r=r, R=0):
    f = (r-R)**(n-1)*sp.exp(-zeta*(r-R))
    N = sp.sqrt(1 / sp.integrate(f*f*r*r, (r, 0, +oo)))
    return N * f


# S Overlap Integrate
def S_int(f1, f2):
    return sp.integrate(f1*f2*r*r, (r, 0, +oo))


# H core = kinetics energy + electron and nuclear potential energy
def H_int(f1, f2, Z):
    return sp.integrate(f1 * (- ((1 / 2) * (1 / r) * diff(diff(r * f2, r), r)) - ((Z / r) * f2)) * r * r, (r, 0, +oo))


# Returns the core hamiltonian matrix
def H_matrix(fs, Z):

    H = np.zeros((len(fs), len(fs)))
    for i in range(len(fs)):
        for j in range(len(fs)):
            H[i, j] = H_int(fs[i], fs[j], Z)

    return H


# Returns the overlap matrix
def S_matrix(fs):

    S = np.zeros((len(fs), len(fs)))
    for i in range(len(fs)):
        for j in range(len(fs)):
            S[i, j] = S_int(fs[i], fs[j])

    return S


def Repulsion_electron(zetas):
    f1 = STO(zetas[0][0], zetas[0][1], r1)
    f2 = STO(zetas[1][0], zetas[1][1], r1)
    f3 = STO(zetas[2][0], zetas[2][1], r2)
    f4 = STO(zetas[3][0], zetas[3][1], r2)

    B = (1 / r1) * sp.integrate(f3 * f4 * r2 * r2, (r2, 0, r1)) + sp.integrate((1 / r2) * f3 * f4 * r2 * r2, (r2, r1, +oo))
    A = sp.integrate(f1 * f2 * r1 * r1 * B, (r1, 0, +oo))
    return A


# Calculates Density matrix
def P_matrix(Co):

    P = np.zeros([Co.shape[0], Co.shape[0]])

    for t in range(Co.shape[0]):
        for u in range(Co.shape[0]):
            for j in range(int(Co.shape[0]/2)):
                P[t][u] += 2 * Co[t][j] * Co[u][j]
    return P


def R_matrix(zetas):
    start = time.time()
    R = np.zeros((len(zetas), len(zetas), len(zetas), len(zetas)))

    rs = list(product(range(len(zetas)), repeat=2))
    tu = list(product(range(len(zetas)), repeat=2))

    for r, s in rs:
        for t, u in tu:
            R[r, s, t, u] = Repulsion_electron((zetas[r], zetas[s], zetas[t], zetas[u]))
    stop = time.time()
    print('time Repu: {:.1f} s'.format(stop-start))
    return R


# Caculate G Matrix
def G_matrix(zetas, Co, R):

    G = np.zeros((Co.shape[0], Co.shape[0]))

    P = P_matrix(Co)

    rs = list(product(range(Co.shape[0]), repeat=2))
    tu = list(product(range(Co.shape[0]), repeat=2))

    for r, s in rs:
        g = 0
        for t, u in tu:
            int1 = R[r, s, t, u]
            int2 = R[r, u, t, s]
            # print('({0}{1}|{2}{3}): {4}'.format(r, s, t, u, int1))
            g += P[t, u] * (int1 - 0.5 * int2)
        G[r, s] = g
    return G


# Returns the Fock matrix
def F_matrix(fs, Z, zetas, Co, R):
    return H_matrix(fs, Z) + G_matrix(zetas, Co, R)


# slove secular equation, return the energy and improved coeffients
# the energy here is molecular orbital energies
def secular_eqn(F, S):
    ei, C = eigh(F, S)
    return ei, C


# return energy of atom
def get_E0(e, P, H):

    E0 = 0
    for i in range(int(e.shape[0]/2)):
        E0 += e[i].real
    E0 = E0 + 0.5 * (P * H).sum()
    return E0
