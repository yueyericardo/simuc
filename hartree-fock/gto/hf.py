import time
import scipy
import scipy.linalg
import numpy as np
import sympy as sp
from scipy.special import erf

# --------- PART 1 STO, CGF, Molecule ---------


class Molecule():
    """
    Parse and build a molecule
    """
    def __init__(self, string):

        zeta_dict = {'H': [1.24], 'He': [2.0925]}
        principle_quantum_number = {'H': 1, 'He': 1}
        charge_dict = {'H': 1, 'He': 2}

        self.num_atoms, self.species, self.coordinates, self.charge = self.mol_parse(string)
        self.zetas = [zeta_dict[a] for a in self.species]
        self.charges = [charge_dict[a] for a in self.species]
        self.num_electron = np.sum(self.charges) - self.charge
        self.ns = [principle_quantum_number[a] for a in self.species]
        self.cgfs = []
        self.stos = []
        for i, a in enumerate(self.species):
            for z in self.zetas[i]:
                self.cgfs.append(CGF(z, self.ns[i], self.coordinates[i]))
                self.stos.append(STO(z, self.ns[i]))

    def mol_parse(self, string):
        atoms = string.split('\n')

        number_of_atoms = 0
        atom_type = []
        atom_coordinates = []
        for a in atoms:
            if a == '':
                continue
            split = a.split()
            if len(split) == 0:
                continue
            if len(split) == 1:
                charge = int(split[0])
                continue
            atom = split[0]
            coordinates = np.array([float(split[1]), float(split[2]), float(split[3])])

            atom_type.append(atom)
            atom_coordinates.append(coordinates)
            number_of_atoms += 1

        return number_of_atoms, atom_type, atom_coordinates, charge


class CGF():
    """
    Build a contracted gaussian function (STO-3G), which is a linear
    combination of three primitive gaussian function.
    """
    def __init__(self, zeta, n, coordinates):
        # Gaussian contraction coefficients (pp157)
        # Going up to 2s orbital (W. J. Hehre, R. F. Stewart, and J. A. Pople. J. Chem. Phys. 51, 2657 (1969))
        # Row represents 1s, 2s etc...
        contract_cos = np.array([[0.444635, 0.535328, 0.154329],
                                 [0.700115, 0.399513, -0.0999672]])

        # Gaussian orbital exponents (pp153)
        # Going up to 2s orbital (W. J. Hehre, R. F. Stewart, and J. A. Pople. J. Chem. Phys. 51, 2657 (1969))
        alphas = np.array([[0.109818, 0.405771, 2.22766],
                          [0.0751386, 0.231031, 0.994203]])

        self.co = contract_cos[n-1]
        self.alpha = alphas[n-1] * zeta ** 2
        self.n = n
        self.zeta = zeta
        self.coordinate = coordinates
        self.gtos = []
        for i, a in enumerate(self.alpha):
            gto = self.get_gto(a, self.n)
            self.gtos.append(gto)
        self.cgf = 0
        for i, g in enumerate(self.gtos):
            self.cgf = self.cgf + self.co[i] * g

    def get_gto(self, alpha, n):
        r = sp.Symbol('r')
        f = (2*alpha/np.pi)**0.75 * r**(n-1)*sp.exp(-alpha*r*r)
        return f

    def __str__(self):
        return str(self.cgf)

    def __repr__(self):
        return str(self.cgf)


class STO():
    """
    Build a Slator Type Orbital.
    """
    def __init__(self, zeta, n):
        self.zeta = zeta
        self.n = n
        self.sto = self.get_sto(self.zeta, self.n)

    def get_sto(self, zeta, n):
        r = sp.Symbol('r')
        f = r ** (n - 1) * sp.exp(-zeta * sp.Abs(r))
        N = sp.sqrt(1 / sp.integrate(4 * sp.pi * f * f * r * r, (r, 0, +sp.oo)))
        return f * N

    def __str__(self):
        return str(self.sto)

    def __repr__(self):
        return str(self.sto)


# --------- PART 2 Compute integrals between two primitive gaussian functions  ---------
# Reference:
# 1. https://github.com/aced125/Hartree_Fock_jupyter_notebook
# 2. Szabo, Ostlund, Modern Quantum Chemistry, 411-416, https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-11-13-Gaussian_Integral.pdf


def gauss_product(A, B):
    """
    The product of two Gaussians gives another Gaussian. (pp411)

    INPUT:
    A: (gaussian_A alpha, gaussian_A coordinate)
    B: (gaussian_B alpha, gaussian_B coordinate)

    OUTPUT:
    p: New gaussian's alpha
    diff: squared difference of the two coordinates
    K: New gaussian's prefactor
    Rp: New gaussian's coordinate
    """
    a, Ra = A
    b, Rb = B
    p = a + b
    diff = np.linalg.norm(Ra - Rb) ** 2
    N = (4 * a * b / (np.pi ** 2)) ** 0.75
    K = N * np.exp(-a * b / p * diff)
    Rp = (a * Ra + b * Rb) / p

    return p, diff, K, Rp


def overlap(A, B):
    """
    Compute overlap integral between two primitive gaussian functions.

    INPUT:
    A: (gaussian_A alpha, gaussian_A coordinate)
    B: (gaussian_B alpha, gaussian_B coordinate)
    """
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (np.pi/p)**1.5
    return prefactor*K


def kinetic(A, B):
    """
    Compute kinetic integral between two primitive gaussian functions.

    INPUT:
    A: (gaussian_A alpha, gaussian_A coordinate)
    B: (gaussian_B alpha, gaussian_B coordinate)
    """
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (np.pi / p) ** 1.5

    a, Ra = A
    b, Rb = B
    reduced_exponent = a * b / p
    return reduced_exponent * (3 - 2 * reduced_exponent * diff) * prefactor * K


def Fo(t):
    """
    Fo function for calculating potential and e-e repulsion integrals.
    Just a variant of the error function
    """
    if t == 0:
        return 1
    else:
        return (0.5 * (np.pi / t) ** 0.5) * erf(t ** 0.5)


def potential(A, B, coordinate, charge):
    """
    Compute Nuclear-electron attraction integral.

    INPUT:
    A: (gaussian_A alpha, gaussian_A coordinate)
    B: (gaussian_B alpha, gaussian_B coordinate)
    coordinate: coordinate of nuclear
    charge: charge of nuclear
    """
    p, diff, K, Rp = gauss_product(A, B)
    Rc = coordinate
    Zc = charge

    return (-2 * np.pi * Zc / p) * K * Fo(p * np.linalg.norm(Rp - Rc) ** 2)


def repulsion(A, B, C, D):
    """
    Compute electron-electron repulsion integral.
    """
    p, diff_ab, K_ab, Rp = gauss_product(A, B)
    q, diff_cd, K_cd, Rq = gauss_product(C, D)
    repul_prefactor = 2 * np.pi ** 2.5 * (p * q * (p + q) ** 0.5) ** -1
    return repul_prefactor*K_ab*K_cd*Fo(p*q/(p+q)*np.linalg.norm(Rp-Rq)**2)


# --------- PART 3 Compute integrals between two contracted gaussian functions  ---------


def S_int(cgf_1, cgf_2):
    """
    Compute overlap integral between two contracted gaussian functions.
    """
    s = 0
    for i, _ in enumerate(cgf_1.alpha):
        for j, _ in enumerate(cgf_2.alpha):
            s += cgf_1.co[i] * cgf_2.co[j] * overlap((cgf_1.alpha[i], cgf_1.coordinate), (cgf_2.alpha[j], cgf_2.coordinate))
    return s


def T_int(cgf_1, cgf_2):
    """
    Compute kinetics integral between two contracted gaussian functions.
    """
    t = 0
    for i, _ in enumerate(cgf_1.alpha):
        for j, _ in enumerate(cgf_2.alpha):
            t += cgf_1.co[i] * cgf_2.co[j] * kinetic((cgf_1.alpha[i], cgf_1.coordinate), (cgf_2.alpha[j], cgf_2.coordinate))
    return t


def V_eN_int(cgf_1, cgf_2, mol):
    """
    Compute electron-nuclear integral between two contracted gaussian functions.
    """
    v = 0
    for i, _ in enumerate(cgf_1.alpha):
        for j, _ in enumerate(cgf_2.alpha):
            for k in range(mol.num_atoms):
                v += cgf_1.co[i] * cgf_2.co[j] * potential((cgf_1.alpha[i], cgf_1.coordinate),
                                                           (cgf_2.alpha[j], cgf_2.coordinate),
                                                           mol.coordinates[k], mol.charges[k])
    return v


def R_int(cgf_1, cgf_2, cgf_3, cgf_4):
    """
    Compute electron-electron repulsion integral.
    """
    repul = 0
    for r, _ in enumerate(cgf_1.alpha):
        for s, _ in enumerate(cgf_2.alpha):
            for t, _ in enumerate(cgf_3.alpha):
                for u, _ in enumerate(cgf_4.alpha):
                    rp = repulsion((cgf_1.alpha[r], cgf_1.coordinate),
                                   (cgf_2.alpha[s], cgf_2.coordinate),
                                   (cgf_3.alpha[t], cgf_3.coordinate),
                                   (cgf_4.alpha[u], cgf_4.coordinate))
                    repul += cgf_1.co[r] * cgf_1.co[s] * cgf_1.co[t] * cgf_1.co[u] * rp
    return repul


# --------- PART 4 Build matrix ---------


def S_matrix(cgfs):
    """
    Compute overlap matrix S.

    INPUT:
        cgfs: basis functions
    OUTPUT:
        S: Overlap matrix
    """
    S = np.zeros([len(cgfs), len(cgfs)])

    for i, cgf_1 in enumerate(cgfs):
        for j, cgf_2 in enumerate(cgfs):
            S[i, j] = S_int(cgf_1, cgf_2)
    return S


def H_matrix(cgfs, mol):
    """
    Compute the core hamiltonian matrix H.
    H_core = electron kinetics energy + electron nuclear potential energy

    INPUT:
        cgfs: basis functions
        mol: which contain the nuclear charge and nuclear coordinate information
    OUTPUT:
        H: core hamiltonian matrix
    """
    T = np.zeros([len(cgfs), len(cgfs)])
    V = np.zeros([len(cgfs), len(cgfs)])
    for i, cgf_1 in enumerate(cgfs):
        for j, cgf_2 in enumerate(cgfs):
            T[i, j] = T_int(cgf_1, cgf_2)
            V[i, j] = V_eN_int(cgf_1, cgf_2, mol)
    H = T + V
    return H


def R_matrix(cgfs):
    """
    Compute the electron repulsion integral matrix R.

    INPUT:
        cgfs: basis functions
    OUTPUT:
        R: repulsion matrix
    """
    start = time.time()

    R = np.zeros([len(cgfs), len(cgfs), len(cgfs), len(cgfs)])
    for r, cgf_1 in enumerate(cgfs):
        for s, cgf_2 in enumerate(cgfs):
            for t, cgf_3 in enumerate(cgfs):
                for u, cgf_4 in enumerate(cgfs):
                    R[r, s, t, u] = R_int(cgf_1, cgf_2, cgf_3, cgf_4)

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
                P[t, u] += 2 * Co[t, j]*Co[u, j]
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


# --------- PART 5 Other Equations ---------


def V_NN(mol):
    """
    Compute Nuclear-Nuclear repulsion energy
    """
    nn = 0

    for i in range(mol.num_atoms):
        for j in range(i+1, mol.num_atoms):
            # Select atoms from molecule
            Ri = mol.coordinates[i]
            Rj = mol.coordinates[j]

            Zi = mol.charges[i]
            Zj = mol.charges[j]

            nn += Zi * Zj / np.linalg.norm(Ri - Rj)

    return nn


def secular_eqn(F, S):
    """
    Slove secular equation, return the MO energies (eigenvalue) and improved coeffients (eigenvector)

    INPUT:
        F: fock matrix
        S: overlap integral
    OUTPUT:
        ei: eigenvalue
        C: eigenvector
    """
    ei, C = scipy.linalg.eigh(F, S)
    return ei, C


def energy_tot(e, N, P, H, Vnn):
    """
    Compute the total energy.

    INPUT:
    e: MO energies
    N: num of electrons
    P: density matrix
    H: h_core matrix
    Vnn: nuclear nuclear repulsion energy
    """
    e_tot = 0
    for i in range(int(N/2)):
        e_tot += e[i].real
    e_tot = e_tot + 0.5 * (P * H).sum() + Vnn
    return e_tot


# --------- PART 6 Utils ---------

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
