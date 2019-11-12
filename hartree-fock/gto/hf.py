import time
import scipy
import numpy as np
import sympy as sp
from scipy.special import erf
from scipy.optimize import minimize


class STO():
    def __init__(self, zeta, n):
        self.zeta = zeta
        self.n = n
        self.sto = self.get_sto(self.zeta, self.n)

    def show(self):
        f = self.get_sto(self.zeta, self.n)
        return f

    def get_sto(self, zeta, n):
        r = sp.Symbol('r')
        f = r ** (n - 1) * sp.exp(-zeta * sp.Abs(r))
        N = sp.sqrt(1 / sp.integrate(f * f * r * r, (r, 0, +sp.oo)))
        return f * N


class CGF():
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

        self.contract_co = contract_cos[n-1]
        self.alpha = alphas[n-1] * zeta ** 2
        self.n = n
        self.zeta = zeta
        self.coordinates = coordinates
        self.gtos = []
        for i, a in enumerate(self.alpha):
            gto = self.get_gto(a, self.n)
            self.gtos.append(gto)
        self.cgf = 0
        for i, g in enumerate(self.gtos):
            self.cgf = self.cgf + self.contract_co[i] * g

    def show(self):
        cgf = 0
        for i, a in enumerate(self.alpha):
            cgf += self.contract_co[i] * self.get_gto(a, self.n)
        return cgf

    def get_gto(self, alpha, n):
        r = sp.Symbol('r')
        f = r**(n-1)*sp.exp(-alpha*r*r)
        return f

    def opt_cgf(self, r1, c1, c2, c3, n):
        f = c1 * r1 ** (n-1)*np.exp(-self.alpha[0]*r1*r1) + c2 * r1 ** (n-1)*np.exp(-self.alpha[1]*r1*r1) + c3 * r1 ** (n-1)*np.exp(-self.alpha[2]*r1*r1)
        r = sp.Symbol('r')
        f_sp = c1 * r ** (n-1)*sp.exp(-self.alpha[0]*r*r) + c2 * r ** (n-1)*sp.exp(-self.alpha[1]*r*r) + c3 * r ** (n-1)*sp.exp(-self.alpha[2]*r*r)
        N = sp.sqrt(1 / sp.integrate(f_sp * f_sp * r * r, (r, 0, +sp.oo)))
        N = N.subs(sp.pi, np.pi)
        return f * N, N

    def opt_sto(self, r1, zeta, n):
        r = sp.Symbol('r')
        f = r ** (n - 1) * sp.exp(-zeta * sp.Abs(r))
        N = sp.sqrt(1 / sp.integrate(f * f * r * r, (r, 0, +sp.oo)))
        f = r1 ** (n - 1) * np.exp(-zeta * np.abs(r1))
        return f * N

    def opt(self):

        r = np.linspace(-5, 5., num=1000)
        sto = self.opt_sto(r, self.zeta, self.n)

        def loss(para):
            import sys
            sys.stdout.write('.')
            sys.stdout.flush()
            pred, _ = self.opt_cgf(r, para[0], para[1], para[2], self.n)
            mse = np.mean((pred - sto)**2)  # mean_squared_error
            return mse

        print('optimizing cgf')
        result = minimize(loss, x0=np.array(self.contract_co), method='Nelder-Mead')
        print('{} iterations: \n', result.nit)
        _, N = self.opt_cgf(r, result.x[0], result.x[1], result.x[2], self.n)
        self.contract_co[1] = N * result.x[1]
        self.contract_co[0] = N * result.x[0]
        self.contract_co[1] = N * result.x[1]

    def update(self):
        self.gtos = []
        for i, a in enumerate(self.alpha):
            gto = self.get_gto(a, self.n)
            self.gtos.append(gto)
        self.cgf = 0
        for i, g in enumerate(self.gtos):
            self.cgf = self.cgf + self.contract_co[i] * g

    def __str__(self):
        return str(self.show())

    def __repr__(self):
        return self.show()


class Molecule():

    def __init__(self, string):
        # Put zeta number in list to accomodate for possibly more basis sets (eg 2s orbital)
        zeta_dict = {'H': [1.24], 'He': [2.0925], 'Li': [2.69, 0.80],
                     'Be': [3.68, 1.15], 'B': [4.68, 1.50],
                     'C': [5.67, 1.72]}
        principle_quantum_number = {'H': 1, 'He': 1, 'Li': 2, 'Be': 2, 'C': 2}
        charge_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'C': 6}

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

# Integrals between Gaussian orbitals (pp410)


def gauss_product(gauss_A, gauss_B):
    # The product of two Gaussians gives another Gaussian (pp411)
    # Pass in the exponent and centre as a tuple
    a, Ra = gauss_A
    b, Rb = gauss_B
    p = a + b
    diff = np.linalg.norm(Ra-Rb)**2             # squared difference of the two centres
    N = (4*a*b/(np.pi**2))**0.75                   # Normalisation
    K = N*np.exp(-a*b/p*diff)                      # New prefactor
    Rp = (a*Ra + b*Rb)/p                        # New centre

    return p, diff, K, Rp


# Overlap integral (pp411)
def overlap(A, B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (np.pi/p)**1.5
    return prefactor*K


# Kinetic integral (pp412)
def kinetic(A, B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (np.pi/p)**1.5

    a, Ra = A
    b, Rb = B
    reduced_exponent = a*b/p
    return reduced_exponent*(3-2*reduced_exponent*diff)*prefactor*K


# Fo function for calculating potential and e-e repulsion integrals.
# Just a variant of the error function
# pp414
def Fo(t):
    if t == 0:
        return 1
    else:
        return (0.5*(np.pi/t)**0.5)*erf(t**0.5)


# Nuclear-electron integral (pp412)
def potential(A, B, coordinate, charge):
    p, diff, K, Rp = gauss_product(A, B)
    Rc = coordinate  # Position of atom C
    Zc = charge  # Charge of atom C

    return (-2*np.pi*Zc/p)*K*Fo(p*np.linalg.norm(Rp-Rc)**2)


# (ab|cd) integral (pp413)
def multi(A, B, C, D):
    p, diff_ab, K_ab, Rp = gauss_product(A, B)
    q, diff_cd, K_cd, Rq = gauss_product(C, D)
    multi_prefactor = 2*np.pi**2.5*(p*q*(p+q)**0.5)**-1
    return multi_prefactor*K_ab*K_cd*Fo(p*q/(p+q)*np.linalg.norm(Rp-Rq)**2)


def S_int(cgf_1, cgf_2):
    s = 0
    for i, _ in enumerate(cgf_1.alpha):
        for j, _ in enumerate(cgf_2.alpha):
            s += cgf_1.contract_co[i] * cgf_2.contract_co[j] * overlap((cgf_1.alpha[i], cgf_1.coordinates), (cgf_2.alpha[j], cgf_2.coordinates))
    return s


def T_int(cgf_1, cgf_2):
    t = 0
    for i, _ in enumerate(cgf_1.alpha):
        for j, _ in enumerate(cgf_2.alpha):
            t += cgf_1.contract_co[i] * cgf_2.contract_co[j] * kinetic((cgf_1.alpha[i], cgf_1.coordinates), (cgf_2.alpha[j], cgf_2.coordinates))
    return t


def V_eN_int(cgf_1, cgf_2, mol):
    v = 0
    for i, _ in enumerate(cgf_1.alpha):
        for j, _ in enumerate(cgf_2.alpha):
            for k in range(mol.num_atoms):
                v += cgf_1.contract_co[i] * cgf_2.contract_co[j] * potential((cgf_1.alpha[i], cgf_1.coordinates),
                                                                             (cgf_2.alpha[j], cgf_2.coordinates),
                                                                             mol.coordinates[k],
                                                                             mol.charges[k])
    return v


def R_int(cgf_1, cgf_2, cgf_3, cgf_4):
    repul = 0
    for r, _ in enumerate(cgf_1.alpha):
        for s, _ in enumerate(cgf_2.alpha):
            for t, _ in enumerate(cgf_3.alpha):
                for u, _ in enumerate(cgf_4.alpha):
                    repul += cgf_1.contract_co[r]*cgf_1.contract_co[s]*cgf_1.contract_co[t]*cgf_1.contract_co[u]*(
                        multi((cgf_1.alpha[r], cgf_1.coordinates),
                              (cgf_2.alpha[s], cgf_2.coordinates),
                              (cgf_3.alpha[t], cgf_3.coordinates),
                              (cgf_4.alpha[u], cgf_4.coordinates)))
    return repul


def S_matrix(cgfs):
    S = np.zeros([len(cgfs), len(cgfs)])
    for i, cgf_1 in enumerate(cgfs):
        for j, cgf_2 in enumerate(cgfs):
            S[i, j] = S_int(cgf_1, cgf_2)
    return S


def H_matrix(cgfs, mol):
    T = np.zeros([len(cgfs), len(cgfs)])
    V = np.zeros([len(cgfs), len(cgfs)])
    for i, cgf_1 in enumerate(cgfs):
        for j, cgf_2 in enumerate(cgfs):
            T[i, j] = T_int(cgf_1, cgf_2)
            V[i, j] = V_eN_int(cgf_1, cgf_2, mol)
    H = T + V
    return H


def R_matrix(cgfs):
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

    P = np.zeros([Co.shape[0], Co.shape[0]])

    for t in range(Co.shape[0]):
        for u in range(Co.shape[0]):
            for j in range(int(N/2)):
                P[t, u] += 2 * Co[t, j]*Co[u, j]
    return P


def G_matrix(P, R):

    G = np.zeros((P.shape[0], P.shape[0]))

    for r in range(P.shape[0]):
        for s in range(P.shape[0]):
            g = 0
            for t in range(P.shape[0]):
                for u in range(P.shape[0]):
                    int1 = R[r, s, t, u]
                    int2 = R[r, u, t, s]
                    g += P[t, u] * (int1 - 0.5 * int2)
            G[r, s] = g
    return G


def F_matrix(H, G):
    return H + G


# Nuclear repulsion
def V_NN(mol):
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


# slove secular equation, return the energy and improved coeffients
# the energy here is orbital energy for 1 electron
def secular_eqn(F, S):
    ei, C = scipy.linalg.eigh(F, S)
    return ei, C


def energy_tot(e, P, H, mol):

    e_tot = 0
    for i in range(int(mol.num_electron/2)):
        e_tot += e[i].real
    e_tot = e_tot + 0.5*(P * H).sum() + V_NN(mol)
    # print('electronic: ', e_tot - V_NN(mol))
    # print('V_NN: ', V_NN(mol))
    return e_tot


# --------- PART 5 Utils ---------

def print_info(e, Co, hf_e, start, stop, delta_e=0, verbose=False):
    """
    Print information while doing SCF interations.
    """
    if(verbose):
        # Co
        print('Coefficients:')
        print(Co)

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