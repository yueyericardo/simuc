import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
from scipy.constants import physical_constants as pc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# just to be pretty
from matplotlib import rc
rc('text', usetex=True)
rc('mathtext', fontset='cm')
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
rc('axes', labelsize=20)
rc('axes.spines', **{'right': False, 'top': False})

# useful constant
amu_to_au = 1.0/(pc['kilogram-atomic mass unit relationship'][0]*pc['atomic unit of mass'][0])  # 1822.888479031408
hartree_to_cm1 = pc['hartree-hertz relationship'][0]/pc['speed of light in vacuum'][0]/100.0  # 2.194746313705e5
sec = pc['atomic unit of time'][0]  # 2.418884326505e-17
cee = 100.0*pc['speed of light in vacuum'][0]  # 2.99792458e10 cm/s
bohr_to_angstroms = pc['atomic unit of length'][0]*1e10
hartree_to_ev = pc['hartree-electron volt relationship'][0]


class State(object):

    def __init__(self, we, De, re, Te):
        we = we  # wavenumbers
        De = De
        re = re  # Angstroms
        Te = Te
        # wexe = we ** 2 / (4.0 * De)
        dissociation = Te + De

        self.De_au = De / hartree_to_cm1
        freq = we * sec * cee  # angular harmonic frequency in au
        k_au = ((2.0 * np.pi * freq) ** 2) * mu_au  # force constant in au
        self.a_au = np.sqrt(k_au / (2.0 * self.De_au))
        self.re_au = re / bohr_to_angstroms
        self.Te_au = Te / hartree_to_cm1

        self.dissociation_au = dissociation / hartree_to_cm1


# includes electronic term energy
def morse(r, state: State, include_Te=True):
    De = state.De_au
    alpha = state.a_au
    re = state.re_au
    if include_Te:
        Te = state.Te_au
    else:
        Te = 0
    return De * (1.0 - np.exp(-alpha * (r - re))) ** 2 + Te


# how many bound eigenstates are there?
def gvmax(state: State):
    we = np.sqrt(2 * state.De_au / mu_au) * state.a_au
    vmax = int(np.floor((2.0 * state.De_au - we) / we))
    return vmax


def g(v, state: State):
    we = np.sqrt(2 * state.De_au / mu_au) * state.a_au
    wexe = we ** 2 / (4 * state.De_au)
    energy = (we*(v+0.5) - wexe*(v+0.5)**2)
    return energy


def energy_diff(r, v, state: State):
    ret = morse(r, state, include_Te=False) - g(v, state)
    return(ret)


def left_position(v, state: State):
    left = fsolve(energy_diff, [state.re_au - 0.1], (v, state))[0]
    return left


def right_position(v, state: State):
    right = fsolve(energy_diff, [state.re_au + 0.1], (v, state))[0]
    return right

# --------------------------- Solve wavefunction -----------------------------


#  This is for numerical calculation of the wavefunctions.
def kron(n, m):
    return(n == m)


def gen_T(x, k):
    """
    Assemble the matrix representation of the kinetic energy (second derivative)
    """
    N = len(x)
    dx = (max(x) - min(x)) / N
    T = np.zeros((N, N)).astype(np.complex)
    for m in range(0, N):
        for n in range(m-1, m+2):
            if n < 0 or n > (N - 1):
                continue
            T[m, n] = k / (dx ** 2) * (
                kron(m + 1, n) - 2 * kron(m, n) +
                kron(m - 1, n))
    return(T)


def gen_V(x, u):
    """
    Assemble the matrix representation of the potential energy
    """
    N = len(x)
    V = np.zeros((N, N)).astype(np.complex)
    for m in range(N):
        V[m, m] = u[m]
    return(V)


def wavefunction_norm(x, psi):
    """
    Calculate the norm of the given wavefunction.
    """
    dx = x[1] - x[0]
    return((psi.conj() * psi).sum() * dx)


def wavefunction_normalize(x, psi):
    """
    Normalize the given wavefunction.
    """
    return(psi / np.sqrt(wavefunction_norm(x, psi)))


def inner_product(x, psi1, psi2):
    """
    Evaluate the inner product of two wavefunctions, psi1 and psi2, on a space
    described by x.
    """
    left = wavefunction_normalize(x, psi1)
    right = wavefunction_normalize(x, psi2)
    dx = x[1] - x[0]
    return((left.conj() * right).sum() * dx)


def mod_sq(x, psi):
    tmp = wavefunction_normalize(x, psi)
    ret = (tmp.conj()*tmp).real
    return(ret)


def mat_x(x, psi1, psi2):
    """
    Evaluate the matrix element of x, over psi1 and psi2, on a space
    described by x.
    """
    left = wavefunction_normalize(x, psi1)
    right = wavefunction_normalize(x, psi2)
    dx = x[1] - x[0]
    return((left.conj() * x * right).sum() * dx)


def mat_x2(x, psi1, psi2):
    """
    Evaluate the matrix element of x^2, over psi1 and psi2, on a space
    described by x.
    """
    left = wavefunction_normalize(x, psi1)
    right = wavefunction_normalize(x, psi2)
    dx = x[1] - x[0]
    return((left.conj()*x*x*right).sum() * dx)


def wrap_potential(u_func, x, args):
    return(u_func(x, args))


def solve_eigenproblem(H):
    """
    Solve an eigenproblem and return the eigenvalues and eigenvectors.
    """
    vals, vecs = np.linalg.eigh(H)
    idx = np.real(vals).argsort()
    vals = vals[idx]
    vecs = vecs.T[idx]
    return(vals, vecs)


if __name__ == "__main__":
    # input
    mass_1 = 10
    mass_2 = 10
    mumass = mass_1 * mass_2 / (mass_1 + mass_2)
    mu_au = mumass * amu_to_au

    # build state
    state_1 = State(we=2500, De=25000, re=2.5, Te=0)
    state_2 = State(we=1900, De=20000, re=2.9, Te=26000)

    # solve wavefunction
    rscale = bohr_to_angstroms
    escale = hartree_to_ev

    k = -1.0/(2.0 * mu_au)   # in atomic units.  Change this if you want different units (Why?)
    x_min = 2.0 / rscale   # Convert from Angstroms
    x_max = 5.5 / rscale
    N = 1250              # The size of the basis grid.  Bigger is more accurate, but slower
    xx = np.linspace(x_min, x_max, N)     # The grid

    u = morse(xx, state_1)
    T = gen_T(xx, k)
    V = gen_V(xx, u)
    H = T + V
    evals1, evecs1 = solve_eigenproblem(H)

    u = morse(xx, state_2)
    T = gen_T(xx, k)
    V = gen_V(xx, u)
    H = T + V
    evals2, evecs2 = solve_eigenproblem(H)

    # plot
    xlow = 2.
    xhigh = 4.5
    ylow = -0.2
    yhigh = 6.0
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    x = np.linspace(xlow / rscale, xhigh / rscale, 1000)

    plt.plot(rscale*x, escale*morse(x, state_1), label=r'\rm X state', color="blue", linewidth=2)
    plt.plot(rscale*x, escale*morse(x, state_2), label=r'\rm A state', color="red", linewidth=2)

    for i in range(gvmax(state_1)):
        plt.hlines(escale*(g(i, state_1)+state_1.Te_au), rscale*left_position(i, state_1), rscale*right_position(i, state_1), linewidth=1.0, color="blue")

    for i in range(gvmax(state_2)):
        plt.hlines(escale*(g(i, state_2)+state_2.Te_au), rscale*left_position(i, state_2), rscale*right_position(i, state_2), linewidth=1.0, color="red")

    overhang = 0.3
    for i in range(0, gvmax(state_1), 3):
        mask = (xx > left_position(i, state_1)-overhang) & (xx < right_position(i, state_1)+overhang)
        plt.plot(xx[mask]*rscale, (0.005*mod_sq(xx[mask], evecs1[i, mask]) + evals1[i].real)*escale, color="blue")

    for i in range(0, gvmax(state_2), 3):
        mask = (xx > left_position(i, state_2)-overhang) & (xx < right_position(i, state_2)+overhang)
        plt.plot(xx[mask]*rscale, (0.005*mod_sq(xx[mask], evecs2[i, mask]) + evals2[i].real)*escale, color="red")

    plt.hlines((state_1.dissociation_au)*escale, xlow, xhigh, linestyle="dashed", linewidth=0.8, color="black")
    plt.hlines((state_2.dissociation_au)*escale, xlow, xhigh, linestyle="dashed", linewidth=0.8, color="black")

    plt.xlim(xlow, xhigh)
    plt.ylim(ylow, yhigh)
    plt.xlabel(r"\rm $R\; $ [\AA]", fontsize=20)
    plt.ylabel(r"\rm Energy [eV]", fontsize=20)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(r'\rm The Potentials', fontsize=24)
    plt.show()

    # overlap
    vmax_1 = gvmax(state_1)
    vmax_2 = gvmax(state_2)
    overlap = np.zeros((vmax_1, vmax_2))
    for i in range(vmax_1):
        for j in range(vmax_2):
            overlap[i, j] = inner_product(x, evecs1[i], evecs2[j]).real
    plt.imshow(overlap, cmap='hot')
    plt.show()

    # Franck-Condon Factor
    plt.imshow(abs(overlap)**2, cmap='viridis')
    plt.show()
