import numpy as np
import hf
import time

verbose = True
MAXITER = 40  # Maximum SCF iterations
E_conv = 1.0e-6  # Energy convergence criterion


def run_hf(fs, Z):
    """
    Run restricted hartree fock for a single atom.

    INPUT:
        fs: basis functions
        Z: nuclear charge of the atom
    """
    print('------------------------------', "Initialization", '------------------------------')
    print('-------------------------', "Ignore repulsion integral", '------------------------')
    N = Z  # num of electron = nuclear charege (since it's atom)
    start = time.time()

    # initialization
    H = hf.H_matrix(fs, Z)
    S = hf.S_matrix(fs)
    e, Co = hf.secular_eqn(H, S)
    P = hf.P_matrix(Co, N)
    Vnn = 0  # A single atom does not have nuclear repulsion
    hf_e = hf.energy_tot(e, N, P, H, Vnn)

    stop = time.time()
    hf.print_info(S, H, e, Co, P, hf_e, start, stop, verbose=verbose)
    print('-----------', "Caculating Electron Repulsion Integral (takes time)", '------------')
    R = hf.R_matrix(fs)
    delta_e = 1
    ITER = 0
    previous_e = hf_e

    # Iterations
    while(delta_e > E_conv and ITER < MAXITER):
        print('------------------------------', "Iteration", ITER + 1, '------------------------------')
        start = time.time()

        # important scf steps
        G = hf.G_matrix(P, R)
        F = H + G
        e, Co = hf.secular_eqn(F, S)
        P = hf.P_matrix(Co, N)
        hf_e = hf.energy_tot(e, N, P, H, Vnn)

        delta_e = np.abs(hf_e - previous_e)
        previous_e = hf_e
        ITER += 1
        stop = time.time()
        hf.print_info(S, H, e, Co, P, hf_e, start, stop, verbose=verbose)

    return hf_e


def test1():
    """
    Test of He (1s)
    """
    # Use 2 Slator Type ourbital to represent Helium 1s orbital.
    # The final Helium 1s orbital is a linear combination of these two STO.
    f1s_1 = hf.STO(zeta=1.45363, n=1)
    f1s_2 = hf.STO(zeta=2.91093, n=1)

    # all basis functions
    fs = [f1s_1, f1s_2]

    #  nuclear charge of He
    Z = 2

    # run hartree fock
    hf_e = run_hf(fs, Z)

    # compare result with reference
    ref_hf_e = -2.8616726
    hf.compare(hf_e, ref_hf_e)


def test2():
    """
    Test of Be (1s, 2s)
    """
    # Use 2 STO to represent Be 1s orbital and another 2 STO for 2s orbital
    # The final 1s orbital is a linear combination of these 4 STO.
    # Same for 2s orbital.
    f1s_1 = hf.STO(zeta=5.59108, n=1)
    f1s_2 = hf.STO(zeta=3.35538, n=1)
    f2s_1 = hf.STO(zeta=1.01122, n=2)
    f2s_2 = hf.STO(zeta=0.61000, n=2)

    # all basis functions
    fs = [f1s_1, f1s_2, f2s_1, f2s_2]

    # nuclear charge of Be
    Z = 4

    # run hartree fock
    hf_e = run_hf(fs, Z)

    # compare result with reference
    ref_hf_e = -14.572369
    hf.compare(hf_e, ref_hf_e)


if __name__ == "__main__":
    test1()
    test2()
