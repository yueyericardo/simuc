import numpy as np
import hf
import time

verbose = False
MAXITER = 40  # Maximum SCF iterations
E_conv = 1.0e-6  # Energy convergence criterion


def run_hf(zetas, Z):

    fs = []
    for i, z in enumerate(zetas):
        f = hf.STO(z[0], z[1])
        fs.append(f)
    N = Z  # num of electron == nuclear charege (since it's atom)
    start = time.time()

    # initialization
    H = hf.H_matrix(fs, Z)
    S = hf.S_matrix(fs)
    e, Co = hf.secular_eqn(H, S)
    P = hf.P_matrix(Co, N)
    hf_e = hf.get_E0(e, P, H)

    stop = time.time()
    print('------------------------------', "Initialization", '------------------------------')
    print('-------------------------', "Ignore repulsion integral", '------------------------')
    hf.print_info(e, Co, hf_e, start, stop, verbose=verbose)
    print('-----------', "Caculating Electron Repulsion Integral (takes time)", '------------')
    R = hf.R_matrix(zetas)
    delta_e = 1
    ITER = 0
    previous_e = hf_e

    while(delta_e > E_conv and ITER < MAXITER):
        print('------------------------------', "Iteration", ITER + 1, '------------------------------')
        start = time.time()
        G = hf.G_matrix(P, R)
        F = hf.F_matrix(H, G)
        e, Co = hf.secular_eqn(F, S)
        P = hf.P_matrix(Co, N)
        hf_e = hf.get_E0(e, P, H)

        delta_e = np.abs(hf_e - previous_e)
        previous_e = hf_e
        ITER += 1
        stop = time.time()
        hf.print_info(e, Co, hf_e, start, stop, delta_e, verbose)

    return hf_e


def test1():
    # 1. test for He
    # input for zeta, format [[zeta1, n1], [zeta2, n2], ...]
    zetas = [[1.45363, 1], [2.91093, 1]]
    # input nuclear charge (element number)
    Z = 2
    hf_e = run_hf(zetas, Z)
    ref_hf_e = -2.8616726
    hf.compare(hf_e, ref_hf_e)


def test2():
    # 2. test for Be
    zetas = [[5.59108, 1], [3.35538, 1], [1.01122, 2], [0.61000, 2]]
    Z = 4
    hf_e = run_hf(zetas, Z)
    ref_hf_e = -14.572369
    hf.compare(hf_e, ref_hf_e)


if __name__ == "__main__":
    test1()
    test2()
