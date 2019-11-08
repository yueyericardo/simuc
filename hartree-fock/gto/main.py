import numpy as np
import hf
import time

np.set_printoptions(precision=6)
print_co = True
print_MO = True

# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-8


def compare(cal, ref, tol=1.0e-4):
    delta = np.abs(ref - cal)
    if delta < tol:
        message = '\33[32m' + 'PASSED' + '\x1b[0m'
    else:
        message = '\033[91m' + 'FAILED' + '\033[0m'
    print('-' * 32, message, '-' * 33)
    print('cal: {:.7f}, ref: {:.7f}\n\n'.format(cal, ref))


def run_hf(mol):

    def print_info():
        if(print_co):
            print('S:')
            print(S)
            print('H:')
            print(H)
            try:
                R
                print('R')
                print(R)
            except NameError:
                pass
            print('Coefficients:')
            print(Co)
        # print energy result
        if(print_MO):
            print('MO energies:')
            message = ', '
            m_list = ['e{} = {:0.3f}'.format(i+1, x) for i, x in enumerate(e)]
            message = message.join(m_list)
            print(message)
        print('HF energy: {:0.5f} (hartree) = {:0.5f} (eV)'.format(hf_e,
                                                                   hf_e * 27.211))
        try:
            delta_e
            print('dE       : {:.2e} \ntime used: {:.1f} s'.format(delta_e,
                                                                   stop-start))
        except NameError:
            pass

    # initialization
    H = hf.H_matrix(mol.cgfs, mol)
    S = hf.S_matrix(mol.cgfs)
    e, Co = hf.secular_eqn(H, S)
    P = hf.P_matrix(Co, mol.num_electron)
    hf_e = hf.energy_tot(e, P, H, mol)
    print('-'*30, "Initialization", '-'*30)
    print('-'*25, "Ignore repulsion integral", '-'*24)
    print_info()
    delta_e = 100
    ITER = 0

    while(delta_e > E_conv and ITER < MAXITER):
        print('-' * 30, "Iteration", ITER + 1, '-' * 30)
        if(ITER == 0):
            print('-' * 7, "Iteration 1 needs more time to caculate Repulsion Integral", '-' * 6)
            start = time.time()
            R = hf.R_matrix(mol.cgfs)
        else:
            start = time.time()
        G = hf.G_matrix(P, R)
        F = H + G
        e, Co = hf.secular_eqn(F, S)
        P = hf.P_matrix(Co, mol.num_electron)
        previous_e = hf_e
        hf_e = hf.energy_tot(e, P, H, mol)
        delta_e = np.abs(hf_e - previous_e)
        ITER += 1

        # print
        stop = time.time()
        print_info()

    return hf_e


def test1():
    # 1. test for H2
    H2 = """
    0
    H 0 0 0
    H 0 0 1.4
    """

    mol = hf.Molecule(H2)
    print(mol.cgfs[0].show())
    print(mol.cgfs[1].show())
    hf_e = run_hf(mol)
    ref_hf_e = -1.11675930740
    compare(hf_e, ref_hf_e)


def test2():
    # 2. test for HeH
    HeH = """
    1
    He 0 0 0
    H 0 0 1.4632
    """

    mol = hf.Molecule(HeH)
    print(mol.cgfs[0].show())
    print(mol.cgfs[1].show())
    hf_e = run_hf(mol)
    ref_hf_e = -2.8606621637
    compare(hf_e, ref_hf_e)


if __name__ == "__main__":
    test1()
    test2()
