import numpy as np
import hf
import time

verbose = False
MAXITER = 40  # Maximum SCF iterations
E_conv = 1.0e-6  # Energy convergence criterion


def run_hf(mol):
    """
    Run restricted hartree fock for 2-electron diatomic molecule.

    INPUT:
        mol: hf.mol object
    """
    start = time.time()

    # initialization
    H = hf.H_matrix(mol.cgfs, mol)
    S = hf.S_matrix(mol.cgfs)
    e, Co = hf.secular_eqn(H, S)
    P = hf.P_matrix(Co, mol.num_electron)
    hf_e = hf.energy_tot(e, P, H, mol)

    stop = time.time()
    print('------------------------------', "Initialization", '------------------------------')
    print('-------------------------', "Ignore repulsion integral", '------------------------')
    hf.print_info(e, Co, hf_e, start, stop, verbose=verbose)
    print('-----------', "Caculating Electron Repulsion Integral (takes time)", '------------')
    R = hf.R_matrix(mol.cgfs)
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
        P = hf.P_matrix(Co, mol.num_electron)
        hf_e = hf.energy_tot(e, P, H, mol)

        delta_e = np.abs(hf_e - previous_e)
        previous_e = hf_e
        ITER += 1
        stop = time.time()
        hf.print_info(e, Co, hf_e, start, stop, delta_e, verbose)

    return hf_e


def test1():
    """
    Test of H2
    """
    H2 = """
    0
    H 0 0 0
    H 0 0 1.4
    """

    mol = hf.Molecule(H2)
    print(mol.cgfs[0].show())
    print(mol.cgfs[1].show())

    # run hartree fock
    hf_e = run_hf(mol)

    # compare result with reference
    ref_hf_e = -1.11675930740
    hf.compare(hf_e, ref_hf_e)


def test2():
    """
    Test of HeH+
    """
    HeH = """
    1
    He 0 0 0
    H 0 0 1.4632
    """

    mol = hf.Molecule(HeH)

    # run hartree fock
    hf_e = run_hf(mol)

    # compare result with reference
    ref_hf_e = -2.8606621637
    hf.compare(hf_e, ref_hf_e)


def test3(dist):
    # 1. test for H2
    H2 = """
    0
    H 0 0 0
    H 0 0 {:.4f}
    """.format(dist)
    # H2 = '0\nH 0 0 0\nH 0 0 {:.4f}'.format(dist)
    mol = hf.Molecule(H2)
    hf_e = run_hf(mol)
    ref_hf_e = -1.11675930740
    compare(hf_e, ref_hf_e)
    return hf_e


def test4():
    import matplotlib
    from matplotlib import pyplot as plt

    distances = np.linspace(0.2, 40., num=300)
    distances = [1.32, 1.34, 1.36, 1.38, 1.40]
    energies = []
    for d in distances:
        energies.append(test3(d))

    for i, d in enumerate(distances):
        print('{:.2f}: {:.6f}'.format(distances[i], energies[i]))

    plt.plot(distances, energies, 'o-', markersize=3)
    plt.xlabel('bond length (A)')
    plt.ylabel('Total Energy')
    plt.title('H2 molecule')
    plt.show()


if __name__ == "__main__":
    test1()
    test2()
