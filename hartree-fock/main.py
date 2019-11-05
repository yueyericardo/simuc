import numpy as np
import hf
import time

np.set_printoptions(precision=6)
print_co = False
print_MO = True

# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6


def print_info():
    if(print_co):
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


# input for zeta
zetas = [[1.45, 1], [2.91, 1]]
# input nuclear charge (element number)
Z = 2

# zetas = [[5.59108, 1], [3.35538, 1], [1.01122, 2], [0.61000, 2]]
# Z = 4

fs = []
for z in zetas:
    f = hf.STO(z[0], z[1])
    fs.append(f)

# initialization
R = np.zeros((len(zetas), len(zetas), len(zetas), len(zetas)))
H = hf.H_matrix(fs, Z)
S = hf.S_matrix(fs)
e, Co = hf.secular_eqn(H, S)
P = hf.P_matrix(Co)
hf_e = hf.get_E0(e, P, H)

print('-'*30, "Initialization", '-'*30)
print('-'*25, "Ignore repulsion integral", '-'*24)
print_info()
delta_e = 1
ITER = 0

while(delta_e > E_conv and ITER < MAXITER):
    print('-' * 30, "Iteration", ITER + 1, '-' * 30)
    if(ITER == 0):
        print('-' * 7, "Iteration 1 needs more time to caculate Repulsion Integral", '-' * 6)
        start = time.time()
        R = hf.R_matrix(zetas)
    else:
        start = time.time()
    F = hf.F_matrix(fs, Z, zetas, Co, R)
    S = hf.S_matrix(fs)
    e, Co = hf.secular_eqn(F, S)
    P = hf.P_matrix(Co)
    previous_e = hf_e
    hf_e = hf.get_E0(e, P, H)
    delta_e = np.abs(hf_e - previous_e)
    ITER += 1

    # print
    stop = time.time()
    print_info()
