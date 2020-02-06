#!/usr/bin/env python
""" Hartree-Fock program for He atom. """

from scipy import *
# import symeig
from pylab import *
from scipy.linalg import eigh

def CmpOverlap(base):
    "Calculates overlap O(a,b)=(Pi/(a+b))^{3/2}"
    Olap = zeros((len(base),len(base)), dtype=float)
    for i,ai in enumerate(base):
        for j,bj in enumerate(base):
            Olap[i,j] = (pi/(ai+bj))**(3/2.)
    return Olap

def CmpHam0(base):
    """ Calculates Hamiltonian K(a,b) = 3*a*b/(a+b)*(Pi/(a+b))^{3/2}
                               Vext(a,b) = -4*Pi/(a+b)  """
    Ham = zeros((len(base),len(base)), dtype=float)
    for i,ai in enumerate(base):
        for j,bj in enumerate(base):
            Ham[i,j] = 3*ai*bj*(pi/(ai+bj))**(3/2.)/(ai+bj) # kinetic part
            Ham[i,j] += -4*pi/(ai+bj)                       # external potential
    return Ham

def CmpU(base):
    "Coulomb matrix is U_{abcd}=2*pi^(5/2)/((a+c)(b+d)*sqrt(a+b+c+d))"
    Uc = zeros((len(base),len(base),len(base),len(base)), dtype=float)
    cp = 2*pi**(5/2.)
    for i,ai in enumerate(base):
        for j,bj in enumerate(base):
            for k,ck in enumerate(base):
                for l,dl in enumerate(base):
                    b1 = ai+ck
                    b2 = bj+dl
                    Uc[i,j,k,l] = cp/(b1*b2*sqrt(b1+b2))
    return Uc

def CmpHeffHartree(Ham, Uc, rho):
    " Computes effective Hartree hamiltonian"
    Heff = zeros(shape(Ham), dtype=float)
    for i in range(shape(Ham)[0]):
        for j in range(shape(Ham)[1]):
            sumH=0 # Hartree
            sumF=0 # Fock
            for k in range(shape(rho)[0]):
                for l in range(shape(rho)[1]):
                    #sumH +=  Uc[k][i][l][j]*2*rho[k][l];
                    #sumF -=  Uc[k][i][j][l]*1*rho[k][l];
                    sumH +=  Uc[k,i,l,j]*rho[k,l]

            Heff[i,j] = Ham[i,j]+sumH+sumF
    return Heff

def Eigensystem(Olap, Heff):
    """ General eigenvalue problem solved"""
    # w,Z = symeig.symeig(Heff, Olap, type=1) # symmetric generalized eigenvalue problem
    w, Z = eigh(Heff, Olap)
    Energy = w[0]    
    return (w, Z)

def TotEnergy(w, rho, Ham):
    return w[0] + sum(rho*Ham)

def CmpDensityM(C, olap):
    """ Computes the density matrix """
    for i in range(len(C)):
        for j in range(len(C)):
            rho[i,j] = C[i]*C[j];
    return rho

def RadialDensity(r, rho, base):
    wsum=0
    for i,ai in enumerate(base):
        for j,bj in enumerate(base):
            wsum += rho[i,j]*exp(-(ai+bj)*r*r)
    return wsum



if __name__ == '__main__':

    base = [0.298073, 1.242567, 5.782948, 38.47497]
    nmax = 100
    
    Olap = CmpOverlap(base)
    Ham = CmpHam0(base)
    Uc = CmpU(base)


    rho = zeros(shape(Ham), dtype=float)

    En = 0
    for i in range(nmax):
        Heff = CmpHeffHartree(Ham, Uc, rho)
        (w, Z) = Eigensystem(Olap, Heff)
        print(w[0], TotEnergy(w,rho,Ham))
        rho = CmpDensityM(Z[:,0], Olap)
        if abs(En-w[0])<1e-6: break
        else: En=w[0]


    R = arange(1e-3, 3.5, 0.01)
    dens = [4*pi*r**2*RadialDensity(r, rho, base) for r in R]

    # title('Electronic density for He atom.')
    # plot(R, dens, label='$4\pi r^2 n(r)$')
    # legend(loc='best')
    # show()
