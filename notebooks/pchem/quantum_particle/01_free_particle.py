# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# # Free Particle

# ### Authors:

# * [Vinícius Wilian D. Cruzeiro](https://scholar.google.com/citations?user=iAK04WMAAAAJ). E-mail: vwcruzeiro@ufl.edu
# * Xiang Gao. E-mail: qasdfgtyuiop@ufl.edu
# * [Valeria D. Kleiman](http://kleiman.chem.ufl.edu/). E-mail: kleiman@ufl.edu
#
# Department of Chemistry
#
# Physical Chemistry Division
#
# University of Florida
#
# P.O. Box 117200
#
# Gainesville, FL 32611-7200
#
# United States

# **Instructions:**
# * The reader should follow this notebook in the order that it is presented, executing code cells in consecutive order.
# * In order to execute a cell you may click on the cell and click the *PLAY* button, press *Shift+Enter*, or got to *Cell-->Run cells*. The user may also execute all cells at once by clicking on *Cell --> Run All* at the toolbar above. 
# * **Important:** Some cells **are only going to execute after the user enters input values in the corresponding boxes**.

from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading. As you scroll through code cell,
they appear as empty cells
with a blue left-side edge
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

# ### Libraries used in this notebook:

# On the next cell we are going to import the libraries used in this notebook as well as call some important functions.

import matplotlib as mpl # matplotlib library for plotting and visualization
import matplotlib.pylab as plt # matplotlib library for plotting and visualization
import numpy as np #numpy library for numerical manipulation, especially suited for data arrays

# In the next cell we are shutting down eventual warnings displayed by IPython. This cell is optional.

import warnings
warnings.filterwarnings('ignore')

# Executing the next cell prints on the screen the versions of IPython, Python and its libraries on your computer. Please check if the versions are up-to-date to facilitate a smooth running of the program.

import sys # checking the version of Python
import IPython # checking the version of IPython
print("Python version = {}".format(sys.version))
print("IPython version = {}".format(IPython.__version__))
print("Matplotlib version = {}".format(plt.__version__))
print("Numpy version = {}".format(np.__version__))


# ### Special calls:

# The next cell configures `matplotlib` to show figures embedded within the cells it was executed from, instead of opening a new window for each figure.

# %matplotlib inline

# ## Describing the Free Particle

# We start by describing a free particle: a particle that is not under the influence of a potential.   
#  As any other particle, the state of a ***Free Particle*** is described with a ket $\left|\psi(x)\right>$. In order to learn about this particle (measure its properties) we must construct *Hermitian Operators*. For example, what is the **momentum operator $\hat P$**?
#  
# The momentum operator most obey the following property (eigenfunction/eigenvalue equation):
#
# $$\hat P \left| \psi_k(x) \right> =p\left | \psi_k(x)\right>  \tag{1}$$ 
#
# where *p* is an eigenvalue of a *Hermitian operator* and therefore it is a real number.
#

# In the $x$ representation, using the momentum operator as $\hat P =-i\hbar \frac{\partial }{\partial x}$, we can solve equation 1 by proposing a function to represent $\left| \psi_k(x) \right>$ as $\psi_k(x) = c\ e^{ikx}$, where $k$ is a real number.
#
# Let's see if it works:  
# $$\hat P \psi_k(x) =p \psi_k(x)$$ 
#
# $$-i\hbar \frac{\partial {c\ e^{ikx}}}{\partial x} =-i\hbar\ c\ ik\ e^{ikx} $$ 
#
# $$\hbar k\ c\ e^{ikx} = \hbar k\ \psi_k(x) \tag{2}$$
# with $p=\hbar k$

# Although **$\psi_k(x)$ can not be normalized**, the constant $c$ is chosen in such a way (see note below) that the function becomes:
# $$\large{\psi_k(x) = \frac{1}{\sqrt{2\pi}} e^{ikx}}$$

# ###  Side note on the normalization factor and Dirac's delta function  

# The normalization of a bra-ket is given by $$\left<\psi_{k'}(x)|\psi_k(x)\right> = \left<k'|k \right> = \int^{+\infty}_{-\infty} \psi_{k'}^*(x)\psi_k(x) dx = \int^{+\infty}_{-\infty} c^* e^{-ik'x} ce^{ikx} dx = |c|^2 \int^{+\infty}_{-\infty} e^{i(k-k')x} dx = |c|^2 2\pi\ \delta(k-k')$$
#
# Where we used the definition of the **Dirac's delta function**:
# $$\delta(x) = \frac{1}{2\pi} \int^{+\infty}_{-\infty} e^{ivx} dv$$
# which obeys the following properties:
# $$\delta(x-a)=0\ if  x \neq a$$
# $$\int^\infty_{-\infty} f(x)\ \delta(x-a) \ dx =f(a)$$
#
# The normalization constant $c$ is chosen as $c = \frac{1}{\sqrt{2\pi}}$ in such a way that:
# $$\left<\psi_{k'}(x)|\psi_k(x)\right> = \delta(k-k')$$
#
# -------------------

# ### Going back to the ***Free Particle***

# Once we know the normalized eigenfunctions, we can ask some questions:  
# * **Q1:** What is the momentum of a free particle?   
# * **Q2:** What are the units of momentum?
# * **Q3:** Is the momentum of a *Free Particle*  quantized?
#

# In SI units, $k$ has units of $\frac{1}{m}$ and $\hbar$ has units of $J.s =\frac{kg.m^{2}}{s}$, therefore from equation 2 we can see that  $p=\hbar k$ and the units are  $\frac{J.s}{m} = \frac{Kg.m}{s}$
#
# To visualize the wavefunction $\psi_k(x)$ we use Euler relationship $[e^{i\theta}=\cos(\theta)+i\sin(\theta)]$ and graph separately the Real and Imaginary contributions to the wavefunction (even though the wavefunction is *complex*, the $\hat P $ is a Hermitian operator, thus the eigenvalues are measurable and real numbers).
#
# On the other hand, the Probability Density [$\psi^*_k(x) \psi_k(x)$] is a constant, since:
#
# $$ \psi^*_k(x) \psi_k(x) = \frac{1}{\sqrt{2 \pi}}e^{-ikx}\frac{1}{\sqrt{2 \pi}}e^{ikx} =\frac{1}{2\pi}$$

# You can use the cell below to  enter different values for the momentum wavevector $k$ and the length of space $x$ 

# +
# Defining Psi and PsiC (complex conjugate) functions
def psi(x,k): return (1.0/np.sqrt(2.0*np.pi))*(np.cos(k*x)+np.sin(k*x)*1j)
def psiC(x,k): return (1.0/np.sqrt(2.0*np.pi))*(np.cos(k*x)-np.sin(k*x)*1j)

# Reading the input variables from the user
k = float(input("Enter the value for k (in Angstroms-1) = "))
xmax = float(input("Enter the maximum value for x (in Angstroms) = "))

# Generating the wavefunction graph
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
x = np.linspace(-xmax, xmax, 900)
fig, ax = plt.subplots()
lim1=1/np.sqrt(2*np.pi) # Maximum value of the wavefunction
ax.axis([-xmax,xmax,-lim1*1.1,lim1*1.1]) # Defining the limits to be plot in the graph
ax.plot(x, psi(x,k).real, label="Real", color="blue") # Ploting the real part of the wavefunction
ax.plot(x, psi(x,k).imag, label="Imag.", color="red") # Ploting the real part of the wavefunction
ax.hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black") # Adding a horizontal line at 0
# Now we define labels, legend, etc
ax.legend(loc=2);
ax.set_xlabel(r'$x$ (Angstroms)')
ax.set_ylabel(r'$\psi_k(x)$')
str1=r"$k$ = "+str(k)+r" A$^{-1}$"
plt.title('Wavefunction for '+str1)
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0)

# Generating the probability density graph
fig, ax = plt.subplots()
ax.axis([-xmax,xmax,0.0,lim1*lim1*1.4]) # Defining the limits to be plot in the graph
ax.plot(x, psi(x,k)*psiC(x,k),label="Probability Density", color="green") # Plotting the probability density
# Now we define labels, legend, etc
ax.legend(loc=2);
ax.set_xlabel(r'$x$ (Angstroms)')
ax.set_ylabel(r'$\left|\psi_k(x)\right|^2$')
str1=r"$k$ = "+str(k)+r" A$^{-1}$"
plt.title('Probability Density for '+str1)
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0)

# Show the plots on the screen once the code reaches this point
plt.show()


# -

# **What do we learn from this graphical representaton?:** The first plot shows that that the wavefunction is completly ***delocalized*** in the x coordinate. However, wherever we look, the value of the probability density ($|\psi^*_k(x) \psi_k(x) |$)  is the same.

# ### An important question to ask is: what is the probability to find a free particle within a $\Delta x$ region in space?  ###
#
# This is crucial for measurements, since we would never measure ALL space at the same time, but instead, our instrument will measure a section of $x$ space 
# $$\Psi^*_k(x) \Psi_k(x) \Delta x = \left|\Psi_k(x)\right|^2 \Delta x = \left(\frac{1}{\sqrt{2\pi}} e^{ikx}\right)^*\frac{1}{\sqrt{2\pi}} e^{ikx} \Delta x = \frac{1}{2\pi} \Delta x $$
#
# The probability of finding it within a $\Delta x$, at **any** place in space, is the same.
#
# *Thus when the particle has a well-defined momentum ($\hbar k$),  the probability of finding it anywhere in space is the same.*

# ***Localization*** of the particle can be obtained by considering not just a single value of momentum ($\hbar k$), but a range of values.  
# Consider the superposition of waves with a range of momenta given by 
# $p=p_0 \pm \Delta p$ $\hspace{0.25cm}$ or $\hspace{0.25cm}$ $\hbar k = \hbar(k_0 \pm \Delta k)$

# Since the momentum varies continuolsy, we use an integration for the superposition of waves, all equally weigthed:   
# $$\Psi_{\Delta k}(x) =N\int^{k_o+\Delta k}_{k_o-\Delta k}\ e^{ikx}dk$$
#
# $$\Psi_{\Delta k}(x) = \frac{2N\sin(\Delta kx)}{x} e^{ik_ox}$$
#
# where $N$ is a normalization constant and is equal to $N=\frac{1}{2\sqrt{\pi\Delta k}}$

# To understand this new wavefunction, we plot its Real and Imaginary components separately.   

# +
# Defining functions
def psi_contour(x,dk): return (np.sin(dk*x)/(np.sqrt(np.pi*dk)*x))
def psi(x,k,dk): return psi_contour(x,dk)*(np.cos(k*x)+np.sin(k*x)*1j)

# Reading the input variables from the user
#print 'To understand this new wavefunction, we plot the Real and Imaginary components separately'
k = float(input("Enter the value for k_o (in Angstroms-1) = "))
dk = float(input("Enter the value for Delta k (in Angstroms-1) = "))
xmax = float(input("Enter the maximum value for x (in Angstroms) = "))

# Generating the wavefunction graph
lim1 = np.sqrt(dk/np.pi)
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
x = np.linspace(-xmax, xmax, 900)
fig, axes = plt.subplots(1, 2, figsize=(13,4))
str1=r"$k_o  \pm \Delta k$ = "+str(k)+r" $\pm$ "+str(dk)+r" A$^{-1}$"
# axes[0] is the graph at the left
axes[0].axis([-xmax,xmax,-1.1*lim1,1.1*lim1])
axes[0].plot(x, psi(x,k,dk).real, label="Real", color="blue", linewidth=1.8)
axes[0].plot(x, psi_contour(x,dk), label="", linestyle ="--",color="blue", linewidth=1.2)
axes[0].plot(x, -psi_contour(x,dk), label="", linestyle ="--",color="blue", linewidth=1.2)
axes[0].hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black")
axes[0].set_xlabel(r'$x$ (Angstroms)')
axes[0].set_ylabel(r'$Real(\Psi_{\Delta k}(x))$')
axes[0].set_title('Real contribution to $\Psi_{\Delta k}(x)$ \n for '+str1)
# axes[1] is the graph at the right
axes[1].axis([-xmax,xmax,-1.1*lim1,1.1*lim1])
axes[1].plot(x, psi(x,k,dk).imag, label="Imag.", color="red", linewidth=1.8)
axes[1].plot(x, psi_contour(x,dk), label="", linestyle ="--",color="red", linewidth=1.2)
axes[1].plot(x, -psi_contour(x,dk), label="", linestyle ="--",color="red", linewidth=1.2)
axes[1].hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black")
axes[1].yaxis.set_label_position("right")
axes[1].set_xlabel(r'$x$ (Angstroms)')
axes[1].set_ylabel(r'$Imag(\Psi_{\Delta k}(x))$')
axes[1].set_title('Imaginary contribution to $\Psi_{\Delta k}(x)$ \n for '+str1)

# Generating the probability density graph
fig, ax = plt.subplots()
ax.axis([-xmax,xmax,0.0,lim1*lim1*1.1]) # Defining the limits to be plot in the graph
ax.plot(x, (psi(x,k,dk).real)**2+(psi(x,k,dk).imag)**2,label="Probability Density", color="green") # Plotting the probability density
# Now we define labels, legend, etc
ax.legend(loc=2);
ax.set_xlabel(r'$x$ (Angstroms)')
ax.set_ylabel(r'$\left|\Psi_{\Delta k}(x)\right|^2$')
plt.title('Probability Density \n for '+str1)
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0)

# Show the plots on the screen once the code reaches this point
plt.show()


# -

# Assuming $k_o > \Delta k$, the $\cos(k_o x)$ and $\sin(k_o x)$ components oscillate with a larger frequency ($k_o$) and they are modulated by a sinusoidal component with smaller frequency ($\Delta k$).   
#
# If $\Delta k$ is very small, the particle is still delocalized. As $\Delta k$ increases, the particle becomes more localized in a specific region around $x = 0$. This leads us to conclude that as the *uncertainty* in the momentum increases (larger $\Delta k$), the *uncertainty* in the position decreases, and the particle becomes more ***localized***.

# The superposition of waves can be accomplished by giving different contributions to different values of momenta. 
#
# For example, we can construct the superposition with a Gaussian weighting function. In this case, we modulate the contribution of each k-wave with a value given by a Gaussian distribution (maximum contribution from $k_o$, with smaller contributions from waves with other $k$ values).  

# $$\Psi_{\Delta k}(x) =N\int^{+\infty}_{-\infty}{e^{-\frac{(k-k_o)^2}{2\Delta k^2}} \   e^{ikx}}dk,$$  where $\Delta k$ is related to the width of the Gaussian distribution.
#
# After integration, we obtain:
#
# $$\Psi_{\Delta k}(x) = \sqrt{2\pi}N\Delta k e^{-\frac{1}{2}x^2\Delta k^2} e^{ik_ox}$$
#  
# where $N$ is a normalization constant and is equal to $N=\frac{1}{\sqrt{2\Delta k\sqrt{\pi^3}}}$

# +
# Defining functions
def psi_contour(x,dk): return dk*np.exp(-0.5*x*x*dk*dk)/np.sqrt(dk*np.sqrt(np.pi))
def psi(x,k,dk): return psi_contour(x,dk)*(np.cos(k*x)+np.sin(k*x)*1j)

# Reading the input variables from the user
print ("We plot the Real and Imaginary components separately")
k = float(input("Enter the value for k_o (in Angstroms-1) = "))
dk = float(input("Enter the value for Delta k (in Angstroms-1) = "))
xmax = float(input("Enter the maximum value for x (in Angstroms) = "))

# Generating the wavefunction graph
lim1 = dk/np.sqrt(dk*np.sqrt(np.pi))
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
x = np.linspace(-xmax, xmax, 900)
fig, axes = plt.subplots(1, 2, figsize=(13,4))
str1=r"$k_o  \pm \Delta k$ = "+str(k)+r" $\pm$ "+str(dk)+r" A$^{-1}$"
# axes[0] is the graph at the left
axes[0].axis([-xmax,xmax,-1.1*lim1,1.1*lim1])
axes[0].plot(x, psi(x,k,dk).real, label="Real", color="blue")
axes[0].plot(x, psi_contour(x,dk), label="", linestyle ="--",color="blue", linewidth=1.8)
axes[0].plot(x, -psi_contour(x,dk), label="", linestyle ="--",color="blue", linewidth=1.8)
axes[0].hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black")
axes[0].set_xlabel(r'$x$ (Angstroms)')
axes[0].set_ylabel(r'$Real(\Psi_{\Delta k}(x))$')
axes[0].set_title('Real contribution to $\Psi_{\Delta k}(x)$ \n for '+str1)
# axes[1] is the graph at the right
axes[1].axis([-xmax,xmax,-1.1*lim1,1.1*lim1])
axes[1].plot(x, psi(x,k,dk).imag, label="Imag.", color="red")
axes[1].plot(x, psi_contour(x,dk), label="", linestyle ="--",color="red", linewidth=1.8)
axes[1].plot(x, -psi_contour(x,dk), label="", linestyle ="--",color="red", linewidth=1.8)
axes[1].hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black")
axes[1].set_xlabel(r'$x$ (Angstroms)')
axes[1].set_ylabel(r'$Imag(\Psi_{\Delta k}(x))$')
axes[1].set_title('Imaginary contribution to $\Psi_{\Delta k}(x)$ \n for '+str1)

# Generating the probability density graph
fig, ax = plt.subplots()
ax.axis([-xmax,xmax,0.0,lim1*lim1*1.1]) # Defining the limits to be plot in the graph
ax.plot(x, (psi(x,k,dk).real)**2+(psi(x,k,dk).imag)**2,label="Probability Density", color="magenta") # Plotting the probability density
# Now we define labels, legend, etc
ax.legend(loc=2);
ax.set_xlabel(r'$x$ (Angstroms)')
ax.set_ylabel(r'$\left|\Psi_{\Delta k}(x)\right|^2$')
plt.title('Probability Density \n for '+str1)
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0)

# Show the plots on the screen once the code reaches this point
plt.show()


# -

# We can compare the effect of the Gaussian distribution with the equally-weigthed $k$ values:

# Here are a couple questions to consider graphically:
#
# * **Q1:** What do you expect to see for a particle with a momentum ($k_o$) with contributions from waves with  very different momenta? (large value of ($\Delta_k$) 
# * **Q2:** What do you expect to see for a a particel with a momentum ($k_o$) with contributions from waves with  similar momenta? (small value of ($\Delta_k$) 

# +
# Defining functions
def psi_contour(x,dk): return np.sin(dk*x)*np.sin(dk*x)/(np.pi*dk*x*x)
def psi_contourG(x,dk): return dk*dk*np.exp(-x*x*dk*dk)/(dk*np.sqrt(np.pi))

# Reading the input variables from the user
print "What is the probability density of finding the particle as a function of position? \n"
k = float(input("Enter the value for k_o (in Angstroms-1) = "))
dk = float(input("Enter the value for Delta k (in Angstroms-1) = "))
xmax = float(input("Enter the maximum value for x (in Angstroms) = "))

# Generating the probability density graphs
lim0 = dk/np.pi
lim1 = dk*dk/(dk*np.sqrt(np.pi))
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
x = np.linspace(-xmax, xmax, 900)
fig, axes = plt.subplots(1, 2, figsize=(13,4))
str1=r"$k_o \pm \Delta k$ = "+str(k)+r" $\pm$ "+str(dk)+r" A$^{-1}$"
axes[0].plot(x, psi_contour(x,dk), label="", linestyle ="-",color="green", linewidth=1.8)
axes[0].hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black")
axes[0].axis([-xmax,xmax,-0.1*lim0,1.1*lim0])
axes[0].set_xlabel(r'$x$ (Angstroms)')
axes[0].set_ylabel(r'$\left|\Psi_k(x)\right|^2$')
axes[0].set_title('Probability Density for equally weigthed k \n '+str1)
axes[1].plot(x, psi_contourG(x,dk), label="", linestyle ="-",color="magenta", linewidth=1.8)
axes[1].hlines(0.0, -xmax, xmax, linewidth=1.8, linestyle='--', color="black")
axes[1].axis([-xmax,xmax,-0.1*lim1,1.1*lim1])
axes[1].set_xlabel(r'$x$ (Angstroms)')
axes[1].set_ylabel(r'$\left|\Psi_k(x)\right|^2$')
axes[1].set_title('Probability Density for Gaussian-weigthed k \n '+str1)

# Show the plots on the screen once the code reaches this point
plt.show()
# -

# **What can we conclude:** From the graphical representation we can learn that for $\Delta k$ very small  compared to $k_o$ (try plotting for $k_o = 4$ and $\Delta k = 0.04$), the particle is still delocalized. As $\Delta k$ increases the particle becomes more localized in a specific region around $x=0$ (i.e $k_o = 4$  and $\Delta k = 0.4$). When $\Delta k$ is large in comparison to $k_o$ (i.e. $k_o = 4$ $ \Delta k = 16$), we oberve the probability density concentrated around $x=0$.   
#
# This leads us to conclude that as the uncertainty in the momentum increases (larger $\Delta k$), the uncertainty in the position decreases, and the particle becomes more localized. This is the exactly the behavior predicted by the **Heisenberg's uncertainty principle**.

# ###   What other properties can we measure?

# A Free particle has no potential (no acting force), thus the Hamiltonian only has a Kinetic term. The Schrödinger equation is:
#
#  $$-\frac{\hbar^2}{2m} \frac{\partial^2\psi(x)}{\partial x^2} =E\psi(x)$$ 
# Since we already know an eigenfunction of the momentum operator, let's work a bit with commutator properties, and how they can be applied to the Free Particle wavefunction.

# One commutator property says that if two Hermitian operators ($\hat A$ and $\hat B$) commute, then they share a common set of eigenfunctions. 
#
# Let's now check if the Hamiltonian and the momentum operators commute: is $\left[\hat P, \hat H \right] =0$?.   
# $\hat H =\frac{\hat P^2}{2m}$, thus we evaluate:

# $$\left[\hat P, \hat H \right] = \hat P \cdot \hat H - \hat H \cdot \hat P$$ 
# $$\hat P \cdot \frac{\hat P^2}{2m} - \frac{\hat P^2}{2m}\cdot \hat P = \frac {1}{2m}\left(\hat P \cdot \hat P^2-\hat P \cdot \hat P^2 \right) = 0$$

# The two operators commute, leding to the existance of a complete set of eigenfunctions common to both operators.
# Since we already know the eigenfunctions of $\hat P$, we can try to see if they are also
# eigenfunctions of $\hat H$. For $\left|k\right> = \frac{1}{\sqrt{2\pi}}e^{ikx}$, let's evaluate what is the result of $\hat H \left|k\right>$:   
# *(note that we switch to a notation where $\Psi_k(x)= \left|k\right>$)* :

# $$\hat H\left|k\right> = -\frac{\hbar^2}{2m} \frac{\partial^2\left|k\right>}{\partial x^2}
# = -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2}\left(\frac{1}{\sqrt{2\pi}}e^{ikx} \right)$$
#
# $$\hspace 1.5cm = -\frac{\hbar^2}{2m\sqrt{2\pi}} \frac{\partial^2e^{ikx}}{\partial x^2}
# = -\frac{\hbar^2}{2m\sqrt{2\pi}} \left(ik\right)^2 e^{ikx}$$ 
#
# $$= \frac{\hbar^2k^2}{2m} \left(\frac{1}{\sqrt{2\pi}}e^{ikx} \right)  = \frac{p^2}{2m}\left|k\right>$$

# The first and last terms in this derivation show that the momentum eigenfunctions are also eigenfunctions of the Hamiltonian, with eigenvalues equal to the classical form of the kinetic energy. However, one has to be beware that although the value of the energy might look  *classical*, the behavior of the particles are very different. For the quantum mechanical systems, the particle can *never be at rest* (if $\Delta k = 0$ the particle is deloclaized; if the particle is very localized, then its momentum has many different values).

# ### We are now ready to tackle "A Particle in a box with infinite-potential walls"###