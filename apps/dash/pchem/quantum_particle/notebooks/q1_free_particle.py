# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Free Particle

# ## Notes
#
# ### Authors:
#
# * [Vinícius Wilian D. Cruzeiro](https://scholar.google.com/citations?user=iAK04WMAAAAJ). E-mail: vwcruzeiro@ufl.edu
# * Xiang Gao. E-mail: qasdfgtyuiop@ufl.edu
# * Jinze (Richard) Xue. E-mail: jinzexue@ufl.edu
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
#
# **Instructions:**
# - The reader should follow this notebook in the order that it is presented, executing code cells in consecutive order.
# - In order to execute a cell you may click on the cell and click the `PLAY` button, press `Shift+Enter`, or got to `Cell-->Run cells`. The user may also execute all cells at once by clicking on `Cell --> Run All` at the toolbar above. 
#
# ### Libraries used in this notebook:
#
# On the next cell we are going to import the libraries used in this notebook as well as call some important functions.

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# In the next cell we are shutting down eventual warnings displayed by IPython. This cell is optional.

import warnings
warnings.filterwarnings('ignore')

# Executing the next cell prints on the screen the versions of IPython, Python and its libraries on your computer. Please check if the versions are up-to-date to facilitate a smooth running of the program.

import sys # checking the version of Python
import IPython # checking the version of IPython
print("Python version = {}".format(sys.version))
print("IPython version = {}".format(IPython.__version__))
print("Plotly version = {}".format(plotly.__version__))
print("Numpy version = {}".format(np.__version__))


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

# ### Going back to the Free Particle

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
def psi(x,k): 
    return (1.0/np.sqrt(2.0*np.pi))*(np.cos(k*x)+np.sin(k*x)*1j)

def psiC(x,k): 
    return (1.0/np.sqrt(2.0*np.pi))*(np.cos(k*x)-np.sin(k*x)*1j)

# Reading the input variables from the user
k = 4
xmax = 5

# calculation (Prepare data)
x = np.linspace(-xmax, xmax, 900)
lim1=1/np.sqrt(2*np.pi) # Maximum value of the wavefunction
y_real = psi(x, k).real
y_imag = psi(x, k).imag
y_prob = (psi(x,k)*psiC(x,k)).real

# Plot
fig = make_subplots(rows=2, cols=1, subplot_titles=("Wavefunction", "Probability Density"))

# 1st subplot
fig.append_trace(go.Scatter(x=x, y=y_real, name="Real"), row=1, col=1, )
fig.append_trace(go.Scatter(x=x, y=y_imag, name="Imag"), row=1, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
fig.update_yaxes(title_text=r'$\psi_k(x)$', row=1, col=1)

# 2nd subplot
fig.append_trace(go.Scatter(x=x, y=y_prob, name="Probability Density"), row=2, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=2, col=1)
fig.update_yaxes(title_text=r'$\left|\psi_k(x)\right|^2$', range=[0, lim1*lim1*1.4], row=2, col=1)

fig.update_layout(height=600)
fig.show()


# -

# <div class="alert alert-info"> 
#     <p><b>Figure 1</b></p>
# </div>
#
# @@@fig@@@

# **What do we learn from this graphical representaton?:** The first plot shows that that the wavefunction is completly ***delocalized*** in the x coordinate. However, wherever we look, the value of the probability density ($|\psi^*_k(x) \psi_k(x) |$)  is the same.

# ### An important question to ask is: what is the probability to find a free particle within a $\Delta x$ region in space? 
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
def psi_contour(x,dk): 
    return (np.sin(dk*x)/(np.sqrt(np.pi*dk)*x))

def psi(x,k,dk): 
    return psi_contour(x,dk)*(np.cos(k*x)+np.sin(k*x)*1j)

k = 4
dk = 2
xmax = 5

# calculation (Prepare data)
lim1 = np.sqrt(dk/np.pi)
x = np.linspace(-xmax, xmax, 900)
y_real = psi(x,k,dk).real
y_imag = psi(x,k,dk).imag
y_prob = (psi(x,k,dk).real)**2+(psi(x,k,dk).imag)**2

# Plot
title1 = r'$ \text {Real contribution to } \Psi_{\Delta k}(x)$'
title2 = r'$ \text {Imaginary contribution to } \Psi_{\Delta k}(x)$'
title3 = r'$ \text {Probability Density }$'
fig = make_subplots(rows=2, cols=2, subplot_titles=(title1, title2, title3))

# 1st subplot
fig.append_trace(go.Scatter(x=x, y=y_real, name="Real", line=dict(color='blue')), row=1, col=1, )
fig.append_trace(go.Scatter(x=x, y=psi_contour(x,dk), showlegend=False, line = dict(color='blue', width=1, dash='dash')), 
                 row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=-psi_contour(x,dk), showlegend=False, line = dict(color='blue', width=1, dash='dash')),
                 row=1, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=1)

# 2nd subplot
fig.append_trace(go.Scatter(x=x, y=y_imag, name="Imag", line=dict(color='red')), row=1, col=2, )
fig.append_trace(go.Scatter(x=x, y=psi_contour(x,dk), showlegend=False, line=dict(color='red', width=1, dash='dash')), 
                 row=1, col=2)
fig.append_trace(go.Scatter(x=x, y=-psi_contour(x,dk), showlegend=False, line=dict(color='red', width=1, dash='dash')),
                 row=1, col=2)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=2)
fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=2)

# 3rd subplot
fig.append_trace(go.Scatter(x=x, y=y_prob, name="Probability Density", line=dict(color='green')), row=2, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=2, col=1)
fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=2, col=1)

# show
title = r"$k_o  \pm \Delta k = {} \pm {} Å^{{-1}}$".format(k, dk)
fig.update_layout(height=600, title_text=title)
fig.show()


# -

# <div class="alert alert-info"> 
#     <p><b>Figure 2</b></p>
# </div>
#
# @@@fig@@@

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
def psi_contour(x,dk): 
    return dk*np.exp(-0.5*x*x*dk*dk)/np.sqrt(dk*np.sqrt(np.pi))

def psi(x,k,dk): 
    return psi_contour(x,dk)*(np.cos(k*x)+np.sin(k*x)*1j)

k = 4
dk = 2
xmax = 5

# calculation (Prepare data)
lim1 = dk/np.sqrt(dk*np.sqrt(np.pi))
x = np.linspace(-xmax, xmax, 900)
y_real = psi(x,k,dk).real
y_imag = psi(x,k,dk).imag
y_prob = (psi(x,k,dk).real)**2+(psi(x,k,dk).imag)**2

# Plot
title1 = r'$ \text {Real contribution to } \Psi_{\Delta k}(x)$'
title2 = r'$ \text {Imaginary contribution to } \Psi_{\Delta k}(x)$'
title3 = r'$ \text {Probability Density }$'
fig = make_subplots(rows=2, cols=2, subplot_titles=(title1, title2, title3))

# 1st subplot
fig.append_trace(go.Scatter(x=x, y=y_real, name="Real", line=dict(color='blue')), row=1, col=1, )
fig.append_trace(go.Scatter(x=x, y=psi_contour(x,dk), showlegend=False, line = dict(color='blue', width=1, dash='dash')), 
                 row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=-psi_contour(x,dk), showlegend=False, line = dict(color='blue', width=1, dash='dash')),
                 row=1, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=1)

# 2nd subplot
fig.append_trace(go.Scatter(x=x, y=y_imag, name="Imag", line=dict(color='red')), row=1, col=2, )
fig.append_trace(go.Scatter(x=x, y=psi_contour(x,dk), showlegend=False, line=dict(color='red', width=1, dash='dash')), 
                 row=1, col=2)
fig.append_trace(go.Scatter(x=x, y=-psi_contour(x,dk), showlegend=False, line=dict(color='red', width=1, dash='dash')),
                 row=1, col=2)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=2)
fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=2)

# 3rd subplot
fig.append_trace(go.Scatter(x=x, y=y_prob, name="Probability Density", line=dict(color='green')), row=2, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=2, col=1)
fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=2, col=1)

# show
title = r"$k_o  \pm \Delta k = {} \pm {} Å^{{-1}}$".format(k, dk)
fig.update_layout(height=600, title_text=title)
fig.show()


# -

# <div class="alert alert-info"> 
#     <p><b>Figure 3</b></p>
# </div>
#
# @@@fig@@@

# We can compare the effect of the Gaussian distribution with the equally-weigthed $k$ values:

# Here are a couple questions to consider graphically:
#
# * **Q1:** What do you expect to see for a particle with a momentum ($k_o$) with contributions from waves with  very different momenta? (large value of ($\Delta_k$) 
# * **Q2:** What do you expect to see for a a particel with a momentum ($k_o$) with contributions from waves with  similar momenta? (small value of ($\Delta_k$) 
#

# +
# Defining functions
def psi_contour(x,dk): 
    return np.sin(dk*x)*np.sin(dk*x)/(np.pi*dk*x*x)

def psi_contourG(x,dk): 
    return dk*dk*np.exp(-x*x*dk*dk)/(dk*np.sqrt(np.pi))

k = 4
dk = 2
xmax = 5

# calculation (Prepare data)
lim1 = np.sqrt(dk/np.pi)
x = np.linspace(-xmax, xmax, 900)

# Plot

title1 = r'$ \text {Probability Density for equally weigthed k}$'
title2 = r'$ \text {Probability Density for Gaussian-weigthed k }$'
fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2))

# 1st subplot
fig.append_trace(go.Scatter(x=x, y=psi_contour(x,dk), name="Probability Density", line=dict(color='green')), row=1, col=1)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=1, col=1)

# 2nd subplot
fig.append_trace(go.Scatter(x=x, y=psi_contourG(x,dk), name="Probability Density", line=dict(color='magenta')), row=1, col=2)
fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=2)
fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=1, col=2)

# show
title = r"$k_o  \pm \Delta k = {} \pm {} Å^{{-1}}$".format(k, dk)
fig.update_layout(height=400, title_text=title)
fig.show()

# -

# <div class="alert alert-info"> 
#     <p><b>Figure 4</b></p>
# </div>
#
# @@@fig@@@

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

# ### We are now ready to tackle "A Particle in a box with infinite-potential walls"
