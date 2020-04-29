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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Particle in a box with infinite-potential walls

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

# +
import matplotlib as mpl # matplotlib library for plotting and visualization
import matplotlib.pylab as plt # matplotlib library for plotting and visualization
import numpy as np #numpy library for numerical manipulation, especially suited for data arrays

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# -

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

# ## Particle in a box with infinite-potential walls
#
# ### Particle in a 1D box

from IPython.display import Image
Image(filename='particle_in_an_infinite_box_diagram.png')


# ![](./assets/particle_in_an_infinite_box_diagram.png)

# Inside the box, the potential is equal to zero, therefore the Schrödinger equation for this system is given by:
# $$\frac{-\hbar^2}{2m}\frac{\partial^2\psi_n(x)}{\partial{x}^2} =E\psi_n(x) $$
#
# Since the potential is infinity outside the box, the wavefunction must obey the following ***Boundary Condition***:
# $$\psi_n(0)=\psi_n(L)=0$$
# where *L* is the length of the box.

# After solving the Schrödinger equation, the eigenfunctions obtained are given by:
#
# $$\psi_n(x) = \sqrt{\frac{2}{L}} sin\left(\frac{n\pi}{L}x\right)$$
# where  $n=1, 2, ..., \infty $.  
# It is important to emphasize that the quantization (*n* being only positive integers) is a consequence of the boundary conditions.

# Here are a few questions to think about before we move on:
#    * **Q1:** Can you infer, just looking at the graphical representations of $\psi_n(x)$ or $|\psi_n(x)|^2$, what is the quantum state (labeled by its quantum number) *n*?
#    * **Q2:** What is a node? How do the nodes relate to the Kinetic Energy of the system?
#    * **Q3:** Is it possible to find the particle outside the box?
#    * **Q4:** Does it matter that $\psi_n(x)$ has negative values?  
#    * **Q5:** What variables and parameters does $\psi_n(x)$ depend on?

# Some of these questions can be answered by plotting the **Wavefunction**, $\psi_n(x)$ and the **Probability Density**, $|\psi_n(x)|^2$ for different values of $n$.

def get_1dbox(n=5, L=10, num_me=1, all_levels=False):
    # Defining the wavefunction
    def psi(x, n, L):
        return np.sqrt(2.0 / L) * np.sin(float(n) * np.pi * x / L)

    def get_energy(n, L, m):
        return (h**2 / (m * 8)) * (1e10) ** 2 * 6.242e+18 * ((n / L)**2)

    N = 200
    h = 6.62607e-34 
    me = 9.1093837e-31
    m = num_me * me
    
    # calculation (Prepare data)
    x = np.linspace(0, L, N)
    wave = psi(x,n,L)
    prob = wave * wave
    xleft = [0, 0]
    xright = [L, L]
    y_vertical = [-1.3, 1.3]
    # energy levels
    if all_levels:
        energy = list(get_energy(np.linspace(1, 10, 10), L, m))
        for i, e in enumerate(energy):
            if i+1 == n:
                energy[i] = dict(energy=e, color="green", n=i+1)
            else:
                energy[i] = dict(energy=e, color="gray", n=i+1)
    else:
        energy = get_energy(n, L, m)
        energy = [dict(energy=energy, color="green", n=n)]

    # nodes
    nodes_x = np.linspace(start=0, stop=L, num=n, endpoint=False)[1:]
    nodes_y = np.zeros_like(nodes_x)

    # Plot
    fig = make_subplots(rows=2, cols=2, 
                        column_widths=[0.75, 0.25],
                        specs=[[{}, {"rowspan": 2}],[{}, None]],
                        subplot_titles=(r"$\text {Wavefunction}$", r"$\text {Energy Level}$", r"$\text {Probability Density}$"))

    # 1st subplot
    fig.append_trace(go.Scatter(x=x, y=wave, name="Wavefunction"), row=1, col=1)
    # nodes
    fig.append_trace(go.Scatter(x=nodes_x, y=nodes_y, name="node", mode="markers", marker=dict(size=6, color='blue'), showlegend=False), row=1, col=1)
    # wall
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    # axis
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-2, 12], showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text=r'$\psi(x)$', range=[-1.2, 1.2], showgrid=False, zeroline=False, row=1, col=1)

    # 2nd subplot
    for e in energy:
        fig.append_trace(go.Scatter(x=[-0.1, 0.5, 1.1], y=[e["energy"], e["energy"], e["energy"]], name="Energy Level", text=[None, r"$E_{{{}}}={:.2f}\; eV$".format(e['n'], e['energy']), None], textfont=dict(color=e["color"]), textposition="top center", mode="lines+text", showlegend=False, line=dict(color=e["color"], width=2 if e['n']==n else 1)), row=1, col=2)
    fig.update_xaxes(range=[0, 1], showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text=r'$eV$', range=[0, energy[-1]["energy"]+2], showgrid=False, zeroline=False, row=1, col=2)

    # 3rd subplot
    fig.append_trace(go.Scatter(x=x, y=prob, name="Probability Density", line=dict(color='red')), row=2, col=1)
    fig.append_trace(go.Scatter(x=nodes_x, y=nodes_y, name="node", mode="markers", marker=dict(size=6, color='red'), showlegend=False), row=2, col=1)
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=2, col=1, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=2, col=1, )
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-2, 12], showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text=r'$\left|\psi(x)\right|^2$', range=[-1.2, 1.2], showgrid=False, zeroline=False, row=2, col=1)

    # annotations
    annotations = list(fig['layout']['annotations'])
    annotations.append(dict(y=0, x=-1, xref='x1', yref='y1', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))
    annotations.append(dict(y=0, x=L+1, xref='x1', yref='y1', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))
    annotations.append(dict(y=0, x=-1, xref='x3', yref='y3', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))
    annotations.append(dict(y=0, x=L+1, xref='x3', yref='y3', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.update_layout(height=600, title_text=r"$\text {{Particle in an 1D Box}} \;(n={})$".format(n))
    return fig


get_1dbox(n=3)

# <div class="alert alert-info"> 
#     <p><b>Figure 1</b></p>
# </div>
#
# @@@fig@@@

# We can explore the changes in the **Wavefunction** and **Probability Density** for a given state *n* in boxes of different length $L$: 

get_1dbox(L=5)

# <div class="alert alert-info"> 
#     <p><b>Figure 2</b></p>
# </div>
#
# @@@fig@@@

# We can also look at the **allowed values of energy**, given by:
# $$E_n = \frac{n^2 h^2}{8mL^2}$$  
# where *m* is the mass of the particle.
#
# **Note:** Did you notice that $\psi_n(x)$ doesn't depend on the mass of the particle?
#
# In contrast to the solution in the free particle system, for a particle confined within the box, not every energy value is allowed. We see that quantization is a direct consequence of the boundary condition. In other words: confinement leads to quantization.
#
# Let's now analyze how the **Energy Levels** $E_n$ for an electron change as a function of the **size of the box**.

get_1dbox(n=4, L=10, all_levels=True)

# <div class="alert alert-info"> 
#     <p><b>Figure 3</b></p>
# </div>
#
# @@@fig@@@

#   
#   
# and how the *Energy Levels*, $E_n$  change as a function of the **mass of the particle**.
#
#
#

get_1dbox(n=4, L=10, all_levels=True, num_me=3)


# <div class="alert alert-info"> 
#     <p><b>Figure 4</b></p>
# </div>
#
# @@@fig@@@

# We can combine the information from the wavefunctions, probability density, and energies into a single plot that compares the wavefunctions and the probability densities for different states, each one represented at its energy value. These plots are made using the electron mass.

def get_1dbox_combined(L=10, num_me=1):
    # Defining the wavefunction
    def psi(x, n, L):
        return np.sqrt(2.0 / L) * np.sin(float(n) * np.pi * x / L)

    def get_energy(n, L, m):
        return (h**2 / (m * 8)) * (1e10) ** 2 * 6.242e+18 * ((n / L)**2)

    N = 200
    h = 6.62607e-34 
    me = 9.1093837e-31
    m = num_me * me
    
    # calculation (Prepare data)
    x = np.linspace(0, L, N)
    nmax = 7
    waves = []
    probs = []
    energies = []
    nodes_x = []
    for n in range(1, nmax+1, 1):
        wave = psi(x,n,L)
        prob = wave * wave
        energy = get_energy(n, L, m)
        nodes = np.linspace(start=0, stop=L, num=n, endpoint=False)[1:]
        waves.append(wave)
        probs.append(prob)
        energies.append(energy)
        nodes_x.append(nodes)


    xleft = [0, 0]
    xright = [L, L]
    y_vertical = [-1, 1000]

    # Plot
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(r"$\text {Wavefunction}$", r"$\text {Probability Density}$"))

    # 1st subplot
    annotations = list(fig['layout']['annotations'])
    for i, w in enumerate(waves):
        fig.append_trace(go.Scatter(x=x, y=w+energies[i], line=dict(color='blue'), showlegend=False), row=1, col=1)
        fig.append_trace(go.Scatter(x=nodes_x[i], y=np.zeros_like(nodes_x[i])+energies[i], name="node", mode="markers", marker=dict(size=6, color='blue'), showlegend=False), row=1, col=1)
        fig.append_trace(go.Scatter(x=[0, L/2, L], y=[energies[i], energies[i], energies[i]], name="Energy Level", mode="lines", showlegend=False, line=dict(color="green", dash='dot')), row=1, col=1)
        annotations.append(dict(y=energies[i], x=L+2.5, xref='x1', yref='y1', text=r"$E_{{{}}}={:.2f}\; eV$".format(i+1, energies[i]), font=dict(size=11, color="green"), showarrow=False))
    # wall
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    # axis
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-3, 14], showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text=r'$eV$', range=[0, energies[-1]+2], showgrid=False, zeroline=False, row=1, col=1)

    # 2nd subplot
    for i, p in enumerate(probs):
        fig.append_trace(go.Scatter(x=x, y=p+energies[i], showlegend=False, line=dict(color='red')), row=1, col=2)
        fig.append_trace(go.Scatter(x=nodes_x[i], y=np.zeros_like(nodes_x[i])+energies[i], name="node", mode="markers", marker=dict(size=6, color='red'), showlegend=False), row=1, col=2)
        fig.append_trace(go.Scatter(x=[0, L/2, L], y=[energies[i], energies[i], energies[i]], name="Energy Level", mode="lines", showlegend=False, line=dict(color="green", dash='dot')), row=1, col=2)
        annotations.append(dict(y=energies[i], x=L+2.5, xref='x2', yref='y2', text=r"$E_{{{}}}={:.2f}\; eV$".format(i+1, energies[i]), font=dict(size=11, color="green"), showarrow=False))
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=2, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=2, )
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-3, 14], showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text=r'$eV$', range=[0, energies[-1]+2], showgrid=False, zeroline=False, row=1, col=2)
    
    # annotations
    annotations.append(dict(y=energies[-1]/2, x=-1.25, xref='x1', yref='y1', text=r"$V = +\infty$", font=dict(size=11, color="black"), showarrow=False))
    annotations.append(dict(y=energies[-1]/2, x=-1.25, xref='x2', yref='y2', text=r"$V = +\infty$", font=dict(size=11, color="black"), showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.update_layout(height=800, title_text=r"$\text {Particle in an 1D Box}$")
    return fig

get_1dbox_combined(L=10, num_me=1)


# <div class="alert alert-info"> 
#     <p><b>Figure 5</b></p>
# </div>
#
# @@@fig@@@

# Once we know the properties of a 1D box, we can use separation of variables to find the solutions to the 2D and 3D box problem.
#

# ### Particle in a 2D box

# Since the Hamiltonian can be separated into two hamiltonians, one depending only on the variable *x* and one depending only on the variable *y*, the solution to the 2D Schroedinger equation will be a wavefunction which is the product of the 1D solutions in the *x* and *y* directions, with **independent quantum numbers** *n* and *m*:
#
# $$\Psi_{n,m}(x,y) = \psi_{n}(x) \  \psi_{m}(y)  =\frac{2}{\sqrt{L_xL_y}} sin\left(\frac{n\pi}{L_x}x\right) \; sin\left(\frac{m\pi}{L_y}y\right)$$

# +
# Defining the wavefunction
def psi2D(x,y): return 2.0*np.sin(n*np.pi*x)*np.sin(m*np.pi*y)

# Here the users inputs the values of n and m
# n = int(input("Let's look at the Wavefunction for a 2D box \nEnter the value for n = "))
# m = int(input("Enter the value for m = "))

n = 3
m = 2

# Generating the wavefunction graph
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
fig, axes = plt.subplots(1, 1, figsize=(8,8))
axes.imshow(psi2D(X,Y), origin='lower', extent=[0.0, 1.0, 0.0, 1.0])
axes.set_title(r'Heat plot of $\sqrt{L_xL_y}\Psi_{n,m}(x,y)$ for $n='+str(n)+r'$ and $m='+str(m)+r'$')
axes.set_ylabel(r'$y/L_y$')
axes.set_xlabel(r'$x/L_x$')

# Plotting the colorbar for the density plots
fig = plt.figure(figsize=(10,3))
colbar = fig.add_axes([0.05, 0.80, 0.7, 0.10])
norm = mpl.colors.Normalize(vmin=-2.0, vmax=2.0)
mpl.colorbar.ColorbarBase(colbar, norm=norm, orientation='horizontal')

# Show the plots on the screen once the code reaches this point
plt.show()
# -

# <div class="alert alert-info"> 
#     <p><b>Figure 6</b></p>
# </div>
#
# @@@fig@@@

# Since the variables are independent, a vertical slice in this plot shows the *y* dependence of the wavefunction, thus it would look like a 1D particle in a box. Similarly, a horizontal slice gives the *x* dependence, and behaves as a 1D wavefunction. Let's see that:

# +
# Here the users inputs the values of n and m
# yo = float(input("Enter the value of y/L_y for the x-axes slice ="))
# xo = float(input("Enter the value of x/L_x for the y-axes slice ="))

yo = 0.25
xo = 0.6


# Generating the wavefunction graph
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
x = np.linspace(0, 1.0, 900)
fig, ax = plt.subplots()
lim1=2.0 # Maximum value of the wavefunction
ax.axis([0.0,1.0,-1.1*lim1,1.1*lim1]) # Defining the limits to be plot in the graph
str1=r"$n = "+str(n)+r", m = "+str(m)+r", y_o = "+str(yo)+r"\times L_y$"
ax.plot(x, psi2D(x,yo), linestyle='--', label=str1, color="orange", linewidth=2.8) # Plotting the wavefunction
ax.hlines(0.0, 0.0, 1.0, linewidth=1.8, linestyle='--', color="black") # Adding a horizontal line at 0
# Now we define labels, legend, etc
ax.legend(loc=2);
ax.set_xlabel(r'$x/L_x$')
ax.set_ylabel(r'$\sqrt{L_xL_y}\Psi_{n,m}(x,y_o)$')
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0)

# Generating the wavefunction graph
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
y = np.linspace(0, 1.0, 900)
fig, ax = plt.subplots()
lim1=2.0 # Maximum value of the wavefunction
ax.axis([0.0,1.0,-1.1*lim1,1.1*lim1]) # Defining the limits to be plot in the graph
str1=r"$n = "+str(n)+r", m = "+str(m)+r", x_o = "+str(xo)+r"\times L_x$"
ax.plot(y, psi2D(xo,y), linestyle='--', label=str1, color="blue", linewidth=2.8) # Plotting the wavefunction
ax.hlines(0.0, 0.0, 1.0, linewidth=1.8, linestyle='--', color="black") # Adding a horizontal line at 0
# Now we define labels, legend, etc
ax.legend(loc=2);
ax.set_xlabel(r'$y/L_y$')
ax.set_ylabel(r'$\sqrt{L_xL_y}\Psi_{n,m}(x_o,y)$')
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0)

# Show the plots on the screen once the code reaches this point
plt.show()


# -

# <div class="alert alert-info"> 
#     <p><b>Figure 7</b></p>
# </div>
#
# @@@fig@@@

# Here are some questions to consider from the plots:
# * **Q1:** How many nodes does $\Psi_{n,m}$ have in the *x* axis?
# * **Q2:** How many nodes does $\Psi_{n,m}$ have in the *y* axis?
# * **Q3:** Can you scketch the equivalent plot for a non-symmetric box (for exampe, with $L_x = 2L_y$)?

# How about the energies?  
# When the Hamiltonian can be separated into independent Hamiltonians, the wavefunction can be built as the product of independent wavefunctions and the energy will be given by the sum of the 1D energies:

# $$E_{n,m} = E_n +E_m = \ \  \frac{ h^2}{8m_p} \frac{n^2}{L_x^2}+ \frac{ h^2}{8m_p}\frac{m^2}{L_y^2} = \ \  \frac{ h^2}{8m_p}\left(\frac{n^2}{L_x^2}+\frac{m^2}{L_y^2}\right)$$  

# Depending on the values of $L_x$ and $L_y$ (the lenght of the box on each side), we may get **degenerated states**: more than one state with the same energy.  
#
# Let's look at these Energy Levels as a function of quantum numbers and box sizes:

# +
# Defining the energy as a function
def En2D(n,m,L1,L2): return 37.60597*((float(n)/L1)**2+ (float(m)/L2)**2)

# Reading data from the user
# L1 = float(input("Can we count DEGENERATE states?\nEnter the value for Lx (in Angstroms) = "))
# nmax1 = int(input("Enter the maximum value of n to consider = "))
# L2 = float(input("Enter the value for Ly (in Angstroms) = "))
# mmax2 = int(input("Enter the maximum value of m to consider = "))

L1 = 5
nmax1 = 3
L2 = 5
mmax2 = 3

# Plotting the energy levels
plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(nmax1*2+2,nmax1*3))
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()
ax.spines['bottom'].set_color('none')
ax.axes.get_xaxis().set_visible(False)
ax.spines['top'].set_color('none')
val = 1.1*(En2D(nmax1,mmax2,L1,L2))
val2= 1.1*max(L1,L2)
ax.axis([0.0,3*nmax1,0.0,val])
ax.set_ylabel(r'$E_n$ (eV)')
for n in range(1,nmax1+1):
    for m in range(1, mmax2+1):
        str1="$"+str(n)+r","+str(m)+r"$"
        str2=" $E = %.3f$ eV"%(En2D(n,m,L1,L2))
        ax.text(n*2-1.8, En2D(n,m,L1,L2)+ 0.005*val, str1, fontsize=20, color="blue")
        ax.hlines(En2D(n,m,L1,L2), n*2-2, n*2-1, linewidth=3.8, color="red")
        ax.hlines(En2D(n,m,L1,L2), 0.0, nmax1*2+1, linewidth=1., linestyle='--', color="black")
        ax.text(nmax1*2+1, En2D(n,m,L1,L2)+ 0.005*val, str2, fontsize=16, color="blue")       
plt.title("Energy Levels for \n ", fontsize=30)
str1=r"$L_x = "+str(L1)+r"$ A, $n_{max} = "+str(nmax1)+r"$     $L_y = "+str(L2)+r"$ A,  $m_{max}="+str(mmax2)+r"$"
ax.text(0.1,val, str1, fontsize=25, color="black")
# Show the plots on the screen once the code reaches this point
plt.show()
# -

# <div class="alert alert-info"> 
#     <p><b>Figure 8</b></p>
# </div>
#
# @@@fig@@@

# In this graph, each state is represented by the quantum numbers $(n,m)$. For example, if $L_x =L_y$ then the state described by $(a,b)$ will be degenerate with the state described by $(b,a)$.  
#
# Going back and plotting the wavefunction for  $(a,b)$ and then for $(b,a)$  you will notice that their properties are different since the number of nodes in one direction will be different from the number of nodes in the other direction (unless $a=b$).  
#
# The quantum numbers identify individual states, whereas the energies are associated with levels. 

# ### We are now ready to tackle "A Particle in a box a box with finite-potential walls" 
