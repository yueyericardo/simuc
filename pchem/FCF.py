import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
from scipy.constants import physical_constants as pc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import bokeh
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.plotting import figure

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

    def __init__(self, we, De, re, Te, mu_au):
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


def calculate_data(mu_au, state_1, state_2):
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

    # potential source
    xlow = 2.
    xhigh = 4.5
    ylow = -0.2
    yhigh = 6.0
    overhang = 0.3
    x = np.linspace(xlow / rscale, xhigh / rscale, 1000)
    source_morse_1 = ColumnDataSource(data=dict(x=rscale*x, y=escale*morse(x, state_1)))
    source_morse_2 = ColumnDataSource(data=dict(x=rscale*x, y=escale*morse(x, state_2)))
    source_energy_list_1 = []
    source_energy_list_2 = []
    source_wave_list_1 = []
    source_wave_list_2 = []
    source_lim_1 = ColumnDataSource(data=dict(x=rscale*x, y=np.ones_like(x)*(state_1.dissociation_au)*escale))
    source_lim_2 = ColumnDataSource(data=dict(x=rscale*x, y=np.ones_like(x)*(state_2.dissociation_au)*escale))

    # source energy
    for i in range(20):
        tmp_N = 1000
        tmp_start = rscale*left_position(i, state_1)
        tmp_stop = rscale*right_position(i, state_1)
        tmp_x = np.linspace(tmp_start, tmp_stop, num=tmp_N)
        tmp_y = np.ones_like(tmp_x) * escale*(g(i, state_1)+state_1.Te_au)
        source_energy_list_1.append(ColumnDataSource(data=dict(x=tmp_x, y=tmp_y)))
        if i % 3 == 0:
            mask = (xx > left_position(i, state_1)-overhang) & (xx < right_position(i, state_1)+overhang)
            tmp_x = xx[mask]*rscale
            tmp_y = (0.005*mod_sq(xx[mask], evecs1[i, mask]) + evals1[i].real)*escale
            source_wave_list_1.append(ColumnDataSource(data=dict(x=tmp_x, y=tmp_y)))

    # source wavefunction
    for i in range(20):
        tmp_N = 1000
        tmp_start = rscale*left_position(i, state_2)
        tmp_stop = rscale*right_position(i, state_2)
        tmp_x = np.linspace(tmp_start, tmp_stop, num=tmp_N)
        tmp_y = np.ones_like(tmp_x) * escale*(g(i, state_2)+state_2.Te_au)
        source_energy_list_2.append(ColumnDataSource(data=dict(x=tmp_x, y=tmp_y)))
        if i % 3 == 0:
            mask = (xx > left_position(i, state_2)-overhang) & (xx < right_position(i, state_2)+overhang)
            tmp_x = xx[mask]*rscale
            tmp_y = (0.005*mod_sq(xx[mask], evecs2[i, mask]) + evals2[i].real)*escale
            source_wave_list_2.append(ColumnDataSource(data=dict(x=tmp_x, y=tmp_y)))

    # overlap source
    vmax_1 = gvmax(state_1)
    vmax_2 = gvmax(state_2)
    overlap = np.zeros((vmax_1, vmax_2))
    for i in range(vmax_1):
        for j in range(vmax_2):
            overlap[i, j] = inner_product(x, evecs1[i], evecs2[j]).real
    source_overlap = ColumnDataSource(data=dict(image=[overlap], dw=[overlap.shape[0]], dh=[overlap.shape[1]]))
    source_FCF = ColumnDataSource(data=dict(image=[abs(overlap)**2], dw=[overlap.shape[0]], dh=[overlap.shape[1]]))

    return [source_morse_1, source_morse_2, source_energy_list_1, source_energy_list_2,
            source_wave_list_1, source_wave_list_2, source_lim_1, source_lim_2,
            source_overlap, source_FCF]


# setup widget
slider_mass_1 = Slider(title="Mass of Atom 1", value=10, start=1, end=15, step=1)
slider_mass_2 = Slider(title="Mass of Atom 2", value=10, start=1, end=15, step=1)

# state 1
slider_we_1 = Slider(title="State 1: we (wavenumbers)", value=2500, start=1000, end=5000, step=100)
slider_De_1 = Slider(title="State 1: De", value=25000, start=10000, end=50000, step=100)
slider_re_1 = Slider(title="State 1: re Å", value=2.5, start=1.0, end=10.0, step=0.1)
slider_Te_1 = Slider(title="State 1: Te", value=0, start=0, end=50000, step=100)
# state 2
slider_we_2 = Slider(title="State 2: we (wavenumbers)", value=1900, start=1000, end=5000, step=100)
slider_De_2 = Slider(title="State 2: De", value=20000, start=10000, end=50000, step=100)
slider_re_2 = Slider(title="State 2: re Å", value=2.9, start=1.0, end=10.0, step=0.1)
slider_Te_2 = Slider(title="State 2: Te", value=26000, start=0, end=50000, step=100)

# input
mass_1 = slider_mass_1.value
mass_2 = slider_mass_2.value
mumass = mass_1 * mass_2 / (mass_1 + mass_2)
mu_au = mumass * amu_to_au

# build state
state_1 = State(we=slider_we_1.value, De=slider_De_1.value, re=slider_re_1.value, Te=slider_Te_1.value, mu_au=mu_au)
state_2 = State(we=slider_we_2.value, De=slider_De_2.value, re=slider_re_2.value, Te=slider_Te_2.value, mu_au=mu_au)

source_morse_1, source_morse_2, source_energy_list_1, source_energy_list_2, \
    source_wave_list_1, source_wave_list_2, source_lim_1, source_lim_2, \
    source_overlap, source_FCF = calculate_data(mu_au, state_1, state_2)

# bokeh plot
plot_potential = figure(plot_height=600, plot_width=1000,
                        title="Title",
                        tools="crosshair,reset,save,wheel_zoom",
                        toolbar_location=None,
                        y_range=[-0.2, 6], x_range=[2, 4.5])
plot_potential.line('x', 'y', source=source_morse_1, line_width=2, line_alpha=1, color='blue', legend='State 1')
plot_potential.line('x', 'y', source=source_morse_2, line_width=2, line_alpha=1, color='red', legend='State 2')

for sc in source_energy_list_1:
    plot_potential.line('x', 'y', source=sc, line_width=1, line_alpha=1, color='blue')

for sc in source_energy_list_2:
    plot_potential.line('x', 'y', source=sc, line_width=1, line_alpha=1, color='red')

for sc in source_wave_list_1:
    plot_potential.line('x', 'y', source=sc, line_width=1, line_alpha=1, color='blue')

for sc in source_wave_list_2:
    plot_potential.line('x', 'y', source=sc, line_width=1, line_alpha=1, color='red')

plot_potential.line('x', 'y', source=source_lim_1, line_width=0.8, line_alpha=1, color="black", line_dash='dashed')
plot_potential.line('x', 'y', source=source_lim_2, line_width=0.8, line_alpha=1, color="black", line_dash='dashed')

# overlap and FCF
plot_overlap = figure(plot_height=350, plot_width=350,
                      title="Title",
                      tools="crosshair,reset,save,wheel_zoom",
                      toolbar_location=None,)

plot_empty = figure(plot_height=350, plot_width=300,
                    tools="crosshair,reset,save,wheel_zoom",
                    toolbar_location=None,)
plot_empty.grid.visible = None
plot_empty.border_fill_color = None
plot_empty.outline_line_color = None

plot_overlap.x_range.range_padding = plot_overlap.y_range.range_padding = 0

plot_FCF = figure(plot_height=350, plot_width=350,
                  title="Title",
                  tools="crosshair,reset,save,wheel_zoom",
                  toolbar_location=None,)
plot_FCF.x_range.range_padding = plot_FCF.y_range.range_padding = 0

plot_overlap.image(image='image', source=source_overlap, x=0, y=0, dw='dw', dh='dh', palette="Spectral11")
plot_FCF.image(image='image', source=source_FCF, x=0, y=0, dw='dw', dh='dh', palette="Spectral11")


def update_source_list(old_source, new_source):
    min_len = len(old_source) if len(old_source) < len(new_source) else len(new_source)
    for i in range(min_len):
        old_source[i].data = new_source[i].data


def update_data(attrname, old, new):
    # input
    mass_1 = slider_mass_1.value
    mass_2 = slider_mass_2.value
    mumass = mass_1 * mass_2 / (mass_1 + mass_2)
    mu_au = mumass * amu_to_au

    # build state
    state_1 = State(we=slider_we_1.value, De=slider_De_1.value, re=slider_re_1.value, Te=slider_Te_1.value, mu_au=mu_au)
    state_2 = State(we=slider_we_2.value, De=slider_De_2.value, re=slider_re_2.value, Te=slider_Te_2.value, mu_au=mu_au)

    source_morse_1_tmp, source_morse_2_tmp, source_energy_list_1_tmp, source_energy_list_2_tmp, \
        source_wave_list_1_tmp, source_wave_list_2_tmp, source_lim_1_tmp, source_lim_2_tmp, \
        source_overlap_tmp, source_FCF_tmp = calculate_data(mu_au, state_1, state_2)

    source_morse_1.data = source_morse_1_tmp.data
    source_morse_2.data = source_morse_2_tmp.data

    update_source_list(source_wave_list_1, source_wave_list_1_tmp)
    update_source_list(source_wave_list_2, source_wave_list_2_tmp)
    update_source_list(source_energy_list_1, source_energy_list_1_tmp)
    update_source_list(source_energy_list_2, source_energy_list_2_tmp)

    source_lim_1.data = source_lim_1_tmp.data
    source_lim_2.data = source_lim_2_tmp.data
    source_overlap.data = source_overlap_tmp.data
    source_FCF.data = source_FCF_tmp.data


for w in [slider_mass_1, slider_mass_2, slider_we_1, slider_De_1, slider_re_1, slider_Te_1, slider_we_2, slider_De_2, slider_re_2, slider_Te_2]:
    w.on_change('value', update_data)

left = column(children=[slider_mass_1, slider_mass_2, slider_we_1, slider_De_1, slider_re_1, slider_Te_1, slider_we_2, slider_De_2, slider_re_2, slider_Te_2],
              sizing_mode='fixed', width=400, height=1000)

# Set up layouts and add to document
middle = column(plot_potential, row(plot_overlap, plot_empty, plot_FCF), sizing_mode='fixed', width=700, height=700)
all_layout = column(row(left, middle), sizing_mode="fixed", width=1400)

curdoc().add_root(all_layout)
curdoc().title = "Franck-Condon Factor"
