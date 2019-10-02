from __future__ import division
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Label, Div
from bokeh.models.widgets import Slider
from surface3d import Surface3d
from bokeh.plotting import figure
import itertools

# Set up widgets
slider_nx = Slider(title="quantum number (nx)", value=2, start=1, end=4, step=1)
slider_ny = Slider(title="quantum number (ny)", value=2, start=1, end=4, step=1)
slider_lx = Slider(title="length of box (Lx) Å", value=4.0, start=4.0, end=7.0, step=1.0)
slider_ly = Slider(title="length of box (Ly) Å", value=4.0, start=4.0, end=7.0, step=1.0)
slider_mass = Slider(title="mass: multiple of electron mass (m)", value=1.0, start=1.0, end=5.0, step=1.0)

h = 6.62607e-34    # planck's constant in joules
me = 9.1093837e-31  # mass of an electron in kg

nx = slider_nx.value
ny = slider_ny.value
lx = slider_lx.value
ly = slider_ly.value


def compute_energy(nx, ny):
    lx = slider_lx.value * 1e-10  # Angstroms to meter
    ly = slider_ly.value * 1e-10
    m = slider_mass.value
    e = h ** 2 / (8 * m * me) * ((nx / lx) ** 2 + (ny / ly) ** 2)  # joules
    e = e * 6.242e+18  # eV
    return e


def compute_wavefunction():
    nx = slider_nx.value
    ny = slider_ny.value
    lx = slider_lx.value
    ly = slider_ly.value

    x = np.linspace(0, lx + 0.001, 70)
    y = np.linspace(0, ly + 0.001, 70)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()

    wave = 2 / np.sqrt(lx * ly) * np.sin(np.pi * nx * xx / lx) * np.sin(np.pi * ny * yy / ly)
    probability = wave * wave
    return [dict(x=xx, y=yy, z=wave), dict(x=xx, y=yy, z=probability)]


def update_data(attrname, old, new):
    source_wave.data, source_prob.data = compute_wavefunction()
    nx = slider_nx.value
    ny = slider_ny.value

    # energies
    for i, p in enumerate(pairs):
        if p[0] == p[1]:
            x = np.linspace(2.1, 2.9, 500)
            x_lable = 2.4
        elif p[0] < p[1]:
            x = np.linspace(1.1, 1.9, 500)
            x_lable = 1.35
            x_lable = 0.8
        elif p[0] > p[1]:
            x = np.linspace(3.1, 3.9, 500)
            x_lable = 3.35
            x_lable = 4
        energy = compute_energy(p[0], p[1]) * np.ones_like(x)
        source_energies[i].data = dict(x=x, y=energy)
        y_lable = energy[0] - 0.5
        if p[0] != p[1]:
            y_lable -= 2
        p_energies_lable[i].x = x_lable
        p_energies_lable[i].y = y_lable

    for p in p_energies:
        p.glyph.line_color = 'gray'
        p.glyph.line_width = 1
    this_energy_number = (nx - 1) * n_num[-1] + ny - 1
    p_energies[this_energy_number].glyph.line_color = 'red'
    p_energies[this_energy_number].glyph.line_width = 3
    for p in p_energies_lable:
        p.text_color = 'gray'
    p_energies_lable[this_energy_number].text_color = 'blue'


source_energies = []
source_energies_lable = []
n_num = [1, 2, 3, 4]
pairs = list(itertools.product(n_num, repeat=2))  # [(1, 1), (1, 2), (2, 1), (2, 2)]

for p in pairs:
    if p[0] == p[1]:
        x = np.linspace(2.1, 2.9, 500)
        x_lable = 2.4
    elif p[0] < p[1]:
        x = np.linspace(1.1, 1.9, 500)
        x_lable = 1.35
        x_lable = 0.8
    elif p[0] > p[1]:
        x = np.linspace(3.1, 3.9, 500)
        x_lable = 3.35
        x_lable = 4
    energy = compute_energy(p[0], p[1]) * np.ones_like(x)
    source_energies.append(ColumnDataSource(data=dict(x=x, y=energy)))
    y_lable = energy[0] - 0.5
    if p[0] != p[1]:
        y_lable -= 2
    source_energies_lable.append([x_lable, y_lable])


plot_energy = figure(plot_height=300, plot_width=600, title="Energy Level",
                     tools="crosshair,reset,save,wheel_zoom", toolbar_location=None,
                     x_range=[0, 5], y_range=[0, 83])
plot_energy.xaxis.visible = False
# plot_energy.outline_line_color = None
plot_energy.grid.visible = None
plot_energy.yaxis.axis_label = "eV"
p_energies = []
p_energies_lable = []
for sc in source_energies:
    p = plot_energy.line('x', 'y', source=sc, line_width=1, line_alpha=1, color='gray')
    p_energies.append(p)

for i, data in enumerate(source_energies_lable):
    x_lable = data[0]
    y_lable = data[1]
    nx = pairs[i][0]
    ny = pairs[i][1]
    text = '{}, {}'.format(nx, ny)
    energy_label = Label(x=x_lable, y=y_lable,
                         text=text, render_mode='css', background_fill_alpha=1.0,
                         text_font_size='8pt', text_color='gray')
    plot_energy.add_layout(energy_label)
    p_energies_lable.append(energy_label)

nx = slider_nx.value
ny = slider_ny.value
this_energy_number = (nx - 1) * n_num[-1] + ny - 1
p_energies[this_energy_number].glyph.line_color = 'red'
p_energies[this_energy_number].glyph.line_width = 3
p_energies_lable[this_energy_number].text_color = 'blue'

data_wave, data_prob = compute_wavefunction()
source_wave = ColumnDataSource(data=data_wave)
source_prob = ColumnDataSource(data=data_prob)
plot_wavefunction = Surface3d(x="x", y="y", z="z", data_source=source_wave)
plot_prob = Surface3d(x="x", y="y", z="z", data_source=source_prob)

for w in [slider_lx, slider_ly, slider_nx, slider_ny, slider_mass]:
    w.on_change('value', update_data)

title_prob = Div(text="Probability Density", style={'font-size': '120%', 'width': '300px', 'font-family': 'serif', 'color': 'black', 'text-align': 'center'}, sizing_mode="stretch_width")
title_wave = Div(text="Wavefunction", style={'font-size': '120%', 'width': '300px', 'font-family': 'serif', 'color': 'black', 'text-align': 'center'}, sizing_mode="stretch_width")
inputs = column(children=[slider_nx, slider_ny, slider_lx, slider_ly, slider_mass], width=400)
prob_wave_plot = row(column(plot_wavefunction, title_wave), column(plot_prob, title_prob))
curdoc().add_root(row(inputs, column(plot_energy, prob_wave_plot), width=1000))
curdoc().title = "Particle in a 2D box"
