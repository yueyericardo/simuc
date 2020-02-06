import numpy as np
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, CheckboxButtonGroup
from bokeh.plotting import figure

# Set up widgets
text = TextInput(title="title", value='Particle in a 1D box')
quantum_number = Slider(title="quantum number (n)", value=2, start=1, end=5, step=1)
length_of_box = Slider(title="length of box (L)", value=5.0, start=3.0, end=7.0, step=1.0)
mass = Slider(title="mass of particle: multiple of electron mass (m)", value=1.0, start=1.0, end=5.0, step=1.0)

h = 6.62607e-34    # planck's constant in joules
me = 9.1093837e-31  # mass of an electron in kg


def psi(x, n, L):
    return np.sqrt(2.0 / L) * np.sin(float(n) * np.pi * x / L)


def get_energy(n, L, m):
    return (h**2 / (m * 8)) * (1e10) ** 2 * 6.242e+18 * ((float(n) / L)**2)


# Set up data
L = 5
n = 2
N = 200
x = np.linspace(0, L, N)
y_wave = psi(x, n, L)
source_wave = ColumnDataSource(data=dict(x=x, y=y_wave))
y_prob = psi(x, n, L)**2
source_prob = ColumnDataSource(data=dict(x=x, y=y_prob))
source_energies = []
for i in range(1, 6):
    energy = get_energy(i, L, me) * np.ones_like(x)
    sc = ColumnDataSource(data=dict(x=x, y=energy))
    source_energies.append(sc)
wall_x = L * np.ones_like(x)
wall_y = np.linspace(-1.5, 120, num=N)
source_wall = ColumnDataSource(data=dict(x=wall_x, y=wall_y))

# Set up plot
plot_prob = figure(plot_height=300, plot_width=600, title="Probability Density",
                   tools="crosshair,reset,save,wheel_zoom", toolbar_location=None,
                   x_range=[0, 7], y_range=[-1.5, 1.5])
plot_prob.grid.visible = None
plot_prob.xaxis.visible = False
plot_prob.outline_line_color = None
p_prob = plot_prob.line('x', 'y', source=source_prob, line_width=1, line_alpha=0.6, color='red')
plot_prob.line('x', 'y', source=source_wall, line_width=1, line_alpha=0.6, color='black')

plot_wave = figure(plot_height=300, plot_width=600, title="Wavefunction",
                   tools="crosshair,reset,save,wheel_zoom", toolbar_location=None,
                   x_range=[0, 7], y_range=[-1.5, 1.5])
plot_wave.grid.visible = None
plot_wave.xaxis.visible = False
plot_wave.outline_line_color = None
p_wave = plot_wave.line('x', 'y', source=source_wave, line_width=1, line_alpha=0.6, color='blue')
plot_wave.line('x', 'y', source=source_wall, line_width=1, line_alpha=0.6, color='black')

plot_energy = figure(plot_height=200, plot_width=600, title="Energy Level",
                     tools="crosshair,reset,save,wheel_zoom", toolbar_location=None,
                     x_range=[0, 7], y_range=[0, 120])
plot_energy.xaxis.visible = False
plot_energy.outline_line_color = None
plot_energy.grid.visible = None
plot_energy.yaxis.axis_label = "eV"
p_energies = []
for sc in source_energies:
    p = plot_energy.line('x', 'y', source=sc, line_width=1, line_alpha=0.6, color='gray')
    p_energies.append(p)
p_energies[n - 1].glyph.line_color = 'green'
p_energies[n - 1].glyph.line_width = 2
plot_energy.line('x', 'y', source=source_wall, line_width=1, line_alpha=0.6, color='black')


# Set up callbacks
def update_title(attrname, old, new):
    p_wave.title.text = text.value


text.on_change('value', update_title)


def update_data(attrname, old, new):
    # Get the current slider values
    n = quantum_number.value
    L = length_of_box.value
    m = mass.value

    # Generate the new curve
    x = np.linspace(0, L, N)
    y_wave = psi(x, n, L)
    source_wave.data = dict(x=x, y=y_wave)
    y_prob = psi(x, n, L)**2
    source_prob.data = dict(x=x, y=y_prob)
    # energies
    for i, level in enumerate(range(1, 6)):
        energy = get_energy(level, L, m * me) * np.ones_like(x)
        source_energies[i].data = dict(x=x, y=energy)
    for p in p_energies:
        p.glyph.line_color = 'gray'
        p.glyph.line_width = 1
    p_energies[n - 1].glyph.line_color = 'green'
    p_energies[n - 1].glyph.line_width = 2

    wall_x = L * np.ones_like(x)
    source_wall.data = dict(x=wall_x, y=wall_y)


for w in [quantum_number, length_of_box, mass]:
    w.on_change('value', update_data)

title = bokeh.models.Div(text="Demo: Particle in a 1D box<br><br>")
energy_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-08-29-CodeCogsEqn-1.png'><br><br><br>")
prob_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-08-29-CodeCogsEqn%20-2--1.png'><br><br><br>")
wave_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-08-29-CodeCogsEqn-1.gif'><br><br><br>")
# Set up layouts and add to document
inputs = column(children=[title, quantum_number, length_of_box, mass, energy_png, prob_png, wave_png], sizing_mode='stretch_width')

curdoc().add_root(row(inputs, column(plot_energy, plot_prob, plot_wave), width=1000))
curdoc().title = "Particle in a 1D box"

