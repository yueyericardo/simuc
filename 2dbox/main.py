from __future__ import division
import bokeh
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from surface3d import Surface3d

# Set up widgets
slider_nx = Slider(title="quantum number (nx)", value=2, start=1, end=5, step=1)
slider_ny = Slider(title="quantum number (ny)", value=2, start=1, end=5, step=1)
slider_lx = Slider(title="length of box (Lx)", value=5.0, start=3.0, end=7.0, step=1.0)
slider_ly = Slider(title="length of box (Ly)", value=5.0, start=3.0, end=7.0, step=1.0)
mass = Slider(title="mass of particle: multiple of electron mass (m)", value=1.0, start=1.0, end=5.0, step=1.0)

nx = slider_nx.value
ny = slider_ny.value
lx = slider_lx.value
ly = slider_ly.value
x = np.arange(0, 5.1, 0.1)
y = np.arange(0, 5.1, 0.1)
xx, yy = np.meshgrid(x, y)
xx = xx.ravel()
yy = yy.ravel()


def compute():
    nx = slider_nx.value
    ny = slider_ny.value
    lx = slider_lx.value
    ly = slider_ly.value
    value = 2 / np.sqrt(lx * ly) * np.sin(np.pi * nx * xx / lx) * np.sin(np.pi * ny * yy / ly)
    return dict(x=xx, y=yy, z=value)


def update_data(attrname, old, new):
    source.data = compute()


source = ColumnDataSource(data=compute())
plot_surface = Surface3d(x="x", y="y", z="z", data_source=source)

for w in [slider_lx, slider_ly, slider_nx, slider_ny]:
    w.on_change('value', update_data)

inputs = column(children=[slider_nx, slider_ny], sizing_mode='stretch_width')
curdoc().add_root(row(inputs, column(plot_surface), width=1000))
curdoc().title = "Particle in a 2D box"
