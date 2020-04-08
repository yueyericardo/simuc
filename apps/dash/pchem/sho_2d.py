# -*- coding: utf-8 -*-
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import util
from server import server

external_stylesheets = ['https://codepen.io/yueyericardo/pen/OJyLrKR.css']
external_scripts = ['https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2020-03-21-iframeResizer.contentWindow.min.js']
app = dash.Dash(name='sho_2d',
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                server=server,
                routes_pathname_prefix='/sho_2d/')

Markdown_text = r"""
<h2><center>2D Simple Harmonic Oscillator</center></h2>

"""

Markdown_text = util.convert_latex(Markdown_text)


#####################################################################

import numpy as np
from scipy.constants import physical_constants as pc
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly
import plotly.graph_objects as go

amu_to_au = 1.0/(pc['kilogram-atomic mass unit relationship'][0]*pc['atomic unit of mass'][0])  # 1822.888479031408
hartree_to_cm1 = pc['hartree-hertz relationship'][0]/pc['speed of light in vacuum'][0]/100.0  # 2.194746313705e5
sec = pc['atomic unit of time'][0]  # 2.418884326505e-17
cee = 100.0*pc['speed of light in vacuum'][0]  # 2.99792458e10 cm/s
bohr_to_angstroms = pc['atomic unit of length'][0]*1e10
hartree_to_ev = pc['hartree-electron volt relationship'][0]

mumass = 1.0
we = 400.00   # Harmonic frequency in wavenumbers
nmu = mumass * amu_to_au
nfreq = we * sec * cee  # angular harmonic frequency in au
nkeq = ((2.0*np.pi*nfreq)**2)*nmu  # force constant in au
k = -1.0/(2.0*nmu)   # Use this if you want to change the units (don't)

NUM = 40
half_r = 2

x_ = np.linspace(-half_r, half_r, NUM)
y_ = np.linspace(-half_r, half_r, NUM)
x, y = np.meshgrid(x_, y_)
xy = np.stack((x, y), axis=-1)

xy = np.reshape(xy, (-1, 2))


def potential(point, k):
    x, y = point
    r = np.sqrt(x*x + y*y)
    u = 1/2 * k * (r) ** 2
    return u


def gen_V(x, u):
    """
    Assemble the matrix representation of the potential energy
    """
    V = np.zeros((NUM*NUM, NUM*NUM)).astype(np.complex)
    for m in range(NUM*NUM):
        V[m, m] = u[m]
    return(V)


def gen_T(xy, k):
    dx = xy[1][0] - xy[0][0]
    dy = xy[NUM][1] - xy[0][1]

    T = np.zeros((NUM*NUM, NUM*NUM)).astype(np.complex)
    row_len = int(np.sqrt(NUM*NUM))

    for m in range(0, NUM*NUM):
        for n in [m-NUM, m-1, m, m+1, m+NUM]:
            if (n >= 0 and n < NUM*NUM):
                factor = laplace_factor(m, n, row_len, dx, dy)
                T[m, n] = k * factor
    return(T)


def convert(xy, row_len):
    return xy // row_len, xy % row_len


def laplace_factor(idxy1, idxy2, row_len, dx, dy):

    idx1, idy1 = convert(idxy1, row_len)
    idx2, idy2 = convert(idxy2, row_len)
    iddx = np.abs(idx2 - idx1)
    iddy = np.abs(idy2 - idy1)

    if (iddx == 1 and iddy == 0):
        return 1 / (dx ** 2)
    elif (iddx == 0 and iddy == 1):
        return 1 / (dy ** 2)
    elif (iddx == 0 and iddy == 0):
        return -2 / (dy ** 2) - 2 / (dx ** 2)
    else:
        return 0


def solve_eigenproblem(H):
    """
    Solve an eigenproblem and return the eigenvalues and eigenvectors.
    """
    vals, vecs = np.linalg.eigh(H)
    idx = np.real(vals).argsort()
    vals = vals[idx]
    vecs = vecs.T[idx]
    return(vals, vecs)


u = np.zeros(xy.shape[0])

for i, point in enumerate(xy):
    u[i] = potential(point, nkeq)

V = gen_V(xy, u)
T = gen_T(xy, k)
H = T + V
evals, evecs = solve_eigenproblem(H)

# hartree to ev
potential_surface = u.reshape(NUM, NUM) * hartree_to_ev

fig = go.Figure(data=[go.Surface(z=potential_surface, x=x_, y=y_, colorscale="ylgn", showscale=False)])

fig.update_layout(title='Potential Surface', autosize=False,
                  width=500, height=500,
                  scene=dict(
                        xaxis=dict(nticks=5, range=[-half_r, half_r],),
                        yaxis=dict(nticks=5, range=[-half_r, half_r],),
                        # zaxis=dict(nticks=3, range=[-0.12, 0.12],),
                        xaxis_title='x (Å)',
                        yaxis_title='y (Å)',
                        zaxis_title='Energy (eV)'
                  ),
                  margin=dict(l=60, r=60, b=60, t=60))
N = 100
colors = ['red', 'blue', 'green', 'orange', 'purple']
color_idx = 0

for idx in range(15):
    scale = np.sqrt(abs(2 * evals[idx] / nkeq))
    t = np.linspace(0, 2*np.pi, N)
    x_t, y_t, z_t = scale * np.cos(t), scale * np.sin(t), evals[idx] * hartree_to_ev *  np.ones_like(t)
    if (idx > 0 and (evals[idx] - evals[idx-1] > 0.001)):
        color_idx += 1
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines',
                               marker=dict(size=5, color=colors[color_idx], opacity=0.8),
                               name="State {}".format(idx+1)
                               ))


app.layout = html.Div([
    dcc.Markdown(Markdown_text, dangerously_allow_html=True),
    html.Div([
        dcc.Graph(figure=fig),
        dcc.Graph(id='graph-with-slider')],
        style={'columnCount': 2}
    ),
    html.Div([
        html.Div([html.Br(), html.Br()],
                 style={'min-height': '70px'}),
        html.Div([
            html.Label('Choose State'),
            dcc.Slider(
                id='state-slider',
                min=1,
                max=15,
                value=1,
                marks={str(x): str(x) for x in np.arange(1, 16, 1)},
                step=1)],
                 )],
             style={'columnCount': 2, 'padding': '0'}),
    html.Div([html.Br(), html.Br()],
             style={'min-height': '50px'})
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('state-slider', 'value'),
     ])
def update_figure(idx):
    wave = evecs[idx-1].reshape([NUM, NUM])

    fig = go.Figure(data=[go.Surface(z=wave.real, x=x_, y=y_, cmin=-0.12, cmax=0.12, showscale=False)])

    fig.update_layout(title='Wavefunction', autosize=False,
                      width=500, height=500,
                      scene=dict(
                          xaxis=dict(nticks=5, range=[-half_r, half_r],),
                          yaxis=dict(nticks=5, range=[-half_r, half_r],),
                          zaxis=dict(nticks=3, range=[-0.12, 0.12],),
                          xaxis_title='x (Å)',
                          yaxis_title='y (Å)',
                          ),
                      margin=dict(l=60, r=60, b=60, t=60))
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
