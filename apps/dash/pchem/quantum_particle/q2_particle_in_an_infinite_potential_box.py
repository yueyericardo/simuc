# -*- coding: utf-8 -*-
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_defer_js_import as dji
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
from server import server
import util
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import os

filepath = os.path.split(os.path.realpath(__file__))[0]

external_stylesheets = ['https://codepen.io/yueyericardo/pen/OJyLrKR.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css']
external_scripts = [
    'https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2020-03-21-iframeResizer.contentWindow.min.js']


app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                server=server,
                routes_pathname_prefix='/q2_particle_in_an_infinite_potential_box/')

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Simuc</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
'''

f = open(os.path.join(filepath, "q2_particle_in_an_infinite_potential_box.md"), "r")
text_list = util.convert(f.read(), lateximg=True, addbutton=True, addtoc=True)
mds = []
print("Total {} blocks of Markdown".format(len(text_list)))
for t in text_list:
    tmp = dcc.Markdown(t, dangerously_allow_html=True)
    mds.append(tmp)


#####################################################################


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
    wave = psi(x, n, L)
    prob = wave * wave
    xleft = [0, 0]
    xright = [L, L]
    y_vertical = [-1.3, 1.3]
    # energy levels
    if all_levels:
        energy = list(get_energy(np.linspace(1, 8, 8), L, m))
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
                        specs=[[{}, {"rowspan": 2}], [{}, None]],
                        subplot_titles=(r"$\text {Wavefunction}$", r"$\text {Energy Level}$", r"$\text {Probability Density}$"))

    # 1st subplot
    fig.append_trace(go.Scatter(x=x, y=wave, name="Wavefunction", showlegend=False), row=1, col=1)
    # nodes
    fig.append_trace(go.Scatter(x=nodes_x, y=nodes_y, name="node", mode="markers", marker=dict(size=6, color='blue'), showlegend=False), row=1, col=1)
    # wall
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    # axis
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-2, 12], showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text=r'$\psi(x)$', range=[-1.2, 1.2], showgrid=False, zeroline=False, row=1, col=1)

    # 2nd subplot
    annotations = list(fig['layout']['annotations'])
    for e in energy:
        fig.append_trace(go.Scatter(x=[-0.1, 0.5, 1.1], y=[e["energy"], e["energy"], e["energy"]], name="Energy Level", mode="lines", showlegend=False, line=dict(color=e["color"], width=2 if e['n'] == n else 1)), row=1, col=2)
        annotations.append(dict(y=e["energy"]+1.5, x=0.5, xref='x2', yref='y2', text=r"$E_{{{}}}={:.2f}\; eV$".format(e["n"], e["energy"]), font=dict(size=11, color=e["color"]), showarrow=False))
    fig.update_xaxes(range=[0, 1], showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text=r'$eV$', range=[0, 102], showgrid=False, zeroline=False, row=1, col=2)

    # 3rd subplot
    fig.append_trace(go.Scatter(x=x, y=prob, name="Probability Density", line=dict(color='red'), showlegend=False), row=2, col=1)
    fig.append_trace(go.Scatter(x=nodes_x, y=nodes_y, name="node", mode="markers", marker=dict(size=6, color='red'), showlegend=False), row=2, col=1)
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=2, col=1, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=2, col=1, )
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-2, 12], showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text=r'$\left|\psi(x)\right|^2$', range=[-1.2, 1.2], showgrid=False, zeroline=False, row=2, col=1)

    # annotations
    annotations.append(dict(y=0, x=-1, xref='x1', yref='y1', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))
    annotations.append(dict(y=0, x=L+1, xref='x1', yref='y1', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))
    annotations.append(dict(y=0, x=-1, xref='x3', yref='y3', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))
    annotations.append(dict(y=0, x=L+1, xref='x3', yref='y3', text=r"$V = +\infty$", font=dict(size=14, color="black"), showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.update_layout(height=800, title_text=r"$\text {{Particle in an 1D Box}} \;(n={})$".format(n))
    return fig


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
        wave = psi(x, n, L)
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
        fig.append_trace(go.Scatter(x=x, y=w*2+energies[i], line=dict(color='blue'), showlegend=False), row=1, col=1)
        fig.append_trace(go.Scatter(x=nodes_x[i], y=np.zeros_like(nodes_x[i])+energies[i], name="node",
                                    mode="markers", marker=dict(size=6, color='blue'), showlegend=False), row=1, col=1)
        fig.append_trace(go.Scatter(x=[0, L/2, L], y=[energies[i], energies[i], energies[i]], name="Energy Level",
                                    mode="lines", showlegend=False, line=dict(color="green", dash='dot')), row=1, col=1)
        annotations.append(dict(y=energies[i], x=L+2.5, xref='x1', yref='y1',
                                text=r"$E_{{{}}}={:.2f}\; eV$".format(i+1, energies[i]), font=dict(size=11, color="green"), showarrow=False))
    # wall
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=1, )
    # axis
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-3, 14], showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text=r'$eV$', range=[0, 35], showgrid=False, zeroline=False, row=1, col=1)

    # 2nd subplot
    for i, p in enumerate(probs):
        fig.append_trace(go.Scatter(x=x, y=p*2+energies[i], showlegend=False, line=dict(color='red')), row=1, col=2)
        fig.append_trace(go.Scatter(x=nodes_x[i], y=np.zeros_like(nodes_x[i])+energies[i], name="node",
                                    mode="markers", marker=dict(size=6, color='red'), showlegend=False), row=1, col=2)
        fig.append_trace(go.Scatter(x=[0, L/2, L], y=[energies[i], energies[i], energies[i]], name="Energy Level",
                                    mode="lines", showlegend=False, line=dict(color="green", dash='dot')), row=1, col=2)
        annotations.append(dict(y=energies[i], x=L+2.5, xref='x2', yref='y2',
                                text=r"$E_{{{}}}={:.2f}\; eV$".format(i+1, energies[i]), font=dict(size=11, color="green"), showarrow=False))
    fig.append_trace(go.Scatter(x=xleft, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=2, )
    fig.append_trace(go.Scatter(x=xright, y=y_vertical, showlegend=False, line=dict(color='white', width=2)), row=1, col=2, )
    fig.update_xaxes(title_text=r"$x (Å)$", range=[-3, 14], showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text=r'$eV$', range=[0, 35], showgrid=False, zeroline=False, row=1, col=2)

    # annotations
    annotations.append(dict(y=35/2, x=-1.25, xref='x1', yref='y1', text=r"$V = +\infty$", font=dict(size=11, color="black"), showarrow=False))
    annotations.append(dict(y=35/2, x=-1.25, xref='x2', yref='y2', text=r"$V = +\infty$", font=dict(size=11, color="black"), showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.update_layout(height=800, title_text=r"$\text {Particle in an 1D Box}$")
    return fig


#####################################################################
# Layout
def empty_space(h=50):
    return html.Div([html.Br()], style={'min-height': '{}px'.format(h)})


my_script = dji.Import(src="https://codepen.io/yueyericardo/pen/OJyLrKR.js")
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

# fig1
fig1 = dcc.Graph(figure=get_1dbox(n=4), id="fig1")
sliders1 = html.Div([
    html.Label('The value for the quantum number n'),
    dcc.Slider(id='fig1_n_slider', min=1, max=10, value=4, marks={str(x): str(x) for x in np.arange(1, 11, 1)}, step=1),
    ], style={'columnCount': 1, 'padding': '0'})


# fig2
fig2 = dcc.Graph(figure=get_1dbox(), id="fig2")
sliders2 = html.Div([
    html.Label('The value for the quantum number n'),
    dcc.Slider(id='fig2_n_slider', min=1, max=10, value=4, marks={str(x): str(x) for x in np.arange(1, 9, 1)}, step=1),
    html.Label('The length for the box L (in Å)'),
    dcc.Slider(id='fig2_l_slider', min=1, max=10, value=10, marks={str(x): str(x) for x in np.arange(5, 11, 1)}, step=1),
    ], style={'columnCount': 2, 'padding': '0'})


# fig3
fig3 = dcc.Graph(figure=get_1dbox(n=4, L=5, all_levels=True), id="fig3")
sliders3 = html.Div([
    html.Label('The value for the quantum number n'),
    dcc.Slider(id='fig3_n_slider', min=1, max=10, value=4, marks={str(x): str(x) for x in np.arange(1, 9, 1)}, step=1),
    html.Label('The length for the box L (in Å)'),
    dcc.Slider(id='fig3_l_slider', min=1, max=10, value=5, marks={str(x): str(x) for x in np.arange(3, 9, 1)}, step=1),
    ], style={'columnCount': 2, 'padding': '0'})

# fig4
fig4 = dcc.Graph(figure=get_1dbox(n=4, L=5, all_levels=True), id="fig4")
sliders4 = html.Div([
    html.Label('The value for the quantum number n'),
    dcc.Slider(id='fig4_n_slider', min=1, max=10, value=4, marks={str(x): str(x) for x in np.arange(1, 9, 1)}, step=1),
    html.Label('The length for the box L (in Å)'),
    dcc.Slider(id='fig4_l_slider', min=1, max=10, value=5, marks={str(x): str(x) for x in np.arange(3, 6, 1)}, step=1),
    html.Label('The mass of particle (in mass of electron)'),
    dcc.Slider(id='fig4_m_slider', min=1, max=5, value=1, marks={str(x): str(x) for x in np.arange(1, 4, 1)}, step=1),
    ], style={'columnCount': 3, 'padding': '0'})

# fig5
fig5 = dcc.Graph(figure=get_1dbox_combined(L=10, num_me=1), id="fig5")
sliders5 = html.Div([
    html.Label('The length for the box L (in Å)'),
    dcc.Slider(id='fig5_l_slider', min=1, max=10, value=10, marks={str(x): str(x) for x in np.arange(5, 11, 1)}, step=1),
    html.Label('The mass of particle (in mass of electron)'),
    dcc.Slider(id='fig5_m_slider', min=1, max=5, value=1, marks={str(x): str(x) for x in np.arange(1, 4, 1)}, step=1),
    ], style={'columnCount': 2, 'padding': '0'})

app.layout = html.Div([
    mds[0],
    html.Div([fig1, sliders1], className="my-whole-fig"),
    mds[1],
    html.Div([fig2, sliders2], className="my-whole-fig"),
    mds[2],
    html.Div([fig3, sliders3], className="my-whole-fig"),
    mds[3],
    html.Div([fig4, sliders4], className="my-whole-fig"),
    mds[4],
    html.Div([fig5, sliders5], className="my-whole-fig"),
    mds[5],
    mds[6],
    mds[7],
    mds[8],
    empty_space(),
    my_script,
    mathjax_script,
])


# update_fig1
@app.callback(
    Output('fig1', 'figure'),
    [Input('fig1_n_slider', 'value')])
def update_fig1(n):
    fig = get_1dbox(n=n)
    return fig


# update_fig2
@app.callback(
    Output('fig2', 'figure'),
    [Input('fig2_n_slider', 'value'),
     Input('fig2_l_slider', 'value'),
     ])
def update_fig2(n, L):
    fig = get_1dbox(n=n, L=L)
    return fig


# update_fig3
@app.callback(
    Output('fig3', 'figure'),
    [Input('fig3_n_slider', 'value'),
     Input('fig3_l_slider', 'value'),
     ])
def update_fig3(n, L):
    fig = get_1dbox(n=n, L=L, all_levels=True)
    return fig


# update_fig4
@app.callback(
    Output('fig4', 'figure'),
    [Input('fig4_n_slider', 'value'),
     Input('fig4_l_slider', 'value'),
     Input('fig4_m_slider', 'value'),
     ])
def update_fig4(n, L, num_me):
    fig = get_1dbox(n, L, num_me, all_levels=True)
    return fig


# update_fig4
@app.callback(
    Output('fig5', 'figure'),
    [Input('fig5_l_slider', 'value'),
     Input('fig5_m_slider', 'value'),
     ])
def update_fig5(L, num_me):
    fig = get_1dbox_combined(L, num_me)
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
