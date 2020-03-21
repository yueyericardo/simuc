# -*- coding: utf-8 -*-
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import util
from server import server

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ['https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2020-03-21-iframeResizer.contentWindow.min.js']
app = dash.Dash(name='test2',
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                server=server,
                routes_pathname_prefix='/test2/')

Markdown_text = r"""
<h2><center>Simulation of Enzyme Kinetics - Background Information</center></h2>
<center> <a target='_blank' href='http://simuc.chem.ufl.edu/enzyme_kinetics'>simuc.chem.ufl.edu/enzyme_kinetics</a></center>

--- 
$$
\large
\mathrm{E}+\mathrm{S} \stackrel{K_{\mathrm{s}}}{\rightleftharpoons} \mathrm{ES} \stackrel{k_{cat}}{\longrightarrow} \mathrm{E}+\mathrm{P}
$$
![](https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-18-Induced_fit_diagram.svg)
<center>From <a href="https://en.wikibooks.org/wiki/Principles_of_Biochemistry/Enzymes#/media/File:Induced_fit_diagram.svg">Wikipedia</a></center>

---

### 1. Without Inhibition:

$$
\large
E + S \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} ES \stackrel{k_{cat}}{\longrightarrow} \mathrm{E}+\mathrm{P}
$$

$$
\large
K_{m} \stackrel{\text { def }}{=} \frac{k_{cat}+k_{-1}}{k_{1}}
$$

$$
\large
V_{m} \stackrel{\text { def }}{=} k_{\text {cat }}[\mathrm{E}]_{\text {tot}}
$$

-------------------
"""

Markdown_text = util.convert_latex(Markdown_text)

app.layout = html.Div([
    dcc.Markdown(Markdown_text, dangerously_allow_html=True),
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='year-slider',
        min=-5.0,
        max=5.0,
        value=3,
        marks={str(x): str(x) for x in np.arange(-5, 6, 1)},
        step=0.1
    ),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # in milliseconds
        n_intervals=0
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('year-slider', 'value'),
     Input('interval-component', 'n_intervals')])
def update_figure(a, sec):
    N = 200
    x = np.linspace(0, 12, N)
    k = 1
    print(sec)
    w = 1 - 0.3 * sec
    b = 0
    y = a * np.sin(k * x + w) + b

    return {
        'data': [dict(
            x=x,
            y=y,
            mode='lines'
        )],
        'layout': dict(
            xaxis={'range': [0, 12]},
            yaxis={'title': 'Wavefunction', 'range': [-5, 5]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            # legend={'x': 0, 'y': 1},
            # hovermode='closest',
            # transition={'duration': 500},
        )
    }


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
