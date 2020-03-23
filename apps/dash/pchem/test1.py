# -*- coding: utf-8 -*-
import dash
import sys
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
sys.path.append('.')
sys.path.append('..')
import util
from server import server

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ['https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2020-03-21-iframeResizer.contentWindow.min.js']
app = dash.Dash(name='test1',
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                server=server,
                routes_pathname_prefix='/test1/')

Markdown_text = r"""
## Describing the Free Particle

We start by describing a free particle: a particle that is not under the influence of a potential.   
 As any other particle, the state of a ***Free Particle*** is described with a ket $\left|\psi(x)\right>$. In order to learn about this particle (measure its properties) we must construct *Hermitian Operators*. For example, what is the **momentum operator $\hat P$**?
 
The momentum operator most obey the following property (eigenfunction/eigenvalue equation):

$$\hat P \left| \psi_k(x) \right> =p\left | \psi_k(x)\right>  \tag{1}$$ 

where *p* is an eigenvalue of a *Hermitian operator* and therefore it is a real number.

In the $x$ representation, using the momentum operator as $\hat P =-i\hbar \frac{\partial }{\partial x}$, we can solve equation 1 by proposing a function to represent $\left| \psi_k(x) \right>$ as $\psi_k(x) = c\ e^{ikx}$, where $k$ is a real number.

Let's see if it works:  
$$\hat P \psi_k(x) =p \psi_k(x)$$ 
$$-i\hbar \frac{\partial {c\ e^{ikx}}}{\partial x} =-i\hbar\ c\ ik\ e^{ikx} $$ 
$$\hbar k\ c\ e^{ikx} = \hbar k\ \psi_k(x) \tag{2}$$
with $p=\hbar k$

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
        interval=0.2 * 1000,  # in milliseconds
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
