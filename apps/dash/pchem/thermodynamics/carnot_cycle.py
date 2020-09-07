# -*- coding: utf-8 -*-
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_defer_js_import as dji
from server import server
import util
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import sys
import os
sys.path.append('.')
sys.path.append('..')

filepath = os.path.split(os.path.realpath(__file__))[0]

external_stylesheets = ['https://codepen.io/yueyericardo/pen/OJyLrKR.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css']
external_scripts = [
    'https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2020-03-21-iframeResizer.contentWindow.min.js']


app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                server=server,
                routes_pathname_prefix='/carnot_cycle/')

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
text = """
<br><br>
<p style="text-align: center;">
<video preload="auto" src="assets/Carnot_Cycle.mp4" loop="" muted="" autoplay="" style="width: 50%" controls="controls">
</video>
</p>
<center>
Edited from 
<a href="https://www.youtube.com/watch?v=cJxF6JqCsJA" target="_blank">
    physics-Thermodynamics -Carnot Engine- basic introduction - YouTube
</a>
</center>
<br><br>
"""
mds = []
tmp = dcc.Markdown(text, dangerously_allow_html=True)
mds.append(tmp)
Vstp = 22.41396954


#####################################################################


def getfig1(p1=2.0, v1=11.2, gamma=5./3.,  p2_iso=1.0, p3_adi=0.4):
    def solveP(p1, V1, V2, n):    # Pressure of the gas assuming a path of pV^n
        p = p1 * (V1 / V2)**n
        return p

    v2_iso = (p1 / p2_iso) * v1
    # Plot
    fig = make_subplots(rows=1, cols=1, subplot_titles=(r"$\text{Carnot Cycle}$", ))

    ####################################################################################
    V = np.linspace(1., 100., 1000)

    # Plot the Isotherm in red
    fig.append_trace(go.Scatter(x=V, y=solveP(p1, v1, V, 1), name="Isotherm",
                                line=dict(width=1.5, color='blue', dash='dot')), row=1, col=1, )

    # Plot the Adiabat in blue
    fig.append_trace(go.Scatter(x=V, y=solveP(p1, v1, V, gamma), name="Adiabat",
                                line=dict(width=1.5, color='red', dash='dot')), row=1, col=1, )

    # plot marker
    fig.append_trace(go.Scatter(x=[v1, v2_iso], y=[p1, p2_iso], name=None, mode='markers',
                                marker=dict(color='black', size=8), opacity=0.5, showlegend=False),  row=1, col=1, )
    ############################################################################################


    ############################################################################################
    # finish the codes to make Figure 15, you could also change codes outside of the fence
    v3_adi = (p2_iso / p3_adi)**(1. / gamma) * v2_iso
    v4_iso = (p1 * v1**gamma / (p3_adi * v3_adi)) ** (1 / (gamma - 1))
    p4_iso = (v1 / v4_iso)**gamma * p1

    fig.append_trace(go.Scatter(x=[v3_adi, v4_iso], y=[p3_adi, p4_iso], name=None, mode='markers',
                                marker=dict(color='black', size=8), opacity=0.5, showlegend=False),  row=1, col=1, )

    # dashed lines
    fig.append_trace(go.Scatter(x=V, y=solveP(p2_iso, v2_iso, V, gamma), showlegend=False,
                                line=dict(width=1.5, color='firebrick', dash='dot')), row=1, col=1, )
    fig.append_trace(go.Scatter(x=V, y=solveP(p3_adi, v3_adi, V, 1), showlegend=False,
                                line=dict(width=1.5, color='#2593ff', dash='dot')), row=1, col=1, )


    # stages plot
    V_stage1 = np.linspace(v1, v2_iso, 1000)
    V_stage2 = np.linspace(v2_iso, v3_adi, 1000)
    V_stage3 = np.linspace(v3_adi, v4_iso, 1000)
    V_stage4 = np.linspace(v4_iso, v1, 1000)

    P_stage1 = solveP(p1, v1, V_stage1, 1)
    P_stage2 = solveP(p2_iso, v2_iso, V_stage2, gamma)
    P_stage3 = solveP(p3_adi, v3_adi, V_stage3, 1)
    P_stage4 = solveP(p4_iso, v4_iso, V_stage4, gamma)

    fig.append_trace(go.Scatter(x=V_stage1, y=P_stage1, name=r"Stage 1",
                                line=dict(width=2, color='blue',)), row=1, col=1, )
    fig.append_trace(go.Scatter(x=V_stage2, y=P_stage2, name="Stage 2",
                                line=dict(width=2, color='firebrick' )), row=1, col=1, )
    fig.append_trace(go.Scatter(x=V_stage3, y=P_stage3, name="Stage 3",
                                line=dict(width=2, color='#2593ff',)), row=1, col=1, )
    fig.append_trace(go.Scatter(x=V_stage4, y=P_stage4, name="Stage 4",
                                line=dict(width=2, color='red',)), row=1, col=1, )

    # Region fill
    fig.add_trace(go.Scatter(x=np.flip(np.concatenate([V_stage3, V_stage4])), y=np.flip(np.concatenate([P_stage3, P_stage4])),
                             fill=None, showlegend=False,  mode='lines',
                             # line_color='rgba(0, 0, 250, 0.1)'
                             ))
    fig.add_trace(go.Scatter(x=np.concatenate([V_stage1, V_stage2]), y=np.concatenate([P_stage1, P_stage2]),
                             fill='tonexty', mode='none', showlegend=False,
                             # line=dict(width=0.5, color='rgb(131, 90, 241)'),
                             # line_color='rgba(0, 0, 250, 0.1)'
                             ))
    ############################################################################################

    fig.update_xaxes(title_text=r"$\overline{V} \;\text{(L)}$ ", row=1, col=1, range=[0, 100]
                     #                  showline=True, linewidth=1, linecolor='black', ticks='outside',
                     #                  showgrid=False, zeroline=False
                     )
    fig.update_yaxes(title_text=r'$p \;\text{(atm)}$ ', row=1, col=1, range=[0, 6]
                     #                  showline=True, linewidth=1, linecolor='black', ticks='outside',
                     #                  showgrid=False, zeroline=False
                     )

    fig.update_layout(height=600, legend={'traceorder': 'normal'}, paper_bgcolor='rgba(0,0,0,0)',
                      #                   plot_bgcolor='rgba(0,0,0,0)'
                      )
    return fig


#####################################################################
# Layout
def empty_space(h=50):
    return html.Div([html.Br()], style={'min-height': '{}px'.format(h)})


my_script = dji.Import(src="https://codepen.io/yueyericardo/pen/OJyLrKR.js")
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

# fig1
fig1 = dcc.Graph(figure=getfig1(), id="fig1")
sliders1_1 = html.Div([
    html.Label('Initial Pressure $P_1$ (in $ atm $)'),
    dcc.Slider(id='p1_slider', min=2.5, max=4, value=3, marks={str(x): str(x) for x in np.arange(2.5, 5, 0.5)}, step=0.5),
    html.Label('Initial Volume $V_1$ (in $ L $)'),
    dcc.Slider(id='v1_slider', min=10, max=15, value=12, marks={str(x): str(x) for x in np.arange(10, 16, 1)}, step=1),
    html.Label('Heat capacity ratio $\gamma$'),
    dcc.Slider(id='gamma_slider', min=1.1, max=1.7, value=1.6, marks={'{:.1f}'.format(x): '{:.1f}'.format(x) for x in np.arange(1.1, 1.8, 0.1)}, step=0.1),
    ], style={'columnCount': 3, 'padding': '0'})
sliders1_2 = html.Div([
    html.Label('Pressure after Stage1 $P_2$ (in $ atm $)'),
    dcc.Slider(id='p2_slider', min=1, max=2.2, value=1.5, marks={'{:.1f}'.format(x): '{:.1f}'.format(x) for x in np.arange(1, 2.5, 0.3)}, step=0.3),
    html.Label('Pressure after Stage2 $P_3$ (in $ atm $)'),
    dcc.Slider(id='p3_slider', min=0.5, max=0.9, value=0.4, marks={'{:.1f}'.format(x): '{:.1f}'.format(x) for x in np.arange(0.5, 1.1, 0.1)}, step=0.1),
    ], style={'columnCount': 2, 'padding': '0'})


app.layout = html.Div([
    mds[0],
    html.Div([fig1, sliders1_1, sliders1_2], className="my-whole-fig"),
    empty_space(),
    my_script,
    mathjax_script,
])


# update_fig1
@app.callback(
    Output('fig1', 'figure'),
    [Input('p1_slider', 'value'),
     Input('v1_slider', 'value'),
     Input('gamma_slider', 'value'),
     Input('p2_slider', 'value'),
     Input('p3_slider', 'value'),
     ])
def update_fig1(p1, v1, gamma,  p2_iso, p3_adi):
    fig = getfig1(p1=p1, v1=v1, gamma=gamma,  p2_iso=p2_iso, p3_adi=p3_adi)
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
