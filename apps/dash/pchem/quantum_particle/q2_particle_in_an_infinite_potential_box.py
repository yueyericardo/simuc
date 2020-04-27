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


app = dash.Dash(name='q2_particle_in_an_infinite_potential_box',
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


def getfig1(k=4, xmax=5):
    # Defining Psi and PsiC (complex conjugate) functions
    def psi(x, k):
        return (1.0/np.sqrt(2.0*np.pi))*(np.cos(k*x)+np.sin(k*x)*1j)

    def psiC(x, k):
        return (1.0/np.sqrt(2.0*np.pi))*(np.cos(k*x)-np.sin(k*x)*1j)

    # calculation (Prepare data)
    x = np.linspace(-xmax, xmax, 900)
    lim1 = 1/np.sqrt(2*np.pi)  # Maximum value of the wavefunction
    y_real = psi(x, k).real
    y_imag = psi(x, k).imag
    y_prob = (psi(x, k)*psiC(x, k)).real

    # Plot
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Wavefunction", "Probability Density"))

    # 1st subplot
    fig.append_trace(go.Scatter(x=x, y=y_real, name="Real"), row=1, col=1, )
    fig.append_trace(go.Scatter(x=x, y=y_imag, name="Imag"), row=1, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
    fig.update_yaxes(title_text=r'$\psi_k(x)$', row=1, col=1)

    # 2nd subplot
    fig.append_trace(go.Scatter(
        x=x, y=y_prob, name="Probability Density"), row=2, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=2, col=1)
    fig.update_yaxes(title_text=r'$\left|\psi_k(x)\right|^2$',
                     range=[0, lim1*lim1*1.4], row=2, col=1)

    fig.update_layout(height=600)
    return fig


def getfig2(k=4, dk=2, xmax=5):
    # Defining functions
    def psi_contour(x, dk):
        return (np.sin(dk*x)/(np.sqrt(np.pi*dk)*x))

    def psi(x, k, dk):
        return psi_contour(x, dk)*(np.cos(k*x)+np.sin(k*x)*1j)

    # calculation (Prepare data)
    x = np.linspace(-xmax, xmax, 900)
    y_real = psi(x, k, dk).real
    y_imag = psi(x, k, dk).imag
    y_prob = (psi(x, k, dk).real)**2+(psi(x, k, dk).imag)**2

    # Plot
    title1 = r'$ \text {Real contribution to } \Psi_{\Delta k}(x)$'
    title2 = r'$ \text {Imaginary contribution to } \Psi_{\Delta k}(x)$'
    title3 = r'$ \text {Probability Density }$'
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=(title1, title2, title3))

    # 1st subplot
    fig.append_trace(go.Scatter(x=x, y=y_real, name="Real",
                                line=dict(color='blue')), row=1, col=1)
    fig.append_trace(go.Scatter(x=x, y=psi_contour(x, dk), showlegend=False, line=dict(
        color='blue', width=1, dash='dash')), row=1, col=1)
    fig.append_trace(go.Scatter(x=x, y=-psi_contour(x, dk), showlegend=False,
                                line=dict(color='blue', width=1, dash='dash')), row=1, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
    fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=1)

    # 2nd subplot
    fig.append_trace(go.Scatter(x=x, y=y_imag, name="Imag",
                                line=dict(color='red')), row=1, col=2, )
    fig.append_trace(go.Scatter(x=x, y=psi_contour(x, dk), showlegend=False, line=dict(
        color='red', width=1, dash='dash')), row=1, col=2)
    fig.append_trace(go.Scatter(x=x, y=-psi_contour(x, dk), showlegend=False,
                                line=dict(color='red', width=1, dash='dash')), row=1, col=2)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=2)
    fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=2)

    # 3rd subplot
    fig.append_trace(go.Scatter(
        x=x, y=y_prob, name="Probability Density", line=dict(color='green')), row=2, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=2, col=1)
    fig.update_yaxes(
        title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=2, col=1)

    # show
    title = r"$k_o  \pm \Delta k = {} \pm {} Å^{{-1}}$".format(k, dk)
    fig.update_layout(height=600, title_text=title)
    return fig


def getfig3(k=4, dk=2, xmax=5):
    # Defining functions
    def psi_contour(x, dk):
        return dk*np.exp(-0.5*x*x*dk*dk)/np.sqrt(dk*np.sqrt(np.pi))

    def psi(x, k, dk):
        return psi_contour(x, dk)*(np.cos(k*x)+np.sin(k*x)*1j)

    # calculation (Prepare data)
    x = np.linspace(-xmax, xmax, 900)
    y_real = psi(x, k, dk).real
    y_imag = psi(x, k, dk).imag
    y_prob = (psi(x, k, dk).real)**2+(psi(x, k, dk).imag)**2

    # Plot
    title1 = r'$ \text {Real contribution to } \Psi_{\Delta k}(x)$'
    title2 = r'$ \text {Imaginary contribution to } \Psi_{\Delta k}(x)$'
    title3 = r'$ \text {Probability Density }$'
    fig = make_subplots(rows=2, cols=2, subplot_titles=(title1, title2, title3))

    # 1st subplot
    fig.append_trace(go.Scatter(x=x, y=y_real, name="Real", line=dict(color='blue')), row=1, col=1, )
    fig.append_trace(go.Scatter(x=x, y=psi_contour(x, dk), showlegend=False, line=dict(color='blue', width=1, dash='dash')),
                     row=1, col=1)
    fig.append_trace(go.Scatter(x=x, y=-psi_contour(x, dk), showlegend=False, line=dict(color='blue', width=1, dash='dash')),
                     row=1, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
    fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=1)

    # 2nd subplot
    fig.append_trace(go.Scatter(x=x, y=y_imag, name="Imag", line=dict(color='red')), row=1, col=2, )
    fig.append_trace(go.Scatter(x=x, y=psi_contour(x, dk), showlegend=False, line=dict(color='red', width=1, dash='dash')),
                     row=1, col=2)
    fig.append_trace(go.Scatter(x=x, y=-psi_contour(x, dk), showlegend=False, line=dict(color='red', width=1, dash='dash')),
                     row=1, col=2)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=2)
    fig.update_yaxes(title_text=r'$\Psi_{\Delta k}(x)$', row=1, col=2)

    # 3rd subplot
    fig.append_trace(go.Scatter(x=x, y=y_prob, name="Probability Density", line=dict(color='green')), row=2, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=2, col=1)
    fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=2, col=1)

    # show
    title = r"$k_o  \pm \Delta k = {} \pm {} Å^{{-1}}$".format(k, dk)
    fig.update_layout(height=600, title_text=title)
    return fig


def getfig4(k=4, dk=2, xmax=5):
    # Defining functions
    def psi_contour(x, dk):
        return np.sin(dk*x)*np.sin(dk*x)/(np.pi*dk*x*x)

    def psi_contourG(x, dk):
        return dk*dk*np.exp(-x*x*dk*dk)/(dk*np.sqrt(np.pi))

    # calculation (Prepare data)
    x = np.linspace(-xmax, xmax, 900)

    # Plot
    title1 = r'$ \text {Probability Density for equally weigthed k}$'
    title2 = r'$ \text {Probability Density for Gaussian-weigthed k }$'
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2))

    # 1st subplot
    fig.append_trace(go.Scatter(x=x, y=psi_contour(x, dk), name="Probability Density", line=dict(color='green')), row=1, col=1)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=1)
    fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=1, col=1)

    # 2nd subplot
    fig.append_trace(go.Scatter(x=x, y=psi_contourG(
        x, dk), name="Probability Density", line=dict(color='magenta')), row=1, col=2)
    fig.update_xaxes(title_text=r"$x (Å)$", row=1, col=2)
    fig.update_yaxes(title_text=r'$\left|\Psi_{\Delta k}(x)\right|^2$', row=1, col=2)

    # show
    title = r"$k_o  \pm \Delta k = {} \pm {} Å^{{-1}}$".format(k, dk)
    fig.update_layout(height=400, title_text=title)
    return fig


#####################################################################
# Layout
def empty_space(h=50):
    return html.Div([html.Br()], style={'min-height': '{}px'.format(h)})


my_script = dji.Import(src="https://codepen.io/yueyericardo/pen/OJyLrKR.js")
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

# fig1
fig1 = dcc.Graph(figure=getfig1(), id="fig1")
sliders1 = html.Div([
    html.Label('The value for k (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig1_k_slider', min=1, max=15, value=4, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('the maximum value for x (in Å)'),
    dcc.Slider(id='fig1_xmax_slider', min=1, max=15, value=5, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    ], style={'columnCount': 2, 'padding': '0'})

# fig2
fig2 = dcc.Graph(figure=getfig2(), id="fig2")
sliders2 = html.Div([
    html.Label('The value for $k_0$ (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig2_k_slider', min=1, max=15, value=4, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('The value for $\Delta k$ (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig2_dk_slider', min=1, max=15, value=2, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('the maximum value for x (in Å)'),
    dcc.Slider(id='fig2_xmax_slider', min=1, max=15, value=5, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    ], style={'columnCount': 3, 'padding': '0'})


# fig3
fig3 = dcc.Graph(figure=getfig3(), id="fig3")
sliders3 = html.Div([
    html.Label('The value for $k_0$ (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig3_k_slider', min=1, max=15, value=4, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('The value for $\Delta k$ (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig3_dk_slider', min=1, max=15, value=2, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('the maximum value for x (in Å)'),
    dcc.Slider(id='fig3_xmax_slider', min=1, max=15, value=5, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    ], style={'columnCount': 3, 'padding': '0'})

# fig4
fig4 = dcc.Graph(figure=getfig4(), id="fig4")
sliders4 = html.Div([
    html.Label('The value for $k_0$ (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig4_k_slider', min=1, max=15, value=4, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('The value for $\Delta k$ (in $ Å^{-\;1} $)'),
    dcc.Slider(id='fig4_dk_slider', min=1, max=15, value=2, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    html.Label('the maximum value for x (in Å)'),
    dcc.Slider(id='fig4_xmax_slider', min=1, max=15, value=5, marks={str(x): str(x) for x in np.arange(1, 16, 1)}, step=1),
    ], style={'columnCount': 3, 'padding': '0'})

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
    [Input('fig1_k_slider', 'value'),
     Input('fig1_xmax_slider', 'value'),])
def update_fig1(k, xmax):
    fig = getfig1(k=k, xmax=xmax)
    return fig


# update_fig2
@app.callback(
    Output('fig2', 'figure'),
    [Input('fig2_k_slider', 'value'),
     Input('fig2_dk_slider', 'value'),
     Input('fig2_xmax_slider', 'value'),
     ])
def update_fig2(k, dk, xmax):
    fig = getfig2(k=k, dk=dk, xmax=xmax)
    return fig


# update_fig3
@app.callback(
    Output('fig3', 'figure'),
    [Input('fig3_k_slider', 'value'),
     Input('fig3_dk_slider', 'value'),
     Input('fig3_xmax_slider', 'value'),
     ])
def update_fig3(k, dk, xmax):
    fig = getfig3(k=k, dk=dk, xmax=xmax)
    return fig


# update_fig4
@app.callback(
    Output('fig4', 'figure'),
    [Input('fig4_k_slider', 'value'),
     Input('fig4_dk_slider', 'value'),
     Input('fig4_xmax_slider', 'value'),
     ])
def update_fig4(k, dk, xmax):
    fig = getfig4(k=k, dk=dk, xmax=xmax)
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
