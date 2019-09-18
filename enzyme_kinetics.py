import numpy as np
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.plotting import figure


# rate constant slider
slider_Ks = Slider(title="Ks", value=10, start=1, end=20, step=3)
slider_kcat = Slider(title="kcat * 1e-3", value=100, start=60, end=200, step=20)
slider_Ki = Slider(title="Ki", value=0, start=0, end=20, step=3)
slider_K_ES_I = Slider(title="Kii", value=0, start=0, end=100, step=10)
slider_K_EI_S = Slider(title="Kss", value=0, start=0, end=100, step=10)
# Initial concentration
slider_E0 = Slider(title="E0", value=200, start=0, end=300, step=50)
slider_S0 = Slider(title="S0", value=500, start=400, end=600, step=50)
slider_Inhi0 = Slider(title="I0", value=0, start=0, end=700, step=50)

# Eq constant
Ki = slider_Ki.value
K_ES_I = slider_K_ES_I.value
K_EI_S = slider_K_EI_S.value
# rate constant
kr = 1/10000
kf = slider_Ks.value * kr
kcat = slider_kcat.value / 1000
kir = 1 / 10000
ki = slider_Ki.value * kir
kesir = 1 / 10000
kesi = slider_K_ES_I.value * kesir
keisr = 1 / 10000
keis = slider_K_EI_S.value * keisr

# initial condition
E0 = slider_E0.value
S0 = slider_S0.value
Inhi0 = slider_Inhi0.value
ES = 0
EI = 0
ESI = 0
P = 0
S = S0
Inhi = Inhi0
E = E0
# converge_tolerance
tol = 1
dt = 0.01
ESI_scale = 50


def get_newdata(kr, kf, kcat, kir, ki, kesir, kesi, keisr, keis, S, ES, EI, ESI, Inhi, E, P, Inhi0, E0, S0, dt):
    x_t = [0]
    y_S = [S]
    y_ES = [ES]
    y_EI = [EI]
    y_ESI = [ESI]
    y_Inhi = [Inhi]
    y_E = [E]
    y_P = [P]
    steps = 0

    for i in range(0, int(150/dt)):
        d_ES = kf * E * S - kr * ES - kcat * ES - kesi * ES * Inhi + kesir * ESI
        d_EI = ki * E * Inhi - kir * EI + keisr * ESI - keis * EI * S
        d_ESI = kesi * ES * Inhi - kesir * ESI + keis * EI * S - keisr * ESI
        d_S = kr * ES - kf * E * S + keisr * ESI - keis * EI * S

        ES = ES + d_ES * dt
        EI = EI + d_EI * dt
        ESI = ESI = d_ESI * dt
        S = S + d_S * dt

        E = E0 - ES - EI - ESI
        Inhi = Inhi0 - EI - ESI
        P = S0 - S - ES - ESI

        steps += 1
        x_t.append(steps * dt)
        y_S.append(S)
        y_ES.append(ES)
        y_EI.append(EI)
        y_ESI.append(ESI * ESI_scale)
        y_Inhi.append(Inhi)
        y_E.append(E)
        y_P.append(P)

    datas = [x_t, y_S, y_ES, y_EI, y_ESI, y_Inhi, y_E, y_P]
    for i, d in enumerate(datas):
        datas[i] = np.array(d)

    return datas


x_t, y_S, y_ES, y_EI, y_ESI, y_Inhi, y_E, y_P = get_newdata(kr, kf, kcat, kir, ki, kesir, kesi, keisr, keis, S, ES, EI, ESI, Inhi, E, P, Inhi0, E0, S0, dt)

source_S = ColumnDataSource(data=dict(x=x_t, y=y_S))
source_ES = ColumnDataSource(data=dict(x=x_t, y=y_ES))
source_EI = ColumnDataSource(data=dict(x=x_t, y=y_EI))
source_ESI = ColumnDataSource(data=dict(x=x_t, y=y_ESI))
source_Inhi = ColumnDataSource(data=dict(x=x_t, y=y_Inhi))
source_E = ColumnDataSource(data=dict(x=x_t, y=y_E))
source_P = ColumnDataSource(data=dict(x=x_t, y=y_P))

plot_con_time = figure(plot_height=300, plot_width=600,
                       title="Concentrations over Time",
                       tools="crosshair,reset,save,wheel_zoom",
                       toolbar_location=None,
                       y_range=[0, 600], x_range=[0, 150])

p_P = plot_con_time.line('x', 'y', source=source_P, line_width=5, line_alpha=0.5, color='red', legend='P')
p_S = plot_con_time.line('x', 'y', source=source_S, line_width=5, line_alpha=0.5, color='green', legend='S')
p_Inhi = plot_con_time.line('x', 'y', source=source_Inhi, line_width=5, line_alpha=0.5, color='orange', legend='I')
p_E = plot_con_time.line('x', 'y', source=source_E, line_width=3, line_alpha=1, color='deepskyblue', line_dash='dashed', legend='E')
p_ES = plot_con_time.line('x', 'y', source=source_ES, line_width=1, line_alpha=1, color='green', line_dash='dashed', legend='ES')
p_EI = plot_con_time.line('x', 'y', source=source_EI, line_width=1, line_alpha=1, color='tomato', line_dash='dashed', legend='EI')
p_ESI = plot_con_time.line('x', 'y', source=source_ESI, line_width=1, line_alpha=1, color='black', line_dash='dashed', legend='ESI * {}'.format(ESI_scale))
plot_con_time.yaxis.axis_label = "Concentration  (mol / L)"
plot_con_time.xaxis.axis_label = "Time  (s)"


def get_1_over_v0(x_S0, x_1_over_S0, kcat, kf, kr, E0, I0, Ki, K_ES_I, K_EI_S):
    Km = (kcat + kr)/kf
    Vm = kcat * E0
    inhi_type = 'None Inhibition'
    text_Km = 'Km = {:.2f}'.format(Km)
    text_Vm = 'Vm = {:.2f}'.format(Vm)

    # competitive inhibition
    if (Ki > 0 and K_ES_I == 0):
        Km = Km * (1 + I0 / Ki)
        inhi_type = 'Competitive Inhibition'
        text_Km = 'Km\' = {:.2f}'.format(Km)

    # non-competitive inhibition
    if (Ki > 0 and K_ES_I > 0):
        Vm = Vm / (1 + I0 / K_ES_I)
        Km = Km * (1 + I0 / Ki) / (1 + I0 / K_ES_I)
        text_Km = 'Km\' = {:.2f}'.format(Km)
        text_Vm = 'Vm\' = {:.2f}'.format(Vm)
        inhi_type = 'Non-Competitive Inhibition'

    # un-competitive inhibition
    if (Ki == 0 and K_ES_I > 0):
        Km = Km / (1 + I0 / K_ES_I)
        Vm = Vm / (1 + I0 / K_ES_I)
        text_Km = 'Km\' = {:.2f}'.format(Km)
        text_Vm = 'Vm\' = {:.2f}'.format(Vm)
        inhi_type = 'Un-Competitive Inhibition'

    x_over = 1 / x_1_over_S0
    V_1_over_S0 = Vm * x_over / (Km + x_over)
    V0 = Vm * x_S0 / (Km + x_S0)

    x_circle_x = -1 / Km
    y_circle_y = 1 / Vm
    xy_circle_half_Vm = (Km, Vm/2)

    return V0, 1/V_1_over_S0, inhi_type, text_Km, text_Vm, x_circle_x, y_circle_y, xy_circle_half_Vm


x_S0 = np.linspace(0.01, 2000, 10000)
x_1_over_S0 = np.linspace(-0.02, 0.02, 900)
y_V0, y_1_over_S0, inhi_type, text_Km, text_Vm, x_circle_x, y_circle_y, xy_circle_half_Vm = get_1_over_v0(x_S0, x_1_over_S0, kcat, kf, kr, E0, Inhi0, Ki, K_ES_I, K_EI_S)
source_V_S = ColumnDataSource(data=dict(x=x_S0, y=y_V0))
source_1_over = ColumnDataSource(data=dict(x=x_1_over_S0, y=y_1_over_S0))
source_circle_x = ColumnDataSource(data=dict(x=[x_circle_x], y=[0]))
source_circle_y = ColumnDataSource(data=dict(x=[0], y=[y_circle_y]))
source_circle__half_Vm = ColumnDataSource(data=dict(x=[xy_circle_half_Vm[0]], y=[xy_circle_half_Vm[1]]))

label_km = bokeh.models.Label(x=10, y=540,
                              text=text_Km, render_mode='css',
                              border_line_color='black', border_line_alpha=1.0,
                              background_fill_color='white', background_fill_alpha=1.0)
label_vm = bokeh.models.Label(x=10, y=480,
                              text=text_Vm, render_mode='css',
                              border_line_color='black', border_line_alpha=1.0,
                              background_fill_color='white', background_fill_alpha=1.0)
plot_con_time.add_layout(label_km)
plot_con_time.add_layout(label_vm)

label_inhi_type = bokeh.models.Label(x=10, y=420,
                                     text=inhi_type, render_mode='css',
                                     border_line_color='black', border_line_alpha=1.0,
                                     background_fill_color='white', background_fill_alpha=1.0)
plot_con_time.add_layout(label_inhi_type)

plot_V_S = figure(plot_height=200, plot_width=600,
                  title="Initial Rate vs Substrate",
                  tools="crosshair,reset,save,wheel_zoom",
                  toolbar_location=None,
                  y_range=[0, 45])
p_V_S = plot_V_S.line('x', 'y', source=source_V_S, line_width=2, line_alpha=1, color='violet', legend='Initial Rate')
plot_V_S.circle('x', 'y', source=source_circle__half_Vm, fill_color="purple", line_color="black", size=8, legend='(Km, Vm/2)')

plot_V_S.yaxis.axis_label = "V0  (mol / (s*L))"
plot_V_S.xaxis.axis_label = "[S]  (mol / L)"

plot_1_over = figure(plot_height=200, plot_width=600,
                     title="1/V0 vs 1/[S]",
                     tools="crosshair,reset,save,wheel_zoom",
                     toolbar_location=None,
                     y_range=[-1, 1],
                     x_range=[-0.02, 0.02])
p_1_over = plot_1_over.line('x', 'y', source=source_1_over, line_width=2, line_alpha=1, color='blue', legend='1/V0')
plot_1_over.circle('x', 'y', source=source_circle_x, fill_color="green", line_color="black", size=8, legend='- 1/Km')
plot_1_over.circle('x', 'y', source=source_circle_y, fill_color='red', line_color="black", size=8, legend='1/Vm')

plot_1_over.yaxis.axis_label = "1 / V0"
plot_1_over.xaxis.axis_label = "1 / [S]"


def update_data(attrname, old, new):
    # Eq constant
    Ki = slider_Ki.value
    K_ES_I = slider_K_ES_I.value
    K_EI_S = slider_K_EI_S.value
    # rate constant
    kr = 1/10000
    kf = slider_Ks.value * kr
    kcat = slider_kcat.value / 1000
    kir = 1 / 10000
    ki = slider_Ki.value * kir
    kesir = 1 / 10000
    kesi = slider_K_ES_I.value * kesir
    keisr = 1 / 10000
    keis = slider_K_EI_S.value * keisr

    # initial condition
    E0 = slider_E0.value
    S0 = slider_S0.value
    Inhi0 = slider_Inhi0.value
    ES = 0
    EI = 0
    ESI = 0
    P = 0
    S = S0
    Inhi = Inhi0
    E = E0
    # converge_tolerance
    dt = 0.01

    x_t, y_S, y_ES, y_EI, y_ESI, y_Inhi, y_E, y_P = get_newdata(kr, kf, kcat, kir, ki, kesir, kesi, keisr, keis, S, ES, EI, ESI, Inhi, E, P, Inhi0, E0, S0, dt)
    source_S.data = dict(x=x_t, y=y_S)
    source_ES.data = dict(x=x_t, y=y_ES)
    source_EI.data = dict(x=x_t, y=y_EI)
    source_ESI.data = dict(x=x_t, y=y_ESI)
    source_Inhi.data = dict(x=x_t, y=y_Inhi)
    source_E.data = dict(x=x_t, y=y_E)
    source_P.data = dict(x=x_t, y=y_P)

    y_V0, y_1_over_S0, inhi_type, text_Km, text_Vm, x_circle_x, y_circle_y, xy_circle_half_Vm = get_1_over_v0(x_S0, x_1_over_S0, kcat, kf, kr, E0, Inhi0, Ki, K_ES_I, K_EI_S)
    label_km.text = text_Km
    label_vm.text = text_Vm
    source_V_S.data = dict(x=x_S0, y=y_V0)
    source_1_over.data = dict(x=x_1_over_S0, y=y_1_over_S0)
    label_inhi_type.text = inhi_type
    source_circle_x.data = dict(x=[x_circle_x], y=[0])
    source_circle_y.data = dict(x=[0], y=[y_circle_y])
    source_circle__half_Vm.data = dict(x=[xy_circle_half_Vm[0]], y=[xy_circle_half_Vm[1]])


for w in [slider_Ks, slider_kcat, slider_Ki, slider_K_ES_I, slider_K_EI_S, slider_E0, slider_Inhi0, slider_S0]:
    w.on_change('value', update_data)

title_left = bokeh.models.Div(text="Without Inhibition<br><br>")
title_inhi = bokeh.models.Div(text="Inhibition<br><br>")
title = bokeh.models.Div(text="<h1>Simulation of Enzyme Kinetics</h1><a target='_blank' href='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-18-Background_Information%20-2-.html'>Background Information</a><br><hr>",
                         style={'font-size': '120%', 'width': '1450px', 'font-family': 'serif', 'color': 'black', 'text-align': 'center'})
footer = bokeh.models.Div(text="<br><hr><br>Richard (Jinze) Xue <br>2019.09.18<br> <a target='_blank' href='https://github.com/yueyericardo/simuc'>Source code on Github</a><br>Department of Chemistry, Physical Chemistry Division, University of Florida",
                          style={'font-size': '100%', 'width': '1450px', 'font-family': 'serif', 'color': 'black', 'text-align': 'center'})
reaction_png = bokeh.models.Div(text="<br><br><br><br><br><br><br><br><br><br><br><br><br>Symbol Definition: <img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-18-143043.png' style='width: 310px; margin: auto; display: block'><br>")
mm_equation_png = bokeh.models.Div(text="<br><br><br><br><br><br><br><br><br><br><br><br><br>Michaelis-Menten Equation:<br><br> <img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-18-mm-equation.png' style='width: 150px; margin: auto; display: block'><br>")

# Set up layouts and add to document
up_title = row(children=[title], sizing_mode='fixed', height=150, width=1600)
down_footer = row(children=[footer], sizing_mode='fixed', height=130, width=1600)
left = column(children=[title_left, slider_Ks, slider_kcat, slider_E0, slider_S0, reaction_png], sizing_mode='fixed', width=400)
middle = column(children=[plot_con_time, plot_V_S, plot_1_over], sizing_mode='fixed', width=700)
right = column(children=[title_inhi, slider_Ki, slider_K_ES_I, slider_K_EI_S, slider_Inhi0, mm_equation_png], sizing_mode='fixed', width=400)
all_layout = column(up_title, row(left, middle, right), down_footer)

curdoc().add_root(all_layout)
curdoc().title = "Enzyme Kinetics"
