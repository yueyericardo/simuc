import numpy as np
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, CheckboxButtonGroup
from bokeh.plotting import figure


# rate constant slider
slider_kf = Slider(title="kf * 1e-4  ", value=10, start=1, end=20, step=3)
slider_kr = Slider(title="kr * 1e-4  ", value=1, start=1, end=1000, step=100)
slider_kcat = Slider(title="kcat * 1e-3", value=100, start=60, end=200, step=20)
slider_ki = Slider(title="ki * 1e-4  ", value=2, start=1, end=20, step=3)
slider_kir = Slider(title="kir * 1e-4  ", value=1, start=1, end=1000, step=100)
# Initial concentration
slider_E0 = Slider(title="E0", value=200, start=100, end=300, step=50)
slider_S0 = Slider(title="S0", value=500, start=400, end=600, step=50)
slider_Inhi0 = Slider(title="I0", value=100, start=100, end=600, step=50)

# rate constant
kf = slider_kf.value / 10000
kr = slider_kr.value / 10000
kcat = slider_kcat.value / 1000
ki = slider_ki.value / 10000
kir = slider_kir.value / 10000

# initial condition
E0 = slider_E0.value
S0 = slider_S0.value
Inhi0 = slider_Inhi0.value
ES = 0
S = S0
# get E and P based on ES and S
E = E0 - ES
P = S0 - S - ES
Inhi = Inhi0
EI = 0
# converge_tolerance
tol = 1
dt = 0.01


def step():
    global kf
    global kr
    global kcat
    global ki
    global kir

    global S
    global ES
    global EI
    global Inhi
    global E
    global P

    global Inhi0
    global E0
    global S0

    d_ES = kf * E * S - kr * ES - kcat * ES
    d_EI = ki * E * Inhi - kir * EI
    d_S = kr * ES - kf * E * S

    ES = ES + d_ES * dt
    S = S + d_S * dt
    EI = EI + d_EI * dt
    E = E0 - ES - EI
    Inhi = Inhi0 - EI
    P = S0 - S - ES


x_t = [0]
y_ES = [ES]
y_S = [S]
y_E = [E]
y_P = [P]
steps = 0

# while(abs(P - S0) > tol):
for i in range(0, int(150/dt)):
    step()
    steps += 1
    x_t.append(steps * dt)
    y_ES.append(ES)
    y_S.append(S)
    y_E.append(E)
    y_P.append(P)

x_t = np.array(x_t)
y_ES = np.array(y_ES)
y_S = np.array(y_S)
y_E = np.array(y_E)
y_P = np.array(y_P)

source_ES = ColumnDataSource(data=dict(x=x_t, y=y_ES))
source_S = ColumnDataSource(data=dict(x=x_t, y=y_S))
source_E = ColumnDataSource(data=dict(x=x_t, y=y_E))
source_P = ColumnDataSource(data=dict(x=x_t, y=y_P))

plot_con_time = figure(plot_height=300, plot_width=600,
                       title="Concentrations over Time",
                       tools="crosshair,reset,save,wheel_zoom",
                       toolbar_location=None,
                       y_range=[0, 600], x_range=[0, 150])

p_ES = plot_con_time.line('x', 'y', source=source_ES, line_width=1, line_alpha=1, color='purple', line_dash='dashed', legend='ES')
p_S = plot_con_time.line('x', 'y', source=source_S, line_width=2, line_alpha=0.6, color='green', legend='S')
p_E = plot_con_time.line('x', 'y', source=source_E, line_width=1, line_alpha=1, color='blue', line_dash='dashed', legend='E')
p_P = plot_con_time.line('x', 'y', source=source_P, line_width=2, line_alpha=0.6, color='red', legend='P')

Km = (kcat + kr)/kf
Vm = kcat * E
label_km = bokeh.models.Label(x=10, y=540,
                              text='Km = {:.2f}'.format(Km), render_mode='css',
                              border_line_color='black', border_line_alpha=1.0,
                              background_fill_color='white', background_fill_alpha=1.0)
plot_con_time.add_layout(label_km)


def get_V0(x_S0):
    V0 = Vm * x_S0/(Km + x_S0)
    return V0


x_S0 = np.linspace(0.01, 2000, 10000)
V0 = get_V0(x_S0)
x_S0_neg = np.linspace(-100, 100, 900)
V0_neg = get_V0(x_S0_neg)
source_V_S = ColumnDataSource(data=dict(x=x_S0, y=V0))
source_Vneg_Sneg = ColumnDataSource(data=dict(x=1/x_S0_neg, y=1/V0_neg))

plot_V_S = figure(plot_height=200, plot_width=600,
                  title="Initial Rate vs Substrate",
                  tools="crosshair,reset,save,wheel_zoom",
                  toolbar_location=None,
                  y_range=[0, 45])
p_V_S = plot_V_S.line('x', 'y', source=source_V_S, line_width=2, line_alpha=1, color='orange', legend='Initial Rate')

plot_Vneg_Sneg = figure(plot_height=200, plot_width=600,
                        title="1/V0 vs 1/[S]",
                        tools="crosshair,reset,save,wheel_zoom",
                        toolbar_location=None,
                        y_range=[-0.5, 0.5],
                        x_range=[-0.02, 0.02])
p_Veg_Sneg = plot_Vneg_Sneg.line('x', 'y', source=source_Vneg_Sneg, line_width=2, line_alpha=1, color='pink', legend='1/V0')


def update_data(attrname, old, new):
    # Get the current slider values
    # rate constant
    global kf
    global kr
    global kcat
    global ki
    global kir

    global S
    global ES
    global EI
    global Inhi
    global E
    global P

    global Inhi0
    global E0
    global S0

    # rate constant
    kf = slider_kf.value / 10000
    kr = slider_kr.value / 10000
    kcat = slider_kcat.value / 1000
    ki = slider_ki.value / 10000
    kir = slider_kir.value / 10000

    # initial condition
    E0 = slider_E0.value
    S0 = slider_S0.value
    Inhi0 = slider_Inhi0.value
    ES = 0
    S = S0
    # get E and P based on ES and S
    E = E0 - ES
    P = S0 - S - ES
    Inhi = Inhi0
    EI = 0
    # converge_tolerance

    x_t = [0]
    y_ES = [ES]
    y_S = [S]
    y_E = [E]
    y_P = [P]
    steps = 0

    # while(abs(P - S0) > tol):
    for i in range(0, int(150/dt)):
        step()
        steps += 1
        x_t.append(steps * dt)
        y_ES.append(ES)
        y_S.append(S)
        y_E.append(E)
        y_P.append(P)

    x_t = np.array(x_t)
    y_ES = np.array(y_ES)
    y_S = np.array(y_S)
    y_E = np.array(y_E)
    y_P = np.array(y_P)

    source_ES.data = dict(x=x_t, y=y_ES)
    source_S.data = dict(x=x_t, y=y_S)
    source_E.data = dict(x=x_t, y=y_E)
    source_P.data = dict(x=x_t, y=y_P)

    Km = (kcat + kr)/kf
    Vm = kcat * E
    label_km.text = 'Km = {:.2f}'.format(Km)

    def get_V0(x_S0):
        V0 = Vm * x_S0/(Km + x_S0)
        return V0

    x_S0 = np.linspace(0.01, 2000, 10000)
    V0 = get_V0(x_S0)
    x_S0_neg = np.linspace(-100, 100, 900)
    V0_neg = get_V0(x_S0_neg)
    source_V_S.data = dict(x=x_S0, y=V0)
    source_Vneg_Sneg.data = dict(x=1/x_S0_neg, y=1/V0_neg)


for w in [slider_kf, slider_kr, slider_kcat, slider_ki, slider_kir, slider_E0, slider_S0, slider_Inhi0]:
    w.on_change('value', update_data)

title = bokeh.models.Div(text="Enzyme Kinetics<br><br>")
energy_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-06-CodeCogsEqn%20-1-.png'><br><br><br>")
wave_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-06-CodeCogsEqn%20-3-.png'><br><br><br>")
prob_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-06-CodeCogsEqn%20-2-.png'><br><br><br>")
v_net_png = bokeh.models.Div(text="<img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-06-CodeCogsEqn%20-4-.png'><br><br><br>")

# Set up layouts and add to document

inputs = column(children=[title, slider_kf, slider_kr, slider_kcat,
                          slider_ki, slider_kir, slider_E0, slider_Inhi0,
                          slider_S0], sizing_mode='fixed', width=500)
con_time = row(inputs, column(plot_con_time), width=1000)
inputs1 = column(children=[energy_png, wave_png, prob_png, v_net_png], sizing_mode='fixed', width=500)
V_S = row(inputs1, column(plot_V_S, plot_Vneg_Sneg), width=1000)
all_layout = column(con_time, V_S)
curdoc().add_root(all_layout)
curdoc().title = "Enzyme Kinetics"
