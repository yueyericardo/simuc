import bokeh
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, RadioButtonGroup, Button, PreText
from bokeh.plotting import figure


class Condition(object):

    def __init__(self, Ks, kcat, E0, S0, Ki, Kii, Kss, I0):
        # Eq constant
        self.Ks = Ks
        self.Ki = Ki
        self.Kii = Kii
        self.Kss = Kss

        # rate constant
        self.kr = 1/10000
        self.kf = Ks * self.kr
        self.kcat = kcat
        self.kir = 1 / 10000
        self.ki = Ki * self.kir
        self.kesir = 1 / 10000
        self.kesi = Kii * self.kesir
        self.keisr = 1 / 10000
        self.keis = Kss * self.keisr

        # initial condition
        self.E0 = E0
        self.S0 = S0
        self.I0 = I0
        self.ES = 0
        self.EI = 0
        self.ESI = 0
        self.P = 0
        self.S = S0
        self.Inhi = self.I0
        self.E = E0

        # converge_tolerance
        self.tol = 1
        self.dt = 0.01
        self.ESI_scale = 50

        # gen info
        self.gen_info()

    def gen_plot_data(self):
        self.x_t = [0]
        self.y_S = [self.S]
        self.y_ES = [self.ES]
        self.y_EI = [self.EI]
        self.y_ESI = [self.ESI]
        self.y_Inhi = [self.Inhi]
        self.y_E = [self.E]
        self.y_P = [self.P]
        steps = 0

        for i in range(0, int(150/self.dt)):
            d_ES = self.kf * self.E * self.S - self.kr * self.ES - self.kcat * self.ES - self.kesi * self.ES * self.Inhi + self.kesir * self.ESI
            d_EI = self.ki * self.E * self.Inhi - self.kir * self.EI + self.keisr * self.ESI - self.keis * self.EI * self.S
            d_ESI = self.kesi * self.ES * self.Inhi - self.kesir * self.ESI + self.keis * self.EI * self.S - self.keisr * self.ESI
            d_S = self.kr * self.ES - self.kf * self.E * self.S + self.keisr * self.ESI - self.keis * self.EI * self.S

            self.ES = self.ES + d_ES * self.dt
            self.EI = self.EI + d_EI * self.dt
            self.ESI = self.ESI + d_ESI * self.dt
            self.S = self.S + d_S * self.dt

            self.E = self.E0 - self.ES - self.EI - self.ESI
            self.Inhi = self.I0 - self.EI - self.ESI
            self.P = self.S0 - self.S - self.ES - self.ESI

            steps += 1
            self.x_t.append(steps * self.dt)
            self.y_S.append(self.S)
            self.y_ES.append(self.ES)
            self.y_EI.append(self.EI)
            self.y_ESI.append(self.ESI * self.ESI_scale)
            self.y_Inhi.append(self.Inhi)
            self.y_E.append(self.E)
            self.y_P.append(self.P)

        self.x_t = np.array(self.x_t)
        self.y_S = np.array(self.y_S)
        self.y_ES = np.array(self.y_ES)
        self.y_EI = np.array(self.y_EI)
        self.y_ESI = np.array(self.y_ESI)
        self.y_Inhi = np.array(self.y_Inhi)
        self.y_E = np.array(self.y_E)
        self.y_P = np.array(self.y_P)

    def gen_more_data(self, x_S0, x_1_over_S0):
        Km = (self.kcat + self.kr) / self.kf
        Vm = self.kcat * self.E0
        self.inhi_type = 'None Inhibition'
        self.text_Km = 'Km = {:.2f}'.format(Km)
        self.text_Vm = 'Vm = {:.2f}'.format(Vm)

        # competitive inhibition
        if (self.Ki > 0 and self.Kii == 0):
            Km = Km * (1 + self.I0 / self.Ki)
            self.inhi_type = 'Competitive Inhibition'
            self.text_Km = 'Km\' = {:.2f}'.format(Km)

        # non-competitive inhibition
        if (self.Ki > 0 and self.Kii > 0):
            Vm = Vm / (1 + self.I0 / self.Kii)
            Km = Km * (1 + self.I0 / self.Ki) / (1 + self.I0 / self.Kii)
            self.text_Km = 'Km\' = {:.2f}'.format(Km)
            self.text_Vm = 'Vm\' = {:.2f}'.format(Vm)
            if self.Ki == self.Kii:
                self.inhi_type = 'Non-Competitive Inhibition'
            else:
                self.inhi_type = 'Mixed Inhibition'

        # un-competitive inhibition
        if (self.Ki == 0 and self.Kii > 0):
            Km = Km / (1 + self.I0 / self.Kii)
            Vm = Vm / (1 + self.I0 / self.Kii)
            self.text_Km = 'Km\' = {:.2f}'.format(Km)
            self.text_Vm = 'Vm\' = {:.2f}'.format(Vm)
            self.inhi_type = 'Un-Competitive Inhibition'

        self.V0 = Vm * x_S0 / (Km + x_S0)
        self.V0_over = 1 / Vm + (Km / Vm) * x_1_over_S0

        self.x_circle_x = -1 / Km
        self.y_circle_y = 1 / Vm
        self.xy_circle_half_Vm = (Km, Vm/2)

    def gen_info(self):
        all_information_text = "All Initial Parameters: <br><br>Rate Constants: <br>(Unit for k1 is L/(mol*s), Unit for k-1 is 1/s)<br>Ks&nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;{: 3d}&nbsp;&nbsp;(k1 = {: .4f}, k-1 = {: .4f})<br>kcat =&nbsp;&nbsp;{: 6.2f}&nbsp;&nbsp;<br>Ki&nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;{: 3d}&nbsp;&nbsp;(k1 = {: .4f}, k-1 = {: .4f})<br>Kii&nbsp;&nbsp;=&nbsp;&nbsp;{: 3d}&nbsp;&nbsp;(k1 = {: .4f}, k-1 = {: .4f})<br>Kss&nbsp;&nbsp;=&nbsp;&nbsp;{: 3d}&nbsp;&nbsp;(k1 = {: .4f}, k-1 = {: .4f})<br><br>Initial Concentration:<br>E0   = {: .0f} mol/L, S0   = {: .0f} mol/L, I0   = {: .0f} mol/L<br>"
        self.info = all_information_text.format(self.Ks, self.kf, self.kr, self.kcat, self.Ki, self.ki, self.kir,
                                                self.Kii, self.kesi, self.kesir, self.Kss, self.keis, self.keisr,
                                                self.E0, self.S0, self.I0)


# load preset conditions
# preset_cond = RadioButtonGroup(labels=["Faster", "Slower", "Default", "Competitive Inhibition", "Uncompetitive Inhibition", "Non-competitive Inhibition"], active=2)
preset_cond = RadioButtonGroup(labels=["Faster", "Slower", "Default", "Competitive Inhibition", "Uncompetitive Inhibition"], active=2)

cond_default = Condition(Ks=10, kcat=0.1, E0=200, S0=500, Ki=0, Kii=0, Kss=0, I0=0)
cond_faster = Condition(Ks=19, kcat=0.15, E0=300, S0=500, Ki=0, Kii=0, Kss=0, I0=0)
cond_slower = Condition(Ks=4, kcat=0.08, E0=120, S0=500, Ki=0, Kii=0, Kss=0, I0=0)
cond_comp_ih = Condition(Ks=10, kcat=0.1, E0=200, S0=500, Ki=3, Kii=0, Kss=0, I0=100)
cond_uncomp_ih = Condition(Ks=10, kcat=0.1, E0=200, S0=500, Ki=0, Kii=10, Kss=0, I0=100)
cond_noncomp_ih = Condition(Ks=10, kcat=0.1, E0=200, S0=500, Ki=2, Kii=10, Kss=0, I0=100)

cond_all = [cond_faster, cond_slower, cond_default, cond_comp_ih, cond_uncomp_ih]

# ------------------------------- Bokeh -------------------------------
# sliders
slider_Ks = Slider(title="Ks", value=cond_default.Ks, start=1, end=20, step=3, sizing_mode="stretch_width")
slider_kcat = Slider(title="kcat", value=cond_default.kcat, start=0.06, end=0.2, step=0.01, sizing_mode="stretch_width")
slider_Ki = Slider(title="Ki", value=cond_default.Ki, start=0, end=20, step=2, sizing_mode="stretch_width")
slider_K_ES_I = Slider(title="Kii", value=cond_default.Kii, start=0, end=50, step=10, sizing_mode="stretch_width")
slider_K_EI_S = Slider(title="Kss", value=cond_default.Kss, start=0, end=50, step=10, sizing_mode="stretch_width")
slider_E0 = Slider(title="E0", value=cond_default.E0, start=0, end=300, step=50, sizing_mode="stretch_width")
slider_S0 = Slider(title="S0", value=cond_default.S0, start=400, end=600, step=50, sizing_mode="stretch_width")
slider_Inhi0 = Slider(title="I0", value=cond_default.I0, start=0, end=700, step=10, sizing_mode="stretch_width")

# generate source data for first plot
cond_default.gen_plot_data()
source_S = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_S))
source_ES = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_ES))
source_EI = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_EI))
source_ESI = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_ESI))
source_Inhi = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_Inhi))
source_E = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_E))
source_P = ColumnDataSource(data=dict(x=cond_default.x_t, y=cond_default.y_P))
# generate source data for 2nd and 3rd plot
x_S0 = np.linspace(0.01, 2000, 10000)
x_1_over_S0 = np.linspace(-0.02, 0.02, 900)
cond_default.gen_more_data(x_S0, x_1_over_S0)
source_V_S = ColumnDataSource(data=dict(x=x_S0, y=cond_default.V0))
source_1_over = ColumnDataSource(data=dict(x=x_1_over_S0, y=cond_default.V0_over))
source_circle_x = ColumnDataSource(data=dict(x=[cond_default.x_circle_x], y=[0]))
source_circle_y = ColumnDataSource(data=dict(x=[0], y=[cond_default.y_circle_y]))
source_circle__half_Vm = ColumnDataSource(data=dict(x=[cond_default.xy_circle_half_Vm[0]], y=[cond_default.xy_circle_half_Vm[1]]))

# first plot
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
p_ESI = plot_con_time.line('x', 'y', source=source_ESI, line_width=1, line_alpha=1, color='black', line_dash='dashed', legend='ESI * {}'.format(cond_default.ESI_scale))
plot_con_time.yaxis.axis_label = "Concentration  (mol / L)"
plot_con_time.xaxis.axis_label = "Time  (s)"
# add some label
label_km = bokeh.models.Label(x=10, y=540,
                              text=cond_default.text_Km, render_mode='css',
                              border_line_color='black', border_line_alpha=1.0,
                              background_fill_color='white', background_fill_alpha=1.0)
label_vm = bokeh.models.Label(x=10, y=480,
                              text=cond_default.text_Vm, render_mode='css',
                              border_line_color='black', border_line_alpha=1.0,
                              background_fill_color='white', background_fill_alpha=1.0)
label_inhi_type = bokeh.models.Label(x=10, y=420,
                                     text=cond_default.inhi_type, render_mode='css',
                                     border_line_color='black', border_line_alpha=1.0,
                                     background_fill_color='white', background_fill_alpha=1.0)
plot_con_time.add_layout(label_km)
plot_con_time.add_layout(label_vm)
plot_con_time.add_layout(label_inhi_type)

# 2nd plot
plot_V_S = figure(plot_height=200, plot_width=600,
                  title="Initial Rate vs Substrate",
                  tools="crosshair,reset,save,wheel_zoom",
                  toolbar_location=None,
                  y_range=[0, 45],
                  x_range=[0, 2010])
p_V_S = plot_V_S.line('x', 'y', source=source_V_S, line_width=2, line_alpha=1, color='violet', legend='Initial Rate')
plot_V_S.circle('x', 'y', source=source_circle__half_Vm, fill_color="purple", line_color="black", size=8, legend='(Km, Vm/2)')
plot_V_S.yaxis.axis_label = "V0  (mol / (s*L))"
plot_V_S.xaxis.axis_label = "[S]  (mol / L)"

# 3rd plot
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


# update function
def update_data(attrname, old, new):
    cond_update = Condition(Ks=slider_Ks.value, kcat=slider_kcat.value, E0=slider_E0.value,
                            S0=slider_S0.value, Ki=slider_Ki.value, Kii=slider_K_ES_I.value,
                            Kss=slider_K_EI_S.value, I0=slider_Inhi0.value)

    cond_update.gen_plot_data()
    source_S.data = dict(x=cond_update.x_t, y=cond_update.y_S)
    source_ES.data = dict(x=cond_update.x_t, y=cond_update.y_ES)
    source_EI.data = dict(x=cond_update.x_t, y=cond_update.y_EI)
    source_ESI.data = dict(x=cond_update.x_t, y=cond_update.y_ESI)
    source_Inhi.data = dict(x=cond_update.x_t, y=cond_update.y_Inhi)
    source_E.data = dict(x=cond_update.x_t, y=cond_update.y_E)
    source_P.data = dict(x=cond_update.x_t, y=cond_update.y_P)

    cond_update.gen_more_data(x_S0, x_1_over_S0)
    label_km.text = cond_update.text_Km
    label_vm.text = cond_update.text_Vm
    label_inhi_type.text = cond_update.inhi_type

    source_V_S.data = dict(x=x_S0, y=cond_update.V0)
    source_1_over.data = dict(x=x_1_over_S0, y=cond_update.V0_over)
    source_circle_x.data = dict(x=[cond_update.x_circle_x], y=[0])
    source_circle_y.data = dict(x=[0], y=[cond_update.y_circle_y])
    source_circle__half_Vm.data = dict(x=[cond_update.xy_circle_half_Vm[0]], y=[cond_update.xy_circle_half_Vm[1]])

    all_information.text = cond_update.info


def update_slider(cond):
    for s in [slider_Ks, slider_kcat, slider_Ki, slider_K_ES_I, slider_K_EI_S, slider_E0, slider_Inhi0, slider_S0]:
        s.value = getattr(cond, s.title)


def load_preset(attrname, old, new):
    active_cond = preset_cond.active
    update_slider(cond_all[active_cond])


for w in [slider_Ks, slider_kcat, slider_Ki, slider_K_ES_I, slider_K_EI_S, slider_E0, slider_Inhi0, slider_S0]:
    w.on_change('value', update_data)


for w in [preset_cond]:
    w.on_change('active', load_preset)


# export button
def export():
    info = {}
    for s in [slider_Ks, slider_kcat, slider_Ki, slider_K_ES_I, slider_K_EI_S, slider_E0, slider_Inhi0, slider_S0]:
        info[s.title] = s.value
    export_text.text = str(info)
    export_text.style = {"color": "black"}


export_button = Button(label="Export", button_type="success", width=100)
export_text = PreText(text=" ", width=400)
export_button.on_click(export)


# quiz button
def submit_quiz2():
    if (slider_Ki.value > 0 and slider_K_ES_I.value > 0):
        export_text.text = "Right!"
        export_text.style = {"color": "green"}
    else:
        export_text.text = "Wrong!"
        export_text.style = {"color": "red"}


quiz2_button = Button(label="Submit Quiz 2", button_type="success", width=100)
quiz2_button.on_click(submit_quiz2)

# other information
title_left = bokeh.models.Div(text="Without Inhibition<br><br>", sizing_mode="stretch_width")
title_inhi = bokeh.models.Div(text="Inhibition<br><br>", sizing_mode="stretch_width")

all_information = bokeh.models.Div(text=cond_default.info, style={'margin-top': '200px', 'width': '1450px', 'font-family': 'serif', 'color': 'black', 'text-align': 'left'}, sizing_mode="stretch_width")

reaction_png = bokeh.models.Div(text="<br><br><br><br><br>Symbol Definition: <img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-18-143043.png' style='width: 310px; margin: auto; display: block'> <br><br>Michaelis-Menten Equation:<br><br> <img src='https://yyrcd-1256568788.cos.na-siliconvalley.myqcloud.com/yyrcd/2019-09-18-mm-equation.png' style='width: 150px; display: block'>", sizing_mode="stretch_width")

# Set up layouts and add to document
# first row
left = column(children=[title_left, slider_Ks, slider_kcat, slider_E0, slider_S0, reaction_png], sizing_mode='fixed', width=400, height=730)
middle = column(children=[plot_con_time, plot_V_S, plot_1_over], sizing_mode='fixed', width=700, height=730)
right = column(children=[title_inhi, slider_Ki, slider_K_ES_I, slider_K_EI_S, slider_Inhi0, all_information], sizing_mode='fixed', width=400, height=730)
# second row
emptydiv = bokeh.models.Div(text=" ", sizing_mode="stretch_width")
empty = column(emptydiv, sizing_mode='fixed', width=400)
bottom = column(preset_cond, sizing_mode='fixed', width=1100, height=40)
# third row
button = column(row(export_button, quiz2_button), sizing_mode='fixed', width=300, height=40)
text = column(export_text, sizing_mode='fixed', width=500, height=40)

all_layout = column(row(left, middle, right), row(empty, bottom), row(empty, button), row(empty, text))

curdoc().add_root(all_layout)
curdoc().title = "Enzyme Kinetics"
