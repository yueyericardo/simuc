---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.constants import physical_constants as pc
```

#### Physical constants
`pc` is a dictionary of physical constants. There are a ton of them.  No need to remember them or get them wrong.  
You could find all avaiable constant at [Constants (scipy.constants)](https://docs.scipy.org/doc/scipy/reference/constants.html#scipy.constants.physical_constants)

```python
# physical constants usage example
print(pc['molar gas constant'])  # We call this "R"
print(pc['molar volume of ideal gas (273.15 K, 101.325 kPa)'])

# molar Volume at standard temprature pressure
Vstp = pc['molar volume of ideal gas (273.15 K, 101.325 kPa)'][0] * 1000.    # converted to Liters
print(Vstp)
```

```python
def solveP(p1, V1, V2, n):    # Pressure of the gas assuming a path of pV^n
    p = p1 * (V1 / V2)**n
    return p


# Check to see if it works
solveP(1., Vstp, Vstp / 2., 1.)
```

<p><img src="assets/Carnot_Cycle.gif" alt="" style="width: 80%" ></p>

<center>
Edited from 
<a href="https://www.youtube.com/watch?v=cJxF6JqCsJA" target="_blank">
    physics-Thermodynamics -Carnot Engine- basic introduction - YouTube
</a>
</center>

```python
p1 = 2.0
v1 = Vstp / 2.
gamma = 5. / 3.

p2_iso = 1.0
v2_iso = (p1 / p2_iso) * v1

# Plot
fig = make_subplots(rows=1, cols=1, subplot_titles=(r"$\text{Carnot Cycle}$", ))

####################################################################################
V = np.linspace(10., 40., 1000)

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
p3_adi = 0.4  # you are free to change p3
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


fig.update_xaxes(title_text=r"$\overline{V} \;\text{(L)}$ ", row=1, col=1, 
#                  showline=True, linewidth=1, linecolor='black', ticks='outside',
#                  showgrid=False, zeroline=False
                )
fig.update_yaxes(title_text=r'$p \;\text{(atm)}$ ', row=1, col=1,
#                  showline=True, linewidth=1, linecolor='black', ticks='outside',
#                  showgrid=False, zeroline=False
                )

fig.update_layout(height=600, legend={'traceorder':'normal'}, paper_bgcolor='rgba(0,0,0,0)', 
#                   plot_bgcolor='rgba(0,0,0,0)'
                 )
fig.show()
```

<div class="alert alert-info"> 
    <p><b>Carnot Cycle</b></p>
</div>

@@@fig@@@
