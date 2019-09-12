import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, CheckboxButtonGroup
from bokeh.plotting import figure

# Set up widgets
text = TextInput(title="title", value='Sound Interference')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)
speed = Slider(title="speed", value=0.1, start=0.0, end=2, step=0.1)
reflection = CheckboxButtonGroup(labels=["1st Reflection", "2nd Reflection", "3rd Reflection", "Total Interference"], active=[])

# Set up data
N = 200
x = np.linspace(0, 12, N)
# Incident Wave
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))
# First Reflection
x1 = np.linspace(12, 24, N)
y1 = np.flip(np.sin(x1))
source1 = ColumnDataSource(data=dict(x=x, y=y1))
# Second Reflection
x2 = np.linspace(24, 36, N)
y2 = np.sin(x2)
source2 = ColumnDataSource(data=dict(x=x, y=y1))
# Third Reflection
x3 = np.linspace(36, 48, N)
y3 = np.flip(np.sin(x2))
source3 = ColumnDataSource(data=dict(x=x, y=y3))
# Total Interference
y4 = y + y1 + y2 + y3
source4 = ColumnDataSource(data=dict(x=x, y=y4))


# Set up plot
plot = figure(plot_height=600, plot_width=400, title="Sound Interference",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 12], y_range=[-3.5, 3.5])

p = plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6, color='blue', legend='Incident Wave')
p1 = plot.line('x', 'y', source=source1, line_width=1, line_alpha=0.6, color='red', legend='First Reflection')
p2 = plot.line('x', 'y', source=source2, line_width=1, line_alpha=0.6, color='green', legend='Second Reflection')
p3 = plot.line('x', 'y', source=source3, line_width=1, line_alpha=0.6, color='purple', legend='Third Reflection')
p4 = plot.line('x', 'y', source=source4, line_width=1, line_alpha=0.6, color='black', legend='Total Interference')


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value


text.on_change('value', update_title)

sec = 0.0


def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = sec
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 12, N)
    y = a*np.sin(k*x + w) + b
    source.data = dict(x=x, y=y)
    # First Reflection
    x1 = np.linspace(12, 24, N)
    y1 = np.flip(a*np.sin(k*x1 + w) + b)
    source1.data = dict(x=x, y=y1)
    # Second Reflection
    x2 = np.linspace(24, 36, N)
    y2 = a*np.sin(k*x2 + w) + b
    source2.data = dict(x=x, y=y2)
    # Third Reflection
    x3 = np.linspace(36, 48, N)
    y3 = np.flip(a*np.sin(k*x3 + w) + b)
    source3.data = dict(x=x, y=y3)
    # Total Interference
    y4 = y + y1 + y2 + y3
    source4.data = dict(x=x, y=y4)


def update_live():
    global sec

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = sec
    sec -= speed.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 12, N)
    y = a*np.sin(k*x + w) + b
    source.data = dict(x=x, y=y)
    # First Reflection
    x1 = np.linspace(12, 24, N)
    y1 = np.flip(a*np.sin(k*x1 + w) + b)
    source1.data = dict(x=x, y=y1)
    # Second Reflection
    x2 = np.linspace(24, 36, N)
    y2 = a*np.sin(k*x2 + w) + b
    source2.data = dict(x=x, y=y2)
    # Third Reflection
    x3 = np.linspace(36, 48, N)
    y3 = np.flip(a*np.sin(k*x3 + w) + b)
    source3.data = dict(x=x, y=y3)

    # Total Interference
    y4 = y + y1 + y2 + y3
    source4.data = dict(x=x, y=y4)

    if 0 in reflection.active:
        p1.visible = True
    else:
        p1.visible = False

    if 1 in reflection.active:
        p2.visible = True
    else:
        p2.visible = False

    if 2 in reflection.active:
        p3.visible = True
    else:
        p3.visible = False

    if 3 in reflection.active:
        p4.visible = True
    else:
        p4.visible = False


for w in [offset, amplitude, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(children=[text, offset, amplitude, freq, speed, reflection], sizing_mode='stretch_width')
curdoc().add_periodic_callback(update_live, 100)
curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sound"

