import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def setup_figure(style):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax

def render(data, style, **kwargs):
    fig, ax = setup_figure(style)
    if style == 'pie':
        return render_pie(data, ax, **kwargs)
    elif style == 'bar':
        return render_bar(data, ax, **kwargs)
    else:
        return None

def render_pie(data, ax, alpha=0.7):
    for color, size, start_angle in data:
        wedge = Wedge(center=ax.center, r=1, theta1=start_angle, theta2=start_angle + size, color=color, alpha=alpha)
        ax.add_artist(wedge)
    plt.show()

def render_bar(data, ax, colors=None):
    labels, sizes, color = data
    if colors is None:
        colors = [f'C{i}' for i in range(len(labels))]
    ax.bar(labels, sizes, color=colors)
    plt.show()