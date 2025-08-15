import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def render(pieces, style):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    
    if style == 'pie':
        for index, (color, size, start_angle) in enumerate(pieces):
            wedge = Wedge(center=ax.center, r=1, theta1=start_angle, theta2=start_angle + size, color=color, alpha=0.7)
            ax.add_artist(wedge)
    elif style == 'bar':
        fig, ax = plt.subplots()
        colors = [piece[0] for piece in pieces]
        sizes = [piece[1] for piece in pieces]
        labels = [f'Category {index+1}' for index in range(len(pieces))]
        ax.bar(labels, sizes, color=colors)
    else:
        raise ValueError("Unsupported style")
    
    plt.show()