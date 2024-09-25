import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero


def axes(limits=[0, 0, 1, 1], margin=0.1, x_label="x", y_label="y(x)"):
    """Create a figure with math axes.

    Arguments
    ---------
    limits : list
        Limits of the axis as [x_min, y_min, x_max, y_max].
    margin : float
        Margin to add to the x limits (y limits are adjusted according to
        the aspect ratio of the figure).
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    """
    # Generate figure and axis
    fig = plt.figure(figsize=(4, 3))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    # Set axis properties
    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("->")
        ax.axis[direction].set_visible(True)
        ax.axis[direction].set_zorder(0)

    # Hide top and right axes
    for direction in ["top", "right", "left", "bottom"]:
        ax.axis[direction].set_visible(False)

    # Set ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set limits
    fig_ratio = fig.get_figwidth() / fig.get_figheight()
    x_margin = margin
    y_margin = margin * (limits[3] - limits[1]) * fig_ratio
    ax.set_xlim(limits[0] - x_margin, limits[2] + x_margin)
    ax.set_ylim(limits[1] - y_margin, limits[3] + y_margin)

    # Labels
    x = -x_margin * 0.3
    y = -y_margin * 0.3
    ax.text(x, y, r"$0$", ha="right", va="top")
    ax.text(limits[2] + x_margin, y, rf"${x_label}$", va="top", ha="right")
    ax.text(x, limits[3] - y, rf"${y_label}$", ha="right")

    return fig, ax


def plot_system(input_label="$x(t)$", output_label="$y(t)$"):
    """Plot the system impulse response.

    Arguments
    ---------
    input_label : str
        Label for the input signal.
    output_label : str
        Label for the output signal.

    Returns
    -------
    fig : Figure
        Figure with the system.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 1.7))
    ax.axis("off")

    # Draw rectangle
    ax.add_patch(plt.Rectangle((0.4, 0.4), 0.2, 0.5, ec="k", fc="w"))
    ax.text(0.5, 0.65, r"$\mathcal{S}$", ha="center", va="center")

    # Arrow properties
    arrowprops = dict(shrinkA=5, shrinkB=5)

    # Input
    ax.annotate(
        input_label,
        xy=(0.4, 0.65),
        xytext=(0.2, 0.65),
        arrowprops=dict(arrowstyle="->", **arrowprops),
        va="center",
        ha="right",
    )

    # Output
    ax.annotate(
        output_label,
        xy=(0.6, 0.65),
        xytext=(0.8, 0.65),
        arrowprops=dict(arrowstyle="<-", **arrowprops),
        va="center",
        ha="left",
    )

    return fig
