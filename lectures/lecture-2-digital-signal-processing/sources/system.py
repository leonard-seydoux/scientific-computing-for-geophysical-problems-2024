import matplotlib.pyplot as plt
import numpy as np

from display import plot_system

plt.rcParams.update({"savefig.bbox": "standard"})


def inset_signal(fig, side="input", x_shift=0, y_shift=0):
    """Add an inset with a signal to a figure."""
    if side == "input":
        inset_position = [0.17 + x_shift, 0 + y_shift, 0.15, 0.5]
    else:
        inset_position = [0.76 + x_shift, 0 + y_shift, 0.15, 0.5]
    ax = fig.add_axes(inset_position)
    ax.axis("off")
    ax.set_ylim(-1, 1)
    return ax


def plot_translated_signal(ax, t, x, δt):
    """Plot a translated signal."""
    t_max = t[np.argmax(x)]
    ax.plot(t, x)
    ax.plot(t + δt, x)
    arrowprops = dict(
        arrowstyle="<|-,head_width=0.07,head_length=0.22",
        lw=0.7,
        color="0.6",
        shrinkB=4,
    )
    ax.annotate(
        "",
        xy=(t_max, 0.7),
        xytext=(t_max + δt, 0.7),
        arrowprops=arrowprops,
        va="center",
        ha="left",
    )


def main():
    plot_system().savefig("images/system/system")


def linear():
    plot_system(
        input_label="$ax_1(t) + bx_2(t)$",
        output_label="$ay_1(t) + by_2(t)$",
    ).savefig("images/system/linear-system")


def time_invariant():

    # Plot system
    fig = plot_system(
        input_label=r"$x(t - \tau)$", output_label=r"$y(t - \tau)$"
    )

    # Add axes for output data
    t = np.linspace(0, 1, 100)
    x = np.exp(-t * 20) * (t > 0.2)
    x /= x.max()
    y = np.exp(-t * 7) * np.sin(t * 40)

    # Add axes for output data
    ax = inset_signal(fig, side="input")
    plot_translated_signal(ax, t, x, 0.5)

    # Add axes for output data
    ax = inset_signal(fig, side="output")
    plot_translated_signal(ax, t, y, 0.5)

    # Save
    fig.savefig("images/system/time-invariant-system")


def impulse_response():
    # Plot system
    fig = plot_system(input_label=r"$\delta(t)$", output_label=r"$h(t)$")

    # Add axes for output data
    t = np.linspace(0, 1, 100)
    x = np.zeros_like(t)
    x[np.argmin(np.abs(t - 0.2))] = 1
    x /= x.max()
    y = np.exp(-t * 7) * np.sin(t * 40)
    y = np.roll(y, 20)

    # Add axes for output data
    ax = inset_signal(fig, side="input", x_shift=0.02)
    ax.plot(t, x)

    # Add axes for output data
    ax = inset_signal(fig, side="output", x_shift=-0.03)
    ax.plot(t, y)

    # Save
    fig.savefig("images/system/impulse-response")


def impulse_response_translated():
    # Plot system
    fig = plot_system(
        input_label=r"$\delta(t - \tau)$", output_label=r"$h(t - \tau)$"
    )

    # Add axes for output data
    t = np.linspace(0, 1, 100)
    x = np.zeros_like(t)
    x[np.argmin(np.abs(t - 0.2))] = 1
    x /= x.max()
    y = np.exp(-t * 7) * np.sin(t * 40)
    y = np.roll(y, 20)

    # Add axes for output data
    ax = inset_signal(fig, side="input", x_shift=-0.03)
    plot_translated_signal(ax, t, x, 0.5)

    # Add axes for output data
    ax = inset_signal(fig, side="output", x_shift=-0.02)
    plot_translated_signal(ax, t, y, 0.5)

    # Save
    fig.savefig("images/system/impulse-response-translated")


def impulse_response_super():
    # Plot system
    fig = plot_system(
        input_label=r"$a\delta(t) + b\delta(t - \tau)$",
        output_label=r"$ah(t) + bh(t - \tau)$",
    )

    # Add axes for output data
    t = np.linspace(0, 1, 100)
    x = np.zeros_like(t)
    x[np.argmin(np.abs(t - 0.2))] = 1
    x[np.argmin(np.abs(t - 0.4))] = 0.6
    x /= x.max()
    y = np.exp(-t * 7) * np.sin(t * 40)
    y = np.roll(y, 10)
    y += np.roll(y, 30) * 0.6

    # Add axes for output data
    ax = inset_signal(fig, side="input", x_shift=-0.04)
    ax.plot(t, x)

    # Add axes for output data
    ax = inset_signal(fig, side="output", x_shift=-0.03)
    ax.plot(t, y)

    # Save
    fig.savefig("images/system/impulse-response-super")


def impulse_response_sum():
    # Plot system
    fig = plot_system(
        input_label=r"$\Sigma_{i} a_i\delta(t - \tau_i)$",
        output_label=r"$\Sigma_{i} a_ih(t - \tau_i)$",
    )

    # Add axes for output data
    t = np.linspace(0, 1, 100)
    x = np.zeros_like(t)
    y = np.exp(-t * 7) * np.sin(t * 40)
    for i in range(0, len(t), 7):
        x[i] = np.sin(2 * np.pi * 2 * t[i]) * np.exp(-t[i] * 2)
        y += np.roll(y, i) * x[i]
    x /= x.max()
    y /= y.max()

    # Add axes for output data
    ax = inset_signal(fig, side="input", x_shift=-0.04)
    ax.plot(t, x)

    # Add axes for output data
    ax = inset_signal(fig, side="output", x_shift=-0.03)
    ax.plot(t, y)

    # Save
    fig.savefig("images/system/impulse-response-sum")


def convolution():
    # Plot system
    fig = plot_system(
        output_label=r"$y(t) = x(t) * h(t)$",
    )

    # Add axes for output data
    np.random.seed(0)
    t = np.linspace(0, 1, 100)
    x = np.random.randn(len(t))
    x /= np.abs(x).max()
    y = np.convolve(x, np.exp(-t * 7) * np.sin(t * 40), mode="same")
    y /= np.abs(y).max()

    # Add axes for output data
    ax = inset_signal(fig, side="input")
    ax.plot(t, x)

    # Add axes for output data
    ax = inset_signal(fig, side="output")
    ax.plot(t, y)

    # Save
    fig.savefig("images/system/convolution")


if __name__ == "__main__":
    main()
    linear()
    time_invariant()
    impulse_response()
    impulse_response_translated()
    impulse_response_super()
    impulse_response_sum()
    convolution()
