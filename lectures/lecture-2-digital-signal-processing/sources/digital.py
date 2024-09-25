import numpy as np
from numpy import pi as π

from display import axes


def generate_signal(Δt=0.9, f=1.7, A=0.35, Φ=π, n=200, dc=0.6, τ=0.1):
    """Generate a sine wave signal.

    Arguments
    ---------
    Δt : float
        Duration of the signal.
    f : float
        Frequency of the signal.
    A : float
        Amplitude of the signal.
    Φ : float
        Phase of the signal.
    n : int
        Number of samples.
    dc : float
        DC component of the signal.
    τ : float
        Time offset.

    Returns
    -------
    t : ndarray
        Time samples.
    y : ndarray
        Signal samples.
    """
    t = np.linspace(0, Δt, n) + τ
    y = A * np.cos(2 * π * f * t + Φ) + dc
    return t, y


def digitize_signal(t, x, n_bins=7, sampling_step=25, bin_margin=0.1):
    """Digitize a continuous signal.

    Arguments
    ---------
    t : ndarray
        Time samples.
    x : ndarray
        Signal samples.
    n_bins : int
        Number of bins.
    sampling_step : int
        Sampling step.
    bin_margin : float
        Margin for the bins.

    Returns
    -------
    bins : ndarray
        Bin centers.
    t : ndarray
        Time samples.
    digital : ndarray
        Digital signal samples.
    """
    boundaries = x.min() - bin_margin, x.max() + bin_margin
    bins = np.linspace(*boundaries, n_bins)
    half_bin_step = (bins[1] - bins[0]) / 2
    digital = np.digitize(x, bins, right=False)
    digital = bins[digital] - half_bin_step
    bins -= half_bin_step
    return bins[1:], t[::sampling_step], digital[::sampling_step]


def main():

    # Create figure and axis
    fig, ax = axes(x_label="t", y_label="x(t)")
    fig.set_facecolor("none")

    # Continuous case
    t, x_t = generate_signal()
    digits, t_n, x_n = digitize_signal(t, x_t)

    # Grid lines
    ax.set_yticks(digits, labels=[])
    ax.tick_params(axis="y", width=0)
    ax.grid(axis="y", color="w", zorder=-1, lw=1)

    # Plot digital signal
    ax.stem(t_n, x_n, markerfmt=".", basefmt=" ")

    # Plot continuous signal
    ax.plot(t, x_t)

    # # Delta t
    t1, t2 = t_n[[1, 2]]
    y0 = -0.05
    arrowprops = dict(arrowstyle="<->", clip_on=False, shrinkA=0, shrinkB=0)
    kwargs = dict(xycoords="data", arrowprops=arrowprops)
    ax.annotate("", (t1, y0), xytext=(t2, y0), **kwargs)
    ax.text(t1 + (t2 - t1) / 2, 2 * y0, r"$\Delta t$", ha="center", va="top")

    # Save
    fig.savefig("images/signal/digital")


if __name__ == "__main__":
    main()
