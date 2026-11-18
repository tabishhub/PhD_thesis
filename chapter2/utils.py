## utility functions
from contextlib import contextmanager
from typing import Union

import matplotlib.pyplot as plt


@contextmanager
def nice_2d_plot(xlabel=None, ylabel=None, xticks=None, yticks=None):
    fig, ax = plt.subplots()
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=30)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=30)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    ax.axis("auto")
    yield fig, ax
    plt.show()


@contextmanager
def nice_3d_plot(
    xlabel=None, ylabel=None, zlabel=None, xticks=None, yticks=None, zticks=None
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=25, labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=25, labelpad=10)
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontsize=25, labelpad=10)

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xlim(min(xticks), max(xticks))

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim(min(yticks), max(yticks))

    if zticks is not None:
        ax.set_zticks(zticks)
        ax.set_zlim(min(zticks), max(zticks))

    ax.tick_params(axis="both", labelsize=20)
    yield fig, ax
    plt.show()
    plt.close(fig)


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11
    LEGEND_SIZE = 6

    FONT = {"family": "serif", "serif": ["Computer Modern Serif"], "size": MEDIUM_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        # "axes.spines.left": True,
        # "axes.spines.bottom": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "figure.constrained_layout.use": True,
    }

    return rc_params


def set_size(
    width: float = 434.55125,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
    adjust_height: Union[None, float] = None,
) -> tuple:
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if adjust_height is not None:
        golden_ratio += adjust_height

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


# import numpy as np
# from scipy.integrate import solve_ivp

# def ruoff_model_original(t_span, y0, Temperature):
#     # Constants
#     Tref = 286.5  # K
#     R = 0.0083144  # kJ/K/mol

#     # Prefactors at Tref (units mM/min except K1 in mM)
#     x_temp = np.array([
#         2.5,
#         100.,
#         6.0,
#         16.0,
#         100.0,
#         1.28,
#         12.0,
#         1.8,
#         13.0,
#         0.52
#     ])

#     # Activation energy barriers (kJ/mol)
#     Ebarriers = np.array([
#         16.2,  # EJ0
#         13.8,  # Ek1
#         60.7,  # Ek2
#         44.9,  # Ek3
#         58.7,  # Ek4
#         41.2,  # Ek5
#         31.4,  # Ek6
#         24.3,  # Ek
#         15.9,  # Ekappa
#         47.0   # EK1
#     ])

#     # Temperature-adjusted rate constants
#     prefactors = x_temp * np.exp(Ebarriers / (R * Tref))
#     x_temp_new = prefactors * np.exp(-Ebarriers / (R * Temperature))

#     # Assign rate constants
#     J0, k1, k2, k3, k4, k5, k6, k, kappa, K1 = x_temp_new
#     q = 4.0
#     psi = 0.1

#     # Parameters passed to the ODE function
#     params = [J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi]

#     # ODE solver
#     sol = solve_ivp(
#         fun=lambda t, y: ruoff_model_derivs(t, y, params),
#         t_span=(t_span[0], t_span[-1]),
#         y0=y0,
#         t_eval=t_span,
#         method='RK45'
#     )

#     # Compute derivatives at each time point
#     y_derivs = np.array([ruoff_model_derivs(ti, yi, params) for ti, yi in zip(sol.t, sol.y.T)])

#     return sol.t, sol.y.T, y_derivs, params

# def ruoff_model_derivs(t, y, params):
#     # Unpack parameters
#     J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi = params

#     # Pooled variable totals
#     N = 1.0  # NADH pool
#     A = 4.0  # Adenosine pool

#     # Map concentrations
#     S1, S2, S3, S4, N2, A3, S5 = y

#     # Reaction rates
#     v1 = k1 * S1 * A3 / (1 + (A3 / K1) ** q)
#     v2 = k2 * S2 * (N - N2)
#     v3 = k3 * S3 * (A - A3)
#     v4 = k4 * S4 * N2
#     v5 = k5 * A3
#     v6 = k6 * S2 * N2
#     v7 = k * S5
#     J = kappa * (S4 - S5)

#     # ODEs
#     dS1dt = J0 - v1
#     dS2dt = 2 * v1 - v2 - v6
#     dS3dt = v2 - v3
#     dS4dt = v3 - v4 - J
#     dN2dt = v2 - v4 - v6
#     dA3dt = -2 * v1 + 2 * v3 - v5
#     dS5dt = psi * J - v7

#     return [dS1dt, dS2dt, dS3dt, dS4dt, dN2dt, dA3dt, dS5dt]

# # Time span (in minutes)
# t = np.linspace(0, 3, 1000)

# # Initial conditions for the 7 state variables
# y0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]

# # Example temperature (in Kelvin)
# Temperature = 298.15  # 25°C

# T, Y, dYdt, params = ruoff_model_original(t, y0, Temperature)
