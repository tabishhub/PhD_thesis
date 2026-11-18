## utility functions
from contextlib import contextmanager

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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


def inverse(x, epsilon=1e-10, ret_sqrt=False):
    eigval, eigvec = jnp.linalg.eigh(x)

    eigval = jnp.maximum(eigval, epsilon)

    if ret_sqrt:
        diag = jnp.diag(jnp.sqrt(1.0 / eigval))
    else:
        diag = jnp.diag(1.0 / eigval)

    return eigvec @ diag @ eigvec.T


exact_eigv = [np.exp(-0.5 * i) for i in range(5)]


def exact_eigf_ou(x, i):
    if i == 0:  # Exact eigenfunctions
        return np.ones(x.shape[1])
    if i == 1:
        return 2 * x
    if i == 2:
        return (4 * x**2 - 1) / np.sqrt(2)
    if i == 3:
        return (8 * x**3 - 6 * x) / np.sqrt(6)
    if i == 4:
        return (16 * x**4 - 24 * x**2 + 3) / np.sqrt(24)


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
