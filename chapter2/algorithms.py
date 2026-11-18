import numpy as np
import scipy as sp
from scipy.integrate import odeint

## SINDy


def sindy_ode(X, Y, psi_x, eps=0.0001, iterations=10):
    # PhiX = phi_sindy(X)
    XiT = Y @ sp.linalg.pinv(psi_x)  # least−squares initial guess

    for k in range(iterations):
        s = abs(XiT) < eps  # find coefficients less than eps ...
        XiT[s] = 0  # ... and set them to zero
        for ind in range(X.shape[0]):
            b = ~s[ind, :]  # consider only coefficients greater than eps
            XiT[ind, b] = Y[ind, :] @ sp.linalg.pinv(psi_x[b, :])
    return XiT


# Cost function of SINDy


def cost_sindy(Psi, X_dot, psi_x):
    """Returns the cost of SINDy for a given Psi matrix
    Parameters-
    Psi: Estimated Psi matrix for a given basis
    X_dot: Derivatives matrix
    psi_x: transformed X matrix using basis functions
    """
    return (1 / 2) * (np.linalg.norm(X_dot - Psi @ psi_x, "fro")) ** 2


# Class to generate data from ODE


class generate_ode_data:
    # Initialization of the class
    def __init__(self, system) -> None:
        self.system = system

    # function to solve the system of ODE
    def solve_system(self, yzero, t_span):
        X_sol = odeint(self.system, yzero, t_span)
        return X_sol

    # Get data matrices X and X_dot
    def data_matrices(self, X_sol, t):
        m, d = X_sol.shape
        X_dot = np.zeros((m, d))
        for i in range(m):
            X_dot[i, :] = self.system(X_sol[i, :], t[i])
        return X_sol.T, X_dot.T

    # Save the data
    def save_data(self, X, X_dot):
        return sp.io.savemat("odeData", {"Xdot": X_dot, "X": X})


## PDE-FIND

"""

This part of tbe code is inspired from the original PDE-FIND code, which is available on GitHub at:

S. H. Rudy, S. L. Brunton, J. L. Proctor, and J. N. Kutz. Data-driven discovery of
partial differential equations. Science Advances, 3(4):e1602614, 2017. doi:10.1126/
sciadv.1602614

For a more detailed and optimized code please refer to
Link to original code: https://github.com/snagcliffs/PDE-FIND/blob/master/PDE_FIND.py

Link to original paper: https://www.science.org/doi/10.1126/sciadv.1602614

"""


def finite_difference(u, dx, d):
    """
    computes the dth derivative using data and second order FD method,upto d=3

    Parameters:
    u = data to be differentiated
    dx = spacing of the grid points with default set to uniform

    Output: Derivatives
    """

    n = u.size
    ux = np.zeros(n, dtype=np.complex64)

    if d == 1:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

        ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
        ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
        return ux

    if d == 2:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2

        ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx**2
        ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx**2
        return ux

    if d == 3:
        for i in range(2, n - 2):
            ux[i] = (u[i + 2] / 2 - u[i + 1] + u[i - 1] - u[i - 2] / 2) / dx**3

        ux[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / dx**3
        ux[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / dx**3
        ux[n - 1] = (
            2.5 * u[n - 1]
            - 9 * u[n - 2]
            + 12 * u[n - 3]
            - 7 * u[n - 4]
            + 1.5 * u[n - 5]
        ) / dx**3
        ux[n - 2] = (
            2.5 * u[n - 2]
            - 9 * u[n - 3]
            + 12 * u[n - 4]
            - 7 * u[n - 5]
            + 1.5 * u[n - 6]
        ) / dx**3
        return ux

    if d > 3:
        return finite_difference(finite_difference(u, dx, 3), dx, d - 3)


def build_linear_system(
    u,
    dt,
    dx,
    D=3,
    P=3,
):
    """
    Create a linear system to find the PDE using regression

    Parameters:  u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
            t
    Output:
        ut = column vector of time derivative
        R = theta matrix with ((D+1)*(P+1)) of column (shape same as of ut)
        rhs_des = terms present in rhs
    """

    n, m = u.shape
    m2 = m
    offset_t = 0

    n2 = n
    offset_x = 0

    ut = np.zeros((n2, m2), dtype=np.complex64)

    # gradients using finite difference
    for i in range(n2):
        ut[i, :] = finite_difference(u[i + offset_x, :], dt, 1)

    ut = np.reshape(ut, (n2 * m2, 1), order="F")

    u2 = u[offset_x : n - offset_x, offset_t : m - offset_t]
    Theta = np.zeros((n2 * m2, (D + 1) * (P + 1)), dtype=np.complex64)
    ux = np.zeros((n2, m2), dtype=np.complex64)
    rhs_description = ["" for i in range((D + 1) * (P + 1))]

    for d in range(D + 1):
        if d > 0:
            for i in range(m2):
                ux[:, i] = finite_difference(u[:, i + offset_t], dx, d)
        else:
            ux = np.ones((n2, m2), dtype=np.complex64)

        for p in range(P + 1):
            Theta[:, d * (P + 1) + p] = np.reshape(
                np.multiply(ux, np.power(u2, p)), (n2 * m2), order="F"
            )

            if p == 1:
                rhs_description[d * (P + 1) + p] = (
                    rhs_description[d * (P + 1) + p] + "u"
                )
            elif p > 1:
                rhs_description[d * (P + 1) + p] = (
                    rhs_description[d * (P + 1) + p] + "u^" + str(p)
                )
            if d > 0:
                rhs_description[d * (P + 1) + p] = (
                    rhs_description[d * (P + 1) + p]
                    + "u_{"
                    + "".join(["x" for _ in range(d)])
                    + "}"
                )

    return ut, Theta, rhs_description


def sindy_pde(theta, Ut, lam=0.001, eps=0.04, iterations=10):
    """PDE-FIND algorithm"""
    n, d = theta.shape

    w_best = np.linalg.inv(theta.T @ theta + lam * np.eye(d)) @ theta.T @ Ut

    for k in range(iterations):
        s = abs(w_best) < eps  # find coefficients less than eps ...
        w_best[s] = 0  # ... and set them to zero

    return w_best


def print_pde(w, rhs_description, ut="u_t"):
    w = np.array(w).flatten()
    # for k in range(11):
    #     s = abs(w) < 0.04  # iterative hard thresholding
    #     w[s] = 0
    pde = ut + " = "
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + " + "
            pde = (
                pde
                + "(%05f %+05fi)" % (w[i].real, w[i].imag)
                + rhs_description[i]
                + "\n   "
            )
            first = False
    print(pde)
