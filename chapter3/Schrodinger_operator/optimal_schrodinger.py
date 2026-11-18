import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from jax.example_libraries import optimizers

# ============================================================================================================================================================================================


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


# ============================================================================================================================================================================================


# multivariate normal
def multivariate_gaussian(x, sigma, mu):
    d = x.shape[0]
    xm = x - mu
    return np.exp(-0.5 * (xm.T.dot(xm)) / sigma**2)


# generating the surface
def generate_surface(X, sigma, mu):
    m = X.shape[1]
    x1 = X[0, :]
    x2 = X[1, :]
    x1s, x2s = np.meshgrid(x1, x2)
    pdf = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            pdf[i, j] = multivariate_gaussian(
                np.matrix([[x1s[i, j]], [x2s[i, j]]]), sigma, mu
            )
    return x1s, x2s, pdf


def sortEig(A, evs=5, which="LM"):
    """
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    """
    n = A.shape[0]
    if evs < n:
        d, V = sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1]
    return (d[ind], V[:, ind])


def basis_func_1D(X, sigma, c):
    X_flat = X.flatten()
    squared_distances = (X_flat - c) ** 2
    return jnp.exp(-squared_distances / (2 * sigma**2))


def psi_1D(X, W):
    X_flat = X.flatten()

    def apply_single_basis(w):
        sigma, c = w[0], w[1]
        return basis_func_1D(X_flat, sigma, c)

    TX = jax.vmap(apply_single_basis)(W)
    return jnp.squeeze(TX)


def basis_func_nD(x, sigma, mu):
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    mu = jnp.asarray(mu).reshape(-1, 1)
    sigma = jnp.asarray(sigma).reshape(1, 1)
    diff = x - mu  # (d, m)
    basis_values = 0.5 * jnp.sum((diff / sigma) ** 2, axis=0, keepdims=True)  # (1, m)
    return jnp.exp(-basis_values)  # (1, m)


def psi_nD(X, W):
    X = jnp.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # sigmas = W[:, 0]
    # mus = W[:, 1:]

    def single_basis(w):
        sigma = w[0]
        mu = w[1:]
        return basis_func_nD(X, sigma, mu)

    TX = jax.vmap(single_basis)(W)
    TX = jnp.squeeze(TX, axis=1)
    if TX.shape[1] == 1:
        return jnp.squeeze(TX, axis=1)
    return TX  # (n, m)


def plot_sphere(ax, center, radius, color="b", alpha=0.2, n_steps=30):
    u = jnp.linspace(0, 2 * jnp.pi, n_steps)
    v = jnp.linspace(0, jnp.pi, n_steps)
    x = radius * jnp.outer(jnp.cos(u), jnp.sin(v)) + center[0]
    y = radius * jnp.outer(jnp.sin(u), jnp.sin(v)) + center[1]
    z = radius * jnp.outer(jnp.ones_like(u), jnp.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


# ==============================================================================================================================================


def inverse(x, epsilon=1e-10, ret_sqrt=False):
    """Utility function that returns the inverse of a matrix.

    Parameters-
    x: numpy array with shape [m,m]
        matrix to be inverted

    Returns-
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval_all, eigvec_all = jnp.linalg.eigh(x)
    eigval = eigval_all[eigval_all > epsilon]
    eigvec = eigvec_all[:, eigval_all > epsilon]

    # of the filtered eigenvalues according to the parameter
    if ret_sqrt:
        diag = jnp.diag(jnp.sqrt(1 / eigval))
    else:
        diag = jnp.diag(1 / eigval)

    # Rebuild the square root of the inverse matrix
    eigvec = eigvec.astype(jnp.float32)
    diag = diag.astype(jnp.float32)
    x_inv = jnp.matmul(eigvec, jnp.matmul(diag, eigvec.T))

    return x_inv


def orthogonalize_rows(X, alpha=0.3):
    Q, _ = jnp.linalg.qr(X.T, mode="reduced")
    X_orth = Q.T
    return (1 - alpha) * X + alpha * X_orth


# ============================================================================================================================================


def qedmd(X, psi, c0, c2, evs=5):
    """
    gEDMD for quantum

    params -
    X: data
    psi: basis functions
    c0: potential function

    returns -
    A: Hamiltonian approximation
    d, V: eigenvalues and eigenvectors of A
    """
    PsiX = psi(X)

    ddPsiX = psi.ddiff(X)  # Hessian (n,d,d,m)
    Vx = c0(X)  # potential (1, m)
    dPsiY = np.multiply(Vx, PsiX)

    # for i in range(n):
    #     dPsiY[i, :] +=  c2(X) * np.trace(ddPsiX, axis1=1, axis2=2)

    dPsiY = dPsiY + c2(X) * np.trace(ddPsiX, axis1=1, axis2=2)

    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ dPsiY.T

    A = sp.linalg.pinv(C_0) @ C_1

    d, V = sortEig(A, evs, which="SM")

    return (A, d, V, C_0, C_1)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class OptimalSchrodinger:
    """
    Class to approximate optimal Schrodinger matrix while also optimizing the basis functions using gradient-based algorithms

    Initialization-
    epss: range within which to approximate
    loops: Number of iterations of the algorithms
    X, Y: Data matrices
    C: centers of the Gaussian basis functions
    """

    def __init__(self, epss, loops, psi_params, c0, c2) -> None:
        self.epss = epss
        self.loops = loops
        self.psi_params = psi_params
        self.c0 = c0
        self.c2 = c2

    def diff(self, x, W):
        return jax.vmap(self._diff, in_axes=(1, None), out_axes=2)(x, W)

    def ddiff(self, X, W):
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_samples = X.T  # (m, d)

        hess = jax.vmap(self._ddiff, in_axes=(0, None))(X_samples, W)  # (m, n, d, d)
        return jnp.transpose(hess, (1, 2, 3, 0))  # (n, d, d, m)

    # def ddiff(self, x, W):
    #     return jax.vmap(self._ddiff, in_axes=(0, None), out_axes=3)(x.T, W)

    def _diff(self, x, W):
        return jax.jacobian(self.psi_params, argnums=0)(x, W)

    def _ddiff(self, x, W):
        return jax.hessian(self.psi_params, argnums=0)(x, W)

    def param_qedmd(self, X, W, evs=10, epsilon=1e-7):
        """
        gEDMD for quantum

        params -
        X: data
        psi: basis functions

        returns -
        A: Hamiltonian approximation
        d, V: eigenvalues and eigenvectors of A
        """
        PsiX = self.psi_params(X, W)

        ddPsiX = self.ddiff(X, W)  # Hessian (n,d,d,m)
        Vx = self.c0(X)  # potential (1, m)
        dPsiY = jnp.multiply(Vx, PsiX)  #

        dPsiY = dPsiY + self.c2(X) * jnp.trace(ddPsiX, axis1=1, axis2=2)

        C_0 = PsiX @ PsiX.T + epsilon * jnp.identity(PsiX.shape[0])
        C_1 = PsiX @ dPsiY.T

        S = jnp.linalg.pinv(C_0) @ C_1

        d, V = sortEig(np.array(S), evs, which="SM")

        recons_error = jnp.linalg.norm(dPsiY - S @ PsiX, "fro")

        return (S, d, V, recons_error)

    def cost_qedmd(self, S, W, X):
        """
        Reconstruction error
        Parameters-
        S: Schrodinger matrix
        W: basis parameters

        Returns-
        Reconstruction error in Frobenius norm
        """

        PsiX = self.psi_params(X, W)

        ddPsiX = self.ddiff(X, W)  # Hessian (n,d,d,m)
        Vx = self.c0(X)  # potential (1, m)
        dPsiY = jnp.multiply(Vx, PsiX)

        dPsiY = dPsiY + self.c2(X) * jnp.trace(ddPsiX, axis1=1, axis2=2)

        # subtract the mean
        # psi_x = psi_x - jnp.mean(psi_x, axis=1, keepdims=True)
        # psi_y = psi_y - jnp.mean(psi_y, axis=1, keepdims=True)

        z = (1 / 2) * (
            jnp.linalg.norm(dPsiY - S @ PsiX, "fro")  # TODO: see this transpose here!
        ) ** 2  # + lam * (jnp.linalg.cond(psi_x)
        return jnp.array(z, dtype="float64")

    def cost_var(self, W, X, epsilon=1e-7):
        """
        Calculates the VAMP-2 score with respect to the new basis functions.

        Parameters-
        W: concatenated \Psi_x and \Psi_y matrices in shape [m, 2n],  m - # of data points, n - # of basis functions

        Returns-
        VAMP-2 score
        """
        PsiX = self.psi_params(X, W)
        # n = PsiX.shape[0]
        ddPsiX = self.ddiff(X, W)  # Hessian (n,d,d,m)
        Vx = self.c0(X)  # potential (1, m)
        dPsiY = jnp.multiply(
            Vx, PsiX
        )  # potential multiplied basis (one of the terms in Hamiltonian operator)

        dPsiY = dPsiY + self.c2(X) * jnp.trace(ddPsiX, axis1=1, axis2=2)

        C_0 = PsiX @ PsiX.T + epsilon * jnp.identity(dPsiY.shape[0])
        C_1 = PsiX @ dPsiY.T

        C_0_inv = jnp.linalg.pinv(C_0)

        var_matrix = jnp.matmul(C_0_inv, C_1)

        # vamp_score = jnp.linalg.eigh(vamp_matrix)[0]  # sortEig(np.array(vamp_matrix), evs=3)[0]

        var_score = jnp.linalg.trace(var_matrix)  # TODO: check the transpose here

        return jnp.square(var_score)  # jnp.sum(jnp.real(vamp_score[:5]))

    def gradient_qedmd(self, S, W, X):
        """
        Gradient of the reconstruction error
        Parameters-
        K: Koopman matrix
        W: sigmas

        Returns:
        Gradient of the reconstruction error in shape [n,n]
        """
        grad_calc_k = jax.grad(self.cost_qedmd)
        return grad_calc_k(S, W, X)

    def gradient_params_var(self, W, X, epsilon):
        grad_calc_params_vamp = jax.grad(self.cost_var)
        return grad_calc_params_vamp(W, X, epsilon)

    def gradient_params_qedmd(self, S, W, X):
        grad_calc_params_edmd = jax.grad(self.cost_qedmd, 1)
        return grad_calc_params_edmd(S, W, X)

    def optimizer(
        self,
        optim: str = "adam",
        hs: float = 0.01,
        hw: float = 0.01,
        mass: float = 0.1,
        decay_rate: float = None,
        decay_steps: int = None,
    ):
        """
        Optimizer to be used for training

        Params -
        optim: optimizer to be used for training, available options are 'adam', 'nesterov', 'gd'

        returns -
        update functions for the optimizer in JAX
        """
        # lr_schedule = hk

        # if decay_rate is not None and decay_steps is not None:
        #     lr_schedule = optimizers.exponential_decay(
        #         init_value=hk,  # Starting learning rate
        #         transition_steps=decay_steps,  # How often to decay
        #         decay_rate=decay_rate,  # How much to decay
        #         staircase=False,  # Smooth decay if False, stepwise if True
        #     )

        if optim == "gd":
            # Initialize the optimizer with the initial parameters
            opt_init_S, opt_update_S, get_S = optimizers.sgd(hs)
            opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        elif optim == "sgd":
            # Initialize the optimizer with the initial parameters
            opt_init_S, opt_update_S, get_S = optimizers.sgd(hs)
            opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        elif optim == "nesterov":
            # Initialize the optimizer with the initial parameters
            opt_init_S, opt_update_S, get_S = optimizers.nesterov(hs, mass)
            opt_init_w, opt_update_w, get_w = optimizers.nesterov(hw, mass)

        elif optim == "adam":
            # Initialize the optimizer with the initial parameters
            opt_init_S, opt_update_S, get_S = optimizers.adam(hs)
            opt_init_w, opt_update_w, get_w = optimizers.adam(hw)

        else:
            raise ValueError(
                f"Optimizer {optim} is not available; available options are 'adam', 'nesterov', 'gd', 'rmsprop'."
            )
        return opt_init_S, opt_update_S, get_S, opt_init_w, opt_update_w, get_w

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def jax_gd(self, S_init, w_init, X, optim="adam", hs=1e-3, hw=1e-3, epsilon=1e-7):
        """
        returns the minimizer S* of the qEDMD cost function
         using gradient descent algorithm.
        """
        # Initialize the optimizer with the initial parameters
        opt_init_S, opt_update_S, get_S, opt_init_w, opt_update_w, get_w = (
            self.optimizer(optim, hs, hw)
        )

        opt_state_S = opt_init_S(S_init)
        opt_state_w = opt_init_w(w_init)

        def update_S(S, w, opt_state):
            """Perform one update step of the optimizer."""
            grads_S = self.gradient_qedmd(S, w, X)
            opt_state = opt_update_S(0, grads_S, opt_state)
            new_S = get_S(opt_state)
            return new_S, opt_state

        def update_w(w, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.gradient_params_var(w, X, epsilon)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        S_gd = S_init
        w_gd = w_init
        f_vals_gd_S = []
        f_vals_gd_w = []

        loss_S = self.cost_qedmd(S_gd, w_gd, X)
        f_vals_gd_S.append(loss_S)
        loss_w = self.cost_var(w_gd, X)
        f_vals_gd_w.append(loss_w)

        l = 1

        while (
            jnp.linalg.norm(self.gradient_qedmd(S_gd, w_gd, X)) > self.epss
            and l < self.loops
        ):
            S_gd, opt_state_S = update_S(S_gd, w_gd, opt_state_S)
            loss_S = self.cost_qedmd(S_gd, w_gd, X)
            f_vals_gd_S.append(loss_S)

            w_gd, opt_state_w = update_w(w_gd, opt_state_w)
            loss_w = self.cost_var(w_gd, X, epsilon)
            f_vals_gd_w.append(loss_w)

            l = l + 1

        return S_gd, w_gd, f_vals_gd_S, f_vals_gd_w


# def cartesian_to_spherical(X):
#     x, y, z = X
#     r = jnp.sqrt(x**2 + y**2 + z**2) + 1e-8
#     theta = jnp.arccos(z / r)
#     phi = jnp.arctan2(y, x)
#     return r, theta, phi


# def basis_func_hydrogen(X, alpha, l, m):

#     r, theta, phi = cartesian_to_spherical(X)

#     radial = jnp.exp(-alpha * r**2)

#     Ylm = sph_harm(int(m), int(l), phi, theta)
#     Ylm = jnp.real(Ylm)

#     return radial * Ylm


# def psi_params(X, W):

#     X = jnp.asarray(X)

#     if X.ndim == 1:
#         X = X.reshape(-1, 1)

#     def single_basis(w):
#         alpha = w[0]
#         l = int(w[1])
#         m = int(w[2])
#         return basis_func_hydrogen(X, alpha, l, m)

#     TX = jax.vmap(single_basis)(W)

#     if TX.shape[1] == 1:
#         return jnp.squeeze(TX, axis=1)

#     return TX
