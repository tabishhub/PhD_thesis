import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from scipy.linalg import fractional_matrix_power
from utils import inverse

##======================================================================================================================
## RaNNDy in flax linen


class NeuralNetwork(nn.Module):
    hidden_sizes: list
    activation: str = "tanh"
    init_weights_dist: str = "lecun_normal"
    init_weights_std: float = 1.0
    init_bias_dist: str = "zeros"
    init_bias_std: float = 1.0
    batch_norm: bool = False
    softmax_output: bool = False

    def activate(self, x):
        # if ws is not None:
        #     scale = ws
        # else:
        #     scale = 1.0
        acts = {
            "relu": nn.relu,
            "sigmoid": nn.sigmoid,
            "tanh": nn.tanh,
            "elu": nn.elu,
            "softmax": lambda y: nn.softmax(y, axis=-1),
            "gelu": nn.gelu,
            "cos": jnp.cos,
            "sin": jnp.sin,
        }
        if self.activation not in acts:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return acts[self.activation](x)

    def get_kernel_init(self, ww=None):
        if ww is not None:
            scale = ww
        else:
            scale = self.init_weights_std
        inits = {
            "normal": nn.initializers.normal(scale),
            "uniform": nn.initializers.uniform(scale),
            "glorot_uniform": nn.initializers.glorot_uniform(),
            "glorot_normal": nn.initializers.glorot_normal(),
            "he_normal": nn.initializers.he_normal(),
            "he_uniform": nn.initializers.he_uniform(),
            "lecun_uniform": nn.initializers.lecun_uniform(),
            "lecun_normal": nn.initializers.lecun_normal(),
        }
        if self.init_weights_dist not in inits:
            raise ValueError(f"Unsupported weight init: {self.init_weights_dist}")
        return inits[self.init_weights_dist]

    def get_bias_init(self, wb=None):
        if wb is not None:
            scale = wb
        else:
            scale = self.init_bias_std
        inits = {
            "normal": nn.initializers.normal(scale),
            "uniform": nn.initializers.uniform(scale),
            "zeros": nn.initializers.zeros,
            "ones": nn.initializers.ones,
            "constant": nn.initializers.constant(self.init_bias_std),
        }
        if self.init_bias_dist not in inits:
            raise ValueError(f"Unsupported bias init: {self.init_bias_dist}")
        return inits[self.init_bias_dist]

    @nn.compact
    def __call__(self, x, w, training=True):
        ww, wb = w
        # input: (features, batch)
        x = x.T  # (batch, features)

        for i, hidden_size in enumerate(self.hidden_sizes[:-1]):
            x = nn.Dense(
                hidden_size,
                kernel_init=self.get_kernel_init(ww=ww),
                bias_init=self.get_bias_init(wb=wb),
                name=f"dense_{i}",
            )(x)

            if self.batch_norm:
                x = nn.BatchNorm(
                    use_running_average=not training,
                    momentum=0.9,
                    epsilon=1e-5,
                    axis=-1,
                    name=f"bn_{i}",
                )(x)

            x = self.activate(x)

        x = nn.Dense(
            self.hidden_sizes[-1],
            kernel_init=self.get_kernel_init(ww=ww),
            bias_init=self.get_bias_init(wb=wb),
            name=f"dense_{len(self.hidden_sizes) - 1}",
        )(x)

        if self.softmax_output:
            x = nn.softmax(x, axis=-1)
        else:
            x = self.activate(x)

        return x.T  # (features, batch)


class RaNNDy(nn.Module):
    X: jnp.ndarray
    operator: str
    hidden_sizes: list
    activation: str = "tanh"
    init_weights_dist: str = "lecun_normal"
    init_weights_std: float = 1.0
    init_bias_dist: str = "normal"
    init_bias_std: float = 1.0
    batch_norm: bool = False
    random_state: int = 0
    key = jax.random.PRNGKey(seed=random_state)

    def setup(self):

        self.model = NeuralNetwork(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            init_weights_dist=self.init_weights_dist,
            init_weights_std=self.init_weights_std,
            init_bias_dist=self.init_bias_dist,
            init_bias_std=self.init_bias_std,
            batch_norm=self.batch_norm,
            softmax_output=False,
        )

    def __call__(self, X, w, training=True):
        return self.model(X, w, training=training)

    def build_params(self, w, key=None):
        """
        Initializes new parameters for the model using a given random key.
        #"""
        # for i in range(len(self.hidden_sizes)):
        #     params["params"]["model"][f"dense_{i}"]["kernel"] = w[
        #         0
        #     ] * jax.random.normal(
        #         key, shape=params["params"]["model"][f"dense_{i}"]["kernel"].shape
        #     )
        #     params["params"]["model"][f"dense_{i}"]["bias"] = w[1] * jax.random.uniform(
        #         key, shape=params["params"]["model"][f"dense_{i}"]["bias"].shape
        #    )
        if key is None:
            key = self.key
        params = jax.jit(lambda w, key: self.init(key, self.X, w=w))(w, key)
        return params

    def compute_jacobian_nn(self, params, x, w):
        def model_apply_fn(x_, w):
            y = self.apply(params, x_[..., None], w, training=False)
            return jnp.squeeze(y, axis=1)

        return jax.jacobian(model_apply_fn, argnums=0)(x, w)

    def jacobian_all_nn(self, params, data, w):
        """Returns the Jacobian matrix evaluated for all the data points"""
        return jax.vmap(self.compute_jacobian_nn, in_axes=(None, 1, None), out_axes=2)(
            params, data, w
        )

    def compute_hessian_nn(self, params, x, w):
        def model_apply_fn(x_, w):
            # y = psi_nn.apply(params, x.reshape(x.shape[0], 1), training=False)
            # jnp.squeeze(y)   this could also be done in the model_apply_fn
            y = self.apply(params, x_[..., None], w, training=False)
            return jnp.squeeze(y, axis=1)

        return jax.hessian(model_apply_fn, argnums=0)(x, w)

    def hessian_all_nn(self, params, data, w):
        """Returns the Hessian matrix evaluated for all the data points"""
        return jax.vmap(self.compute_hessian_nn, in_axes=(None, 1, None), out_axes=3)(
            params, data, w
        )

    def koopman_eig_decomp(
        self,
        w: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        # params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        output: str = "spectrum",
    ):
        params = self.build_params(w=w, key=self.key)

        N_x = self.apply(params, X, w, training=False)
        N_y = self.apply(params, Y, w, training=False)

        ## subtract the mean
        # psi_x = psi_x - jnp.mean(psi_x, axis=1, keepdims=True)
        # psi_y = psi_y - jnp.mean(psi_y, axis=1, keepdims=True)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T  # (N_o, N_o)

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])

        A = jnp.linalg.pinv(C_xx_reg) @ C_xy

        if output == "spectrum":
            eigvals, eigvecs = jnp.linalg.eig(A)

            idx = jnp.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            E = eigvecs[:, :n].T
            ECxE_T = E @ C_xx @ E.T

            ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)

            E_whitened = ECxE_T_inv_sqrt @ E

            return eigvals[:n], E_whitened

        elif output == "matrix":
            return A

    def koopman_generator_eig_decomp(
        self,
        w: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray,
        # params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        output: str = "spectrum",
    ):
        params = self.build_params(w=w, key=self.key)

        Y1 = Y  # kwargs.get("Y1")
        Z1 = Z  # kwargs.get("Z1")

        PsiX = self.apply(params, X, w, training=False)
        jacobX = self.jacobian_all_nn(params, X, w)

        dPsiY = jnp.einsum("ijk,jk->ik", jacobX, Y1)

        N_o = PsiX.shape[0]
        hessX = self.hessian_all_nn(params, X, w)
        S = jnp.einsum("ijk,ljk->ilk", Z1, Z1)
        # dPsiY = self.compute_dPsiY(dPsiY, hessX, S, n)
        for i in range(N_o):
            dPsiY = dPsiY.at[i, :].add(
                0.5 * jnp.sum(hessX[i, :, :, :] * S, axis=(0, 1))
            )

        N_x_dot = dPsiY
        N_x = PsiX

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_x_dot.T

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])

        A = jnp.linalg.pinv(C_xx_reg) @ C_xy

        if output == "spectrum":
            eigvals, eigvecs = jnp.linalg.eig(A)

            idx = jnp.argsort(eigvals)[::1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            E = eigvecs[:, :n].T

            # ## whitening transformation
            ECxE_T = E @ C_xx @ E.T

            ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)

            E_whitened = ECxE_T_inv_sqrt @ E

            return eigvals[:n], E_whitened

        elif output == "matrix":
            return A

    def schrodinger_eig_decomp(
        self,
        w: jnp.ndarray,
        X: jnp.ndarray,
        # params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        output: str = "spectrum",
        **kwargs,
    ):
        params = self.build_params(w=w, key=self.key)

        c0 = kwargs.get("c0")
        c2 = kwargs.get("c2")
        PsiX = self.apply(params, X, w, training=False)
        Hessian = self.hessian_all_nn(params, X, w)
        Vx = c0(X)
        dPsiY = Vx * PsiX

        lap = jnp.trace(Hessian, axis1=1, axis2=2)  # shape (m,)

        dPsiY = dPsiY + c2(X) * lap
        N_x_dot = dPsiY
        N_x = PsiX

        C_xx = N_x @ N_x.T
        C_xy = N_x @ N_x_dot.T

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])

        A = jnp.linalg.pinv(C_xx_reg) @ C_xy

        if output == "spectrum":
            eigvals, eigvecs = jnp.linalg.eig(A)

            idx = jnp.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            E = eigvecs[:, :n].T
            ECxE_T = E @ C_xx @ E.T

            ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)

            E_whitened = ECxE_T_inv_sqrt @ E

            return eigvals[:n], E_whitened

        elif output == "matrix":
            return A

    def forward_backward_eig_decomp(
        self,
        w: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        # params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        output: str = "spectrum",
    ):
        params = self.build_params(w=w, key=self.key)

        N_x = self.apply(params, X, w, training=False)
        N_y = self.apply(params, Y, w, training=False)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T
        C_yy = N_y @ N_y.T

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        C_yy_reg = C_yy + epsilon * jnp.identity(C_yy.shape[0])

        A = jnp.linalg.pinv(C_xx_reg) @ C_xy @ jnp.linalg.pinv(C_yy_reg) @ C_xy.T

        if output == "spectrum":
            eigvals, eigvecs = jnp.linalg.eig(A)

            idx = jnp.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            E = eigvecs[:, :n].T
            ECxE_T = E @ C_xx @ E.T

            ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)

            E_whitened = ECxE_T_inv_sqrt @ E

            return eigvals[:n], E_whitened

        elif output == "matrix":
            return A

    def operator_eig_decomp(
        self,
        w: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray = None,
        Z: jnp.ndarray = None,
        # params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        output: str = "spectrum",
        **kwargs,
    ):
        """
        Calculates the eigenvalues and eigenvectors of the approximated operator matrix.
        Params:
        n - number of eigenvalues and eigenvectors to return
        epsilon - regularization parameter
        kwargs - extra arguments for different operators

        Returns:
        eigvals - eigenvalues of the operator matrix
        eigvecs - eigenvectors of the operator matrix
        """

        if self.operator == "koopman":
            result = self.koopman_eig_decomp(
                w=w,
                X=X,
                Y=Y,
                # params=params,
                n=n,
                epsilon=epsilon,
                output=output,
            )
            return result

        elif self.operator == "koopman_generator":
            result = self.koopman_generator_eig_decomp(
                w=w,
                X=X,
                Y=Y,
                Z=Z,
                # params=params,
                n=n,
                epsilon=epsilon,
                output=output,
            )
            return result

        elif self.operator == "schrodinger":
            result = self.schrodinger_eig_decomp(
                w=w,
                X=X,
                # params=params,
                n=n,
                epsilon=epsilon,
                output=output,
                **kwargs,
            )
            return result

        elif self.operator == "forward_backward":
            result = self.forward_backward_eig_decomp(
                w=w,
                X=X,
                Y=Y,
                # params=params,
                n=n,
                epsilon=epsilon,
                output=output,
            )
            return result

        else:
            raise ValueError(
                f"Operator {self.operator} is not available. Please select one of {'koopman', 'koopman_generator', 'schrodinger', 'forward_backward'}."
            )

    def lr_scheduler(self, schedule="constant", decay_steps=1, lr=1e-3, **kwargs):
        if schedule == "constant":
            return lr

        elif schedule == "exponential":
            decay_rate = kwargs.get("decay_rate", 0.99)
            return optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )

        elif schedule == "cosine":
            return optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=decay_steps,
            )

        elif schedule == "linear":
            end_lr = kwargs.get("end_lr", 0.0)
            return optax.linear_schedule(
                init_value=lr,
                end_value=end_lr,
                transition_steps=decay_steps,
            )

        else:
            raise ValueError(f"Unknown lr schedule {schedule}")

    def optimizer(self, optim="adam", lr=1e-3, lr_schedule=None):
        lr_or_schedule = lr_schedule if lr_schedule is not None else lr

        if optim == "adam":
            base = optax.adam(lr_or_schedule)
        elif optim == "gd":
            base = optax.sgd(lr_or_schedule)
        elif optim == "nesterov":
            base = optax.sgd(lr_or_schedule, momentum=0.1, nesterov=True)
        elif optim == "rmsprop":
            base = optax.rmsprop(lr_or_schedule)
        else:
            raise ValueError(f"Unknown optimizer {optim}")

        return base

    def optimize_activation(
        self,
        w_init,
        X,
        Y=None,
        Z=None,
        # params=None,
        n=5,
        epsilon=1e-5,
        optim="adam",
        lr=1e-2,
        lr_schedule="constant",
        decay_steps=1,
        epochs=100,
        get_epoch=10,
        tol=1e-3,
        **kwargs,
    ):
        @jax.jit
        def loss_fn(w):
            A = self.operator_eig_decomp(
                w=w,
                X=X,
                Y=Y,
                Z=Z,
                # params=params,
                n=n,
                epsilon=epsilon,
                output="matrix",
                **kwargs,
            )
            if self.operator in ["schrodinger", "koopman_generator"]:
                return jnp.linalg.trace(A)  # jnp.sum(jnp.real(eigvals))
            else:
                return -jnp.linalg.trace(A)  # -jnp.sum(jnp.real(eigvals))

        schedule = self.lr_scheduler(
            schedule=lr_schedule,
            decay_steps=decay_steps,
            lr=lr,
            # epochs=epochs,
            **kwargs,
        )

        tx = self.optimizer(
            optim=optim,
            lr=lr,
            lr_schedule=schedule,
        )

        opt_state = tx.init(w_init)
        w = w_init

        def train_step(w, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(w)
            updates, opt_state = tx.update(grads, opt_state, w)
            w = optax.apply_updates(w, updates)
            return w, opt_state, loss

        loss_vals = []
        w_vals = []
        times_ranndy = []
        t = 0
        epoch = 0
        # for epoch in range(epochs):
        while epoch < epochs and (
            len(loss_vals) == 0 or abs(loss_vals[-1] - loss_fn(w)) > tol
        ):
            rtime1 = time.time()
            w, opt_state, loss = train_step(w, opt_state)
            rtime2 = time.time()
            one_step_ranndy_time = rtime2 - rtime1
            w_vals.append(w)
            t += one_step_ranndy_time

            if epoch % get_epoch == 0:
                loss_vals.append(loss)
                times_ranndy.append(t)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                print(
                    f"Time for one step of RaNNDy optimization: {one_step_ranndy_time:.4f} seconds"
                )
            epoch += 1

        return w, loss_vals, w_vals, times_ranndy

    def get_model_params(self, w):
        params = self.init(self.key, jnp.ones_like(self.X), w=w)
        return params

    def eigenfunctions(self, w, eigvecs, domain: jnp.ndarray = None):
        """
        Computes the eigenfunctions of the operator using the eigenvectors and input data.
        Params:
        eigvecs - Eigenvectors of the operator
        domain - Domain for the eigenfunctions (optional)

        Returns:
        eigfuncs - Eigenfunctions of the operator
        """
        params = self.init(self.key, jnp.ones_like(self.X), w=w)
        if domain is None:
            H_aug = self.apply(params, self.X, w, training=False)
        else:
            H_aug = self.apply(params, domain, w, training=False)
        eigfuncs = eigvecs @ H_aug

        return eigfuncs

    def ranks_cov_matrices(self, X, w, Y=None, epsilon=1e-5, **kwargs):
        params = self.init(self.key, jnp.ones_like(self.X), w=w)

        if self.operator == "schrodinger":
            c0 = kwargs.get("c0")
            c2 = kwargs.get("c2")
            PsiX = self.apply(params, X, w, training=False)
            Hessian = self.hessian_all_nn(params, X, w)
            Vx = c0(X)
            dPsiY = (
                Vx * PsiX
            )  # potential multiplied basis (one of the terms in Hamiltonian operator)

            # use jnp.trace for the Hessian trace
            lap = jnp.trace(Hessian, axis1=1, axis2=2)  # shape (m,)

            dPsiY = dPsiY + c2(X) * lap
            N_x_dot = dPsiY
            N_x = PsiX

            C_xx = N_x @ N_x.T  # (N_o, N_o)
            C_xy = N_x @ N_x_dot.T  # (N_o, N_o)

        else:
            # Random data matrices
            N_x = self.apply(params, X, w, training=False)
            N_y = self.apply(params, Y, w, training=False)

            C_xx = N_x @ N_x.T  # (N_o, N_o)
            C_xy = N_x @ N_y.T  # (N_o, N_o)
            C_yy = N_y @ N_y.T

        # Regularization
        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        C_yy_reg = C_yy + epsilon * jnp.identity(C_yy.shape[0])

        print(
            "Rank of C_00 and C_00_reg matrix: ",
            jnp.linalg.matrix_rank(C_xx),
            jnp.linalg.matrix_rank(C_xx_reg),
        )
        print("Rank of C_01 matrix: ", jnp.linalg.matrix_rank(C_xy))
        print(
            "Rank of C_11 and C_11_reg matrix: ",
            jnp.linalg.matrix_rank(C_yy),
            jnp.linalg.matrix_rank(C_yy_reg),
        )


class VAMPnets(nn.Module):
    X: jnp.ndarray
    operator: str
    hidden_sizes: list
    activation: str = "tanh"
    init_weights_dist: str = "lecun_normal"
    init_weights_std: float = 1.0
    init_bias_dist: str = "normal"
    init_bias_std: float = 1.0
    softmax_output: bool = True
    random_state: int = 0
    key = jax.random.PRNGKey(seed=random_state)

    def setup(self):
        self.model = NeuralNetwork(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            init_weights_dist=self.init_weights_dist,
            init_weights_std=self.init_weights_std,
            init_bias_dist=self.init_bias_dist,
            init_bias_std=self.init_bias_std,
            softmax_output=self.softmax_output,
        )

    def __call__(self, X, w=[None, None], training=True):
        return self.model(X, w, training=training)

    def lr_scheduler(self, schedule="constant", decay_steps=1, lr=1e-3, **kwargs):
        if schedule == "constant":
            return lr

        elif schedule == "exponential":
            decay_rate = kwargs.get("decay_rate", 0.99)
            return optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )

        elif schedule == "cosine":
            return optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=decay_steps,
            )

        elif schedule == "linear":
            end_lr = kwargs.get("end_lr", 0.0)
            return optax.linear_schedule(
                init_value=lr,
                end_value=end_lr,
                transition_steps=decay_steps,
            )

        else:
            raise ValueError(f"Unknown lr schedule {schedule}")

    def optimizer(self, optim="adam", lr=1e-3, lr_schedule=None):
        lr_or_schedule = lr_schedule if lr_schedule is not None else lr

        if optim == "adam":
            base = optax.adam(lr_or_schedule)
        elif optim == "gd":
            base = optax.sgd(lr_or_schedule)
        elif optim == "nesterov":
            base = optax.sgd(lr_or_schedule, momentum=0.1, nesterov=True)
        elif optim == "rmsprop":
            base = optax.rmsprop(lr_or_schedule)
        else:
            raise ValueError(f"Unknown optimizer {optim}")

        return base

    def cost_vamp2(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        epsilon: float = 1e-5,
    ):
        """
        Calculates the VAMP-2 score with respect to the new basis functions.

        Parameters:
        params : jnp.ndarray
            Parameters of the model.
        X : jnp.ndarray
            Input data
        Y : jnp.ndarray
            Input data
        Returns:
        VAMP-2 score
        """

        psi_x = self.apply(params, X)
        psi_y = self.apply(params, Y)
        # psi_x = psi_x - jnp.mean(psi_x, axis=1, keepdims=True)
        # psi_y = psi_y - jnp.mean(psi_y, axis=1, keepdims=True)

        data_points = psi_x.shape[1]

        cov_01 = 1 / (data_points - 1) * psi_x @ psi_y.T
        cov_00 = 1 / (data_points - 1) * psi_x @ psi_x.T + epsilon * jnp.eye(
            psi_x.shape[0]
        )
        cov_11 = 1 / (data_points - 1) * psi_y @ psi_y.T + epsilon * jnp.eye(
            psi_x.shape[0]
        )

        cov_00_inv = inverse(cov_00, ret_sqrt=True)
        cov_11_inv = inverse(cov_11, ret_sqrt=True)

        vamp_matrix = jnp.matmul(jnp.matmul(cov_00_inv, cov_01), cov_11_inv)

        vamp_score = jnp.linalg.norm(vamp_matrix, "fro")

        return -jnp.square(vamp_score)

    def training(
        self,
        params_init,
        X,
        Y,
        n=5,
        epochs=100,
        optim="adam",
        lr=1e-3,
        lr_schedule="constant",
        decay_steps=1,
        epsilon=1e-5,
        get_epoch=10,
        **kwargs,
    ):
        schedule = self.lr_scheduler(
            schedule=lr_schedule,
            decay_steps=decay_steps,
            lr=lr,
            # epochs=epochs,
            **kwargs,
        )

        tx = self.optimizer(
            optim=optim,
            lr=lr,
            lr_schedule=schedule,
        )
        opt_state = tx.init(params_init)

        @jax.jit
        def loss_fn(params):
            return self.cost_vamp2(params, X, Y, epsilon)

        def train_step(params, opt_state, X, Y):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        params = params_init
        loss_vals = []
        times_vampnet = []
        t = 0
        for epoch in range(epochs):
            rtime1 = time.time()
            params, opt_state, loss = train_step(params, opt_state, X, Y)
            A, _, _ = self.spectrum_approximation(params=params, X=X, Y=Y, n=n)
            rtime2 = time.time()
            one_step_vampnet_time = rtime2 - rtime1
            t += one_step_vampnet_time
            # loss_vals.append(jnp.trace(A))
            if epoch % get_epoch == 0:
                loss_vals.append(-jnp.trace(A))
                times_vampnet.append(t)
                print(
                    f"Time for one step of VAMPnet optimization: {one_step_vampnet_time:.4f} seconds"
                )
                print(f"Epoch {epoch}, Loss: {loss_vals[-1]:.4f}")

        return params, loss_vals, times_vampnet

    def spectrum_approximation(
        self,
        params: dict,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        n: int = 5,
        epsilon: float = 1e-5,
        operator: str = "forward_backward",
    ):

        # Random data matrices
        N_x = self.apply(params, X)
        N_y = self.apply(params, Y)

        ## subtract the mean
        # N_x = N_x - jnp.mean(N_x, axis=1, keepdims=True)
        # N_y = N_y - jnp.mean(N_y, axis=1, keepdims=True)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T  # (N_o, N_o)
        C_yy = N_y @ N_y.T

        C_yy_reg = C_yy + epsilon * jnp.identity(C_yy.shape[0])

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])

        if operator == "forward_backward":
            A = jnp.linalg.pinv(C_xx_reg) @ C_xy @ jnp.linalg.pinv(C_yy_reg) @ C_xy.T

        else:
            A = jnp.linalg.pinv(C_xx_reg) @ C_xy  # Shape: (N_o, N_o)

        eigvals, eigvecs = jnp.linalg.eig(A)

        idx = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return A, eigvals[:n], eigvecs[:, :n]

    def eigenfunctions(
        self, params: dict, eigvecs: jnp.ndarray, domain: jnp.ndarray = None
    ):
        """
        Computes the eigenfunctions of the operator using the eigenvectors and input data.
        Params:
        eigvecs - Eigenvectors of the operator
        domain - Domain for the eigenfunctions
        params - Parameters of the model

        Returns:
        eigfuncs - Eigenfunctions of the operator
        """

        if domain is None:
            H_aug = self.apply(params, self.X)  # , training=False, mutable=False)
        else:
            H_aug = self.apply(params, domain)  # , training=False, mutable=False)
        eigfuncs = eigvecs @ H_aug

        return eigfuncs
