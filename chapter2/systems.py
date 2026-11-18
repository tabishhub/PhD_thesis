import numpy as np


def drift_ou(x, alpha=1.0):
    return -(alpha * x)


def drift_triple_well_2d(x):
    return np.array(
        [
            6 * x[0] * np.exp(-(x[0] ** 2) - (x[1] - 1 / 3) ** 2)
            - 6 * x[0] * np.exp(-(x[0] ** 2) - (x[1] - 5 / 3) ** 2)
            - 10 * (x[0] - 1) * np.exp(-((x[0] - 1) ** 2) - (x[1] ** 2))
            - 10 * (x[0] + 1) * np.exp(-((x[0] + 1) ** 2) - (x[1] ** 2))
            - (4 / 5) * x[0] ** 3,
            6 * (x[1] - 1 / 3) * np.exp(-(x[0] ** 2) - (x[1] - 1 / 3) ** 2)
            - 6 * (x[1] - 5 / 3) * np.exp(-(x[0] ** 2) - (x[1] - 5 / 3) ** 2)
            - 10 * x[1] * np.exp(-((x[0] - 1) ** 2) - (x[1] ** 2))
            - 10 * x[1] * np.exp(-((x[0] + 1) ** 2) - (x[1] ** 2))
            - (4 / 5) * (x[1] - 1 / 3) ** 3,
        ]
    )


def drift_double_well_2d(x):
    return np.array([-4 * x[0] ** 3 + 4 * x[0], -2 * x[1]])


def drift_himmelblau(x):
    return np.array(
        [
            -4 * x[0] * (x[0] ** 2 + x[1] - 11) - 2 * (x[0] + x[1] ** 2 - 7),
            -2 * (x[0] ** 2 + x[1] - 11) - 4 * x[1] * (x[0] + x[1] ** 2 - 7),
        ]
    )


def drift_triple_well_1d(x):
    return -(6 * x**5 - 20 * x**3 + 12 * x + 1)


def drift_double_well_1d(x):
    return -(8 * x**3 - 10 * x)


def drift_lemon_slice(x, n=5):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = np.arctan2(x[1], x[0])
    return np.array(
        [
            -(n * np.sin(n * theta) * x[1] / r**2 + 20 * (r - 1) * x[0] / r),
            -(-n * np.sin(n * theta) * x[0] / r**2 + 20 * (r - 1) * x[1] / r),
        ]
    )


def drift_quadruple_well_2d(x):
    return np.array(
        [
            4 * x[0] ** 3 - (3 / 16) * x[0] ** 2 - 4 * x[0] + 3 / 16,
            4 * x[1] ** 3 - (3 / 8) * x[1] ** 2 - 4 * x[1] + 3 / 8,
        ]
    )


DRIFT_MAP = {
    "ou": drift_ou,
    "triple2d": drift_triple_well_2d,
    "double2d": drift_double_well_2d,
    "himmelblau": drift_himmelblau,
    "triple1d": drift_triple_well_1d,
    "double1d": drift_double_well_1d,
    "lemonslice": drift_lemon_slice,
    "quadruple2d": drift_quadruple_well_2d,
}


class EulerMaruyama:
    def __init__(self, x0: np.ndarray, h: float, iters: int = 500):
        """
        x0 : array of shape (d, m)
        h  : time-step
        iters : Euler-Maruyama steps
        """
        self.x0 = x0
        self.h = h
        self.iters = iters
        self.d = x0.shape[0]
        self.m = x0.shape[1]

    def _simulate(self, drift_func, beta, **kwargs):
        """
        Generic EM integrator for any potential.
        """
        x = np.zeros((self.d, self.m, self.iters))
        x[:, :, 0] = self.x0

        noise_scale = np.sqrt(2 / beta) * np.sqrt(self.h)

        for t in range(1, self.iters):
            drift_val = drift_func(x[:, :, t - 1], **kwargs)
            x[:, :, t] = (
                x[:, :, t - 1]
                + drift_val * self.h
                + noise_scale * np.random.normal(size=(self.d, self.m))
            )
        return x[:, :, -1]

    def sample(self, potential_name: str, beta: float, **kwargs):
        drift_func = DRIFT_MAP[potential_name]
        return self._simulate(drift_func, beta, **kwargs)

    def get_long_trajectory(self, T: int, potential_name: str, beta: float, **kwargs):
        drift_func = DRIFT_MAP[potential_name]

        x = np.zeros((self.d, T))
        x[:, 0] = self.x0[:, 0]

        noise_scale = np.sqrt(2 / beta) * np.sqrt(self.h)

        for t in range(1, T):
            drift_val = drift_func(x[:, t - 1 : t], **kwargs).reshape(self.d)
            x[:, t] = (
                x[:, t - 1]
                + drift_val * self.h
                + noise_scale * np.random.normal(size=self.d)
            )
        return x
