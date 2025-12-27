"""Lightweight optimizer implementations for 2D quadratic demos."""

from __future__ import annotations

import numpy as np


class SGD:
    """Plain stochastic gradient descent: theta_{t+1} = theta_t - lr * g_t."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self.lr * grads


class Momentum:
    """Momentum: maintains velocity to smooth updates."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.v = np.zeros_like(grads_template())

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.v = self.beta * self.v + (1 - self.beta) * grads
        return params - self.lr * self.v


class Nesterov:
    """Nesterov accelerated gradient: looks ahead with momentum."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.v = np.zeros_like(grads_template())

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        v_prev = self.v.copy()
        self.v = self.beta * self.v + (1 - self.beta) * grads
        # Use lookahead velocity for the update to anticipate curvature
        return params - self.lr * (self.beta * v_prev + (1 - self.beta) * grads)


class AdaGrad:
    """AdaGrad: per-parameter learning-rate decay via accumulated squares."""

    def __init__(self, lr: float = 0.1, eps: float = 1e-8) -> None:
        self.lr = lr
        self.eps = eps
        self.G = np.zeros_like(grads_template())

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.G += grads ** 2
        return params - self.lr * grads / (np.sqrt(self.G) + self.eps)


class RMSProp:
    """RMSProp: exponential moving average of squared gradients."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.G = np.zeros_like(grads_template())

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.G = self.beta * self.G + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.G) + self.eps)


class Adam:
    """Adam: combines momentum and RMS scaling with bias correction."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros_like(grads_template())
        self.v = np.zeros_like(grads_template())
        self.t = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def grads_template() -> np.ndarray:
    """Utility to initialize state vectors for 2D demos."""
    return np.zeros(2, dtype=float)
