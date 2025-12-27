import numpy as np

def loss_fn(params, a=1, b=10):
    """
    Convex quadratic loss function:
    f(x, y) = a*x^2 + b*y^2
    """
    x, y = params
    return a * x**2 + b * y**2

def grad_fn(params, a=1, b=10):
    """
    Gradient of the loss function
    """
    x, y = params
    return np.array([2 * a * x, 2 * b * y])
