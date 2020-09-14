import numpy as np
from .algorithms.de import DE


def _aerofoil_similarity(x_u, y_u, x_l, y_l, aerofoil):
    if aerofoil.parameterization.constraint_violation:
        return -np.inf
    else:
        y_u_trial = np.interp(x_u, aerofoil.x_u, aerofoil.y_u)
        y_l_trial = np.interp(x_l, aerofoil.x_l, aerofoil.y_l)
        return -(np.linalg.norm(y_l - y_l_trial) + np.linalg.norm(y_u - y_u_trial))


def fit(optimizer: DE, x_u: np.ndarray, y_u: np.ndarray, x_l: np.ndarray, y_l: np.ndarray):

    """
    Runs optimisation algorithm to obtain parameters which best fit the given target aerofoil coordinates.

    Parameters
    ----------
    optimizer : DE
        Optimisation algorithm object.
    x_u : numpy array
        Upper surface x coordinates of target aerofoil.
    y_u : numpy array
        Upper surface y coordinates of target aerofoil.
    x_l : numpy array
        Lower surface x coordinates of target aerofoil.
    y_l : numpy array
        Lower surface y coordinates of target aerofoil.
    """

    def func(a): return _aerofoil_similarity(x_u, y_u, x_l, y_l, a)
    optimizer.optimize(func)
