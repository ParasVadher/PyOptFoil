import numpy as np


def b3(u: np.ndarray, p0: float, p1: float, p2: float, p3: float):
    """
    Calculates cubic bezier curve points.

    Parameters
    ----------
    u : parameter 0 <= u <= 1.
    p0 : bezier control point.
    p1 : bezier control point.
    p2 : bezier control point.
    p3 : bezier control point.

    Returns
    -------
    b : cubic bezier points.
    """

    b = p0 * (1 - u) ** 3 + 3 * p1 * u * (1 - u) ** 2 + 3 * p2 * u ** 2 * (1 - u) + p3 * u ** 3
    return b


def db3(u: np.ndarray, p0: float, p1: float, p2: float, p3: float):
    """
    Calculates points for the gradient of a cubic bezier curve.

    Parameters
    ----------
    u : parameter 0 <= u <= 1.
    p0 : bezier control point.
    p1 : bezier control point.
    p2 : bezier control point.
    p3 : bezier control point.

    Returns
    -------
    db_du : cubic bezier curve gradients.
    """

    db_du = -3 * p0 * (1 - u) ** 2 + 3 * p1 * (1 - 4 * u + 3 * u ** 2) + 3 * p2 * (2 * u - 3 * u ** 2) + 3 * p3 * u ** 2
    return db_du
