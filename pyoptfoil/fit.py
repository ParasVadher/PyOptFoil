import numpy as np


def _aerofoil_similarity(x_u, y_u, x_l, y_l, aerofoil):
    if aerofoil.parameterization.constraint_violation:
        return -np.inf
    else:
        y_u_trial = np.interp(x_u, aerofoil.x_u, aerofoil.y_u)
        y_l_trial = np.interp(x_l, aerofoil.x_l, aerofoil.y_l)
        return -(np.linalg.norm(y_l - y_l_trial) + np.linalg.norm(y_u - y_u_trial))
