from .algorithms.de import DE
from .aerofoil import Aerofoil
from .utils.xfoil_tools import write_dat, run_xfoil, read_out
import numpy as np
from scipy.interpolate import CubicSpline
import os


def _drag_reward(cl_des: float, re: float, m: float, alpha_range: tuple, xfoil_path: str, itermax: int,
                 aerofoil: Aerofoil):
    if aerofoil.parameterization.constraint_violation:
        return -np.inf

    try:
        write_dat(aerofoil)
        proc = run_xfoil(xfoil_path, 'xfoil.dat', alpha_range, re, m, itermax)
        arr = read_out()
        os.remove('xfoil.out')
    except:
        return -1e12

    try:
        alpha = arr[:, 0]
        cl = arr[:, 1]
        cd = arr[:, 2]
    except:
        return -1e11

    try:
        alpha = alpha[np.argmin(cl):np.argmax(cl) + 1]
        cl = cl[np.argmin(cl):np.argmax(cl) + 1]
        cd = cd[np.argmin(cl):np.argmax(cl) + 1]

        lift_spline = CubicSpline(cl, alpha)
        alpha_req = float(lift_spline(cl_des, extrapolate=False))
        drag_spline = CubicSpline(alpha, cd)
        cd_des = float(drag_spline(alpha_req, extrapolate=False))
    except:
        return -1e9

    if np.isnan(cd_des):
        return -1e8
    else:
        return -cd_des


def opt(optimizer: DE, cl_des: float, re: float, m: float, alpha_range: tuple, xfoil_path: str = 'xfoil.exe',
        itermax: int = 100):

    """
    Runs optimisation algorithm to obtain parameters which minimise drag at a given lift coefficient, Reynolds number
    and Mach number within a given incidence range.

    Parameters
    ----------
    optimizer : DE
        Optimisation algorithm object.
    cl_des : float
        Desired lift coefficient.
    re : float
        Reynolds number.
    m : float
        Mach number.
    alpha_range : tuple
        Incidence range (alpha_start, alpha_stop, alpha_increment).
    xfoil_path : str
        Path to XFOIL executable file.
    itermax : int
        XFOIL viscous solution iteration limit.
    """
    
    def func(a): return _drag_reward(cl_des, re, m, alpha_range, xfoil_path, itermax, a)

    optimizer.optimize(func)
