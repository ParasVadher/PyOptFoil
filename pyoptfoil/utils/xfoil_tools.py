import subprocess as sp
import numpy as np
from ..aerofoil import Aerofoil


def write_dat(aerofoil: Aerofoil):

    """
    Creates XFOIL labeled coordinate file for given aerofoil.
    Parameters
    ----------
    aerofoil : Aerofoil
        Aerofoil object for which to generate coordinate file
    """

    x = np.concatenate((np.flip(aerofoil.x_l), aerofoil.x_u[1:]))
    y = np.concatenate((np.flip(aerofoil.y_l), aerofoil.y_u[1:]))

    lines = ['{0} {1}\n'.format(x[i], y[i]) for i in range(len(x))]

    with open('xfoil.dat', 'w') as f:
        f.write(aerofoil.name + '\n')
        f.writelines(lines)


def run_xfoil(xfoil_path: str, datfile_path: str, alfas: tuple, re: float, m: float, itermax: int):

    """
    Runs XFOIL. Commands turn off graphics, load coordinate file, set panelling, and run incidence sweep at requested
    conditions.

    Parameters
    ----------
    xfoil_path : str
        Path to the XFOIL executable file.
    datfile_path : str
        Path to the aerofoil coordinate file.
    alfas : tuple
        Incidence range. (alpha_start, alpha_stop, alpha_increment)
    re : float
        Reynolds number.
    m: float
        Mach number
    itermax : int
        XFOIL viscous solution iteration limit.
    """

    inputs = ['plop', 'g', '', 'load', datfile_path, 'pane', 'oper', 'v', str(re), 'm', str(m), 'pacc',
              'xfoil.out', ' ', 'iter', str(itermax), 'aseq', str(alfas[0]), str(alfas[1]), str(alfas[2]), ' ', 'quit']

    xfoil_proc = sp.run(xfoil_path, input='\n'.join(inputs), capture_output=True, text=True, timeout=10)

    return xfoil_proc


def read_out():
    """
    Reads XFOIL polar save file.

    Returns
    -------
    Numpy array containing angle of attack, lift coefficient and drag coefficient data.
    """
    with open('xfoil.out', 'r') as f:
        lines = f.readlines()

    skiprows = [i for i in range(len(lines)) if '------' in lines[i]][0] + 1
    arr = np.loadtxt('xfoil.out', skiprows=skiprows)

    return arr
