import subprocess as sp
import numpy as np
from ..aerofoil import Aerofoil


def write_dat(aerofoil: Aerofoil):
    pass


def run_xfoil(xfoil_path: str, datfile_path: str, alfas: tuple, re: float, m: float, itermax: int):
    inputs = ['load', datfile_path, 'pane', 'oper', 'v', str(re), 'm', str(m), 'pacc', 'xfoil.out', ' ', 'iter',
              str(itermax), 'aseq',
              str(alfas[0]), str(alfas[1]),
              str(alfas[2]), ' ', 'quit']
    xfoil_proc = sp.run(xfoil_path, input='\n'.join(inputs), capture_output=True, text=True)

    with open('xfoil.out', 'r') as f:
        lines = f.readlines()

    skiprows = [i for i in range(len(lines)) if '------' in lines[i]][0] + 1
    arr = np.loadtxt('xfoil.out', skiprows=skiprows)

    return arr, xfoil_proc
