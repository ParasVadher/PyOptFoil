import numpy as np
import math
from scipy.optimize import fsolve
from .bezier_curves import b3, db3


class BP3333:
    def __init__(self, name: str, params: dict):

        """
        Bezier-PARSEC 3333 aerofoil parameterization class.

        Parameters
        ----------
        name : Geometry name.
        params : Dict containing BP3333 parameters. Should contain the following keys: 'x_t', 'y_t', 'r_le', 'k_t',
        'beta_te' , 'dz_te', 'gamma_le', 'x_c', 'y_c', 'k_c', 'alpha_te', 'z_te'.

        """

        self.name = name
        self.params = params
        self.constraint_violation = False
        self._rt()
        self._rc()

    def _rt(self):
        def quartic_func(r_t, x_t, y_t, r_le, k_t):
            expr = 27 * k_t ** 2 * r_t ** 4 / 4
            expr -= 27 * k_t ** 2 * x_t * r_t ** 3
            expr += (9 * k_t * y_t + 81 * k_t ** 2 * x_t ** 2 / 2) * r_t ** 2
            expr += (2 * r_le - 18 * k_t * x_t * y_t - 27 * k_t ** 2 * x_t ** 3) * r_t
            expr += 3 * y_t ** 2 + 9 * k_t * x_t ** 2 * y_t + 27 * k_t ** 2 * x_t ** 4 / 4
            return expr

        r_t = fsolve(quartic_func, 0.15, (self.params['x_t'], self.params['y_t'], self.params['r_le'], self.params['k_t']))

        def rt_constraint_check(params, p):
            retbool1 = 0 < p < params['x_t']
            retbool2 = p > params['x_t'] - (-2 * params['y_t'] / (3 * params['k_t'])) ** 0.5
            retbool3 = 1 + (params['dz_te'] - (3 * params['k_t'] * (params['x_t'] - p) ** 2 / 2 + params['y_t'])) \
                       / math.tan(params['beta_te']) > 2 * params[
                           'x_t'] - p  # ensures that x is monotonically increasing

            return retbool1 and retbool2 and retbool3

        r_t = list(filter(lambda i: rt_constraint_check(self.params, i), r_t))

        if len(r_t) is 0:
            self.constraint_violation = True
        else:
            self.r_t = min(r_t)

    def _rc(self):
        summand1 = 16 + 3 * self.params['k_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1) * (
                           1 + self.params['z_te'] * math.tan(self.params['alpha_te']) ** -1)

        summand2 = 6 * self.params['k_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1)
        summand2 *= 1 - self.params['y_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1) + self.params[
                        'z_te'] * math.tan(self.params['alpha_te']) ** -1
        summand2 = np.array([-1, 1]) * 4 * (16 + summand2) ** 0.5

        r_c = (summand1 + summand2) / (3 * self.params['k_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1) ** 2)

        def rc_constraint_check(params, p):
            retbool1 = 0 < p < params['y_c']
            retbool2 = params['x_c'] - (2 * (p - params['y_c']) / (3 * params['k_c'])) ** 0.5 > p / math.tan(
                params['gamma_le'])  # ensures that x is monotonically increasing
            retbool3 = 1 + (params['z_te'] - p) * math.tan(params['alpha_te']) ** -1 > params['x_c'] + (
                    2 * (p - params['y_c']) / (3 * params['k_c'])) ** 0.5  # ensures that x is monotonically increasing
            retbool4 = np.isreal(p)
            return retbool1 and retbool2 and retbool3 and retbool4

        r_c = list(filter(lambda i: rc_constraint_check(self.params, i), r_c))

        if len(r_c) is 0:
            self.constraint_violation = True
        else:
            self.r_c = max(r_c)

    def xy_lt(self):

        """Calculates cubic bezier control points for the leading edge thickness curve"""

        x0 = 0
        x1 = 0
        x2 = self.r_t
        x3 = self.params['x_t']

        y0 = 0
        y1 = 3 * self.params['k_t'] * (self.params['x_t'] - self.r_t) ** 2 / 2 + self.params['y_t']
        y2 = self.params['y_t']
        y3 = self.params['y_t']

        x_lt = (x0, x1, x2, x3)
        y_lt = (y0, y1, y2, y3)

        return x_lt, y_lt

    def xy_tt(self):

        """Calculates cubic bezier control points for the trailing edge thickness curve"""

        x0 = self.params['x_t']
        x1 = 2 * self.params['x_t'] - self.r_t
        x2 = 1 + (self.params['dz_te'] - (
                3 * self.params['k_t'] * (self.params['x_t'] - self.r_t) ** 2 / 2 + self.params['y_t'])) \
             / math.tan(self.params['beta_te'])
        x3 = 1

        y0 = self.params['y_t']
        y1 = self.params['y_t']
        y2 = 3 * self.params['k_t'] * (self.params['x_t'] - self.r_t) ** 2 / 2 + self.params['y_t']
        y3 = self.params['dz_te']

        x_tt = (x0, x1, x2, x3)
        y_tt = (y0, y1, y2, y3)

        return x_tt, y_tt

    def xy_lc(self):

        """Calculates cubic bezier control points for the leading edge camber curve"""

        x0 = 0
        x1 = self.r_c * math.tan(self.params['gamma_le']) ** -1
        x2 = self.params['x_c'] - (2 * (self.r_c - self.params['y_c']) / (3 * self.params['k_c'])) ** 0.5
        x3 = self.params['x_c']

        y0 = 0
        y1 = self.r_c
        y2 = self.params['y_c']
        y3 = self.params['y_c']

        x_lc = (x0, x1, x2, x3)
        y_lc = (y0, y1, y2, y3)

        return x_lc, y_lc

    def xy_tc(self):
        """Calculates cubic bezier control points for the trailing edge camber curve"""

        x0 = self.params['x_c']
        x1 = self.params['x_c'] + (2 * (self.r_c - self.params['y_c']) / (3 * self.params['k_c'])) ** 0.5
        x2 = 1 + (self.params['z_te'] - self.r_c) * math.tan(self.params['alpha_te']) ** -1
        x3 = 1

        y0 = self.params['y_c']
        y1 = self.params['y_c']
        y2 = self.r_c
        y3 = self.params['z_te']

        x_tc = (x0, x1, x2, x3)
        y_tc = (y0, y1, y2, y3)

        return x_tc, y_tc

    def xy(self):
        """Calculates aerofoil x,y coordinates"""

        x_lt, y_lt = self.xy_lt()
        x_tt, y_tt = self.xy_tt()
        x_lc, y_lc = self.xy_lc()
        x_tc, y_tc = self.xy_tc()

        u = np.linspace(0, 1, 100)  # bezier parameter u

        x_t = np.concatenate((b3(u, *x_lt), b3(u, *x_tt)))
        y_t = np.concatenate((b3(u, *y_lt), b3(u, *y_tt)))
        x_c = np.concatenate((b3(u, *x_lc), b3(u, *x_tc)))
        y_c = np.concatenate((b3(u, *y_lc), b3(u, *y_tc)))

        dyc_du = np.concatenate((db3(u, *y_lc), db3(u, *y_tc)))
        dxc_du = np.concatenate((db3(u, *x_lc), db3(u, *x_tc)))
        dyc_dxc = dyc_du / dxc_du
        theta = np.arctan(dyc_dxc)

        y_t = np.interp(x_c, x_t, y_t)  # interpolating for thickness at camber x points

        x_u = x_c - y_t / 2 * np.sin(theta)
        x_l = x_c + y_t / 2 * np.sin(theta)
        y_u = y_c + y_t / 2 * np.cos(theta)
        y_l = y_c - y_t / 2 * np.cos(theta)

        return x_u, y_u, x_l, y_l
