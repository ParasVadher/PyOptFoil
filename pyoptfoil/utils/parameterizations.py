import numpy as np
import math
from scipy.optimize import fsolve


class BezierCurves:
    @staticmethod
    def b3(u: np.ndarray, p0: float, p1: float, p2: float, p3: float):
        """
        Calculates cubic bezier curve.

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


class BP3333:
    def __init__(self, name: str, params: dict):

        """
        Bezier-PARSEC 3333 aerofoil parameterization class

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
        def func(r_t, x_t, y_t, r_le, k_t):
            expr = 27 * k_t ** 2 * r_t ** 4 / 4
            expr -= 27 * k_t ** 2 * x_t * r_t ** 3
            expr += (9 * k_t * y_t + 81 * k_t ** 2 * x_t ** 2 / 2) * r_t ** 2
            expr += (2 * r_le - 18 * k_t * x_t * y_t - 27 * k_t ** 2 * x_t ** 3) * r_t
            expr += 3 * y_t ** 2 + 9 * k_t * x_t ** 2 * y_t + 27 * k_t ** 2 * x_t ** 4 / 4
            return expr

        r_t = fsolve(func, 0.15, (self.params['x_t'], self.params['y_t'], self.params['r_le'], self.params['k_t']))

        def rt_constraint_check(params, p):
            min_exp = params['x_t'] - (-2 * params['y_t'] / (3 * params['k_t'])) ** 0.5
            retbool = p > 0
            retbool = retbool and p > min_exp
            retbool = retbool and p < params['x_t']
            return retbool

        r_t = list(filter(lambda i: rt_constraint_check(self.params, i), r_t))

        if len(r_t) is 0:
            self.constraint_violation = True
        else:
            self.r_t = np.sort(r_t)[0]

    def _rc(self):
        summand1 = 16 + 3 * self.params['k_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1) * (
                           1 + self.params['z_te'] * math.tan(self.params['alpha_te']) ** -1)
        summand1 /= 3 * self.params['k_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1)

        summand2 = 6 * self.params['k_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1)
        summand2 *= 1 - self.params['y_c'] * (
                math.tan(self.params['gamma_le']) ** -1 + math.tan(self.params['alpha_te']) ** -1) + self.params[
                        'z_te'] * math.tan(self.params['alpha_te']) ** -1
        summand2 = np.array([-1, 1]) * 4 * (16 + summand2) ** 0.5

        r_c = summand1 + summand2

        def rc_constraint_check(params, p):
            return 0 < p < params['y_c']

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
        y2 = self.params['y_t']
        y3 = self.params['y_t']
        y1 = 3 * self.params['k_t'] * (x3 - x2) ** 2 / 2 + y2

        u = np.linspace(0, 1, 100)
        x_lt = BezierCurves.b3(u, x0, x1, x2, x3)
        y_lt = BezierCurves.b3(u, y0, y1, y2, y3)

        return x_lt, y_lt

    def xy_tt(self):

        """Calculates cubic bezier control points for the trailing edge thickness curve"""

        x0 = self.params['x_t']
        x1 = 2 * x0 - self.r_t
        x2 = 1 + (self.params['dz_te'] - (3 * self.params['k_t'] * (x0 - self.r_t) ** 2 / 2 + self.params['y_t'])) \
             / math.tan(self.params['beta_te'])
        x3 = 1

        y0 = self.params['y_t']
        y1 = y0
        y2 = 3 * self.params['k_t'] * (x0 - self.r_t) ** 2 / 2 + y0
        y3 = self.params['dz_te']

        u = np.linspace(0, 1, 100)
        x_tt = BezierCurves.b3(u, x0, x1, x2, x3)
        y_tt = BezierCurves.b3(u, y0, y1, y2, y3)

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
        y3 = y2

        u = np.linspace(0, 1, 100)
        x_lc = BezierCurves.b3(u, x0, x1, x2, x3)
        y_lc = BezierCurves.b3(u, y0, y1, y2, y3)

        return x_lc, y_lc

    def xy_tc(self):
        """Calculates cubic bezier control points for the trailing edge camber curve"""

        x0 = self.params['x_c']
        x1 = self.params['x_c'] + (2 * (self.r_c - self.params['y_c']) / (3 * self.params['k_c'])) ** 0.5
        x2 = 1 + (self.params['z_te'] - self.r_c) * math.tan(self.params['alpha_te']) ** -1
        x3 = 1

        y0 = self.params['y_c']
        y1 = y0
        y2 = self.r_c
        y3 = self.params['z_te']

        u = np.linspace(0, 1, 100)
        x_tc = BezierCurves.b3(u, x0, x1, x2, x3)
        y_tc = BezierCurves.b3(u, y0, y1, y2, y3)

        return x_tc, y_tc
