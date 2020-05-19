import numpy as np
from scipy.optimize import fsolve


class BezierCurves:
    @staticmethod
    def b3(u: np.ndarray, p0: float, p1: float, p2: float, p3: float):
        """
        Calculates cubic bezier curve.

        Parameters
        ----------
        u : parameter 0 <= u <= 1
        p0 : bezier control point
        p1 : bezier control point
        p2 : bezier control point
        p3 : bezier control point

        Returns
        -------
        b : cubic bezier points
        """
        b = p0 * (1 - u) ** 3 + 3 * p1 * u * (1 - u) ** 2 + 3 * p2 * u ** 2 * (1 - u) + p3 * u ** 3
        return b


class BP3333:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def _rt(self):
        def func(r_t, x_t, y_t, r_le, k_t):
            expr = 27 * k_t ** 2 * r_t ** 4 / 4
            expr -= 27 * k_t ** 2 * x_t * r_t ** 3
            expr += (9 * k_t * y_t + 81 * k_t ** 2 * x_t ** 2 / 2) * r_t ** 2
            expr += (2 * r_le - 18 * k_t * x_t * y_t - 27 * k_t ** 2 * x_t ** 3) * r_t
            expr += 3 * y_t ** 2 + 9 * k_t * x_t ** 2 * y_t + 27 * k_t ** 2 * x_t ** 4 / 4
            return expr

        rt = fsolve(func, 0.15, (self.params['x_t'], self.params['y_t'], self.params['r_le'], self.params['k_t']))

        def check(params, p):
            min_exp = params['x_t'] - (-2 * params['y_t'] / (3 * params['k_t'])) ** 0.5
            retbool = p > 0
            retbool = retbool and p > min_exp
            retbool = retbool and p < params['x_t']
            return retbool

        rt = list(filter(lambda i: check(self.params,i), rt))

        return np.sort(rt)[0]

    def xy_lt(self):
        x0 = 0
        x1 = 0
        x2 = self._rt()
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
        x0 = self.params['x_t']
        x1 = 2 * x0 - self._rt()
        x2 = 1 + (self.params['dz_te'] - (3 * self.params['k_t'] * (x0 - self._rt()) ** 2 / 2 + self.params['y_t'])) \
             / np.tan(self.params['beta_te'])
        x3 = 1

        y0 = self.params['y_t']
        y1 = y0
        y2 = 3 * self.params['k_t'] * (x0 - self._rt()) ** 2 / 2 + y0
        y3 = self.params['dz_te']

        u = np.linspace(0, 1, 100)
        x_tt = BezierCurves.b3(u, x0, x1, x2, x3)
        y_tt = BezierCurves.b3(u, y0, y1, y2, y3)

        return x_tt, y_tt
