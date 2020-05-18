from scipy.optimize import fsolve

class Aerofoil:
    def __init__(self, name: str):

        self.name = name

    @staticmethod
    def calc_rt(r_le,x_t,y_t,k_t):
        func = lambda r_t,r_le,x_t,y_t,k_t: 27*k_t**2*r_t**4/4 - 27*k_t**2*x_t*r_t**3 + (9*k_t*y_t+81*k_t**2*x_t**2/2)*r_t**2 + r_t*(2*r_le-18*k_t*x_t*y_t -27*k_t**2*x_t**3) + 3*y_t**2+9*k_t*x_t**2*y_t+27*k_t**2*x_t**4/4
        return fsolve(func,0.5,(r_le,x_t,y_t,k_t))