from .parameterizations.bezier_parsec import BP3333


class Aerofoil:
    def __init__(self, name: str, param_method: str, params: dict):

        """
        Class containing information about an individual aerofoil.

        Parameters
        ----------
        name : str
            Name of individual.
        param_method : str
            Parametrisation method.
        params : dict
            Contains parameters defining location in search space. To be used with parametrisation method to determine
            aerofoil coordinates.
        """

        self.name = name
        self.position = params

        if param_method is 'BP3333':
            self.parameterization = BP3333(self.name, self.position)
        else:
            raise ValueError("Invalid parameterization method")

        if not self.parameterization.constraint_violation:
            x_u, y_u, x_l, y_l = self.parameterization.xy()
            self.x_u = x_u
            self.y_u = y_u
            self.x_l = x_l
            self.y_l = y_l

        self.fitness = None
