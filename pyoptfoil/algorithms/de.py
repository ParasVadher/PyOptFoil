from ..aerofoil import Aerofoil
import numpy as np
import copy
import random


class DE:
    def __init__(self, bounds: dict, pop_size: int, n_iterations: int, param_method: str):
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_iterations = n_iterations
        self.param_method = param_method

        self.best_individual = None
        self.population = []

        bounds_arr = np.array(self.bounds.values())
        self.lb = bounds_arr[:, 0]
        self.ub = bounds_arr[:, 1]

    def generate_individual(self, name):
        params_values = np.random.uniform(self.lb, self.ub)
        params = dict(zip(self.bounds.keys(), params_values))

        individual = Aerofoil(name, self.param_method, params)

        return individual

    def initialise_population(self):
        for i in range(self.pop_size):
            name = 'Population Member No. ' + str(i)

            individual = self.generate_individual(name)
            while individual.parameterization.constraint_violation:
                # ensures all members of the initial population meet constraints
                individual = self.generate_individual(name)

            self.population.append(individual)

    def evaluate_population(self, func):
        for individual in self.population:
            individual.fitness = func(individual)
            if self.best_individual is None or individual.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(individual)

    def optimize(self, func):
        self.initialise_population()
        self.evaluate_population(func)

        for iter in range(self.n_iterations):
            raise NotImplementedError()
