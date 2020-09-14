from ..aerofoil import Aerofoil
import numpy as np
import copy
import random
import sys


class DE:
    def __init__(self, bounds: dict, pop_size: int, n_generations: int, param_method: str, f: float, cr: float):

        """
        DE (Differential Evolution) class.

        Parameters
        ----------
        bounds : dict
            Lower and upper bounds of parameters in the search space.
        pop_size : int
            Population size
        n_generations : int
            Total number of generations in the optimisation process.
        param_method : str
            Parametrisation method.
        f : float
            Differential weight/mutation factor
        cr : float
            Crossover probability
        """

        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.param_method = param_method
        self.f = f
        self.cr = cr

        self.best_individual = None
        self.population = []
        self.best_fitness_history = [None]
        self.best_individual_history = [None]

        bounds_arr = np.array(list(self.bounds.values()))
        self.lb = bounds_arr[:, 0]
        self.ub = bounds_arr[:, 1]

    def generate_individual(self, name):

        """Generates an individual Aerofoil object with parameters randomly selected from given bounds."""

        params_values = np.random.uniform(self.lb, self.ub)
        params = dict(zip(self.bounds.keys(), params_values))

        individual = Aerofoil(name, self.param_method, params)

        return individual

    def initialise_population(self):

        """Generates the initial population."""

        for i in range(self.pop_size):
            name = 'Population Member No. ' + str(i)

            individual = self.generate_individual(name)
            while individual.parameterization.constraint_violation:
                # ensures all members of the initial population meet constraints
                individual = self.generate_individual(name)

            self.population.append(individual)

    def evaluate_population(self, func):

        """Evaluates fitness of every individual in the population with the given fitness function."""

        print('Evaluating Gen 0')

        for individual in self.population:
            sys.stdout.write(individual.name + '\r')

            individual.fitness = func(individual)
            if self.best_individual is None or individual.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(individual)
                self.best_fitness_history[0] = self.best_individual.fitness
                self.best_individual_history[0] = self.best_individual

    def mutate(self, idx):

        """Mutation step of the differential evolution algorithm. (DE/rand-to-best/1 scheme)."""

        pot_index = [i for i in range(self.pop_size) if i != idx]
        random_index = random.sample(pot_index, 2)

        x0 = self.population[idx]
        x1 = self.population[random_index[0]]
        x2 = self.population[random_index[1]]

        v_donor = np.fromiter(x0.position.values(), float)
        v_donor += self.f * (np.fromiter(self.best_individual.position.values(), float) - v_donor)
        v_donor += self.f * (np.fromiter(x1.position.values(), float) - np.fromiter(x2.position.values(), float))

        for i in range(len(v_donor)):
            if v_donor[i] < self.lb[i]:
                v_donor[i] = self.lb[i]
            elif v_donor[i] > self.ub[i]:
                v_donor[i] = self.ub[i]

        return v_donor

    def crossover(self, idx, v_donor):

        """Crossover step of the differential evolution algorithm (binary crossover)."""

        v_trial = copy.deepcopy(list(self.population[idx].position.values()))
        for k in range(len(v_donor)):
            if np.random.random() < self.cr:
                v_trial[k] = v_donor[k]

        return v_trial

    def selection(self, idx, v_trial, func):

        """Selection step of the differential evolution algorithm."""

        target_individual = self.population[idx]

        trial_params = dict(zip(self.bounds.keys(), v_trial))
        trial_individual = Aerofoil(target_individual.name, self.param_method, trial_params)

        trial_individual.fitness = func(trial_individual)

        replaced = False
        if trial_individual.fitness > target_individual.fitness:
            self.population[idx] = trial_individual
            replaced = True

        return replaced

    def optimize(self, func):
        """
        Runs the entire optimisation process with the given fitness function.

        Parameters
        ----------
        func : function
            Function used for evaluating fitness.
        """

        self.initialise_population()
        self.evaluate_population(func)

        for gen in range(self.n_generations - 1):
            print('Evaluating Gen {}'.format(gen + 1))

            for idx in range(self.pop_size):
                sys.stdout.write(self.population[idx].name + '\r')
                v_donor = self.mutate(idx)
                v_trial = self.crossover(idx, v_donor)
                replaced = self.selection(idx, v_trial, func)

                if replaced and self.population[idx].fitness > self.best_individual.fitness:
                    self.best_individual = copy.deepcopy(self.population[idx])

            self.best_fitness_history.append(self.best_individual.fitness)
            self.best_individual_history.append(self.best_individual)
