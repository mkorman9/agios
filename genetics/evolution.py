from typing import Tuple
import random

import numpy as np

POPULATION_SIZE = 20
BEST_INDIVIDUALS_TO_TAKE = 3
MUTATION_CHANCE = 0.05


class Solver(object):
    def __init__(self, blueprint: np.array):
        self._blueprint = blueprint
        self._current_best_sample = Sample(shape=blueprint.shape)

        print("Initial loss: {}".format(self._current_best_sample.get_loss(blueprint)))

    def step(self) -> np.array:
        population = self._create_population()
        best_individuals = self._resolve_best_individuals(population)
        effect = self._perform_crossing(best_individuals)

        if effect.get_loss(self._blueprint) < self._current_best_sample.get_loss(self._blueprint):
            self._current_best_sample = effect

        print('Current loss: {}'.format(self._current_best_sample.get_loss(self._blueprint)))

        return self._current_best_sample.get_state()

    def _create_population(self):
        population = []
        for individual in range(POPULATION_SIZE):
            sample = Sample(self._current_best_sample.get_state())
            sample.perform_random_mutation()
            population.append(sample)
        return population

    def _resolve_best_individuals(self, population):
        ranking = sorted(
            population,
            key=lambda individual: individual.get_loss(self._blueprint),
            reverse=True
        )

        return ranking[:BEST_INDIVIDUALS_TO_TAKE]

    def _perform_crossing(self, best_individuals):
        current = best_individuals[0]
        for i in range(1, len(best_individuals)):
            current = current.perform_crossing(best_individuals[i])
        return current


class Sample(object):
    def __init__(self, parent: np.array=None, shape: Tuple[int, int]=None):
        self._state = np.copy(parent) if parent is not None else np.zeros(shape)

    def get_state(self) -> np.array:
        return self._state

    def get_loss(self, blueprint: np.array):
        return np.sqrt(
            np.sum((self._state - blueprint) ** 2)
        )

    def perform_random_mutation(self):
        #for pixel in np.nditer(self._state, op_flags=['readwrite']):
        #    if random.random() <= MUTATION_CHANCE:
        #        pixel[...] = random.random()

        color = random.random()
        max_w, max_h = self._state.shape
        x, y = random.randint(0, max_w), random.randint(0, max_h)
        w, h = random.randint(10, 50), random.randint(10, 50)

        for i in range(x, min(x + w, max_w)):
            for j in range(y, min(y + h, max_h)):
                self._state[i, j] = color

    def perform_crossing(self, other_sample: 'Sample') -> 'Sample':
        return Sample((self._state + other_sample._state) / 2)
