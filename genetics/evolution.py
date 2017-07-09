from typing import Tuple
import random

import numpy as np

POPULATION_SIZE = 200
BEST_INDIVIDUALS_TO_TAKE = 2
MUTATION_CHANCE = 0.001


class Solver(object):
    def __init__(self, blueprint: np.array):
        self._blueprint = blueprint
        self._current_best_sample = None
        self._population = [Sample(shape=blueprint.shape) for _ in range(0, POPULATION_SIZE)]

    def step(self) -> np.array:
        best_score = self._current_best_sample.get_loss(self._blueprint) if self._current_best_sample else float('inf')
        self._sort_population_by_best_scores()
        best_sample = self._perform_crossing_and_get_best()

        if best_sample.get_loss(self._blueprint) < best_score:
            self._current_best_sample = best_sample

        print('Loss: {}'.format(best_score))

        self._mutate_population()
        return self._current_best_sample.get_state()

    def _mutate_population(self):
        for i in range(POPULATION_SIZE):
            self._population[i].perform_random_mutation()

    def _sort_population_by_best_scores(self):
        self._population = sorted(
            self._population,
            key=lambda individual: individual.get_loss(self._blueprint)
        )

    def _perform_crossing_and_get_best(self):
        best_individuals = self._population[:BEST_INDIVIDUALS_TO_TAKE]
        best_genome_sample = best_individuals[0]
        for i in range(1, len(best_individuals)):
            best_genome_sample = best_genome_sample.perform_crossing(best_individuals[i])

        for i in range(BEST_INDIVIDUALS_TO_TAKE, POPULATION_SIZE):
            self._population[i] = self._population[i].perform_crossing(best_genome_sample)

        return best_genome_sample


class Sample(object):
    def __init__(self, parent: np.array=None, shape: Tuple[int, int]=None):
        self._state = np.copy(parent) if parent is not None else np.random.random(shape)

    def get_state(self) -> np.array:
        return self._state

    def get_loss(self, blueprint: np.array):
        return np.sqrt(
            np.sum(
                (self._state - blueprint) ** 2
            )
        )

    def perform_random_mutation(self):
        color = random.random()
        max_w, max_h = self._state.shape
        x, y = random.randint(0, max_w), random.randint(0, max_h)
        w, h = random.randint(10, 50), random.randint(10, 50)

        for i in range(x, min(x + w, max_w)):
            for j in range(y, min(y + h, max_h)):
                self._state[i, j] = (self._state[i, j] + color) / 2

    def perform_crossing(self, other_sample: 'Sample') -> 'Sample':
        return Sample((self._state + other_sample._state) / 2)
