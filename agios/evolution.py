import abc
from collections import namedtuple
from typing import Tuple
import random

import numpy as np

# Loss calculators


class LossCalculator(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, sample1: 'SampleGeneric', sample2: 'SampleGeneric') -> float:
        pass


class SquaredMeanMatrixLossCalculator(LossCalculator):
    def calculate(self, sample1: 'NumpyArraySample', sample2: 'NumpyArraySample') -> float:
        return np.sqrt(
            np.sum(
                (sample1.state() - sample2.state()) ** 2
            )
        )

# Mutators


class Mutator(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def mutate(self, sample: 'SampleGeneric') -> 'SampleGeneric':
        pass


class RandomMatrixFieldChangeMutator(Mutator):
    def mutate(self, sample: 'NumpyArraySample') -> 'NumpyArraySample':
        sample_to_create = sample.clone()
        matrix = sample_to_create.state()

        max_w, max_h = matrix.shape
        x, y = random.randint(0, max_w - 1), random.randint(0, max_h - 1)
        matrix[x, y] = (matrix[x, y] + random.random()) / 2

        return sample_to_create


class RandomMatrixAreasMutator(Mutator):
    def __init__(self, horizontal_size_range: Tuple[int, int], vertical_size_range: Tuple[int, int]):
        self._horizontal_size_range = horizontal_size_range
        self._vertical_size_range = vertical_size_range

    def mutate(self, sample: 'NumpyArraySample') -> 'NumpyArraySample':
        sample_to_create = sample.clone()

        value_to_mix_with = random.random()
        max_w, max_h = sample_to_create.state().shape
        x, y = random.randint(0, max_w), random.randint(0, max_h)
        w, h = random.randint(*self._vertical_size_range), random.randint(*self._horizontal_size_range)

        for i in range(x, min(x + w, max_w)):
            for j in range(y, min(y + h, max_h)):
                sample_to_create.state()[i, j] = (sample_to_create.state()[i, j] + value_to_mix_with) / 2

        return sample_to_create


class SimplePaintbrushMatrixMutator(Mutator):
    def __init__(self, brush_widths_range=(1, 4), moves_length_range=(1, 10)):
        self._brush_widths_range = brush_widths_range
        self._moves_length_range = moves_length_range

    def mutate(self, sample: 'NumpyArraySample') -> 'NumpyArraySample':
        sample_to_create = sample.clone()
        matrix = sample_to_create.state()
        W, H = matrix.shape

        brush_position = np.array([random.randint(0, W - 1), random.randint(0, H - 1)])
        brush_width = random.randint(*self._brush_widths_range)
        move_length = random.randint(*self._moves_length_range)
        move_directions = np.array([random.randint(-1, 1), random.randint(-1, 1)])
        value = random.random()

        moves_done = 0
        while moves_done != move_length:
            x, y = brush_position[0], brush_position[1]
            if x < 0 or x >= W or y < 0 or y >= H:
                break
            matrix[x, y] = (matrix[x, y] + value) / 2
            self._fill_vertical(matrix, x, y, W, brush_width, value)
            self._fill_horizontal(matrix, x, y, H, brush_width, value)
            brush_position += move_directions
            moves_done += 1

        return sample_to_create

    def _fill_vertical(self, matrix, x, y, max_x, length, value):
        for i in range(x, min(x + (length // 2), max_x)):
            distance_covered = abs(x - i) or 1
            matrix[i, y] = (matrix[i, y] + (value / distance_covered)) / 2
        for i in range(x, max(x - (length // 2), 0, -1)):
            distance_covered = abs(x - i) or 1
            matrix[i, y] = (matrix[i, y] + (value / distance_covered)) / 2

    def _fill_horizontal(self, matrix, x, y, max_y, length, value):
        for i in range(y, min(y + (length // 2), max_y)):
            distance_covered = abs(y - i) or 1
            matrix[x, i] = (matrix[x, i] + (value / distance_covered)) / 2
        for i in range(y, max(y - (length // 2), 0, -1)):
            distance_covered = abs(y - i) or 1
            matrix[x, i] = (matrix[x, i] + (value / distance_covered)) / 2


# Crossers


class Crosser(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def cross(self, sample1: 'SampleGeneric', sample2: 'SampleGeneric') -> 'SampleGeneric':
        pass


class MeanValueMatrixCrosser(Crosser):
    def cross(self, sample1: 'NumpyArraySample', sample2: 'NumpyArraySample') -> 'SampleGeneric':
        return sample1.factory().create(
            (sample1.state() + sample2.state()) / 2
        )

# Sample generics


class SampleGeneric(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def state(self):
        pass

    def clone(self) -> 'SampleGeneric':
        return self.factory().clone(self)

    def factory(self) -> 'SampleFactory':
        return GenericFactory(self.__class__)

    def mutated(self, mutator: 'Mutator') -> 'SampleGeneric':
        return mutator.mutate(self)

    def cross_with(self, sample2: 'SampleGeneric', crosser: 'Crosser') -> 'SampleGeneric':
        return crosser.cross(self, sample2)

    def calculate_loss_to(self, blueprint: 'SampleGeneric', loss_calculator: LossCalculator):
        return loss_calculator.calculate(self, blueprint)


class NumpyArraySample(SampleGeneric):
    def __init__(self, state: np.array):
        self._state = np.copy(state)

    def state(self) -> np.array:
        return self._state


class SampleFactory(object):
    @abc.abstractmethod
    def create(self, *args, **kwargs) -> 'SampleGeneric':
        pass

    @abc.abstractmethod
    def clone(self, sample: 'SampleGeneric') -> 'SampleGeneric':
        pass


class GenericFactory(SampleFactory):
    def __init__(self, proxied_type: callable):
        self.proxied_type = proxied_type

    def create(self, *args, **kwargs) -> 'SampleGeneric':
        return self.proxied_type(*args, **kwargs)

    def clone(self, sample: 'SampleGeneric') -> 'SampleGeneric':
        return self.create(state=sample.state())


# State generator

class SampleStateGenerator(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self) -> object:
        pass


class RandomMatrixGenerator(SampleStateGenerator):
    def __init__(self, shape=(100, 100)):
        self._shape = shape

    def generate(self) -> object:
        return np.random.random(self._shape)


class ZeroMatrixGenerator(SampleStateGenerator):
    def __init__(self, shape=(100, 100)):
        self._shape = shape

    def generate(self) -> object:
        return np.zeros(self._shape)


# Algorithm itself

SampleAndItsLoss = namedtuple('SampleAndItsLoss', ['sample', 'loss'])


class Algorithm(object):
    def __init__(self,
                 population_size: int,
                 best_samples_to_take: int,
                 blueprint: 'SampleGeneric',
                 mutator: 'Mutator',
                 crosser: 'Crosser',
                 loss_calculator: 'LossCalculator',
                 initial_sample_state_generator: 'SampleStateGenerator'):
        self._population_size = population_size
        self._best_samples_to_take = best_samples_to_take
        self._blueprint = blueprint
        self._mutator = mutator
        self._crosser = crosser
        self._loss_calculator = loss_calculator
        self._initial_sample_state_generator = initial_sample_state_generator

        self._statistics = Statistics()
        self._population = self._generate_population(blueprint.factory())
        self._best_sample_and_loss = SampleAndItsLoss(
            sample=self._population[0],
            loss=self._loss_calculator.calculate(self._population[0], self._blueprint)
        )

    def step(self):
        currently_lowest_loss = self._evaluate_lowest_loss()
        self._sort_population_by_best_scores()
        tournament_winner = self._perform_crossing_and_get_best()
        currently_best_sample, is_better = self._evaluate_best_sample(tournament_winner, currently_lowest_loss)
        self._mutate_population()

        if is_better or self._best_sample_and_loss is None:
            self._best_sample_and_loss = SampleAndItsLoss(
                sample=currently_best_sample,
                loss=currently_lowest_loss
            )

        self._statistics.add_observation(currently_lowest_loss)

    def get_best(self) -> 'SampleAndItsLoss':
        return self._best_sample_and_loss

    def statistics(self) -> 'Statistics':
        return self._statistics

    def _generate_population(self, samples_factory):
        population = []
        for _ in range(self._population_size):
            population.append(
                samples_factory.create(state=self._initial_sample_state_generator.generate())
            )
        return population

    def _evaluate_best_sample(self, tournament_winner, currently_lowest_loss):
        currently_best_sample = None
        loss = self._loss_calculator.calculate(tournament_winner, self._blueprint)
        is_better = False

        if loss < currently_lowest_loss:
            currently_best_sample = tournament_winner
            is_better = True

        return currently_best_sample, is_better

    def _evaluate_lowest_loss(self):
        return self._loss_calculator.calculate(self._best_sample_and_loss.sample, self._blueprint)

    def _mutate_population(self):
        for i in range(self._population_size):
            self._population[i] = self._population[i].mutated(self._mutator)

    def _sort_population_by_best_scores(self):
        self._population = sorted(
            self._population,
            key=lambda individual: self._loss_calculator.calculate(individual, self._blueprint)
        )

    def _perform_crossing_and_get_best(self):
        best_individuals = self._population[:self._best_samples_to_take]
        best_genome_sample = best_individuals[0]
        for i in range(1, len(best_individuals)):
            best_genome_sample = best_genome_sample.cross_with(best_individuals[i], self._crosser)

        for i in range(self._best_samples_to_take, self._population_size):
            self._population[i] = self._population[i].cross_with(best_genome_sample, self._crosser)

        return best_genome_sample


# Statistics


class Statistics(object):
    def __init__(self):
        self.iterations = 0
        self.current_loss = 0
        self.current_speed = 0
        self.average_speed = 0

        self._last_loss = 0

    def to_dict(self):
        return {
            'iterations': self.iterations,
            'current_loss': self.current_loss,
            'current_speed': self.current_speed,
            'average_speed': self.average_speed
        }

    def add_observation(self, loss: float):
        self.iterations += 1
        self.current_loss = loss
        if self.iterations > 1:
            self.current_speed = abs(self.current_loss - self._last_loss)
            self.average_speed = self.average_speed + ((self.current_speed - self.average_speed) / self.iterations)
        self._last_loss = loss
