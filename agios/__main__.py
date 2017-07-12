import sys

from agios import extras
from agios import windowing
from agios import evolution

if __name__ == '__main__':
    blueprint = extras.load_normalized_greyscale_image(sys.argv[1])

    evolution_problem_solver = evolution.Algorithm(
        population_size=200,
        best_samples_to_take=2,
        blueprint=evolution.NumpyArraySample(blueprint),
        mutator=evolution.SimplePaintbrushMatrixMutator((10, 15), (10, 50)),
        crosser=evolution.MeanValueMatrixCrosser(),
        loss_calculator=evolution.SquaredMeanMatrixLossCalculator(),
        initial_sample_state_generator=evolution.RandomMatrixGenerator(blueprint.shape)
    )

    renderer = windowing.Application(blueprint.shape)
    renderer.start(evolution_problem_solver)
