from agios import evolution
from agios import extras
from examples.util import windowing

if __name__ == '__main__':
    colorspace = extras.Greyscale
    blueprint = extras.load_normalized_image('input/mona_lisa.jpg', colorspace)

    evolution_problem_solver = evolution.BasicSolver(
        population_size=100,
        best_samples_to_take=2,
        blueprint=evolution.NumpyArraySample(blueprint),
        mutator=evolution.SimplePaintbrushMatrixMutator((10, 15), (10, 50)),
        crosser=evolution.MeanValueMatrixCrosser(),
        loss_calculator=evolution.SquaredMeanMatrixLossCalculator(),
        initial_sample_state_generator=evolution.RandomMatrixGenerator(blueprint.shape)
    )

    renderer = windowing.Application(blueprint.shape, colorspace)
    renderer.start(evolution_problem_solver)
