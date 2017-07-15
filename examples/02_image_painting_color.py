from agios import evolution
from agios import extras
from examples.util import windowing

if __name__ == '__main__':
    colorspace = extras.CombinedChannels
    blueprints = extras.load_normalized_image_channels('input/lena.png')

    evolution_problem_solver = evolution.MultidimensionalSolver(
        population_size=100,
        best_samples_to_take=2,
        blueprints=[evolution.NumpyArraySample(b) for b in blueprints],
        mutator=evolution.SimplePaintbrushMatrixMutator((10, 15), (10, 50)),
        crosser=evolution.MeanValueMatrixCrosser(),
        loss_calculator=evolution.LinearMatrixLossCalculator(),
        initial_sample_state_generator=evolution.RandomMatrixGenerator(blueprints[0].shape),
        combiner=evolution.MatrixElementsCombiner(),
        step_performer=evolution.ParallelStepPerformer()
    )

    renderer = windowing.Application(blueprints[0].shape, colorspace)
    renderer.start(evolution_problem_solver)
