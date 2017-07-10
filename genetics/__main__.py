import sys

from genetics import imaging
from genetics import windowing
from genetics import evolution

if __name__ == '__main__':
    blueprint_image_processor = imaging.ImageProcessor()
    blueprint_image_processor.load(sys.argv[1])

    evolution_problem_solver = evolution.Algorithm(
        population_size=200,
        best_samples_to_take=2,
        blueprint=evolution.NumpyArraySample(blueprint_image_processor.get_pixels()),
        mutator=evolution.RandomMatrixAreasMutator((10, 50), (10, 50)),
        crosser=evolution.MeanValueMatrixCrosser(),
        loss_calculator=evolution.SquaredMeanMatrixLossCalculator(),
        initial_sample_state_generator=evolution.RandomMatrixGenerator(*blueprint_image_processor.get_size())
    )

    renderer = windowing.Application(blueprint_image_processor.get_size())
    renderer.start(evolution_problem_solver)
