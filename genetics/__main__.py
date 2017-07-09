import sys

from genetics import imaging
from genetics import windowing
from genetics import evolution

if __name__ == '__main__':
    blueprint_image_processor = imaging.ImageProcessor()
    blueprint_image_processor.load(sys.argv[1])

    evolution_problem_solver = evolution.Solver(blueprint_image_processor.get_pixels())

    renderer = windowing.Application(blueprint_image_processor.get_size())
    renderer.start(evolution_problem_solver.step)
