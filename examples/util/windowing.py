from typing import Tuple
import time

from agios import extras
from agios import evolution

import pygame


class Application(object):
    def __init__(self, screen_size: Tuple[int, int], colorspace: extras.ImageFormat):
        self._screen_size = screen_size[1], screen_size[0]
        self._colorspace = colorspace

    def start(self, algorithm: evolution.GenericSolver):
        screen = self._initialize_screen()
        done = False
        iteration = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            image_to_render, loss = self._perform_step_and_get_result(algorithm)
            print(algorithm.statistics().to_dict())

            screen.fill((0, 0, 0))
            screen.blit(image_to_render, (0, 0))

            pygame.display.flip()
            time.sleep(0)
            iteration += 1

    def _initialize_screen(self):
        pygame.init()
        pygame.display.set_caption('agios example')
        screen = pygame.display.set_mode(self._screen_size)
        return screen

    def _perform_step_and_get_result(self, algorithm):
        algorithm.step()
        best_result = algorithm.best()

        image = extras.create_rgb_image_from_matrix(best_result.sample.state(), self._colorspace)

        return pygame.image.fromstring(
            image.tobytes(),
            image.size,
            image.mode
        ), best_result.loss
