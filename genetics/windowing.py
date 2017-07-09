from typing import Tuple
import time

from genetics import imaging

import pygame


class Application(object):
    def __init__(self, screen_size: Tuple[int, int]):
        self._screen_size = screen_size

    def start(self, step_function: callable):
        screen = self._initialize_screen()
        done = False
        iteration = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            image_to_render, loss = self._perform_step_and_get_image(step_function)
            print("[{}] {}".format(iteration, loss))

            screen.fill((0, 0, 0))
            screen.blit(image_to_render, (0, 0))

            pygame.display.flip()
            time.sleep(0)
            iteration += 1

    def _initialize_screen(self):
        pygame.init()
        pygame.display.set_caption('genetics algorithm')
        screen = pygame.display.set_mode(self._screen_size)
        return screen

    def _perform_step_and_get_image(self, step_function):
        sample, loss = step_function()

        current_image_state = imaging.ImageProcessor()
        current_image_state.from_pixels(sample)
        image = current_image_state.get_image()

        return pygame.image.fromstring(
            image.tobytes(),
            image.size,
            image.mode
        ), loss
