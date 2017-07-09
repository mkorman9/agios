from typing import Tuple

from PIL import Image
import numpy as np


class ImageProcessor(object):
    MAX_COLOR = 255

    def load(self, file_path: str):
        image = Image.open(file_path).convert('L')
        image_width, image_height = image.size
        source_pixels = self._normalize_colors(image.getdata())

        self._pixels = self._create_image_data(image_height, image_width, source_pixels)
        self._size = image_width, image_height

    def from_pixels(self, pixels: np.array):
        self._pixels = pixels
        self._size = pixels.shape

    def get_pixels(self) -> np.array:
        return self._pixels

    def get_image(self) -> Image:
        return Image.fromarray(np.uint8(self._pixels * self.MAX_COLOR)).convert('RGB')

    def get_size(self) -> Tuple[int, int]:
        return self._size

    def _create_image_data(self, image_height, image_width, source_pixels):
        return np.array([source_pixels[i * image_width:(i + 1) * image_width] for i in range(image_height)])

    def _normalize_colors(self, source_pixels):
        return [color / self.MAX_COLOR for color in source_pixels]
