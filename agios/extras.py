import numpy as np
from PIL import Image


MAX_COLOR_CHANNEL_VALUE = 255


def load_normalized_greyscale_image(image_path: str) -> np.array:
    image = Image.open(image_path).convert('L')
    image_width, image_height = image.size

    source_pixels = _normalize_colors(image.getdata())
    return _create_matrix_from_pixels_array(image_width, image_height, source_pixels)


def create_rgb_image_from_matrix(matrix: np.array) -> Image:
    return Image.fromarray(np.uint8(matrix * MAX_COLOR_CHANNEL_VALUE)).convert('RGB')


def _normalize_colors(source_pixels: list):
    return [color / MAX_COLOR_CHANNEL_VALUE for color in source_pixels]


def _create_matrix_from_pixels_array(image_width, image_height, source_pixels):
    return np.array([source_pixels[i * image_width:(i + 1) * image_width] for i in range(image_height)])
