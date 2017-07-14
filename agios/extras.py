from collections import namedtuple

import numpy as np
from PIL import Image


ImageFormat = namedtuple('ImageFormat', ['symbol', 'max_channel_value', 'pack_function', 'unpack_function'])
Greyscale = ImageFormat(
    symbol='L',
    max_channel_value=0xff,
    pack_function=lambda color: color,
    unpack_function=lambda color: color
)
RGB = ImageFormat(
    symbol='RGB',
    max_channel_value=0xffffff,
    pack_function=lambda color: ((color[0] << 16) + (color[1] << 8) + color[2]),
    unpack_function=lambda color: ((color & 0xff0000) >> 16, (color & 0x00ff00) >> 8, (color & 0x0000ff))
)


def load_normalized_image(image_path: str, img_format: ImageFormat) -> np.array:
    image = _convert_image_format(Image.open(image_path), img_format.symbol)
    image_width, image_height = image.size

    source_pixels = _normalize_colors(image.getdata(), img_format)
    return _create_matrix_from_pixels_array(image_width, image_height, source_pixels)


def create_rgb_image_from_matrix(matrix: np.array, img_format: ImageFormat) -> Image:
    denormalized_matrix = np.uint32(matrix * img_format.max_channel_value)
    pixels = []
    for x in range(matrix.shape[0]):
        row = []
        for y in range(matrix.shape[1]):
            row.append(img_format.unpack_function(denormalized_matrix[x, y]))
        pixels.append(row)
    return Image.fromarray(np.uint8(pixels)).convert('RGB')


def _normalize_colors(source_pixels: list, img_format: ImageFormat):
    return [img_format.pack_function(color) / img_format.max_channel_value for color in source_pixels]


def _create_matrix_from_pixels_array(image_width, image_height, source_pixels):
    return np.array([source_pixels[i * image_width:(i + 1) * image_width] for i in range(image_height)])


def _convert_image_format(image, target_mode):
    if image.mode.lower() != target_mode.lower():
        return image.convert(target_mode)
    return image
