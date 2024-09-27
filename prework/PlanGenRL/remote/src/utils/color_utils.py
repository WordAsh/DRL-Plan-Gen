import random

import numpy as np


def generate_random_colors(n: int, channel: int = 3, seed: int = 42):
    """
    Generates random colors between 0 and 255.
    :param n: num colors to generate
    :param channel: color channel
    :param seed: random seed
    :return:
    """
    random.seed(seed)
    colors = []
    for _ in range(n):
        if channel == 3:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        colors.append(color)
    return colors


