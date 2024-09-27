import math
import time

import numpy as np
from numba import jit


def gaussian(x, mu, sigma, min_value=0, max_value=1):
    """
    高斯函数


    :param x: 自变量
    :param mu: 均值（mean）
    :param sigma: 标准差（standard deviation）
    :param min_value: 函数最小值
    :param max_value: 函数最大值
    :return: 高斯函数值
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    value = coefficient * np.exp(exponent) / coefficient
    value = value * (max_value - min_value) + min_value
    return value

