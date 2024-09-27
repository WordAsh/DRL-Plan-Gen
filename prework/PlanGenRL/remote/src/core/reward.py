import os

import matplotlib.pyplot as plt
import numpy as np

from utils.reward_utils import gaussian


def get_in_bound_reward(ac_area, area):
    """-1 to 0"""
    # ac_area 未被裁剪的area
    percent = area / ac_area  # 0.25 to 1
    value = (percent - 0.25) / 0.75 - 1  # -1 to 0
    return value


def get_area_reward(x: float, target_area=20, loose=1):
    """ -1 to 1"""
    assert x > 0
    x = x / target_area
    if x < 1:
        x = 1 / x
    return gaussian(x, 1.0, 1 * loose, -1, 1)


def get_ratio_reward(x: float):
    """ -1 to 1"""
    assert x > 0
    if x < 1:
        x = 1 / x
    mu = 1.5
    if x < mu:
        return 1.0

    return gaussian(x, mu, 1, -1, 1)


def get_overlay_reward(overlay_area, room_area):
    """ -1 to 0 or 1"""
    if overlay_area == 0:
        return 1
    overlay_percent = overlay_area / room_area
    return -overlay_percent


def get_invalid_reward(invalid_area, room_area):
    """ -1 to 0 or 1"""
    if invalid_area == 0:
        return 1
    invalid_percent = invalid_area / room_area
    return -invalid_percent


if __name__ == '__main__':

    if 'core' in os.getcwd():
        os.chdir(os.path.abspath('..'))
    print(os.getcwd())


    def _test_area_reward():
        x = np.linspace(1, 100, 1000)
        y = [get_area_reward(i, 20, 1) for i in x]

        # 绘制高斯函数
        plt.plot(x, y)
        plt.title('Gaussian Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()


    def _test_ratio_reward():
        x = np.linspace(0.1, 5, 1000)
        y = [get_ratio_reward(i) for i in x]

        # 绘制高斯函数
        plt.plot(x, y)
        plt.title('Gaussian Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()


    _test_ratio_reward()
