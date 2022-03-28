import math
import numpy as np


def _clip(value, _min, _max):
    return min(_max, max(_min, value))


def _binary_embed(value, min_value, max_value, length, power=1.0):
    assert min_value <= value <= max_value, "value: %.2f, min_value: %.2f, max_value: %.2f" % (
        value, min_value, max_value)
    embed = [0.0] * length
    scale = int(math.pow(value - min_value, power) / math.pow(
        max_value - min_value + 1e-8, power) * math.pow(2, length))
    while scale > 0:
        if scale % 2 == 1:
            embed[length - 1] = 1.0
        scale //= 2
        length -= 1
    return embed


def _sqrt_embed(value, min_value, max_value, length, monotonic=True):
    """
        p = sqrt(value - min_value) / sqrt(value / max_value)
        embed[ceil(p / (1 / length))] = 1.0
    """
    assert min_value <= value <= max_value, "value: %.2f, min_value: %.2f, max_value: %.2f" % (
        value, min_value, max_value)
    vid = math.sqrt(value - min_value)
    max_vid = math.sqrt(max_value - min_value)
    if max_vid > 0:
        p = vid / max_vid
    else:
        p = 0.0
    embed = [0.0 for _ in range(length)]
    if p > 0:
        idx = math.ceil(p / (1.0 / length)) - 1
        if monotonic:
            for i in range(0, idx):
                embed[i] = 1.0
            embed[idx] = p * length - idx
        else:
            embed[idx] = 1.0
    return embed


def _power_embed(value, min_value, max_value, length, power=0.5, monotonic=True):
    assert min_value <= value <= max_value, "value: %.2f, min_value: %.2f, max_value: %.2f" % (
        value, min_value, max_value)
    vid = math.pow(value - min_value, power)
    max_vid = math.pow(max_value - min_value, power)
    if max_vid > 0:
        p = vid / max_vid
    else:
        p = 0.0
    embed = [0.0 for _ in range(length)]
    if p > 0:
        idx = math.ceil(p / (1.0 / length)) - 1
        if monotonic:
            for i in range(0, idx):
                embed[i] = 1.0
            embed[idx] = p * length - idx
        else:
            embed[idx] = 1.0
    return embed


def _soft_rot_embed(x, y, length, base=0.0):
    theta = math.atan2(y, x)
    relative_theta = theta - base

    soft_onehot = list(np.cos(np.linspace(0, np.pi * 2, length, False) - relative_theta))
    return soft_onehot
