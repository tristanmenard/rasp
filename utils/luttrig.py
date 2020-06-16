"""Lookup table trigonometry."""

import numba as nb
import numpy as np

from ..constants import HALFPI, TWOPI

LUT_SIZE = 2**16
MAX_INDEX = LUT_SIZE - 1

LUT_ANGLES = TWOPI * np.linspace(0, 1, num=LUT_SIZE, endpoint=False, dtype=float)
LUT_SIN = np.sin(LUT_ANGLES)

LUT_ARCSIN = np.arcsin(np.linspace(-1, 1, num=LUT_SIZE + 1, endpoint=True, dtype=float))


@nb.vectorize('float64(float64)')
def sin(x):
    index_float = LUT_SIZE * x / TWOPI
    index_int = int(index_float)
    weight = index_float - index_int

    y1 = LUT_SIN[index_int & MAX_INDEX]
    y2 = LUT_SIN[(index_int + 1) & MAX_INDEX]

    return y1 + (y2 - y1) * weight


@nb.vectorize('float64(float64)')
def cos(x):
    return sin(x + HALFPI)


@nb.vectorize('float64(float64)')
def tan(x):
    return sin(x) / cos(x)


@nb.vectorize('float64(float64)')
def arcsin(x):
    if x >= 1:
        return HALFPI

    index_float = LUT_SIZE * (x + 1) / 2
    index_int = int(index_float)
    weight = index_float - index_int

    y1 = LUT_ARCSIN[index_int]
    y2 = LUT_ARCSIN[index_int + 1]

    return y1 + (y2 - y1) * weight


@nb.vectorize('float64(float64)')
def arccos(x):
    return HALFPI - arcsin(x)
