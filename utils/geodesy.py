"""Geodetic utilities.

The methods here are mostly simple functions converting
distances and bearings to/from latitudes and longitudes.
"""

import numba as nb
import numpy as np

from numpy import cos, sin, tan
from numpy import arccos, arcsin, arctan2
from numpy import sqrt

from ..constants import EARTH_RADIUS
from ..constants import π, TWOPI
from . import luttrig


__all__ = [
    'azimuth',
    'destination',
    'distance',
    'geospace',
    'waypoints',
]


@nb.vectorize('float64(float64, float64)')
def arccos2(x: float, y: float) -> float:
    """Arcosine, respecting quadrant."""
    val = arccos(max(min(x / y, 1), -1))
    if y <= 0:
        return val + π
    return val


@nb.vectorize('float64(float64, float64, float64, float64)')
def _azimuth(φ1: float, λ1: float, φ2: float, λ2: float) -> float:
    Δλ = λ2 - λ1
    y = sin(Δλ)
    x = cos(φ1) * tan(φ2) - sin(φ1) * cos(Δλ)

    # under this convention, azimuth increases to the east
    α = arctan2(y, x)
    α = α % TWOPI

    return α


def azimuth(φ1: float, λ1: float, φ2: float, λ2: float) -> float:
    """Azimuth between two points on a globe.

    The North-based azimuth convention is used: az(N) = 0, az(E) = π/2, etc.,
    and points from (φ1, λ1) to (φ2, λ2). I.e. the apparent azimuth of the
    latter, as viewed by the former.

    This is one-half of the solution to the "second" (or "reverse") geodetic
    problem, assuming a spherical Earth. `distance` provides the other half.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    φ2, λ2 : float
        final latiutude, longitude

    Returns
    -------
    float
        azimuth, in radians
    """
    return _azimuth(φ1, λ1, φ2, λ2)


@nb.njit
def distance(φ1: float, λ1: float, φ2: float, λ2: float, haversine: bool = False) -> float:
    """Distance between two points on a globe.

    This is one-half of the solution to the "second" (or "reverse") geodetic
    problem, assuming a spherical Earth. `azimuth` provides the other half.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    φ2, λ2 : float
        final latiutude, longitude
    haversine : bool
        use the haversine formula

    Returns
    -------
    float
        distance, in metres
    """
    Δλ = λ2 - λ1

    if haversine:
        Δφ = φ2 - φ1
        β = 2 * arcsin(sqrt(sin(Δφ / 2)**2 + cos(φ1) * cos(φ2) * sin(Δλ / 2)**2))
    else:
        β = arccos(sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ))

    return EARTH_RADIUS * β


def destination(φ1: float, λ1: float, α: float, d: float):
    """Provided an origin, bearing, and distance, compute the destination.

    This is the solution to the "first" (or "forward") geodetic problem,
    assuming a spherical Earth.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    α : float
        azimuthual bearing (north-based)
    d : float
        distance, in metres

    Returns
    -------
    φ2, λ2 : float
        final latiutude, longitude
    """
    β = d / EARTH_RADIUS
    φ2 = arcsin(sin(φ1) * cos(β) + cos(φ1) * sin(β) * cos(α))
    λ2 = λ1 + arctan2(
        sin(α) * sin(β) * cos(φ1),
        cos(β) - sin(φ1) * sin(φ1),
    )
    return φ2, λ2


@nb.njit('Tuple((float64[:], float64[:]))(float64, float64, float64, float64, int64, bool_)')
def _waypoints(φ1: float, λ1: float, φ2: float, λ2: float, num: int, lut: bool):
    Δλ = λ2 - λ1
    α = _azimuth(φ1, λ1, φ2, λ2)

    if lut:
        cos = luttrig.cos
        sin = luttrig.sin
        arccos = luttrig.arccos
        arcsin = luttrig.arcsin

    sinφ1 = sin(φ1)
    cosφ1 = cos(φ1)

    β = np.linspace(0.0, arccos(sinφ1 * sin(φ2) + cosφ1 * cos(φ2) * cos(Δλ)), num)
    sinβ = sin(β)
    cosβ = cos(β)

    sinφ = sinφ1 * cosβ + cosφ1 * sinβ * cos(α)
    φ = arcsin(sinφ)

    y = sin(α) * sinβ * cosφ1
    x = cosβ - sinφ1 * sinφ

    λ = arctan2(y, x) + λ1
    λ[λ > π] -= TWOPI

    return φ, λ


@nb.njit('Tuple((float64[:], float64[:]))(float64, float64, float64, float64, int64)')
def _waypoints_flatearth(φ1: float, λ1: float, φ2: float, λ2: float, num: int):
    return np.linspace(φ1, φ2, num), np.linspace(λ1, λ2, num)


def waypoints(φ1: float, λ1: float, φ2: float, λ2: float, num: int, lut: bool = False, flatearth: bool = False):
    """Evenly-spaced waypoints between two positions on a spherical globe.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    φ2, λ2 : float
        final latiutude, longitude
    num : int
        number of desired samples
    lut : bool
        use a lookup table for the trig functions; offers moderate speedup at expense of some precision
    flatearth : bool
        ignore Earth's curvature and compute evenly-spaced lat/lon values

    Returns
    -------
    ndarray
        latitude, longitude values
    """
    if flatearth:
        return _waypoints_flatearth(φ1, λ1, φ2, λ2, num)

    return _waypoints(φ1, λ1, φ2, λ2, num, lut)


@nb.njit
def geospace(φ1: float, λ1: float, φ2: float, λ2: float, num: int):
    """Evenly-spaced coordinates between two positions on a globe.

    An analogue to numpy's linspace/logspace.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    φ2, λ2 : float
        final latiutude, longitude
    num : int
        number of desired samples

    Returns
    -------
    ndarray
        latitude, longitude values
    """
    dist = np.linspace(0.0, distance(φ1, λ1, φ2, λ2, False, False), num)

    α = azimuth(φ1, λ1, φ2, λ2)
    cos_α = cos(α)
    sin_φ1 = sin(φ1)
    cos_φ1 = cos(φ1)

    β = dist / EARTH_RADIUS
    sin_φ = sin_φ1 * cos(β) + cos_α * sin(β) * cos_φ1
    φ = arcsin(sin_φ)
    y = cos(β) - sin_φ1 * sin_φ
    x = cos_φ1 * cos(φ)

    if α <= π:
        λ = λ1 - arccos2(y, x)
    else:
        λ = λ1 + arccos2(y, x)

    λ[λ < 0] += TWOPI
    λ[λ > π] -= TWOPI

    return φ, λ
