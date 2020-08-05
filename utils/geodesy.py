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


__all__ = [
    'azimuth',
    'destination',
    'distance',
    'geospace',
    'haversine_distance'
]


@nb.vectorize('float64(float64, float64)')
def arccos2(y: float, x: float) -> float:
    """Arcosine, respecting quadrant."""
    val = arccos(max(min(y / x, 1), -1))
    if x <= 0:
        return val + π
    return val


@nb.vectorize('float64(float64, float64, float64, float64)')
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
    Δλ = λ2 - λ1
    y = sin(Δλ)
    x = cos(φ1) * tan(φ2) - sin(φ1) * cos(Δλ)

    # under this convention, azimuth increases to the east
    α = arctan2(y, x)
    α = α % TWOPI

    return α


@nb.vectorize('float64(float64, float64, float64, float64)')
def distance(φ1: float, λ1: float, φ2: float, λ2: float) -> float:
    """Distance between two points on a globe.

    This method uses the spherical law of cosines.

    This is one-half of the solution to the "second" (or "reverse") geodetic
    problem, assuming a spherical Earth. `azimuth` provides the other half.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    φ2, λ2 : float
        final latiutude, longitude

    Returns
    -------
    float
        distance, in metres
    """
    Δλ = λ2 - λ1
    β = arccos(sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ))
    return EARTH_RADIUS * β


@nb.vectorize('float64(float64, float64, float64, float64)')
def haversine_distance(φ1: float, λ1: float, φ2: float, λ2: float) -> float:
    """Distance between two points on a globe, using the haversine formula.

    This is one-half of the solution to the "second" (or "reverse") geodetic
    problem, assuming a spherical Earth. `azimuth` provides the other half.

    Parameters
    ----------
    φ1, λ1 : float
        initial latitude, longitude
    φ2, λ2 : float
        final latiutude, longitude

    Returns
    -------
    float
        distance, in metres
    """
    Δλ = λ2 - λ1
    Δφ = φ2 - φ1
    β = 2 * arcsin(sqrt(sin(Δφ / 2)**2 + cos(φ1) * cos(φ2) * sin(Δλ / 2)**2))
    return EARTH_RADIUS * β


@nb.njit('Tuple((float64, float64))(float64, float64, float64, float64)')
def destination(φ1: float, λ1: float, α: float, d: float):
    """Destination point, provided origin, bearing, and distance.

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
    φ2 = arcsin(sin(φ1) * cos(β) + cos(α) * sin(β) * cos(φ1))
    λ2 = λ1 + arctan2(
        sin(α) * sin(β) * cos(φ1),
        cos(β) - sin(φ1) * sin(φ2),
    )
    return φ2, λ2


@nb.njit('Tuple((float64[:], float64[:]))(float64, float64, float64, float64, int64)')
def geospace(φ1: float, λ1: float, φ2: float, λ2: float, num: int):
    """Evenly-spaced coordinates between two positions on a spherical globe.

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
    dist = np.linspace(0.0, distance(φ1, λ1, φ2, λ2), num)
    β = dist / EARTH_RADIUS
    cos_β = cos(β)

    α = azimuth(φ1, λ1, φ2, λ2)

    sin_φ1 = sin(φ1)
    cos_φ1 = cos(φ1)

    sin_φ = sin_φ1 * cos_β + cos(α) * sin(β) * cos_φ1
    φ = arcsin(sin_φ)

    y = cos_β - sin_φ1 * sin_φ
    x = cos_φ1 * cos(φ)

    if α >= π:
        λ = λ1 - arccos2(y, x)
    else:
        λ = λ1 + arccos2(y, x)

    λ[λ < -π] += TWOPI
    λ[λ > π] -= TWOPI

    return φ, λ
