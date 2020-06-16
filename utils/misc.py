"""Miscellaneous utilities."""
import numba as nb
from numpy import log10


def sex_to_dec(s):
    """Convert sexigisimal-formatted string/integer to a decimal degree value."""
    s = format(int(s), '07d')
    degree, arcmin, arcsec = s[:3], s[3:5], s[5:]
    return float(degree) + float(arcmin) / 60 + float(arcsec) / 3600


def dec_to_sex(d):
    """Convert decimal degree value to sexigisimal-formatted string."""
    degrees, minutes = divmod(60 * d, 60)
    minutes, seconds = divmod(60 * minutes, 60)
    return f'{degrees:3d}{minutes:02d}{seconds:02d}'


@nb.vectorize("float64(float64)")
def watts_to_dBW(power):
    """Convert power from watts to dBW."""
    return 10.0 * log10(power)


@nb.vectorize("float64(float64)")
def dBW_to_watts(dbw):
    """Convert log-power from dBW to watts."""
    return 10 ** (dbw / 10)
