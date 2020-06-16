"""Scattering methods for the Longley-Rice Irregular Terrain Model."""

import numba as nb
from numpy import exp, log10

from .constants import Hs, SQRT2


H01_A = (24, 45, 68, 80, 105)
H01_B = (25, 80, 177, 395, 705)


@nb.njit("float64(float64, float64)")
def Fθd(d, N_s):
    if d < 10e3:
        F0 = 133.4 + 0.332e-3 * d - 10 * log10(d)
    if d < 70e3:
        F0 = 104.6 + 0.212e-3 * d - 2.5 * log10(d)
    else:
        F0 = 71.8 + 0.157e-3 * d + 5 * log10(d)
    return F0 - 0.2 * (N_s - 301) * exp(-d / 40e3)


@nb.njit("float64(float64, int64)")
def H01(r, j):
    return 10 * log10(1.0 + H01_A[j - 1] / r ** 2 + H01_B[j - 1] / r ** 4)


@nb.njit(
    "float64(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)"
)
def scatter_attenuation(s, k, he1, he2, dL1, dL2, N_s, θe1, θe2, θe, γe):
    """Tropospheric scatter attenuation."""
    # find crossover point
    d_s = s - dL1 - dL2  # distance between horizons; (4.64)
    s_s = (dL2 + 0.5 * d_s) / (dL1 + 0.5 * d_s)  # asymmetry factor; (4.65)

    θ = θe + γe * s
    θp = θe1 + θe2 + γe * s
    r1 = 2 * k * θp * he1
    r2 = 2 * k * θp * he2

    z0 = s_s * s * θp / (1 + s_s) ** 2  # height of crossover

    # scatter efficiency
    ηs = (
        z0
        * (1 + (0.031 - 2.32e-3 * N_s + 5.67e-6 * N_s ** 2) * exp(-((z0 / 8e3) ** 6)))
        / 1.756e3
    )

    assert ηs > 0, 'computed scatter efficiency is unphysical'

    # frequency gain
    if ηs < 1:
        H010 = 10 * log10(
            ((1 + SQRT2 / r1) ** 2)
            * ((1 + SQRT2 / r2) ** 2)
            * (r1 + r2)
            / (r1 + r2 + 2 * SQRT2)
        )

        # interpolate between the value of H_00 at j = 0 and j = 1
        H00 = H010 + 0.5 * ηs * (H01(r1, 1) + H01(r2, 1)) - ηs * H010
    elif ηs >= 5:
        H00 = 0.5 * (H01(r1, 5) + H01(r2, 5))
    else:
        # interpolate between the floor and ceiling values of ηs
        iηs = int(ηs)
        H00 = 0.5 * (
            (iηs + 1 - ηs) * (H01(r1, iηs) + H01(r2, iηs))
            + (ηs - iηs) * (H01(r1, iηs + 1) + H01(r2, iηs + 1))
        )

    ΔH0 = 6 * (0.6 - log10(ηs)) * log10(s_s) * log10(r2 / (s_s * r1))
    H0 = H00 + ΔH0

    # overall scatter attenuation
    return 10.0 * log10(k * Hs * (θ ** 4)) + Fθd(θ * s, N_s) + H0
