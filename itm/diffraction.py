"""Diffraction methods for the Longley-Rice Irregular Terrain Model."""

import numba as nb
import numpy as np
from numpy import exp, log10, sqrt
from numpy import pi as π

from .constants import A, C, C1, D, H, α, THIRD


@nb.njit("float64(float64, float64)")
def Δhs(s: float, Δh: float) -> float:
    """Reduced surface irregularity."""
    return Δh * (1 - 0.8 * exp(-s / D))


@nb.njit("float64(float64, float64)")
def σhs(s: float, Δh: float) -> float:
    dhs = Δhs(s, Δh)
    return 0.78 * dhs * exp(-((dhs / H) ** 0.25))


@nb.njit("float64(float64)")
def Fn(v: float) -> float:
    """Approximation of the Fresnel integral, in dB."""
    if v <= 2.40:
        return 6.02 + 9.11 * v - 1.27 * v * v
    return 12.953 + 20.0 * log10(v)


@nb.njit("float64(float64, complex128)")
def F(x: float, K: float) -> float:
    """Approximate height-gain over a smooth Earth."""
    if x <= 200.0:
        # F = F2
        if (abs(K) < 1e-5) or (x * (-log10(abs(K))) ** 3.0 > 450.0):
            # F = F2 = F1
            return 40.0 * log10(max(x, 1)) - 117.0
        return 2.5e-5 * x * x / abs(K) + 20.0 * log10(abs(K)) - 15.0

    G = 0.05751 * x - 10.0 * log10(x)

    if x >= 2000.0:
        return G

    F1 = 40.0 * log10(max(x, 1)) - 117.0
    return G + 0.0134 * x * exp(-x / 200.0) * (F1 - G)


@nb.njit(
    "float64(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, complex128)"
)
def diffraction_attenuation(s, k, he1, he2, hg1, hg2, dL1, dL2, dLs, Δh, θe, γe, Zgnd):
    """Diffraction attenuation.

    The "diffraction attenuation" is a convex combination of the attenuation
    associated with smooth-earth diffraction and knife-edge diffraction. An
    additional (empirical) attenuation is included to account for "clutter" along
    the path.
    """
    # rounded Earth attenuation
    θ = θe + s * γe
    dL = dL1 + dL2
    γj = np.array([θ / (s - dL), 2 * he1 / (dL1 ** 2), 2 * he2 / (dL2 ** 2)])
    αj = (k / γj) ** THIRD
    Kj = 1.0 / (1j * αj * Zgnd)

    x1 = A * (1.607 - abs(Kj[1])) * αj[1] * γj[1] * dL1
    x2 = A * (1.607 - abs(Kj[2])) * αj[2] * γj[2] * dL2
    x0 = A * (1.607 - abs(Kj[0])) * αj[0] * θ + x1 + x2
    G = 0.05751 * x0 - 10.0 * log10(x0)

    Ar = G - F(x1, Kj[1]) - F(x2, Kj[2]) - C1

    # knife-edge diffraction
    v1 = 0.5 * θ * sqrt(k * dL1 * (s - dL) / (π * (s - dL + dL1)))
    v2 = 0.5 * θ * sqrt(k * dL2 * (s - dL) / (π * (s - dL + dL2)))
    Ak = Fn(v1) + Fn(v2)

    # clutter attenuation
    Afo = min(15.0, 5.0 * log10(1.0 + α * k * hg1 * hg2 * σhs(dLs, Δh)))

    # weighting factor
    Q = min(k * Δhs(s, Δh) / (2 * π), 1000.0) * sqrt((he1 * he2 + C) / (hg1 * hg2 + C))
    Q += (dL + θe / γe) / s
    w = 1.0 / (1.0 + 0.1 * sqrt(Q))

    # overall diffraction attenuation
    return (1.0 - w) * Ak + w * Ar + Afo
