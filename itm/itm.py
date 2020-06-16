"""Attenuation in the Longley-Rice Irregular Terrain Model"""

import numba as nb
import numpy as np
from numpy import exp, log, log10, sqrt

from .constants import f0, Ds, Hs, z1, N1, Z0, γa, THIRD, INF
from .diffraction import diffraction_attenuation
from .los import los_coefficients
from .scatter import scatter_attenuation
from ..utils.misc import dBW_to_watts, watts_to_dBW


@nb.njit("float64(float64, float64)")
def reduced_refractivity(N0, zs):
    """Reduced surface refractivity.

    Parameters
    ----------
    N0 : float
        surface refractivity at sea-level, in N-units
    zs : float
        height above sea-level, in metres

    Returns
    -------
    Ns : float
        reduced surface refractivity, in N-units
    """
    return N0 * exp(-zs / z1)


@nb.njit("float64(float64)")
def effective_curvature(Ns):
    """Effective curvature of the Earth.

    Parameters
    ----------
    Ns : float
        reduced surface refractivity, in N-units

    Returns
    -------
    γe : float
        effective curvature of the Earth, in m^-1
    """
    return γa * (1.0 - 0.04665 * exp(Ns / N1))


@nb.njit("complex128(float64, float64, float64, bool_)")
def surface_impedance(frequency, εr, σ, horizontal):
    """Complex surface impedance.

    Parameters
    ----------
    frequency : float
        frequency, in MHz
    εr : float
        relative permittivity (the dielectric constant) of the ground
    σ : float
        conductivity of the ground, in S/m
    horizontal : bool
        polarization is horizontally-polarized

    Returns
    -------
        complex surface impedance (unitless)
    """
    k = frequency / f0
    εrp = εr + 1j * Z0 * σ / k
    if horizontal:
        return sqrt(εrp - 1.0)
    return sqrt(εrp - 1.0) / εrp


@nb.njit(
    nb.float64(
        nb.float64,  # distance
        nb.float64,  # frequency
        nb.float64,  # tx_height
        nb.float64,  # rx_height
        nb.types.Tuple(
            (nb.float64, nb.float64, nb.float64, nb.float64)
        ),  # horizons (θe1, θe2, dL1, dL2)
        nb.float64,  # Δh
        nb.float64,  # Ns
        nb.complex128,  # Zgnd
        nb.float64,  # γe
    )
)
def attenuation(distance, frequency, tx_height, rx_height, horizons, Δh, Ns, Zgnd, γe):
    """Attenuation predicted by the Longley-Rice Irregular Terrain Model.

    Parameters
    ----------
    distance : float
        the distance between transmitter and receiver, in metres
    frequency : float
        the frequency of transmission, in MHz
    tx_height, rx_height : float
        transmitter and receiver heights, in metres
    horizons : Tuple[float, float, float, float]
        radio horizon elevation angles and distances: (θe1, θe2, dL1, dL2)
    Δh : float
        terrain irregularity, in metres
    Ns : float
        reduced surface refractivity, in N-units
    Zgnd : float
        complex surface impedance (unitless)
    γe : float
        effective curvature of the Earth, in m^-1

    Returns
    -------
    float
        predicted attenuation, *including* free-space loss, in dB
    """
    θe1, θe2, dL1, dL2 = horizons
    he1, he2 = tx_height, rx_height  # effective heights
    k = frequency / f0

    dLs1 = sqrt(2.0 * he1 / γe)
    dLs2 = sqrt(2.0 * he2 / γe)
    dLs = dLs1 + dLs2
    dL = dL1 + dL2

    θe = max(θe1 + θe2, -dL * γe)

    X_ae = (k * (γe ** 2)) ** -THIRD
    d3 = max(dLs, dL + 1.3787 * X_ae)
    d4 = d3 + 2.7574 * X_ae

    # print((d3, k, he1, he2, tx_height, rx_height, dL1, dL2, dLs, Δh, θe, γe, Zgnd))
    A3 = diffraction_attenuation(
        d3, k, he1, he2, tx_height, rx_height, dL1, dL2, dLs, Δh, θe, γe, Zgnd
    )
    A4 = diffraction_attenuation(
        d4, k, he1, he2, tx_height, rx_height, dL1, dL2, dLs, Δh, θe, γe, Zgnd
    )

    m_d = (A4 - A3) / (d4 - d3)
    A_ed = A3 - m_d * d3

    d5 = dL + Ds
    d6 = d5 + Ds

    # print((d5, k, he1, he2, dL1, dL2, Ns, θe1, θe2, θe, γe))
    A5 = scatter_attenuation(d5, k, he1, he2, dL1, dL2, Ns, θe1, θe2, θe, γe)
    A6 = scatter_attenuation(d6, k, he1, he2, dL1, dL2, Ns, θe1, θe2, θe, γe)

    if np.isinf(A5) or np.isinf(A6):
        d_x = INF
    else:
        if A5 > 15:
            A5 = A6

        m_s = (A6 - A5) / Ds
        d_x = max(dLs, dL + X_ae * log10(k * Hs), (A5 - A_ed - m_s * d6) / (m_d - m_s))
        A_es = A_ed + (m_d - m_s) * d_x

    if distance < dLs:  # line-of-sight
        A_el, K1, K2 = los_coefficients(A_ed, m_d, k, he1, he2, dL, dLs, Δh, Zgnd)
        # TODO: log or log10 here?
        A_ref = max(0, A_el + K1 * distance + K2 * log(distance / dLs))
        # zone = 1
    elif distance < d_x:  # diffraction
        A_ref = A_ed + m_d * distance
        # zone = 2
    else:  # scatter
        A_ref = A_es + m_s * distance
        # zone = 3

    A_fspl = 20 * log10(distance) + 20 * log10(frequency) - 27.55

    return A_fspl + A_ref  # , zone
