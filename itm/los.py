"""Line-of-sight methods for the Longley-Rice Irregular Terrain Model."""

import numba as nb
from numpy import exp, log, log10, sqrt
from numpy import pi as π

from .constants import D1, D2
from .diffraction import σhs


@nb.njit(
    "float64(float64, float64, float64, float64, float64, float64, float64, float64, complex128)"
)
def los_attenuation(s, Aed, md, k, he1, he2, dLs, Δh, Zgnd):
    """Line-of-sight attenuation.

    The LOS attenuation is a convex combination of
    plane-earth fields and diffracted fields.
    """
    # extended diffraction attenuation
    A_d = Aed + md * s

    # two-way attenuation
    sinψ = (he1 + he2) / sqrt(s ** 2 + (he1 + he2) ** 2)
    Rep = exp(-k * σhs(s, Δh) * sinψ) * (sinψ - Zgnd) / (sinψ + Zgnd)

    if abs(Rep) >= max(0.5, sqrt(sinψ)):
        Re = Rep
    else:
        Re = Rep * sqrt(sinψ) / abs(Rep)

    δp = 2 * k * he1 * he2 / s
    if δp <= 0.5 * π:
        δ = δp
    else:
        δ = π - ((0.5 * π) ** 2) / δp

    A_t = -20.0 * log10(abs(1 + Re * exp(1j * δ)))

    # weighting factor
    w = 1 / (1 + D1 * k * Δh / max(D2, dLs))

    # overall line-of-sight attenuation
    return (1 - w) * A_d + w * A_t


@nb.njit(
    "Tuple((float64, float64, float64))(float64, float64, float64, float64, float64, float64, float64, float64, complex128)"
)
def los_coefficients(Aed, md, k, he1, he2, dL, dLs, Δh, Zgnd):
    """Convex combination coefficients for line-of-sight attenuation.

    We want an attenuation curve of the form:
        A(d) = Ael + K1*d + K2*log(d/dLs)

    Under the constraint that A(d0) = A0, A(d1) = A1, and A(d2) = A2.
    However, we also require that K1 and K2 are non-negative, which
    overconstrains the problem. We may therefore have to abandon the
    constraint of A0 or A1 or both.
    """
    d2 = dLs
    A2 = Aed + md * d2

    if Aed >= 0:
        gc1 = True  # we're in "general case 1"
        d0 = min(0.5 * dL, 1.908 * k * he1 * he2)
        d1 = 0.75 * d0 + 0.25 * dL
    else:
        gc1 = False  # second general case
        d0 = 1.908 * k * he1 * he2
        d1 = max(-Aed / md, 0.25 * dL)

    if gc1 or (d0 < d1):
        A0 = los_attenuation(d0, Aed, md, k, he1, he2, dLs, Δh, Zgnd)
        A1 = los_attenuation(d1, Aed, md, k, he1, he2, dLs, Δh, Zgnd)

        K2p = ((d2 - d0) * (A1 - A0) - (d1 - d0) * (A2 - A0)) / (
            (d2 - d0) * log(d1 / d0) - (d1 - d0) * log(d2 / d0)
        )
        K2p = max(0, K2p)

        if gc1 or (K2p > 0):
            K1p = (A2 - A0 - K2p * log(d2 / d0)) / (d2 - d0)
            if K1p >= 0:
                K1 = K1p
                K2 = K2p
            else:
                K2pp = (A2 - A0) / log(d2 / d0)
                if K2pp >= 0:
                    K1 = 0
                    K2 = K2pp
                else:
                    K1 = md
                    K2 = 0

    if (not gc1) and ((d0 >= d1) or (K2p == 0)):
        K1pp = (A2 - A1) / (d2 - d1)
        if K1pp > 0:
            K1 = K1pp
            K2 = 0
        else:
            K1 = md
            K2 = 0.0

    Ael = A2 - K1 * d2

    return Ael, K1, K2
