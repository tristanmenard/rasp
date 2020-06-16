"""One-dimensional profiles of digital elevation profiles."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from .itm import itm


__all__ = ["Profile"]


@nb.experimental.jitclass(
    (
        ("length", nb.int64),
        ("elevation", nb.float64[:]),
        ("latitude", nb.float64[:]),
        ("longitude", nb.float64[:]),
        ("distance", nb.float64),
    )
)
class JITProfile:
    """Numba-compiled class for terrain profiles."""

    def __init__(self, length, elevation, latitude, longitude, distance):
        self.length = length
        self.elevation = elevation
        self.latitude = latitude
        self.longitude = longitude
        self.distance = distance

    def horizons(self, tx_height, rx_height, γe):
        """Compute LR-ITM radio horizons.

        Parameters
        ----------
        tx_height : float
            height of the transmitting antenna
        rx_height : float
            height of the receiving antenna
        γe : float
            effective curvature of the Earth
        Returns
        -------
        float : θe1, θe2
            elevation angles of the first and second horizon
        float : dL1, dL2
            distances to the first and second horizon
        """
        xi = self.distance / (self.length - 1)
        za = self.elevation[0] + tx_height
        zb = self.elevation[-1] + rx_height

        q = 0.5 * γe * self.distance
        θe2 = (zb - za) / self.distance
        θe1 = θe2 - q
        θe2 = -θe2 - q

        dL1 = self.distance
        dL2 = self.distance

        if self.length > 2:
            sa = 0.0
            sb = self.distance

            wq = True

            for i in range(1, self.length):
                sa += xi
                sb -= xi
                q = self.elevation[i] - (0.5 * γe * sa + θe1) * sa - za

                if q > 0:
                    θe1 += q / sa
                    dL1 = sa
                    wq = False

                if not wq:
                    q = self.elevation[i] - (0.5 * γe * sb + θe2) * sb - zb
                    if q > 0:
                        θe2 += q / sb
                        dL2 = sb

        return θe1, θe2, dL1, dL2

    def detrend(self):
        """Subtract off a linear trendline.

        This code looks ugly because it's been optimized under
        numba's nopython compilation.
        """
        n = self.length

        sum_x = n * (n - 1) / 2
        sum_x2 = (n - 1) * n * (2 * n - 1) / 6

        denom = n * sum_x2 - sum_x ** 2

        if denom == 0:
            return self.elevation

        sum_xy = 0.0
        sum_y = 0.0

        for i in range(n):
            sum_xy += i * self.elevation[i]
            sum_y += self.elevation[i]

        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y * sum_x2 - sum_x * sum_xy) / denom

        detrended = np.empty_like(self.elevation)

        for i in range(n):
            detrended[i] = self.elevation[i] - (m * i + b)

        return detrended

    def detrend_cov(self):
        """Find and subtract off a linear trendline."""
        n = self.length
        x = np.arange(n, dtype=float)
        C = np.cov(x, self.elevation, bias=1)
        m = C[0, 1] / C[0, 0]
        b = np.mean(self.elevation) - m * np.mean(x)
        return self.elevation - (m * x + b)

    def irregularity(self):
        """Compute the profile's irregularity (Δh in the LR-ITM).

        The terrain irregularity is the difference between the 90th and 10th
        elevation percentiles, after a linear trendline has been subtracted.
        """
        detrended = self.detrend()

        # partition the elevation at the 10% and 90% quantiles
        ind_10, ind_90 = int(0.1 * self.length), int(0.9 * self.length)
        detrended = np.partition(detrended, (ind_10, ind_90))

        return detrended[ind_90] - detrended[ind_10]

    def attenuation(self, frequency, tx_height, rx_height, Ns, Zgnd, γe):
        """Longley-Rice Irregular Terrain Model attenuation."""
        horizons = self.horizons(tx_height, rx_height, γe)
        Δh = self.irregularity()
        return itm.attenuation(
            self.distance, frequency, tx_height, rx_height, horizons, Δh, Ns, Zgnd, γe
        )


@dataclass
class Profile:
    """Terrain profile dataclass."""

    length: int

    elevation: np.ndarray = field(repr=False)
    latitude: np.ndarray = field(repr=False)
    longitude: np.ndarray = field(repr=False)
    distance: float = field(repr=False)

    jit: JITProfile = field(repr=False, default=None)

    def __post_init__(self):
        if self.jit is None:
            self.jit = JITProfile(
                self.length,
                self.elevation,
                self.latitude,
                self.longitude,
                self.distance,
            )

    def horizons(self, tx_height, rx_height, γe=None, Ns=250.0):
        """Compute the LR-ITM radio horizons.

        Parameters
        ----------
        tx_height : float
            height of the transmitting antenna
        rx_height : float
            height of the receiving antenna
        γe : float
            effective curvature of the Earth; computed using Ns, if None
        Ns : float
            effective surface refractivity, in N-units

        Returns
        -------
        θe1, θe2 : float
            elevation angles of the first and second horizon
        dL1, dL2 : float
            distances to the first and second horizon
        """
        if γe is None:
            γe = itm.effective_curvature(Ns)

        return self.jit.horizons(tx_height, rx_height, γe)

    def irregularity(self):
        """Compute the profile's irregularity (Δh in the LR-ITM).

        The terrain irregularity is the difference between the 90th and 10th
        elevation percentiles, after a linear trendline has been subtracted.

        Returns
        -------
        Δh : float
        """
        return self.jit.irregularity()

    def attenuation(
        self,
        frequency,
        tx_height,
        rx_height,
        Ns=250.0,
        Zgnd=None,
        εr=15.0,
        σ=0.005,
        γe=None,
        polarization="horizontal",
    ):
        """Longley-Rice Irregular Terrain Model attenuation."""
        if γe is None:
            γe = itm.effective_curvature(Ns)

        if Zgnd is None:
            if polarization.lower().startswith("h"):  # horizontally-polarized
                Zgnd = itm.surface_impedance(frequency, εr, σ, True)
            elif polarization.lower().startswith("v"):  # vertically-polarized
                Zgnd = itm.surface_impedance(frequency, εr, σ, False)
            elif polarization.lower().startswith("u"):  # unpolarized
                Zgnd = (
                    itm.surface_impedance(frequency, εr, σ, True)
                    + itm.surface_impedance(frequency, εr, σ, False)
                ) / 2
            else:
                raise ValueError(
                    f'`polarization` value "{polarization}" not understood; valid '
                    'options are: "horizontal", "vertical", and "unpolarized"'
                )

        return self.jit.attenuation(frequency, tx_height, rx_height, Ns, Zgnd, γe)
