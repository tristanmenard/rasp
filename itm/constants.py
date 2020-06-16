"""Constants used in the Longley-Rice Irregular Terrain Model."""

import numpy as np

from ..constants import EARTH_RADIUS


INF = oo = np.inf
SQRT2 = np.sqrt(2)
THIRD = 1 / 3

# frequency per wavenumber
f0 = 47.70  # MHz*m

# surface refractivity scale-height
z1 = 9_460  # m

# Earth's actual curvature
γa = 1 / EARTH_RADIUS  # ~= 157e-9/m = 157 N-units/km

# effective curvature scale-refractivity
N1 = 179.3  # N-units

# conductivity-normalized surface transfer inductance (???)
Z0 = 376.62  # Ohms

# ???
D = 50e+3  # m
H = 16.0  # m

# diffraction
A = 151.03
C = 10.0
C1 = 20.0
α = 4.77e-4

# line-of-sight
D1 = 47.7  # m
D2 = 10e3  # m

# scattering
Ds = 200e3  # m
Hs = 47.7  # m
