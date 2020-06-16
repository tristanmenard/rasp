"""Digital elevation models."""

from typing import Tuple
from dataclasses import dataclass

import numba as nb
import numpy as np
from numpy import cos, sin, arccos, exp, sqrt

from .utils.geodesy import distance, _waypoints
from .constants import EARTH_RADIUS, HALFPI, DEG_PER_RAD, RAD_PER_DEG
from .profile import JITProfile, Profile
from .itm import itm


@nb.experimental.jitclass((
    ('shape', nb.types.Tuple((nb.int64, nb.int64))),
    ('dem', nb.float64[:, :]),
    ('latitude', nb.float64[:]),
    ('longitude', nb.float64[:]),
    ('resolution', nb.float64),
    ('latitude_span', nb.float64),
    ('longitude_span', nb.float64),
))
class JITElevation:
    """Numba-compiled class for digital elevation models."""

    def __init__(self, shape, dem, latitude, longitude):
        self.shape = (np.int64(shape[0]), np.int64(shape[1]))
        self.dem = dem
        self.latitude = latitude
        self.longitude = longitude

        dφ = latitude[1] - latitude[0]
        dλ = longitude[1] - longitude[0]
        self.resolution = min(abs(dφ), abs(dλ))

        self.latitude_span = latitude[-1] - latitude[0]
        self.longitude_span = longitude[-1] - longitude[0]

    def __call__(self, φ, λ):
        """Retrieve an elevation, assuming latitude and longitude data are sorted."""
        return self.get_elevation(φ, λ)

    def get_elevation(self, φ: float, λ: float) -> float:
        """Retrieve an elevation, assuming latitude and longitude data are sorted.

        Parameters
        ----------
        φ, λ : float
            latitude, longitude

        Returns
        -------
        float
            elevation value
        """
        lat_ind = np.searchsorted(self.latitude, φ)
        lon_ind = np.searchsorted(self.longitude, λ)

        if abs(self.latitude[lat_ind - 1] - φ) < abs(self.latitude[lat_ind] - φ):
            lat_ind -= 1

        if abs(self.longitude[lon_ind - 1] - λ) < abs(self.longitude[lon_ind] - λ):
            lon_ind -= 1

        return self.dem[lat_ind, lon_ind]

    def get_elevation_regular(self, φ: float, λ: float) -> float:
        """Retrieve an elevation, assuming latitude and longitude are regularly-spaced.

        Parameters
        ----------
        φ, λ : float
            latitude, longitude

        Returns
        -------
        float
            elevation value
        """
        if φ <= self.latitude[0]:
            lat_ind = 0
        elif φ >= self.latitude[-1]:
            lat_ind = -1
        else:
            lat, frac = divmod(self.shape[0] * (φ - self.latitude[0]), self.latitude_span)
            lat_ind = int(lat) + int(frac)

        if λ <= self.longitude[0]:
            lon = 0
        elif λ >= self.longitude[-1]:
            lon = -1
        else:
            lon, frac = divmod(self.shape[1] * (λ - self.longitude[0]), self.longitude_span)
            lon_ind = int(lon) + int(frac)

        return self.dem[lat_ind, lon_ind]

    def get_elevations(self, φ_arr: np.ndarray, λ_arr: np.ndarray) -> np.ndarray:
        elev = np.empty(φ_arr.size)

        for i in range(elev.size):
            elev[i] = self.get_elevation(φ_arr[i], λ_arr[i])

        return elev

    def get_elevations_regular(self, φ: np.ndarray, λ: np.ndarray) -> np.ndarray:
        elev = np.empty(φ.size)

        for i in range(elev.size):
            elev[i] = self.get_elevation_regular(φ[i], λ[i])

        return elev

    def get_profile(self, φ1: float, λ1: float, φ2: float, λ2: float, num: int, lut: bool, flatearth: bool):
        Δλ = λ2 - λ1
        β = arccos(sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ))

        # how many points do we want?
        if num == 0:
            num = max(int(β / self.resolution) + 1, 2)

        if flatearth:
            φ = np.linspace(φ1, φ2, num)
            λ = np.linspace(λ1, λ2, num)
        else:
            φ, λ = _waypoints(φ1, λ1, φ2, λ2, num, lut)

        return JITProfile(φ.size, self.get_elevations_regular(φ, λ), φ, λ, EARTH_RADIUS * β)

    def irregularity(self):
        dem_flat = self.dem.flatten()

        ind_10, ind_90 = int(0.1 * dem_flat.size), int(0.9 * dem_flat.size)
        dem_flat = np.partition(dem_flat, (ind_10, ind_90))

        return dem_flat[ind_90] - dem_flat[ind_10]

    def estimate_horizons(self, tx_height, rx_height, Bj, Δh, γe):
        if Bj <= 0.0:
            he1 = tx_height
            he2 = rx_height
        else:
            Bjp = (Bj - 1) * sin(HALFPI * min(tx_height / 5, 1)) + 1
            he1 = tx_height + Bjp * exp(-2 * tx_height / Δh)
            he2 = rx_height + Bjp * exp(-2 * rx_height / Δh)

        dLs1 = sqrt(2 * he1 / γe)
        dLs2 = sqrt(2 * he2 / γe)

        # H3 = 5m
        dL1 = dLs1 * exp(-0.07 * sqrt(Δh / max(he1, 5)))
        dL2 = dLs2 * exp(-0.07 * sqrt(Δh / max(he2, 5)))

        θe1 = (0.65 * Δh * (dLs1 / dL1 - 1) - 2 * he1) / dLs1
        θe2 = (0.65 * Δh * (dLs2 / dL2 - 1) - 2 * he2) / dLs2

        return θe1, θe2, dL1, dL2

    def attenuation(self, φ1: float, λ1: float, φ2: float, λ2: float, tx_height, rx_height,
                    Δh, frequency, Ns, Zgnd, γe, lut, flatearth, area_mode):
        if area_mode:
            horizons = self.estimate_horizons(tx_height, rx_height, 5, Δh, γe)
            dist = distance(φ1, λ1, φ2, λ2, haversine=True)
        else:
            jitprofile = self.get_profile(φ1, λ1, φ2, λ2, 0, lut, flatearth)
            Δh = jitprofile.irregularity()
            horizons = jitprofile.horizons(tx_height, rx_height, γe)
            dist = jitprofile.distance

        return itm.attenuation(dist, frequency, tx_height, rx_height,
                               horizons, Δh, Ns, Zgnd, γe)


@dataclass(repr=False)
class Elevation:
    """Digital elevation model dataclass."""

    shape: Tuple[int, int]
    dem: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None

    roi: dict = None

    jit: JITElevation = None

    def __post_init__(self, force_jit=False):
        self.latitude_deg = DEG_PER_RAD * self.latitude
        self.longitude_deg = DEG_PER_RAD * self.longitude
        self.bounds = ((self.latitude[0], self.longitude[0]), (self.latitude[-1], self.longitude[-1]))
        self.bounds_deg = ((self.latitude_deg[0], self.longitude_deg[0]), (self.latitude_deg[-1], self.longitude_deg[-1]))

        if self.jit is None or force_jit:
            self.jit = JITElevation(self.shape, self.dem, self.latitude, self.longitude)

    def __call__(self, φ: float, λ: float) -> float:
        return self.get_elevation(φ, λ)

    def __repr__(self):
        return f'Elevation(shape={self.shape})'

    @classmethod
    def from_jitelevation(cls, jitelevation: JITElevation):
        return Elevation(shape=jitelevation.shape, dem=jitelevation.dem,
                         latitude=jitelevation.latitude, longitude=jitelevation.longitude,
                         jit=jitelevation)

    @classmethod
    def from_srtm_ascii(cls, filename, bad_val=np.nan):
        """Load SRTM data in an Esri ASCII format."""
        header = {}
        with open(filename, 'r') as file:
            for _ in range(6):
                key, val = file.readline().split()
                header[key] = val

        try:
            shape = (int(header['nrows']), int(header['ncols']))
            xll = float(header['xllcorner'])
            yll = float(header['yllcorner'])
            reso = float(header['cellsize'])
            no_data_val = float(header['NODATA_value'])
        except KeyError:
            raise ValueError(f"header in {filename} appears malformed; expecting: "
                             "'ncols', 'nrows',''xllcorner', 'yllcorner', 'cellsize', and 'NODATA_value'")

        dem = np.loadtxt(filename, dtype=float, skiprows=6)
        if dem.shape != shape:
            raise ValueError("header shape ({}) and data shape ({}) don't match".format(shape, dem.shape))
        dem[dem == no_data_val] = bad_val

        dem = dem[::-1]  # make latitude increase with index
        dem[np.isnan(dem)] = np.nanmin(dem)  # fill NaNs with min value

        latitude = RAD_PER_DEG * (yll + reso * np.arange(shape[0]))
        longitude = RAD_PER_DEG * (xll + reso * np.arange(shape[1]))

        return cls(shape=shape, dem=dem, latitude=latitude, longitude=longitude)

    @classmethod
    def from_cache(cls, bounds):
        from rasp import SRTM_DIR
        from .data import fetch

        inds = list(fetch.srtm_inds(bounds=bounds))
        yy, xx = inds[0]
        elevation = cls.from_srtm_ascii(SRTM_DIR.joinpath(f'srtm_{xx:02d}_{yy:02d}.asc'))

        for yy, xx in inds[1:]:
            e = cls.from_srtm_ascii(SRTM_DIR.joinpath(f'srtm_{xx:02d}_{yy:02d}.asc'))
            elevation = elevation.join(e)

        elevation.subset(bounds, inplace=True)

        return elevation

    @classmethod
    def from_web(cls, bounds):
        from .data import fetch
        fetch.srtm(bounds)
        return cls.from_cache(bounds)

    def join(self, e):
        φmin = min(self.bounds[0][0], e.bounds[0][0])
        φmax = max(self.bounds[1][0], e.bounds[1][0])
        λmin = min(self.bounds[0][1], e.bounds[0][1])
        λmax = max(self.bounds[1][1], e.bounds[1][1])

        φres = self.latitude[1] - self.latitude[0]
        λres = self.longitude[1] - self.longitude[0]

        shape = np.round((
            (φmax - φmin) / φres,
            (λmax - λmin) / λres,
        )).astype(int) + 1

        φ = φmin + φres * np.arange(shape[0])
        λ = λmin + λres * np.arange(shape[1])

        new_dem = np.zeros(shape, float)

        # TODO: do this properly, with error-checking
        ind_y = np.searchsorted(φ, (self.latitude[0] - φres / 2, self.latitude[-1] - φres / 2))
        ind_x = np.searchsorted(λ, (self.longitude[0] - λres / 2, self.longitude[-1] - λres / 2))
        new_dem[ind_y[0]:ind_y[1] + 1, ind_x[0]:ind_x[1] + 1] = self.dem

        ind_y = np.searchsorted(φ, (e.latitude[0] - φres / 2, e.latitude[-1] - φres / 2))
        ind_x = np.searchsorted(λ, (e.longitude[0] - λres / 2, e.longitude[-1] - λres / 2))
        new_dem[ind_y[0]:ind_y[1] + 1, ind_x[0]:ind_x[1] + 1] = e.dem

        return self.__class__(shape, new_dem, φ, λ)

    def get_elevation(self, φ: float, λ: float) -> float:
        if isinstance(φ, (float, int)):
            return self.jit.get_elevation(φ, λ)

        orig_shape = φ.shape
        φ = np.asarray(φ).reshape(-1)
        λ = np.asarray(λ).reshape(-1)

        elev = self.jit.get_elevations(φ, λ)

        return elev.reshape(orig_shape)

    def get_profile(self, φ1: float, λ1: float, φ2: float, λ2: float,
                    num: int = None, flatearth: bool = False, lut: bool = False,
                    jit: bool = False) -> Profile:
        if num is None:
            num = 0

        profile = self.jit.get_profile(φ1, λ1, φ2, λ2, num, lut, flatearth)

        if jit:
            return profile

        return Profile(length=profile.length, elevation=profile.elevation,
                       latitude=profile.latitude, longitude=profile.longitude,
                       distance=profile.distance, jit=profile)

    def irregularity(self):
        return self.jit.irregularity()

    def subset(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]], inplace: bool = False):
        (φmin, λmin), (φmax, λmax) = bounds
        ind_y = np.searchsorted(self.latitude, (φmin, φmax))
        ind_x = np.searchsorted(self.longitude, (λmin, λmax))

        if not inplace:
            dem_subset = self.dem[ind_y[0]:ind_y[1], ind_x[0]:ind_x[1]]
            lat_subset = self.latitude[ind_y[0]:ind_y[1]]
            lon_subset = self.longitude[ind_x[0]:ind_x[1]]
            return Elevation(shape=dem_subset.shape, dem=dem_subset, latitude=lat_subset, longitude=lon_subset)

        self.dem = self.dem[ind_y[0]:ind_y[1], ind_x[0]:ind_x[1]]
        self.shape = self.dem.shape
        self.latitude = self.latitude[ind_y[0]:ind_y[1]]
        self.longitude = self.longitude[ind_x[0]:ind_x[1]]
        self.__post_init__(force_jit=True)

    def side_lengths(self):
        bottom = distance(self.latitude[0], self.longitude[0], self.latitude[0], self.longitude[-1])
        top = distance(self.latitude[-1], self.longitude[0], self.latitude[-1], self.longitude[-1])
        side = distance(self.latitude[0], self.longitude[0], self.latitude[-1], self.longitude[0])
        return top, bottom, side

    def set_region_of_interest(self, bounds=None, location=None, radius=None, full=False):
        if full:
            inds = np.asarray(np.unravel_index(np.arange(self.dem.size), self.shape), dtype=np.int64)
            self.roi = dict(bounds=self.bounds, inds=inds)

        if (bounds is None) and ((location is None) or (radius is None)):
            raise ValueError('either `bounds` or `location` & `radius` must be supplied')

        if bounds is not None:
            (φmin, λmin), (φmax, λmax) = bounds
            ind_y = np.searchsorted(self.latitude, (φmin, φmax))
            ind_y = np.arange(ind_y[0], ind_y[1] + 1)
            ind_x = np.searchsorted(self.longitude, (λmin, λmax))
            ind_x = np.arange(ind_x[0], ind_x[1] + 1)
            inds = np.asarray(np.meshgrid(ind_y, ind_x, indexing='ij')).reshape(2, -1)
            self.roi = dict(bounds=bounds, inds=inds)
        else:
            dist = distance(location[0], location[1], *np.meshgrid(self.latitude, self.longitude, indexing='ij'))
            inds = np.stack(np.where(dist <= radius))
            self.roi = dict(location=location, radius=radius, inds=inds)

    def points_in_roi(self):
        if self.roi is None:
            return self.dem.size
        return self.roi['inds'].shape[1]

    # def map_bounds(self, **kwargs):
    #     from .mapping import elevation_bounds
    #     return elevation_bounds(self, **kwargs)

    def attenuation(self, φ1, λ1, φ2, λ2, tx_height, rx_height=1.0,
                    frequency=100.0, horizontal=True, γe=None,
                    Ns=250.0, Zgnd=None, εr=15.0, σ=0.005, lut=False, flatearth=False):
        if γe is None:
            γe = itm.effective_curvature(Ns)

        if Zgnd is None:
            Zgnd = itm.surface_impedance(frequency, εr, σ, horizontal)

        return self.jit.attenuation(
            φ1, λ1, φ2, λ2, tx_height, rx_height, 0, frequency,
            Ns, Zgnd, γe, lut, flatearth, False,
        )

    def attenuation_map(self, tx_φ, tx_λ, tx_height, rx_height=1, frequency=100,
                        full=False, mode='point', Δh=None, Ns=250, εr=15, σ=0.005,
                        γe=None, Zgnd=None, horizontal=True, lut=False, flatearth=False):
        if γe is None:
            γe = itm.effective_curvature(Ns)

        if full:
            if self.roi is None:
                self.set_region_of_interest(full=True)
        elif self.roi is None:
            raise ValueError('region of interest unset; pass `full=True` if you wish to map the full region.')

        if Zgnd is None:
            Zgnd = itm.surface_impedance(frequency, εr, σ, horizontal)

        if mode.lower().startswith('p'):  # point-to-point mode
            loss = Elevation._attenuation_map(
                self.jit, self.roi['inds'], tx_φ, tx_λ, tx_height, rx_height,
                0.0, frequency, Ns, Zgnd, γe, lut, flatearth, False,
            )
        elif mode.lower().startswith('a'):  # area mode
            if Δh is None:
                Δh = self.irregularity()

            loss = Elevation._attenuation_map(
                self.jit, self.roi['inds'], tx_φ, tx_λ, tx_height, rx_height,
                Δh, frequency, Ns, Zgnd, γe, lut, flatearth, True,
            )
        else:
            raise ValueError('`mode` should be "point" or "area".')

        return loss

    @staticmethod
    @nb.njit(
        nb.float64[:](
            JITElevation.class_type.instance_type,  # jitelevation
            nb.int64[:, :],  # inds
            nb.float64, nb.float64,  # tx_φ, tx_λ
            nb.float64, nb.float64,  # tx_height, rx_height
            nb.float64,  # Δh
            nb.float64,  # frequency
            nb.float64,  # Ns
            nb.complex128,  # Zgnd
            nb.float64,  # γe
            nb.bool_,  # lut
            nb.bool_,  # flatearth
            nb.bool_,  # area_mode
        ),
        parallel=True,
    )
    def _attenuation_map(jitelevation, inds, tx_φ, tx_λ, tx_height, rx_height,
                         Δh, frequency, Ns, Zgnd, γe, lut, flatearth, area_mode):
        loss = np.empty(inds.shape[1])

        for i in nb.prange(len(inds[0])):
            φ2 = jitelevation.latitude[inds[0, i]]
            λ2 = jitelevation.longitude[inds[1, i]]
            loss[i] = jitelevation.attenuation(
                tx_φ, tx_λ, φ2, λ2, tx_height, rx_height, Δh, frequency,
                Ns, Zgnd, γe, lut, flatearth, area_mode
            )

        return loss

    # def map_overlay(self, overlay, **kwargs):
    #     from .mapping import elevation_overlay
    #     return elevation_overlay(self, overlay, **kwargs)
