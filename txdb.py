from dataclasses import dataclass

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rasp import logger
from .itm.itm import effective_curvature, watts_to_dBW


class Tx:
    """Transmitter dataclass."""

    def __init__(self, **kwargs):
        self.frequency = None
        self.latitude = None
        self.longitude = None
        self.service = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        repr_str = ""

        if self.frequency is not None:
            if self.frequency < 5:
                repr_str = repr_str + f"ν={1000*self.frequency:.0f}kHz, "
            else:
                repr_str = repr_str + f"ν={self.frequency:.0f}MHz, "

        if self.latitude is not None:
            repr_str = repr_str + f"φ={self.latitude:.2f}°, "
        if self.longitude is not None:
            repr_str = repr_str + f"λ={self.longitude:.2f}°, "

        if self.service is not None:
            repr_str = repr_str + f"service={self.service}, "

        repr_str = repr_str[:-2]  # remove trailing comma

        return f"Tx({repr_str})"

    def as_dict(self):
        return self.__dict__


@dataclass
class TxDB:
    """Transmitter database."""

    dataframe: pd.DataFrame

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        for tup in self.dataframe.itertuples():
            dic = tup._asdict()
            del dic["Index"]
            yield Tx(**dic)

    def __repr__(self):
        return f"TxDB(len={len(self)})"

    def _repr_html_(self):
        return self.dataframe._repr_html_()

    def head(self, n):
        return self.dataframe.head(n)

    @classmethod
    def from_csv(cls, filename: str, **kwargs):
        """Generate a TxDB instance from a CSV file."""
        return cls(pd.read_csv(filename, low_memory=False, **kwargs))

    @classmethod
    def from_cache(cls, baserad=True, sitedata=True, **kwargs):
        """Generate a TxDB instance from cached transmitter data."""
        from rasp import CACHE_DIR
        paths = set()

        if baserad:
            for fname in ['amstatio.csv', 'fmstatio.csv', 'tvstatio.csv']:
                paths.add(CACHE_DIR.joinpath('baserad', fname))

        if sitedata:
            paths.add(CACHE_DIR.joinpath('site_data_extract.csv'))

        return cls(pd.DataFrame()).concat(*(cls.from_csv(p, **kwargs) for p in paths))

    @classmethod
    def from_web(cls, baserad=True, sitedata=True, **kwargs):
        from .data import fetch

        if baserad:
            fetch.baserad()

        if sitedata:
            fetch.sitedata()

        return cls.from_cache(baserad=baserad, sitedata=sitedata, **kwargs)

    def add_column(self, name, values):
        self.dataframe.loc[:, name] = values

    def fill_null(self, null_column, fill_column):
        fill_inds = pd.isnull(self.dataframe[null_column])
        self.dataframe.loc[fill_inds, null_column] = self.dataframe.loc[
            fill_inds, fill_column
        ]

    def concat(self, *args, **kwargs):
        """Concatenate this TxDB instance with others."""
        dfs = (arg.dataframe for arg in args)

        return TxDB(pd.concat((self.dataframe, *dfs), sort=False, **kwargs))

    def in_bounds(self, bounds, inplace=False):
        (φmin, λmin), (φmax, λmax) = bounds
        good_inds = (
            (φmin <= self.dataframe.latitude)
            & (self.dataframe.latitude <= φmax)
            & (λmin <= self.dataframe.longitude)
            & (self.dataframe.longitude <= λmax)
        )

        if inplace:
            self.dataframe = self.dataframe.loc[good_inds]
            return

        return TxDB(self.dataframe.loc[good_inds])

    def in_band(self, lower, higher, inplace=False):
        good_inds = (lower <= self.dataframe.frequency) & (
            self.dataframe.frequency <= higher
        )

        if inplace:
            self.dataframe = self.dataframe.loc[good_inds]
            return

        return TxDB(self.dataframe.loc[good_inds])

    # def map_txs(self, **kwargs):
    #     """Plot all the transmitters on a leaflet map."""
    #     from .mapping import transmitters
    #     return transmitters(self, **kwargs)

    def attenuations(self, elevation, φ, λ, **kwargs):
        loss = np.empty(len(self), float)

        # TODO: parallel computation here?
        for i, tx in enumerate(self):
            loss[i] = elevation.attenuation(
                tx.latitude,
                tx.longitude,
                φ,
                λ,
                tx.height,
                frequency=tx.frequency,
                **kwargs,
            )

        return loss

    def attenuation_maps(self, elevation, **kwargs):
        if "γe" not in kwargs:
            kwargs["γe"] = effective_curvature(kwargs.get("Ns", 250.0))

        Ntx = len(self)
        logger.info(f'computing attenuation maps for {Ntx} transmitters')
        loss = np.empty((Ntx, elevation.points_in_roi()), float)

        with tqdm(total=Ntx, unit="Tx") as pbar:
            for i, tx in enumerate(self):
                loss[i] = elevation.attenuation_map(
                    tx.latitude, tx.longitude, tx.height, frequency=tx.frequency, **kwargs
                )
                pbar.update()

        return loss

    def power_maps(self, elevation, power="power", **kwargs):
        if isinstance(power, str):
            power = (power,)

        signal = -1 * self.attenuation_maps(elevation, **kwargs)

        for i, tx in enumerate(self):
            for key in power:
                if hasattr(tx, key) and not np.isnan(tx.__dict__[key]):
                    signal[i, :] += watts_to_dBW(tx.__dict__[key])
                    break
            else:
                # TODO: handle this better
                signal[i, :] = np.nan
                continue

        return signal
