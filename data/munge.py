"""Munge transmitter data."""

import argparse
import logging
from pathlib import Path

import dbfread
import numpy as np
import pandas as pd

from ..utils.misc import sex_to_dec


logger = logging.getLogger('rasp')
logger.addHandler(logging.NullHandler())

__all__ = ["tafl", "baserad", "sitedata",]


DROP_COLUMNS = [
    "auto_prog",
    "bc_mode",
    "border",
    "brdr_lat",
    "brdr_long",
    "can_land",
    "cert_numb",
    "clist1",
    "clist10",
    "clist2",
    "clist3",
    "clist4",
    "clist5",
    "clist6",
    "clist7",
    "clist8",
    "clist9",
    "dec_number",
    "doc_file",
    "euvalu",
    "fre_land",
    "ifrbn_d",
    "ifrbn_n",
    "last_mod_date",
    "latitude2",
    "longitude2",
    "network",
    "ok_dump",
    "scmo",
    "st_creat",
    "st_mod",
    "status1",
    "status2",
    "structure_height",
    "tx_pwr_type",
    "unattended",
    "usa_land",
    "zone_enhancer",
]

# for MBS stations
RENAME_COLUMNS = {
    "transmit_freq": "frequency",
    "transmit_lower": "frequency_lower",
    "transmit_upper": "frequency_upper",
    "transmit_bw": "bandwidth",
    "bw_emission": "bandwidth_designator",
    "tx_pwr": "eirp",
    "tx_ant_gain": "gain",
    "tx_line_loss": "line_loss",
    "tx_ant_azim": "azimuth_angle",
    "tx_ant_elev_angle": "elevation_angle",
    "tx_ant_ht": "height",
    "stuct_ht": "structure_height",
    "site_elev": "site_elevation",
    "tx_ant_mfr": "antenna_manufacturer",
    "tx_ant_model": "antenna_model",
    "tx_ant_directional": "directional",
    "tx_mfr": "manufacturer",
    "tx_model": "model",
    "new_licno": "new_licence_number",
    "old_licno": "old_licence_number",
    "prov": "province",
}

NUMERIC_COLUMNS = [
    "frequency",
    "frequency_lower",
    "frequency_upper",
    "bandwidth",
    "latitude",
    "longitude",
    "site_elevation",
    "structure_height",
    "eirp",
    "erpvav",
    "erpvpk",
    "erphav",
    "erphpk",
    "height",
    "azimuth_angle",
    "elevation_angle",
    "gain",
    "line_loss",
]


def dbf_to_df(path, encoding="latin1"):
    return pd.DataFrame(iter(dbfread.DBF(path, encoding=encoding)))


def _tafl(stations, radians=False, allow_duplicates=False, allow_nonoperational=False, allow_unauthorized=False):
    logger.info('Munging transmitter database...')
    transmitters = stations.loc[stations.station_function == 'TX'].copy()

    if not radians:
        # Convert latitudes and longitudes from degrees to radians
        transmitters.loc[:,'latitude'] = np.radians(transmitters[:,'latitude'])
        transmitters.loc[:,'longitude'] = np.radians(transmitters[:,'longitude'])

    if not allow_duplicates:
        # Remove any duplicated tranmitters in database
        n1 = len(transmitters)
        transmitters.drop_duplicates(keep='first', inplace=True)
        n2 = len(transmitters)
        logger.info(f'Found and removed {n1-n2} duplicated transmitters')

    if not allow_nonoperational:
        # Remove non-operational transmitters (including auxiliary transmitters and those under consideration)
        n1 = len(transmitters)
        bad_operational_status = ('AX','AXO','AXP','UX','PUC','UC')
        transmitters = transmitter[~transmitters.operational_status.isin(bad_operational_status)]
        n2 = len(transmitters)
        logger.info(f'Removed {n1-n2} non-operational transmitters (including auxiliary transmitters and those under consideration)')

    if not allow_unauthorized:
        # Remove transmitters with cancelled or "not granted" authorization status
        n1 = len(transmitters)
        bad_authorization_status = ('-4','-2','NG')
        transmitters = transmitters[~transmitters.authorization_status.isin(bad_authorization_status)]
        n2 = len(transmitters)
        logger.info(f'Removed {n1-n2} unauthorized transmitters')

    return transmitters


def tafl(input_path, output_path=None):
    columns = [0,1,10,14,15,16,23,24,25,26,27,28,29,30,31,33,37,39,40,41,42,43,48,49,51,52,54,56,58,59]
    fieldnames = ['station_function','frequency','bandwidth','erp','power','loss','gain','pattern',
                      'half_power_beamwidth','fb_ratio','polarization','haat','azimuth','elevation_angle','location',
                      'call_sign','identical_stations','province','latitude','longitude','ground_elevation','height',
                      'service','subservice','authorization_status','in-service_date','licensee','operational_status',
                      'horizontal_power','vertical_power']
    dtypes = {'frequency': float,'bandwidth': float,'erp': float,'power': float,'loss': float,'gain': float,
                  'pattern': str,'half_power_beamwidth': float,'fb_ratio': float,'haat': float,'azimuth': float,
                  'elevation_angle': float,'latitude': float,'longitude': float,'ground_elevation': float,'height': float,
                  'service': int,'subservice': int,'authorization_status': str,'licensee': str, 'operational_status': str,
                  'horizontal_power': float,'vertical_power': float}
    stations = pd.read_csv(input_path, usecols=columns, names=fieldnames, dtype=dtypes, header=None, na_values=['-'])
    transmitters = _tafl(stations)

    if output_path is None:
        output_path = input_path.with_name(input_path.name.lower())

    logger.info(f'writing munged TAFL data to {output_path}')
    transmitters.to_csv(output_path, index=False)
    return output_path


def _sitedata(stations):
    logger.info('munging records')
    stations.rename(columns={c: c.lower() for c in stations.columns}, inplace=True)
    stations.rename(columns=RENAME_COLUMNS, inplace=True)

    for col in NUMERIC_COLUMNS:
        try:
            stations[col] = pd.to_numeric(stations[col])
        except KeyError:
            continue

    stations.latitude = np.deg2rad(stations.latitude)
    stations.longitude = np.deg2rad(stations.longitude)

    # if power values are EIRP (i.e. type "I"), then we don't have to do anything else there
    if not (stations.tx_pwr_type == "I").all():
        print(f"not all site power values in the SiteData DBF are EIRP, like expected")

    # clean up heights
    bad_height_inds = np.any(
        [np.isnan(stations.height), stations.height == 0.0], axis=0
    )

    stations.loc[bad_height_inds, "height"] = stations.loc[
        bad_height_inds, "structure_height"
    ]

    # convert EIRP to power, baking in the line losses
    stations.loc[:, "power"] = stations.eirp / 10 ** (stations.gain / 10.0)

    stations.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)
    # stations = stations.reindex(sorted(stations.columns), axis=1)

    stations.loc[:, 'service'] = 'MBS'

    return stations


def sitedata(input_path, output_path=None):
    input_path = Path(input_path)

    logger.info(f'reading site data from {input_path}')
    stations = pd.read_csv(input_path, encoding="latin1", dtype=str)
    logger.info(f'{len(stations)} records found')

    stations = _sitedata(stations)

    if output_path is None:
        output_path = input_path.with_name(input_path.name.lower())

    logger.info(f'writing munged site data to {output_path}')
    stations.to_csv(output_path, index=False, encoding="utf8")
    return output_path


def _baserad(stations, params=None, label=False):
    # lower-case column names
    stations.rename(
        columns={c: c.split(",")[0].lower() for c in stations.columns}, inplace=True
    )

    # convert to decimal lat/lon, assuming western hemisphere
    stations.latitude = stations.latitude.map(sex_to_dec)
    stations.longitude = -stations.longitude.map(sex_to_dec)

    if "latitude2" in stations:
        stations.loc[stations.latitude == 0, "latitude"] = stations.loc[
            stations.latitude == 0, "latitude2"
        ]
        stations.loc[stations.longitude == 0, "longitude"] = stations.loc[
            stations.longitude == 0, "longitude2"
        ]

    # convert to radians
    stations.latitude = np.deg2rad(stations.latitude)
    stations.longitude = np.deg2rad(stations.longitude)

    for col in NUMERIC_COLUMNS:
        try:
            stations[col] = pd.to_numeric(stations[col])
        except KeyError:
            continue

    label = label.lower()

    if label == "am":
        if params is None:
            raise ValueError("`params` must be supplied for AM transmitters")

        # convert to MHz, from kHz
        stations.frequency = stations.frequency.astype(float) / 1e3

        # (worst-case) power
        stations = stations.assign(
            power=np.max([stations.powerday, stations.powernight], axis=0)
        )

        # make callsign_banner -> height mapping
        heights = {row.calls_banr: row.height for _, row in params.iterrows()}
        average_height = np.mean(list(heights.values()))

        # add height field
        stations["height"] = stations.apply(
            lambda row: heights.get(f"{row.call_sign},{row.banner}", average_height),
            axis=1,
        )

        # convert height from degrees to metres
        stations["wavelength"] = 299.8 / stations.frequency
        stations["height"] = stations.height * stations.wavelength / 360
    elif label in ["fm", "tv"]:
        erp_columns = ["erpvav", "erpvpk", "erphav", "erphpk"]

        stations["erp"] = np.max(
            stations[[col for col in erp_columns if col in stations]], axis=1
        )
        stations["eirp"] = 1.64 * stations["erp"]

        stations.rename(columns={"overall_h": "height"}, inplace=True)
        stations["height"] = stations["height"].astype(float)

        # missing height stations get set to the average value
        stations.loc[~(stations.height > 0), "height"] = (
            stations.loc[stations.height > 0, "height"].astype(float).mean()
        )
    else:
        raise ValueError(
            f'baserad label `{label}` not understood; expecting "am", "fm", or "tv"'
        )

    stations.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)
    # stations = stations.reindex(sorted(stations.columns), axis=1)

    stations.loc[:, 'service'] = label.upper()

    return stations


def baserad(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    params = dbf_to_df(input_dir.joinpath("params.dbf"))
    params.rename(
        columns={c: c.split(",")[0].lower() for c in params.columns}, inplace=True
    )

    if output_dir is None:
        output_dir = input_dir

    for label in ["am", "fm", "tv"]:
        dbf_path = input_dir.joinpath(label + "statio.dbf")

        logger.info(f'reading site data from {dbf_path}')
        stations = dbf_to_df(dbf_path)

        logger.info(f'{len(stations)} records found')
        stations = _baserad(stations, params, label)

        output_path = output_dir.joinpath(label + "statio.csv")
        logger.info(f'writing munged site data to {output_path}')
        stations.to_csv(
            output_path, index=False, encoding="utf8"
        )

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Munge the Industry Canada transmitter dbf files.")
    parser.add_argument("--tafl", type=str)
    parser.add_argument("--baserad", type=str)
    parser.add_argument("--sitedata", type=str)
    args = parser.parse_args()

    if args.tafl:
        tafl(args.tafl)

    if args.baserad:
        baserad(args.baserad)

    if args.sitedata:
        sitedata(args.sitedata)
