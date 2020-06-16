"""Fetch transmitter and elevation data."""

import argparse
import itertools
import pathlib
import requests
import zipfile

from tqdm.auto import tqdm

from rasp import logger
from rasp import CACHE_DIR, SRTM_DIR
from ..utils import geodesy
from ..constants import DEG_PER_RAD, π


__all__ = [
    "baserad",
    "sitedata",
]

BASERAD_URL = "http://www.ic.gc.ca/engineering/BC_DBF_FILES/baserad.zip"
SITEDATA_URL = "http://www.ic.gc.ca/engineering/SMS_TAFL_Files/Site_Data_Extract.zip"

SRTM_URL_FORMAT = "http://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/ascii/srtm_{xx:02d}_{yy:02d}.zip"


def download(url, dest, force=False, chunk_size=8192):
    """Download file to disk, with maximum memory-footprint.

    Parameters
    ----------
    url: str
        file to download
    dest: path-like
        download destination path
    force: bool
        re-download existing files
    chunk_size: int
        maximum number of bytes to read into memory at a time
        if None, data is read in whatever chunks are sent by the server

    Returns
    -------
    Path
        location of downloaded file
    """
    dest = pathlib.Path(dest)

    if dest.is_dir():
        dest = dest.joinpath(url.split("/")[-1])

    if not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=False)

    dest_size = dest.stat().st_size if dest.exists() else 0
    file_size = int(requests.head(url).headers["Content-Length"])

    # print(requests.head(url).headers)

    if (dest_size == file_size) and (not force):
        logger.info(f'destination ({dest}) exists; skipping download')
        return dest

    # counter = enlighten.Counter(total=file_size, desc=dest.name, unit='B')

    logger.info(f'downloading {url} to {dest}')
    # with open(dest, "wb",) as f, requests.get(url, stream=True) as r:
    #     for chunk in r.iter_content(chunk_size=chunk_size):
    #         f.write(chunk)
    with open(dest, "wb",) as f, requests.get(url, stream=True) as r, tqdm(
        total=file_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    # with open(dest, "wb",) as f, requests.get(url, stream=True) as r:
    #     for chunk in r.iter_content(chunk_size=chunk_size):
    #         f.write(chunk)
    #         counter.update(incr=len(chunk))

    return dest


def unzip(path, remove=False):
    logger.info(f'unzipping {path}')

    with zipfile.ZipFile(path) as zf:
        namelist = zf.namelist()
        zf.extractall(path=path.parent)

    logger.info(f'extracted {len(namelist)} files to {path.parent}')

    if remove:
        logger.info(f'removing {path}')
        path.unlink()

    return [path.parent.joinpath(name) for name in namelist]


def baserad(url=BASERAD_URL, dest=CACHE_DIR, munge=True, force=False, remove=False):
    """Fetch baserad (AM, FM, TV) transmitter data.

    ISED supplies `baserad.zip`, which contains many dbf files and associated
    metadata. This routine fetches, unzips, and (optionally) munges the data
    relevant to rasp.
    """
    dest = download(url, dest, force=force)
    contents = unzip(dest, remove=remove)

    if munge:
        from . import munge
        munge.baserad(input_dir=contents[0].parent)


def sitedata(url=SITEDATA_URL, dest=CACHE_DIR, munge=True, force=False, remove=False):
    """Fetch SiteData (MBS) transmitter data.

    ISED supplies `Site_Data_Extract.zip`, which contains a single CSV file.
    """
    dest = download(url, dest, force=force)
    contents = unzip(dest, remove=remove)

    if munge:
        from . import munge
        munge.sitedata(input_path=contents[0].with_suffix('.csv'))


def srtm_inds(bounds=None, location=None, radius=None):
    """Pick appropriate SRTM digital elevation data indices."""

    def x_ind(λ):
        return int((DEG_PER_RAD * λ + 185) // 5)

    def y_ind(φ):
        return int((65 - DEG_PER_RAD * φ) // 5)

    if bounds is not None:
        (φ1, λ1), (φ2, λ2) = bounds
    elif location is not None and radius is not None:
        φ1, _ = geodesy.destination(*location, 0, radius)  # North
        φ2, _ = geodesy.destination(*location, π, radius)  # South
        _, λ2 = geodesy.destination(*location, π / 2, radius)  # East
        _, λ1 = geodesy.destination(*location, 3 * π / 2, radius)  # West
    else:
        raise ValueError('either `bounds` or both `location` and `radius` must be supplied')

    y1, y2 = y_ind(φ1), y_ind(φ2)
    x1, x2 = x_ind(λ1), x_ind(λ2)

    x_inds = list(range(x1, x2 + 1))
    y_inds = list(range(y2, y1 + 1))

    return itertools.product(y_inds, x_inds)


def srtm(bounds=None, location=None, radius=None, url_format=SRTM_URL_FORMAT, dest=SRTM_DIR, **kwargs):
    """Fetch SRTM digital elevation data."""
    urls = [
        SRTM_URL_FORMAT.format(xx=xx, yy=yy)
        for yy, xx in srtm_inds(bounds=bounds, location=location, radius=radius)
    ]
    logger.info(f'requested area overlaps {len(urls)} SRTM tiles')

    for url in urls:
        dest_dl = download(url, dest, **kwargs)
        if not dest_dl.with_suffix('.asc').exists():
            unzip(dest_dl, remove=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Munge the Industry Canada transmitter dbf files."
    )
    parser.add_argumennt("--baserad", action="store_true")
    parser.add_argumennt("--sitedata", action="store_true")
    parser.add_argumennt("--munge", action="store_true")
    parser.add_argumennt("--output", type=str, default=CACHE_DIR)
    args = parser.parse_args()

    if args.baserad:
        baserad(dest=args.output, munge=args.munge)

    if args.sitedata:
        sitedata(dest=args.output, munge=args.munge)
