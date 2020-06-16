"""rasp computes VHF/UHF radio losses over irregular terrain."""

import numba as nb
import logging.config
import pathlib
import toml

__all__ = [
    'config',
    'constants',
    'data',
    'Elevation',
    'fetch',
    'logger',
    'Profile',
    'TxDB',
    'utils',
]

# look for config file
config_path = pathlib.Path(__file__).parent.joinpath('config.toml')

if not config_path.exists():
    raise RuntimeError(f'rasp configuration file ({config_path}) not found')

config = toml.load(config_path)

# basic logging configuration
logging.config.dictConfig(config['logging'])

logger = logging.getLogger('rasp')

# save directories
if config['data']['cache_dir']:
    CACHE_DIR = pathlib.Path(config['data']['cache_dir'])
else:
    CACHE_DIR = config['data']['cache_dir'] = pathlib.Path.home().joinpath(".cache/rasp/")

if config['data']['srtm_dir']:
    SRTM_DIR = pathlib.Path(config['data']['srtm_dir'])
else:
    SRTM_DIR = CACHE_DIR.joinpath('srtm')

for path in [CACHE_DIR, SRTM_DIR]:
    path.mkdir(parents=True, exist_ok=True)

if config['general']['threads'] > 0:
    nb.set_num_threads(config['general']['threads'])

# import core functionality
from . import constants, data, utils
from .data import fetch
from .elevation import Elevation
from .profile import Profile
from .txdb import TxDB
