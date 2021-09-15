"""Perform file operations on transmitter and elevation data"""

import pickle
import pandas as pd

from ..elevation import Elevation
from ..txdb import TxDB

def save_elevation(elevation_obj, pickle_fname):
	# Convert Elevation object to dictionary
	elevation_dict = {'shape': elevation_obj.shape,
			  'dem': elevation_obj.dem,
			  'latitude': elevation_obj.latitude,
			  'longitude': elevation_obj.longitude,
			  'roi': None
			  }
	if elevation_obj.roi is not None:
		elevation_dict['roi'] = elevation_obj.roi

	# Save the elevation dictionary as pickle
	with open(pickle_fname, 'wb') as f:
		pickle.dump(elevation_dict, f)

def load_elevation(pickled_fname):
	# Load pickled elevation data from file
	with open(pickled_fname, 'rb') as f:
		elevation_dict = pickle.load(f)
	# Convert dictionary to Elevation object
	return Elevation(elevation_dict['shape'],
			 elevation_dict['dem'],
			 elevation_dict['latitude'],
			 elevation_dict['longitude'],
			 roi = elevation_dict['roi'])

def save_txdb(txdb_obj, csv_fname):
	# Save TxDB object as .csv file
	txdb_obj.dataframe.to_csv(csv_fname, index=False)

def load_txdb(csv_fname):
	# Load previously saved TxDB object from .csv file
	return TxDB(pd.read_csv(csv_fname))
