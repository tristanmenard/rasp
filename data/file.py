"""Perform file operations on transmitter and elevation data"""

import os
import pickle
import pandas as pd

from ..elevation import Elevation
from ..txdb import TxDB
from rasp import CACHE_DIR

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

def fix_raw_tafl(tafl_path=CACHE_DIR.joinpath('TAFL_LTAF.csv')):
	# Fix the delimiter known error in TAFL_LTAF.csv
	with open(tafl_path, encoding='utf-8') as f:
		lines = f.readlines()

	for i in range(len(lines)):
		if lines[i] == '"TX","459.6125","0001913061","1","D","A","LM_CAN_UHF_DUPLEX5_12K5","E369"","210634FC","D","7.6","7K60F1W","LMR-DIGITAL","","6.9897","5","0","","","-","-","","","2.1484","","","","","0","","","Creston BC","portable radios","","3","ML","","5","Swan Valley Lodge","BC","49.09814722","-116.51522222","","","L","1","","010883606-001","3","300","S","G","2021-05-12","100000092629","Interior Heath Authority Swan Valley Lodge","818 Vancouver Street,Creston,BC,V0B 1G0","","","0","0",""\n':
			lines[i] = '"TX","459.6125","0001913061","1","D","A","LM_CAN_UHF_DUPLEX5_12K5","E369\'","210634FC","D","7.6","7K60F1W","LMR-DIGITAL","","6.9897","5","0","","","-","-","","","2.1484","","","","","0","","","Creston BC","portable radios","","3","ML","","5","Swan Valley Lodge","BC","49.09814722","-116.51522222","","","L","1","","010883606-001","3","300","S","G","2021-05-12","100000092629","Interior Heath Authority Swan Valley Lodge","818 Vancouver Street,Creston,BC,V0B 1G0","","","0","0",""\n'

	temp = CACHE_DIR.joinpath('temp.csv')
	with open(temp, 'w', encoding='utf-8') as f:
		f.writelines(lines)

	os.remove(tafl_path)
	os.rename(temp, tafl_path)
