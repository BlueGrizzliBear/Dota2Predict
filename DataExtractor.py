import numpy as np
import pandas as pd
from parse import parse_model_data
from decorator import print_colored

def Data_Extractor():
# FOR PRO GAMES - NEW
	print_colored("Reading data from json file . . .", '\033[94m')	
	data = pd.read_json("./resources_data/data_012020_15052020.json")
	data = data.dropna()
	cols = ["slot_" + str(item) for item in range(0, 10)]
	cols.extend(['radiant', 'dire', 'time'])
	cols.extend(["gpm_" + str(item) for item in range(0, 10)])
	print_colored("Done !", '\033[92m')	

	y_parsed_data = data['win'].astype(float)
	y_parsed_data = y_parsed_data.replace(to_replace=0., value=-1.)
	data_team_column = data[['radiant', 'dire']]

	print_colored("Parsing data and creating arrays . . .", '\033[94m')	
	t = parse_model_data(data[cols])
	hero_label = t[0]
	team_label = t[1]
	x_parsed_data = t[2]
	sample_weight = t[3]
	print_colored("Done !", '\033[92m')	

	return hero_label, team_label, x_parsed_data, y_parsed_data, data_team_column, sample_weight