import numpy as np
import pandas as pd
from parse import parse_data
from decorator import print_colored

def Data_Extractor():
# FOR PRO GAMES - NEW
	print_colored("Reading data from json file . . .", '\033[94m')	
	data = pd.read_json("./resources_data/data.json")
	data = data.dropna()
	slot = ["slot_" + str(item) for item in range(0, 10)]
	slot.append('radiant')
	slot.append('dire')
	print_colored("Done !", '\033[92m')	
	
	y_parsed_data = data['win']
	# z_parsed_data = data['time']
	# print (z_parsed_data)

	print_colored("Parsing data and creating arrays . . .", '\033[94m')	
	t = parse_data(data[slot], data['time'])
	hero_label = t[0]
	team_label = t[1]
	x_parsed_data = t[2]
	print_colored("Done !", '\033[92m')	

	return hero_label, team_label, x_parsed_data, y_parsed_data