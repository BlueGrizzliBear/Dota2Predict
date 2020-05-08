import numpy as np
import pandas as pd
from decorator import func_deco
from decorator import print_colored
import time

def	ft_progress(lst):
	start = time.perf_counter()
	for i in lst:
		if i == 0 or i == len(lst) - 1:
			eta = 0
		else:
			eta = round((time.perf_counter() - start) / (i) * len(lst) - (time.perf_counter() - start), 2)
		percent = round(((i + 1) / len(lst)) * 100)
		progress = ""
		progress = progress.rjust(round((i / len(lst)) * 30), '=')
		if i < len(lst) - 1:
			progress += '>'
		else:
			progress += '='
		progress = progress.ljust(31, ' ')
		elapsed = time.perf_counter() - start
		erase_line = '\x1b[2K'
		print(erase_line + "ETA: %.2fs [%3.0f%%][%s] %i/%i | elapsed time %.2fs" % (eta, percent, progress, i + 1, len(lst), elapsed), end='\r', flush=True)
		# print_colored(erase_line + "ETA: %.2fs [%3.0f%%][%s] %i/%i | elapsed time %.2fs" % (eta, percent, progress, i + 1, len(lst), elapsed), end='\r', flush=True, '\033[90m')
		yield i

def parse_data(data):

	def get_hero_labels():	
		heroes = pd.read_csv("./resources_data/dota2_heroes.csv")
		hero_label = pd.DataFrame(np.array(heroes['id']))
		return hero_label

	def get_team_labels(data):
		return pd.DataFrame(data['radiant'].append(data['dire']).unique())
	
	def assimilate_data(data, hero_label, team_label):
		a = len(data)
		b = len(team_label)
		c = len(hero_label)
		shape = (a,b,c)
		last_array = np.zeros(shape)
		
		max_iter = range(len(data))
		for row_index in ft_progress(max_iter):

			player_slot = ["slot_" + str(item) for item in range(0, 10)]
			team_radiant = data['radiant'][row_index]
			team_dire = data['dire'][row_index]
			team_radiant_index = team_label[team_label[0] == team_radiant].index[0]
			team_dire_index = team_label[team_label[0] == team_dire].index[0]
			
			for index, slot in enumerate(player_slot):
				hero_index = hero_label[hero_label[0] == data[slot][row_index]].index[0]
				if index < 5:				
					last_array[row_index][team_radiant_index][hero_index] = 1.
				elif index >= 5:
					last_array[row_index][team_dire_index][hero_index] = -1.
			
		print ("")
		return last_array
		
	hero_label = get_hero_labels()
	team_label = get_team_labels(data)
	parsed_data = assimilate_data(data, hero_label, team_label)

	return hero_label, team_label, parsed_data