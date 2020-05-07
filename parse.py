import numpy as np
import pandas as pd
from decorator import func_deco
import time

# def func_deco(func):
# 	def wrapper(*args, **kwargs):
# 		print (f"Executing : {func.__name__} . . .")
# 		ret = func(*args, **kwargs)
# 		print ("Done")
# 		return ret
# 	return wrapper

# @func_deco

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
		yield i

def convert_draft_str_to_array(data):

	def get_hero_labels():	
		heroes = pd.read_csv("./resources/dota2_heroes.csv")
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

# @func_deco
def	remove_empty_row(data):
	data = data[data.astype(str)['draft'] != '[]']
	return data

# @func_deco
def dataframe_to_draft_str(data):
	draft = np.array(data['draft'])

	match_heroes = []
	for matches in draft:
		str_radiant = ""
		str_dire = ""
		j = 0
		k = 0
		rad = 3
		choose = True
		for i in range(len(matches)):
			if matches[i].get('pick') == True:
				if choose == True:
					if matches[i].get('active_team') == 2:
						if matches[i].get('player_slot') % 2 == 0:
							rad = 0
						else:
							rad = 1
					else:
						if matches[i].get('player_slot') % 2 == 0:
							rad = 1
						else:
							rad = 0
					choose = False

				if matches[i].get('player_slot') % 2 == rad:
					str_radiant += str(matches[i].get('hero_id'))
					j += 1
					if j < 5:
						str_radiant += ", "
				else:
					str_dire += str(matches[i].get('hero_id'))
					k += 1
					if k < 5:
						str_dire += ", "
		match_heroes.append(np.asarray([str_radiant, str_dire], dtype=list))
	match_heroes = np.asarray(match_heroes, dtype=list)
	return match_heroes