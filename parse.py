import numpy as np
import pandas as pd
from decorator import func_deco
from decorator import print_colored
import time

from datetime import datetime
from datetime import timedelta
# from datetime import date
from dateutil.relativedelta import relativedelta

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

def parse_data(data):

	def sigmoid_(current, first, last):
		x = 10 * (60*60*24*60) / (last - first) * (current - first) / (last - first)
		res = 2 * float(1 / float(1 + np.exp(-x))) - 1
		# res = float(1 / float(1 + np.exp(-x)))
		return res

	def get_hero_labels():
		heroes = pd.read_csv("./resources_data/dota2_heroes.csv")
		hero_label = pd.DataFrame(np.array(heroes['id']))
		return hero_label

	def get_team_labels(data):
		team_label = pd.DataFrame(data['radiant'].append(data['dire']).unique())
		team_label.columns = ['Team']
		print (team_label.shape)
		return team_label
	
	def assimilate_data(data, hero_label, team_label):
		a = len(data)
		b = len(team_label)
		c = len(hero_label)
		shape = (a,b,c)
		last_array = np.zeros(shape)
		
		time_min = data.loc[0, 'time']
		time_max = data.loc[len(data) - 1, 'time']

		for row_index, values in enumerate(data.values):

			team_radiant = values[10]
			team_dire = values[11]
			
			# A CHANGER
			team_radiant_index = team_label[team_label['Team'] == team_radiant].index[0]
			team_dire_index = team_label[team_label['Team'] == team_dire].index[0]
			
			for index in range(0, 10):
				
				hero_index = hero_label[hero_label[0] == values[index]].index[0]
				# sig = 1
				sig = sigmoid_(values[12], time_min, time_max)
				if index < 5:				
					last_array[row_index][team_radiant_index][hero_index] = 1. * sig
				elif index >= 5:
					last_array[row_index][team_dire_index][hero_index] = -1. * sig
			
		return last_array
		
	hero_label = get_hero_labels()
	team_label = get_team_labels(data)
	parsed_data = assimilate_data(data, hero_label, team_label)

	return hero_label, team_label, parsed_data