import numpy as np
import pandas as pd
from decorator import func_deco
from decorator import print_colored
import time

def parse_model_data(data):

	def get_elo_rating():
		elo_rating = pd.read_json("./resources_data/elo_rating.json")
		return elo_rating[['name', 'rating']]
	
	def sigmoid_(current, first, last):
		# date correpsonding to 26/01/2020 + 2 months
		after_meta_change = 1580031318 + 60*60*24*30*2
		if after_meta_change < first:
			after_meta_change = last
		x = 10 * ((current - first) / (last - first)) * (((after_meta_change - first) / (last - first)))
		res = (2 * float(1 / float(1 + np.exp(-x))) - 1)
		# res = 1
		return res

	def get_hero_labels():
		heroes = pd.read_csv("./resources_data/dota2_heroes.csv")
		hero_label = pd.DataFrame(np.array(heroes['id']))
		return hero_label

	def get_team_labels(data):
		team_label = pd.DataFrame(data['radiant'].append(data['dire']).unique())
		team_label.columns = ['Team']
		return team_label
	
	def get_pos_from_gpm(row, index):
		if index < 5:
			nb1 = 13
			nb2 = 18
		elif index >= 5:
			nb1 = 18
			nb2 = 23
		gpm = row[13 + index]
		gpm_array = np.sort(row[nb1:nb2].copy(), axis=None)
		pos_rating = 0.
		for ind, item in enumerate(gpm_array):
			if gpm == item:
				pos_rating = 2
				if ind == 1:
					pos_rating *= 2
				if ind == 2:
					pos_rating *= 4
				if ind == 3:
					pos_rating *= 5
				if ind == 4:
					pos_rating *= 4
				# pos_rating = float(((ind + 2) * 5) / 100)
				# if ind == 1:
				# 	pos_rating *= 2
				# if ind == 2:
				# 	pos_rating *= 4
				# if ind == 3:
				# 	pos_rating *= 5
				# if ind == 4:
				# 	pos_rating *= 4
		# print (pos_rating)
		return pos_rating
		
	def assimilate_data(data, hero_label, team_label, elo_rating):
		a = len(data)
		b = len(team_label) + len(hero_label)
		last_array = np.zeros((a,b))

	# ERROR HERE FOR INDEX SELECTION
		sample_weight = data['time'].copy().astype(float)
		time_min = sample_weight.min()
		time_max = sample_weight.max()
		for index, item in enumerate(sample_weight.values):
			sample_weight.values[index] = sigmoid_(item, time_min, time_max)
		
		for row_index, values in enumerate(data.values):

			# for team name
			team_radiant = values[10]
			team_dire = values[11]
			
			# extracting teams elo_rating
			# radiant
			if elo_rating[elo_rating['name'] == team_radiant].rating.values.size == 0:
				radiant_elo_rating = 1000
			elif elo_rating[elo_rating['name'] == team_radiant].rating.values.size > 1:
				radiant_elo_rating = elo_rating[elo_rating['name'] == team_radiant].rating.values[0]
			else:
				radiant_elo_rating = elo_rating[elo_rating['name'] == team_radiant].rating.values
			# dire
			if elo_rating[elo_rating['name'] == team_dire].rating.values.size == 0:
				dire_elo_rating = 1000
			elif elo_rating[elo_rating['name'] == team_dire].rating.values.size > 1:
				dire_elo_rating = elo_rating[elo_rating['name'] == team_dire].rating.values[0]
			else:
				dire_elo_rating = elo_rating[elo_rating['name'] == team_dire].rating.values
			
			# applying team unit * team_elo_rating
			team_radiant_index = team_label[team_label['Team'] == team_radiant].index
			team_dire_index = team_label[team_label['Team'] == team_dire].index
			last_array[row_index][team_radiant_index] = 1. * radiant_elo_rating
			last_array[row_index][team_dire_index] = -1. * dire_elo_rating

			# applying hero unit * pos_rating
			for index in range(0, 10):
				hero_index = hero_label[hero_label[0] == values[index]].index
				if index < 5:				
					pos_rating = get_pos_from_gpm(values, index)
					last_array[row_index][len(team_label) + hero_index] = 1. * pos_rating * radiant_elo_rating
					# print (last_array[row_index][len(team_label) + hero_index])
				elif index >= 5:
					pos_rating = get_pos_from_gpm(values, index)
					last_array[row_index][len(team_label) + hero_index] = -1. * pos_rating * dire_elo_rating
					# print (last_array[row_index][len(team_label) + hero_index])
			
		return last_array, sample_weight
		
	hero_label = get_hero_labels()
	team_label = get_team_labels(data)
	elo_rating = get_elo_rating()
	t = assimilate_data(data, hero_label, team_label, elo_rating)
	parsed_data = t[0]
	sample_weight = t[1]

	return hero_label, team_label, parsed_data, sample_weight


def parse_match_data(data, team_label, hero_label):

	def get_elo_rating():
		elo_rating = pd.read_json("./resources_data/elo_rating.json")
		return elo_rating[['name', 'rating']]	

	def assimilate_data(data, hero_label, team_label, elo_rating):
		a = len(data)
		b = len(team_label) + len(hero_label)
		last_array = np.zeros((a,b))
		
	# ERROR HERE FOR INDEX SELECTION
		for row_index, values in enumerate(data.values):

			team_radiant = values[10]
			team_dire = values[11]
			
			# for radiant team rating
			if elo_rating[elo_rating['name'] == team_radiant].rating.values.size == 0:
				radiant_elo_rating = 1000
			elif elo_rating[elo_rating['name'] == team_radiant].rating.values.size > 1:
				radiant_elo_rating = elo_rating[elo_rating['name'] == team_radiant].rating.values[0]
			else:
				radiant_elo_rating = elo_rating[elo_rating['name'] == team_radiant].rating.values
			# for dire team rating			
			if elo_rating[elo_rating['name'] == team_dire].rating.values.size == 0:
				dire_elo_rating = 1000
			elif elo_rating[elo_rating['name'] == team_dire].rating.values.size > 1:
				dire_elo_rating = elo_rating[elo_rating['name'] == team_dire].rating.values[0]
			else:
				dire_elo_rating = elo_rating[elo_rating['name'] == team_dire].rating.values

			# A CHANGER
			team_radiant_index = team_label[team_label['Team'] == team_radiant].index
			team_dire_index = team_label[team_label['Team'] == team_dire].index
			last_array[row_index][team_radiant_index] = 1. * radiant_elo_rating
			last_array[row_index][team_dire_index] = -1. * dire_elo_rating
			
			for index in range(0, 10):
				
				hero_index = hero_label[hero_label[0] == values[index]].index
				if index < 5:				
					last_array[row_index][len(team_label) + hero_index] = 1. * radiant_elo_rating
				elif index >= 5:
					last_array[row_index][len(team_label) + hero_index] = -1. * dire_elo_rating

		return last_array
		
	elo_rating = get_elo_rating()
	parsed_match = assimilate_data(data, hero_label, team_label, elo_rating)

	return hero_label, team_label, parsed_match