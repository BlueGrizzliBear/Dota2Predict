import numpy as np
import pandas as pd
from decorator import print_colored
from DataExtractor import Data_Extractor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
# import sklearn.metrics
from datetime import date
from parse2 import parse_match_data

def main():
# FOR PRO GAMES
	t = Data_Extractor()
	hero_label = t[0]
	team_label = t[1]
	x = t[2]
	y = t[3]
	data_team_column = t[4]
	sample_weight = t[5]

	# split = 0.85
	# print_colored("Spliting data " + str(round(100 * split)) + "%/" + str(round(100 * (1-split))) + "% . . .", '\033[94m')
	# t = train_test_split(x, y, sample_weight, train_size=split, shuffle=False)
	# xtrain = t[0]
	# xtest = t[1]
	# ytrain = t[2]
	# ytest = t[3]
	# sample_weight_train = t[4]
	# sample_weight_test = t[5]	
	# print_colored("Done !", '\033[92m')	

	alpha = 5e-1
	n_cycle = 100000

	print_colored("Creating Logistic_Regression Model . . .", '\033[94m')
	myLogR = LogisticRegressionCV(penalty='l2', tol=alpha, max_iter=n_cycle, cv=2, solver='lbfgs')
	print_colored("Done !", '\033[92m')	

	print_colored("Fitting model on X . . .", '\033[94m')	
	myLogR.fit(x, y, sample_weight=sample_weight)
	print_colored("Done !", '\033[92m')	

	# print_colored("Calculating accuracy on X after new fit . . .", '\033[94m')
	# display_accuracy(myLogR.score(x, y, sample_weight=sample_weight))
	# print_colored("Done !", '\033[92m')	

	def construct_diff_with_teams(pred, y, data_team_column, team_label):
		result = team_label.copy()
		result['Cumulated_OK'] = np.zeros(result.shape[0])
		result['Total'] = np.zeros(result.shape[0])
		result['Accuracy_score'] = np.zeros(result.shape[0])
		result['Sort_coef'] = np.zeros(result.shape[0])
		for index, value in enumerate(y.values):

			rad_ind = data_team_column.iloc[index]['radiant']
			dir_ind = data_team_column.iloc[index]['dire']

			result.loc[lambda result: result['Team'] == rad_ind, result.columns[2]] += 1
			result.loc[lambda result: result['Team'] == dir_ind, result.columns[2]] += 1
			
			if value == pred[index]:
				if value == 1:
					result.loc[lambda result: result['Team'] == rad_ind, result.columns[1]] += 1
				else:
					result.loc[lambda result: result['Team'] == dir_ind, result.columns[1]] += 1
			result['Accuracy_score'] = result['Cumulated_OK'] / result['Total']			
			result['Sort_coef'] = result['Cumulated_OK'] * result['Accuracy_score']			
			
		# print (result.shape)
		result = result.dropna()
		# print (result.shape)
		pd.set_option('display.max_rows', None)
		# print (result.sort_values(["Sort_coef", "Accuracy_score"], ascending=[True, True]))
		return result

	def display_accuracy(accuracy):
		if accuracy > 0.5:
			print_colored("Accuracy = " + str(accuracy), '\033[92m')	
		else:
			print_colored("Accuracy = " + str(accuracy), '\033[91m')	

# TENDENCIES ON X
	print_colored("====== Applying to X . . . ======", '\033[94m')	

	print_colored("Calculating accuracy . . .", '\033[94m')
	accuracy = myLogR.score(x, y, sample_weight=sample_weight)
	display_accuracy(accuracy)
	print_colored("Done !", '\033[92m')	
	
	print_colored("Predicting model . . .", '\033[94m')	
	pred = myLogR.predict(x)
	print_colored("Done !", '\033[92m')	

	# print_colored("Displaying results . . .", '\033[94m')
	# display_diff(pred, y)
	
	print_colored("Constructing tendancies on Teams . . .", '\033[94m')
	x_result = construct_diff_with_teams(pred, y, data_team_column, team_label)

# # DIFFERENCE OF TENDENCIES
	# x_test_result2 = x_test_result
	x_test_result2 = x_result
	x_result2 = x_result
	frames = [x_result2['Team'], x_result2['Accuracy_score'], x_test_result2['Accuracy_score'], x_result2['Total'], x_test_result2['Total']]

	total = pd.concat(frames, axis=1, sort=False)
	total = total.dropna()
	total.columns = ['Team','Accuracy_score_model','Accuracy_score_test','Total_model', 'Total_test']
	total['Diff'] = np.zeros(total.shape[0])
	total['Diff'] = total['Accuracy_score_test'] - total['Accuracy_score_model']

	# print (total.sort_values(["Total_model", "Accuracy_score_test"], ascending=[True, True]))
	# print (total)

	team_wanted = ['Team Secret','PSG.LGD','Royal Never Give Up','Evil Geniuses','Fnatic',
	'Virtus.Pro','Vici Gaming','INVICTUS GAMING','EHOME','VP.Prodigy',
	'OG','Alliance','ViKin.gg','Team Liquid','CDEC '
	'Team Aster','Quincy Crew','beastcoast','Thunder Predator','Nigma',
	'Adroit Esports','TNC Predator','Chicken Fighters','Sparking Arrow Gaming','Natus Vincere',
	'OG.Seed','Gambit Esports','BOOM','FlyToMoon',
	'HellRaisers','Newbee','NoPing Esports','Ninjas in Pyjamas','Geek Fam']
	
	print (total[total.Team.isin(team_wanted)])
	print (total[total.Team.isin(team_wanted)].Accuracy_score_model.values.mean())

# TESTING ON A MATCH
	print_colored("====== Testing on a match . . . ======", '\033[94m')

# BIG WHILE
	while True:
	# TEST ON RADIANT TEAM WINS
		match_test_x = pd.DataFrame([
			{'slot_0':0,'slot_1':0,'slot_2':0,'slot_3':0,'slot_4':0,
			'slot_5':0,'slot_6':0,'slot_7':0,'slot_8':0,'slot_9':0,
			'radiant':"",
			'dire':"",
			'time':int(date.today().strftime("%Y%m%d%H%M%S"))},
			])

		def get_hero_labels():
			heroes = pd.read_csv("./resources_data/dota2_heroes.csv")
			match_hero_label = heroes[['id','localized_name']]
			return match_hero_label

		match_hero_label = get_hero_labels()
		nb = 0
		for nb in range(0, 10):
			hero_id = False
			while hero_id == False:
				hero_id = input("Pick_" + str(nb) + " : ")
				if match_hero_label[match_hero_label['localized_name'] == hero_id].id.values.size == 1:
					hero_id = match_hero_label[match_hero_label['localized_name'] == hero_id].id.values
				else:
					hero_id = False
			slot = "slot_" + str(nb)
			match_test_x[slot] = hero_id
		match_test_x['radiant'] = input("Radiant team name : ")
		match_test_x['dire'] = input("Dire team name : ")

		t = parse_match_data(match_test_x, team_label, hero_label)
		match_hero_label = t[0]
		match_team_label = t[1]
		match_x = t[2]

		print_colored("Displaying pred's accuracy for teams . . .", '\033[94m')	
		team_1 = match_test_x['radiant'].values
		team_2 = match_test_x['dire'].values

		print_colored("Model's accuracy " + str(team_1) + " . . .", '\033[94m')	
		print (total[total.Team.isin(team_1)]['Accuracy_score_model'].values)
		print_colored("Done !", '\033[92m')

		print_colored("Model's accuracy " + str(team_2) + " . . .", '\033[94m')	
		print (total[total.Team.isin(team_2)]['Accuracy_score_model'].values)
		print_colored("Done !", '\033[92m')	
		
		print_colored("Predicting " + str(team_1) + " wins . . .", '\033[94m')	
		print ("Pred : " + str(myLogR.predict(match_x)))
		print ("Proba : " + str(myLogR.predict_proba(match_x)))
		print_colored("Done !", '\033[92m')	

	# TEST ON DIRE TEAM WINS
		match_test_x = match_test_x.reindex(columns=['slot_5','slot_6','slot_7','slot_8','slot_9',
			'slot_0','slot_1','slot_2','slot_3','slot_4',
			'dire','radiant',
			'time'
			])
		match_test_x.columns = ['slot_0','slot_1','slot_2','slot_3','slot_4',
			'slot_5','slot_6','slot_7','slot_8','slot_9',
			'radiant','dire',
			'time'
			]
		print (match_test_x)
		t = parse_match_data(match_test_x, team_label, hero_label)
		match_hero_label = t[0]
		match_team_label = t[1]
		match_x = t[2]

		print_colored("Displaying pred's accuracy for teams . . .", '\033[94m')	
		team_1 = match_test_x['radiant'].values
		team_2 = match_test_x['dire'].values

		print_colored("Model's accuracy " + str(team_1) + " . . .", '\033[94m')	
		print (total[total.Team.isin(team_1)]['Accuracy_score_model'].values)
		print_colored("Done !", '\033[92m')	

		print_colored("Model's accuracy " + str(team_2) + " . . .", '\033[94m')	
		print (total[total.Team.isin(team_2)]['Accuracy_score_model'].values)
		print_colored("Done !", '\033[92m')	
		
		print_colored("Predicting " + str(team_1) + " wins . . .", '\033[94m')	
		print ("Pred : " + str(myLogR.predict(match_x)))
		print ("Proba : " + str(myLogR.predict_proba(match_x)))
		print_colored("Done !", '\033[92m')	

	pass	

if __name__ == "__main__":
    main()