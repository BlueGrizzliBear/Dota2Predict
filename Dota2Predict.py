import numpy as np
import pandas as pd
from decorator import print_colored
from DataExtractor import Data_Extractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
# FOR PRO GAMES
	t = Data_Extractor()
	hero_label = t[0]
	team_label = t[1]
	x = t[2]
	y = t[3]
	data_team_column = t[4]

	print_colored("Reshaping x(a,b,c) into x(a,b*c) . . .", '\033[94m')
	nsamples, nx, ny = x.shape
	x_reshaped = x.reshape((nsamples,nx*ny))
	print_colored("Done !", '\033[92m')	

	split = 0.8
	print_colored("Spliting data " + str(round(100 * split)) + "%/" + str(round(100 * (1-split))) + "% . . .", '\033[94m')
	t = train_test_split(x_reshaped, y, train_size=0.8, shuffle=False)
	xtrain = t[0]
	xtest = t[1]
	ytrain = t[2]
	ytest = t[3]
	print_colored("Done !", '\033[92m')	

	alpha = 5e-1
	n_cycle = 10000

	print_colored("Creating Logistic_Regression Model . . .", '\033[94m')
	myLogR = LogisticRegression(penalty='none', tol=alpha, max_iter=n_cycle)
	print_colored("Done !", '\033[92m')	

	print_colored("Fitting model . . .", '\033[94m')	
	myLogR.fit(xtrain, ytrain)
	print_colored("Done !", '\033[92m')	
	

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

	def display_diff(pred, ytest):
		for index, value in enumerate(ytest):
			if value == pred[index]:
				print_colored("OK", '\033[92m')	
			else:
				print_colored("NOK: pred(" + str(pred[index]) + ") vs test(" + str(value) + ")", '\033[91m')
		pass

# TENDENCIES ON Xtest
	print_colored("====== Applying to Xtest . . . ======", '\033[94m')	

	print_colored("Predicting model . . .", '\033[94m')	
	pred = myLogR.predict(xtest)
	print_colored("Done !", '\033[92m')	

	# print_colored("Displaying results on test_set . . .", '\033[94m')
	# display_diff(pred, ytest)

	print_colored("Constructing tendancies on Teams . . .", '\033[94m')
	x_test_result = construct_diff_with_teams(pred, ytest, data_team_column, team_label)
	
	print_colored("Calculating accuracy . . .", '\033[94m')
	# accuracy = myLogR.score(xtest, ytest)
	accuracy = myLogR.score(xtest, ytest)
	if accuracy > 0.5:
		print_colored("Accuracy = " + str(accuracy), '\033[92m')	
	else:
		print_colored("Accuracy = " + str(accuracy), '\033[91m')	

# TENDENCIES ON TOTAL X
	print_colored("====== Applying to X . . . ======", '\033[94m')	

	print_colored("Predicting model . . .", '\033[94m')	
	pred = myLogR.predict(x_reshaped)
	print_colored("Done !", '\033[92m')	

	# print_colored("Displaying results . . .", '\033[94m')
	# display_diff(pred, y)
	
	print_colored("Constructing tendancies on Teams . . .", '\033[94m')
	x_result = construct_diff_with_teams(pred, y, data_team_column, team_label)

	print_colored("Calculating accuracy . . .", '\033[94m')
	# accuracy = myLogR.score(xtest, ytest)
	accuracy = myLogR.score(x_reshaped, y)
	if accuracy > 0.5:
		print_colored("Accuracy = " + str(accuracy), '\033[92m')	
	else:
		print_colored("Accuracy = " + str(accuracy), '\033[91m')
	
# DIFFERENCE OF TENDENCIES
	x_test_result2 = x_test_result
	x_result2 = x_result
	frames = [x_result2['Team'], x_result2['Accuracy_score'], x_test_result2['Accuracy_score'], x_result2['Total'], x_test_result2['Total']]

	total = pd.concat(frames, axis=1, sort=False)
	total = total.dropna()
	total.columns = ['Team','Accuracy_score_model','Accuracy_score_test','Total_model', 'Total_test']
	total['Diff'] = np.zeros(total.shape[0])
	total['Diff'] = total['Accuracy_score_test'] - total['Accuracy_score_model']

	print (total.sort_values(["Total_model", "Total_test", "Accuracy_score_test"], ascending=[True, True, True]))
	# print (total)
	pass	

if __name__ == "__main__":
    main()