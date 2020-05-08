import numpy as np
import pandas as pd
from decorator import print_colored
from DataExtractor import Data_Extractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
# FOR PRO GAMES - NEW
	t = Data_Extractor()
	hero_label = t[0]
	team_label = t[1]
	x = t[2]
	y = t[3]

	print_colored("Reshaping x(a,b,c) into x(a,b*c) . . .", '\033[94m')
	nsamples, nx, ny = x.shape
	x_reshaped = x.reshape((nsamples,nx*ny))
	print_colored("Done !", '\033[92m')	

	split = 0.8
	print_colored("Spliting data " + str(round(100 * split)) + "%/" + str(round(100 * (1-split))) + "% . . .", '\033[94m')
	t = train_test_split(x_reshaped, y, train_size=0.8)
	xtrain = t[0]
	xtest = t[1]
	ytrain = t[2]
	ytest = t[3]
	print_colored("Done !", '\033[92m')	

	alpha = 5e-1
	n_cycle = 10000

	print_colored("Creating Logistic_Regression Model . . .", '\033[94m')
	myLogR = LogisticRegression(penalty='none', tol=alpha, max_iter=n_cycle, verbose=1)
	print_colored("Done !", '\033[92m')	

	print_colored("Fitting model . . .", '\033[94m')	
	myLogR.fit(xtrain, ytrain)
	print_colored("Done !", '\033[92m')	
	
	print_colored("Predicting model with xtest . . .", '\033[94m')	
	pred = myLogR.predict(xtest)
	print_colored("Done !", '\033[92m')	

	def display_diff(pred, ytest):
		for index, value in enumerate(ytest):
			if value == pred[index]:
				print_colored("OK", '\033[92m')	
			else:
				print_colored("NOK: pred(" + str(pred[index]) + ") vs test(" + str(value) + ")", '\033[91m')
		pass

	print_colored("Displaying results on test_set . . .", '\033[94m')
	display_diff(pred, ytest)

	print_colored("Calculating accuracy . . .", '\033[94m')
	accuracy = myLogR.score(xtest, ytest)
	if accuracy > 0.5:
		print_colored("Accuracy = " + str(accuracy), '\033[92m')	
	else:
		print_colored("Accuracy = " + str(accuracy), '\033[91m')	

	pass	

if __name__ == "__main__":
    main()