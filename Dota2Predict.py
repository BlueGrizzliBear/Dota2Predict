import numpy as np
import pandas as pd
from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression
from parse import convert_draft_str_to_array, dataframe_to_draft_str, remove_empty_row

def main():

# FOR PUB GAMES
	# data = pd.read_csv("./resources/dota2_pub_matchs.csv")
	# x = np.array(data[['radiant_team', 'dire_team']])
	# y = np.array(data['radiant_win']).astype(float)
	# t = convert_draft_str_to_array(x)

# FOR PRO GAMES
	data = pd.read_json("./resources/data.json")
	data = remove_empty_row(data)
	x = dataframe_to_draft_str(data)
	y = np.array(data['win']).astype(float)
	
	t = convert_draft_str_to_array(x)
	x_label = t[0]
	x = t[1]

	shape = (len(x[0])+ 1,)
	thetas = np.zeros(shape)
	# thetas = np.full_like(x[0], 0.5, dtype=float, shape=(len(x[0])+ 1,))
	thetas[0] = 0

	t = data_spliter(x, y, 0.8)
	xtrain = t[0]
	xtest = t[1]
	ytrain = t[2]
	ytest = t[3]

	alpha = 5e-1
	n_cycle = 100

	MyLR = MyLogisticRegression(thetas, alpha, n_cycle)

	# pred = MyLR.logistic_predict_(xtrain)
	# print (pred)
	cost = MyLR.cost_(xtrain, ytrain)
	print ("Cost = " + str(cost))
	mse = MyLR.mse_(xtest, ytest)
	print ("MSE = " + str(mse))

	new_thetas = MyLR.fit_(xtrain, ytrain)
	# print ("New_thetas :")
	# print (new_thetas)

	cost = MyLR.cost_(xtrain, ytrain)
	print ("Cost = " + str(cost))
	mse = MyLR.mse_(xtest, ytest)
	print ("MSE = " + str(mse))

	pred = np.round_(MyLR.logistic_predict_(xtest), 2)
	# i = 0
	# while i < len(pred):
	# 	print ("Comparing ===")
	# 	print ("Pred. = " + str(pred[i]))
	# 	print ("Truth = " + str(y[i]))
	# 	i += 1
	print (pred)
	print (ytest)
	accuracy = MyLR.accuracy_score_(xtest, ytest) 
	print ("Accuracy = " + str(accuracy))

	# the_x_test = np.array([['86,43,84,114,96', '111,95,110,38,39']])
	# y_test = np.array([1.0])
	# t = parse(the_x_test)
	# x_label_test = t[0]
	# x_test = t[1]
	# print (x_test)

	# print ("new TEST =================")
	# pred = np.round_(MyLR.logistic_predict_(x_test), 2)
	# print (pred)
	# print (y_test)

	pass

if __name__ == "__main__":
    main()