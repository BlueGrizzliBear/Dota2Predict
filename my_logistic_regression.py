import numpy as np
from matrix import Matrix
from vector import Vector
import matplotlib.pyplot as plt
import time
from decorator import func_deco

class MyLogisticRegression():

	def __init__(self, thetas, alpha=5e-5, n_cycle=40000):
		self.alpha = alpha
		self.max_iter = n_cycle
		if isinstance(thetas, list):
			self.thetas = np.array(thetas).flatten()
		else:
			self.thetas = thetas.flatten()	

	def log_gradient(self, x, y, theta):
		# theta = theta.flatten()
		X = self.add_intercept(x)
		X1 = Matrix(X.tolist())
		t = Vector(theta.tolist())
		y_hat = X1*t
		y_hat = Vector(self.sigmoid_(y_hat.values).tolist())
		Xt = X.transpose()
		Xt = Matrix(Xt.tolist())
		Y = Vector(y.tolist())
		X1 = y_hat - Y
		nabla = Xt*X1
		nabla = np.asarray(nabla.values)
		nabla = (nabla/len(y)).astype(float)
		return nabla

	def	ft_progress(self, lst):
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

	@func_deco
	def fit_(self, x, y):
		# y = y.flatten()
		# self.thetas = self.thetas.flatten()
		max_iter = range(self.max_iter)
		for elem in self.ft_progress(max_iter):
			grad = self.log_gradient(x, y, self.thetas)
			self.thetas = self.thetas - self.alpha * grad
			for item in self.thetas:
				if np.isnan(item):
					return self.thetas
		print ("")
		return self.thetas		

	def add_intercept(self, x):
		tmp = np.copy(x)
		res = np.c_[np.ones(x.shape[0]), tmp]
		return res

	def sigmoid_(self, x):
		if isinstance(x, list):
			sig = []
			for item in x:
				res = 1 / (1 + np.exp(-1 * item))
				sig.append(res)
			sig = np.array(sig)
			return sig
		elif isinstance(x, np.ndarray):
			if len(x.shape) == 0:
				sig = 1 / (1 + np.exp(-1 * x))
				sig = np.array(sig)
			else:		
				sig = np.zeros(x.shape)
				for index, value in enumerate(x):
					sig[index] = 1 / (1 + np.exp(-1 * value))
			return sig
		else:
			print ("ERROR")

	def logistic_predict_(self, x):
		# theta = self.thetas.flatten()
		X = self.add_intercept(x)
		X1 = Matrix(X.tolist())
		T1 = Vector(self.thetas.tolist())
		y_hat = X1*T1
		y_hat = np.array(y_hat.values)
		y_hat = self.sigmoid_(y_hat)
		return y_hat		

	def cost_(self, x, y, eps=1e-15):
		y_hat = self.logistic_predict_(x)
		if len(y.shape) > 1:
			y = y.flatten()
		if len(y_hat.shape) > 1:
			y_hat = y_hat.flatten()
		if y.shape != y_hat.shape:
			return None
		# y = y.flatten()
		log_loss = sum(y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps))/(-len(y))
		return log_loss

	def accuracy_score_(self, x, y):
		y_hat = self.logistic_predict_(x)
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for index, game_res in enumerate(y_hat):
			if game_res > 0.5 and y[index] == 1:
				tp += 1
			elif game_res < 0.5 and y[index] == 0:
				tn += 1
			elif game_res > 0.5 and y[index] == 0:
				fp += 1
			elif game_res < 0.5 and y[index] == 1:
				fn += 1
		res = (tp + tn) / (tp + fn + tn + fp)
		return str(round(100 * res, 2)) + "%"

	def mse_(self, x, y):
		y_hat = self.logistic_predict_(x)
		if len(y.shape) > 1:
			y = y.flatten()
		if len(y_hat.shape) > 1:
			y_hat = y_hat.flatten()
		if y.shape != y_hat.shape:
			return None
		m1 = y_hat.shape[0]
		J_mse = sum((1 / m1) * pow(y_hat - y, 2))
		return J_mse