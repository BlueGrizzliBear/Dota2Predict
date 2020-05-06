from vector import Vector

class Matrix:
	def __init__(self, data=None, shape=None):
		if isinstance(data, list) and shape==None:
			for element in data:
				if isinstance(element, list)==False:
					print("List is not a matrice")
					pass
			self.data = data
			self.shape = (len(data), len(data[0]))
		if data==None and isinstance(shape, tuple):
			self.shape = shape
			self.data = []
			element = []
			for y in range(shape[1]):
				element.append(0)
			for x in range(shape[0]):
				self.data.append(element)
		if isinstance(data, list) and isinstance(shape, tuple):
			if len(data)==shape[0] and len(data[0])==shape[1]:
				self.data = data
				self.shape = shape
	def __add__(self, x):
		if isinstance(x, Matrix):
			if x.shape == self.shape:
				temp = []
				for i in range(x.shape[0]):
					element = [a + b for a, b in zip(x.data[i], self.data[i])]
					temp.append(element)
				return Matrix(temp)
			else:
				print("Matrix not the same shape")
		# elif isinstance(x, Matrix) and isinstance(self, Vector):
		# 	temp = Matrix(None, (len(x), len(x[0])))
		# 	if x.shape[0] == 1 and x.shape[1] == self.size:
		# 		for i in range(len(self)):
		# 			temp[1].append(temp[1][i] + self[i])
		else:
			print("Adding with somthing other than a scalar not possible")
			pass
	def __radd__(self, x):
		print("Not possible")
		pass
	def __sub__(self, x):
		if isinstance(x, Matrix):
			if x.shape == self.shape:
				temp = []
				for i in range(x.shape[0]):
					element = [a - b for a, b in zip(x.data[i], self.data[i])]
					temp.append(element)
				return Matrix(temp)
			else:
				print("Matrix not the same shape")
		else:
			print("Substracting with somthing other than a scalar not possible")
			pass

	def __rsub__(self, x):
		print("Not possible")
		pass

	def __truediv__(self, x):
		if isinstance(x, (int, float)):
			if x != 0:
				temp = []
				for i in range(self.shape[0]):
					element = [a / x for a in self.data[i]]
					temp.append(element)
				return Matrix(temp)

			else:
				print("Dividing with 0 not possible")
		else:
			print("Dividing with somthing other than a scalar not possible")

	def __rtruediv__(self, x):
		print("Not possible")
		pass
	def __mul__(self, x):
		if isinstance(x, (int, float)):
			temp = []
			for i in range(self.shape[0]):
				element = [a * x for a in self.data[i]]
				temp.append(element)
			return Matrix(temp)
		elif isinstance(x, Matrix):
			if x.shape[1] == self.shape[0]:
				tmatrice = []
				for i in range(x.shape[1]):
					element = []
					for j in range(x.shape[0]):
						element.append(x.data[j][i])
					tmatrice.append(element)
				temp = []
				for i in range(self.shape[0]):
					elem = []
					for j in range(self.shape[0]):
						element = [a * b for a, b in zip(self.data[i], tmatrice[j])]
						element = sum(element)
						elem.append(element)
					temp.append(elem)
				return Matrix(temp)
			else:
				print("Incompatible size matrix")

		elif isinstance(x, Vector):
			if x.size==self.shape[1]:
				for nbr in x.values:
					if isinstance(nbr, (float, int))==False:
						print("List contains invalid elements")
				temp = []
				for i in range(self.shape[0]):
					element = [a * b for a, b in zip(self.data[i], x.values)]
					element = sum(element)
					temp.append(element)
				return Vector(temp)
			else:
				print("Wrong vector size")
		else:
			print("Invalid element for multiplying")
			pass

	def __rmul__(self, x):
		if isinstance(x, (int, float)):
			temp = []
			for i in range(self.shape[0]):
				element = [a * x for a in self.data[i]]
				temp.append(element)
			return Matrix(temp)
		elif isinstance(x, Vector):
			if x.size==self.shape[1]:
				for nbr in x.values:
					if isinstance(nbr, (float, int))==False:
						print("List contains invalid elements")
				temp = []
				for i in range(self.shape[0]):
					element = [a * b for a, b in zip(self.data[i], x.values)]
					element = sum(element)
					temp.append(element)
				return Vector(temp)
			else:
				print("Wrong vector size")
		else:
			print("Invalid element for multiplying")
			pass

	def __str__(self):
		"""Return the string to print with the recipe info"""
		txt = "Matrix (" + str(self.data) + ")"
		return txt
	def __repr__(self):
		return self.data
