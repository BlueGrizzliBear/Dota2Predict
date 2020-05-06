class Vector:
	def __init__(self, value):
		if isinstance(value, int):
			self.size = value
			self.values = [x for x in range(0, value)]
		if isinstance(value, tuple) and len(value) == 2:
			self.size = value[1] - value[0]
			self.values = [x for x in range(value[0], value[1])]
		if isinstance(value, list):
			self.size = len(value)
			self.values = [x for x in value]
	def __add__(self, x):
		if isinstance(x, Vector):
			if x.size == self.size:
				temp = [a + b for a, b in zip(x.values, self.values)]
				return Vector(temp)
			else:
				print("Different size vectors")
				pass
		# elif isinstance(x, (int, float)):
		# 	temp = [x + val for val in self.values]
		# 	return Vector(temp)
		else:
			print("Not possible")
			pass
	def __radd__(self, x):
		if isinstance(x, (int, float)) and self.size == 1:
			temp = [x + val for val in self.values]
			return Vector(temp)
		else:
			print("Not possible")
			pass
	def __sub__(self, x):
		if isinstance(x, Vector):
			if x.size == self.size:
				temp = [b - a for a, b in zip(x.values, self.values)]
				return Vector(temp)
			else:
				print("Different size vectors")
				pass
		# elif isinstance(x, (int, float)):
		# 	temp = [val - x for val in self.values]
		# 	return Vector(temp)
		else:
			print("Not possible")
			pass
	def __rsub__(self, x):
		if isinstance(x, (int, float)) and self.size == 1:
			temp = [x - val for val in self.values]
			return Vector(temp)
		else:
			print("Not possible")
			pass
	def __truediv__(self, x):
		if isinstance(x, Vector):
			if x.size == 1:
				if x.values[0] != 0:
					temp = [b / x.values[0] for b in self.values]
					return Vector(temp)
				else:
					print("Dividing with 0 not possible")
			else:
				print("Dividing with Vector not possible")
			pass
		elif isinstance(x, (int, float)):
			if x != 0:
				temp = [val / x for val in self.values]
				return Vector(temp)
			else:
				print("Dividing with 0 not possible")
		else:
			print("Not possible")
			pass
	def __rtruediv__(self, x):
		if isinstance(x, (int, float)) and x == 0:
			temp = 0
			return Vector(temp)
		else:
			print("Not possible")
		pass
	def __mul__(self, x):
		if isinstance(x, Vector):
			if x.size == self.size:
				temp = [b * a for a, b in zip(x.values, self.values)]
				temp = sum(temp)
				return temp
			else:
				print("Multiplying different size Vector not possible")
				pass
		elif isinstance(x, (int, float)):
			temp = [val * x for val in self.values]
			return Vector(temp)
		else:
			print("Not possible")
			pass
	def __rmul__(self, x):
		if isinstance(x, (int, float)):
			temp = [x * val for val in self.values]
			return Vector(temp)
		else:
			print("Not possible")
			pass
	def __str__(self):
		"""Return the string to print with the recipe info"""
		txt = "Vector (" + str(self.values) + ")"
		return txt
	def __repr__(self):
		return repr(self.values)
