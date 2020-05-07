def func_deco(func):
	def wrapper(*args, **kwargs):
		print (f"Executing : {func.__name__} . . .")
		ret = func(*args, **kwargs)
		print ("Done")
		return ret
	return wrapper