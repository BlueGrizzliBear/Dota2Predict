# A simple decorator to for displaying info
def func_deco(func):
	def wrapper(*args, **kwargs):
		print (f"Executing : {func.__name__} . . .")
		ret = func(*args, **kwargs)
		print ("Done")
		return ret
	return wrapper

# A simple function to put some colors into display
def print_colored(string, color1):
	print (color1 + string + '\033[0m')