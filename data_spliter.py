import numpy as np

def data_spliter(x, y, proportion):
    rand = np.arange(len(x))
    np.random.shuffle(rand)
    x = x[rand]
    y = y[rand]
    prop = len(x) * proportion
    prop = int(prop)
    Xsplit1 = np.delete(x, range(prop, len(x)), axis=0)
    Xsplit2 = np.delete(x, range(0, prop), axis=0)
    Ysplit1 = np.delete(y, range(prop, len(x)), axis=0)
    Ysplit2 = np.delete(y, range(0, prop), axis=0)
    return (Xsplit1, Xsplit2, Ysplit1, Ysplit2)