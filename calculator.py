# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 
'''
    Original runtime: 1.476s
    Improved runtime: 0.016s
    Speedup: 92.25
    
    What I do is using Numpy-native functions to replace the functions defined
    by ourselves. And it really improved the behavior of our codes a lot.
        
'''
import numpy as np

def add(x,y):
    """
    Add two arrays using a numpy function.
    x and y must be two-dimensional arrays of the same shape.
    """
    return np.add(x,y)


def multiply(x,y):
    """
    Multiply two arrays using a numpy function.
    x and y must be two-dimensional arrays of the same shape.
    """
    return np.multiply(x,y)


def sqrt(x):
    """
    Take the square root of the elements of an arrays using a numpy function.
    """
    return np.sqrt(x)


def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = multiply(x,x)
    yy = multiply(y,y)
    zz = add(xx, yy)
    return sqrt(zz)