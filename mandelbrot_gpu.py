#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:27:24 2017

@author: hxr
"""

# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show
import math

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
    
    '''
    #get the height and width of image
    height = image.shape[0]
    width = image.shape[1]
    #get the pixel size
    pixel_size_y = (max_y - min_y) / height
    pixel_size_x = (max_x - min_x) / width
    
    # obtain the starting x and y coordinates 
    x_start, y_start = cuda.grid(2)

    # calculate the ending x and y coordinates 
    y_end = cuda.gridDim.y * cuda.blockDim.y 
    x_end = cuda.gridDim.x * cuda.blockDim.x
    
    # get the number of block of image
    range_x = int(math.ceil(width / x_end))
    range_y = int(math.ceil(height / y_end))
    # compute the mandel value for each element of the block
    for i in range(range_x):
        x = x_start + x_end * i
        real = min_x + x * pixel_size_x
        for j in range(range_y):
            y = y_start + y_end * j
            # make sure that x,y would not be out of range
            if ((x < width) & (y < height)):
                imag = min_y + y * pixel_size_y
                
                image[y, x] = mandel(real, imag, iters)
    

    
if __name__ == '__main__':
	image = np.zeros((1024, 1536), dtype = np.uint8)
	blockdim = (32, 8)
	griddim = (32, 16)
	
	image_global_mem = cuda.to_device(image)
	compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
	image_global_mem.copy_to_host()
	imshow(image)
	show()