#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:56:23 2017

@author: hxr
"""

from pyspark import SparkContext
from operator import add

# Calculate the average of the square root of all the numbers from 1 to 1000.
# Map all elements in RDD to square roots. 
# Then calculate the average.

if __name__ == '__main__': 
    # configuration
    sc = SparkContext("local", "sqrt avg")
    # create an RDD containing the numbers from 1 to 1000.
    nums = sc.parallelize(range(1, 1001))
    # Map all elements to their square roots.
    roots = nums.map(lambda x: x ** 0.5)
    # calculate the average of the sqrt
    average = roots.fold(0, add) / roots.count()

    print("Average of Square Roots :", average)