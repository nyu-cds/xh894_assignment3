#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:41:03 2017

@author: hxr
"""

from pyspark import SparkContext
from operator import mul

# calculates the product of all the numbers from 1 to 1000 and prints the result.

if __name__ == '__main__':
    # configuration
    sc = SparkContext("local", "Factorial Product")
    # create an RDD containing the numbers from 1 to 1000.
    nums = sc.parallelize(range(1, 1001))
    # Using 1 as default value, fold partitions using multiplication
    product = nums.fold(1, mul)

    print("product:", product)