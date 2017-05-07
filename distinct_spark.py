#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:49:32 2017

@author: hxr
"""

from pyspark import SparkContext
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
    # configuration
	sc = SparkContext("local", "distinct_wordcount")
    # input the file
	text = sc.textFile('pg2701.txt')
    # using flatMap method to apply splitter function to all elements
	words = text.flatMap(splitter)
    # for the same word, we only keep one and remove others
	words_distinct = words.distinct()
    # count distinct words
	counts = words_distinct.count()
	print("distinct words:",counts)
    
