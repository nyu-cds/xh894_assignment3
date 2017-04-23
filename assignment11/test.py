#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:27:08 2017

@author: hxr

Test

"""
import unittest
from parallel_sorter import sort, get_input
from mpi4py import MPI
import numpy as np

class TestSorter(unittest.TestCase):

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
    def tearDown(self): 
        pass
    def test_get_input(self):
        '''
        check the input 
        '''
        if self.rank == 0:
            inp = get_input()
            self.assertIsInstance(inp, int)

    def test_sort(self):
        '''
        For process 0, check whether results are sorted correctly.
        For other process, check whether their results are None.
        '''
        result = sort()
        if self.rank == 0:
            assert np.array_equal(result,sorted(result)), "Wrong Order"
            
        else:
            self.assertEqual(None, result)

if __name__ == '__main__':
    unittest.main()