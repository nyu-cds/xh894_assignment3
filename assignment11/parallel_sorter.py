#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:11:01 2017

@author: hxr
"""
import numpy as np
from mpi4py import MPI
def get_input():
    user_input = input("Please Enter a Number : ")
    # check the input is integer or not
    try:
        result = int(user_input)
    except ValueError:
        print("Input is not a integer!")
        return 
    
    return result
def sort():
    comm=MPI.COMM_WORLD
    # get the rank of the process within the communicator.
    rank=comm.Get_rank() 
    # get the size of the process within the communicator.
    size=comm.Get_size() 
    # get_input() function is used to get the input number validated correctly
    
    
    if rank ==0:
        # set the amount of random data set
        N = get_input()
        # input a number until it is an integer and less than 100
        while not isinstance(N, int):
            print("Try again!")
            N = get_input()
        # generate the unsorted N data set range from 0 to 1000
        data = np.random.randint(0,1000,int(N))
        # get the max and min of data 
        Max = max(data)
        Min = min(data)
        # set the partition
        part = np.linspace(Min,Max+1,size+1)
        # set the data set to scatter
        data_ = []
        for i in range(size):
            data_.append(list(filter(lambda x : part[i] <= x < part[i+1], data)))
    else:
        data_=None
    
    # scatter the data      
    local_data = comm.scatter(data_, root=0)
    # sort sort the data
    local_data_sorted = sorted(local_data)
    # gether the data
    combine_data = comm.gather(local_data_sorted, root=0)
    
    if rank == 0:
        # concatenate the data
        combine_data = np.concatenate(combine_data)
        compare_data = np.sort(data)
        
        assert np.array_equal(combine_data,compare_data), "Wrong Order"
        print(combine_data)
        return combine_data
        
if __name__ == '__main__':
    sort()
    
    
    
