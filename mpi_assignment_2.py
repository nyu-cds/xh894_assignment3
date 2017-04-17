#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:51:54 2017

Requirement:
    
Write an MPI program that does the following for some arbitrary number
of processes:

1.Process 0 reads a value from the user and verifies that it is an integer 
  less than 100.
2.Process 0 sends the value to process 1 which multiplies it by its rank.
3.Process 1 sends the new value to process 2 which multiplies it by its rank.
4.This continues for each process, such that process i sends the value to process 
  i+1 which multiplies it by i+1.
5.The last process sends the value back to process 0, which prints the result.
    

@author: hxr
"""
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
# get the rank of the process within the communicator.
rank = comm.Get_rank()
# get the size of the process within the communicator.
size = comm.Get_size()
# get_input() function is used to get the input number validated correctly
def get_input():
    user_input = input("Please Enter a Number : ")
    # check the input is integer or not
    try:
        result = int(user_input)
    except ValueError:
        print("Input is not a integer!")
        return 
    # check the input is less than 100 or not
    if result >= 100:
        print("Input is out of range!")
    else:
        return result
#initialize an array    
num = np.zeros(1)
    
if rank == 0:
    start_number = get_input()
    # input a number until it is an integer and less than 100
    while not isinstance(start_number, int):
        print("Try again!")
        start_number = get_input()
    # multiply by rank
    num[0] = start_number*(rank+1)
    # send to the next process
    comm.Send(num, dest=rank+1)
    print("Process", rank, "sent num to Process", rank+1)
    # receive from the last process
    comm.Recv(num, source = size - 1)
    # print the result
    print("Result-------",num[0])

if rank > 0 :
    # receive from the former process
    comm.Recv(num, source = rank-1)
    print("Process", rank, "received num from Process",rank-1)
    # multiply by rank
    num[0] *=(rank+1)
    # if not the last process
    if rank < size - 1:    
        comm.Send(num, dest = rank + 1) 
        print("Process", rank, "sent num to Process", rank+1)
    # if the last process
    else:
        comm.Send(num, dest = 0)
        print("Process", rank, "sent num to Process", 0)

    

    
    


