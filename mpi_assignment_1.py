#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requirement:
    Write an MPI program in which the processes with even rank print “Hello” 
    and the processes with odd rank print “Goodbye”. Print the rank along with 
    the message (for example “Goodbye from process 3”)

@author: hxr
"""
from mpi4py import MPI

# get the communicator
comm = MPI.COMM_WORLD

# get the rank of the process within the communicator.
rank = comm.Get_rank()

# if the process has odd rank
if rank % 2 == 1:
    print("Goodbye from process", rank)

# if the process has even rank
if rank % 2 == 0:
    print("Hello from process", rank)