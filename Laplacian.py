"""
Compute Laplacian of a function of a rectangular domain with uniform grid spacing in x and y
"""

from mpi4py import MPI
import sys
import h5py
import numpy as np
import time

import tools as tls

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# definition in indices
i_north = 0; i_south = 1; i_east = 2; i_west = 3; Ncardinaldirs = 4
i_ori = 0; i_N = 1
i_x = 0; i_y = 1
if rank == 0:
    inputfilename = "Laplacian.in"
    f = open(inputfilename,'r')
    inputfile = f.readlines()
    f.close()
    Lx = tls.parser(inputfile,"Lx/(2pi)","FLOAT")
    Lx *= 2*np.pi # Lx= Lx * 2*np.pi
    Ly = tls.parser(inputfile,"Ly/(2pi)","FLOAT")
    Ly *= 2*np.pi
    Nx = tls.parser(inputfile,"Nx","INT")
    Ny = tls.parser(inputfile,"Ny","INT")
    input_function = tls.parser(inputfile,"Function","STR")
    outputfile = tls.parser(inputfile,"Output","STR")
    
    npx = tls.parser(inputfile,"# of processes in x","INT")
    if (npx == 0):
        sys.exit("npx must be large or equal to 1")
    npy = tls.parser(inputfile,"# of processes in y","INT")
    if (npy == 0):
        sys.exit("npy must be large or equal to 1")
    if (npx*npy != size):
        sys.exit("Number of processes in Laplacian.in is not equal to nb of procs. requested")

    
    # Nodes number along i and j
    inodes = np.arange(Nx,dtype=np.int32)
    jnodes = np.arange(Ny,dtype=np.int32)
    # Partioning of inodes and jnodes as a function of requested number of processes npx, and npy
    inodes_split = (np.array_split(inodes,npx))
    jnodes_split = (np.array_split(jnodes,npy))
    # Mapping partitions in 2D defined as offset (i_ori) and length, i_N
    
    #partitions[number of procs,(i_x,i_y),(i_ori,i_N)]
    partitions = np.zeros((size,2,2),dtype=np.int32)
    # definition of process location on the ipx,ipy grid
    proc_to_partition = np.zeros((size,2), dtype = np.int32)
    # location of ipx,ipy proc on number of processes
    partition_to_proc = np.zeros((npx,npy), dtype = np.int32)
    
    # Fill partitions, proc_to_partition, partition_to_proc
    ip = 0
    for ipx in range(npx):
        for ipy in range(npy):
            partitions[ip,i_x,i_ori] = inodes_split[ipx][0]
            partitions[ip,i_x,i_N] = len(inodes_split[ipx])
            partitions[ip,i_y,i_ori] = jnodes_split[ipy][0]
            partitions[ip,i_y,i_N] = len(jnodes_split[ipy])
            partition_to_proc[ipx,ipy] = ip
            proc_to_partition[ip,i_x] = ipx
            proc_to_partition[ip,i_y] = ipy
            ip += 1
    # Definition of neighbors
    
    # neighbors[nb of procs, cardinal dirs]
    neighbors = np.zeros((size,Ncardinaldirs),dtype = np.int32)
    for ip in range(size):
        # get location of process on ipx ipy grid
        ipx = proc_to_partition[ip,i_x]
        ipy = proc_to_partition[ip,i_y]
        if npx > 1: # if npx = 1 periodicity is enforced in derivation function
            if ipx == 0: # takes care periodicity on west boundary
                neighbors[ip,i_west] = partition_to_proc[npx - 1,ipy]
            else: # neighbor is the proc adjacent to west boundary
                neighbors[ip,i_west] = partition_to_proc[ipx - 1,ipy]
            if ipx == npx - 1: # periodity on east boundary
                neighbors[ip,i_east] = partition_to_proc[0,ipy]
            else:
                neighbors[ip,i_west] = partition_to_proc[ipx + 1,ipy]
        if (npy > 1): 
            if ipy == 0:
                neighbors[ip,i_south] = partition_to_proc[ipx,npy - 1]
            else:
                neighbors[ip,i_north] = partition_to_proc[ipx,ipy - 1]
            if ipy == npy - 1:
                neighbors[ip,i_north] = partition_to_proc[ipx,0]
            else:
                neighbors[ip,i_north] = partition_to_proc[ipx,ipy + 1]
        print(neighbors[ip,:])  # debug      
                  
    # Definition of grid metrics and variable
    dx = Lx / Nx
    dy = Lx / Ny
    x = np.linspace(dx/2., Lx - dx/2.,Nx)
    y = np.linspace(dy/2., Ly - dy/2.,Ny)
    X,Y = np.meshgrid(x,y) # -> Ny,Nx matrix
    X = X.T # -> Nx,Ny matrix
    Y = Y.T
    #Definition of the variable to be derived on the entire domain
    if input_function == 'p':
        var = np.cos(2*X) + np.sin(2*Y)
    else:
        sys.exit("Function %s is not defined" %(input_function))
    ip = 0
    for ipx in range(npx):
        for ipy in range (npy):
            if (ip == 0):
                print(partitions[ip,i_x,i_ori],partitions[ip,i_x,i_ori] + partitions[ip,i_x,i_N])
                print(partitions[ip,i_y,i_ori],partitions[ip,i_y,i_ori] + partitions[ip,i_y,i_N])
                my_var = var[partitions[ip,i_x,i_ori]:partitions[ip,i_x,i_ori] + partitions[ip,i_x,i_N],
                       partitions[ip,i_y,i_ori]:partitions[ip,i_y,i_ori] + partitions[ip,i_y,i_N]]
            else:
                buffer = var[partitions[ip,i_x,i_ori]:partitions[ip,i_x,i_ori] + partitions[ip,i_x,i_N],
                           partitions[ip,i_y,i_ori]:partitions[ip,i_y,i_ori] + partitions[ip,i_y,i_N]]
                comm.send(buffer,dest = ip, tag = ip)
            ip += 1
else: # rank > 0
    my_var = comm.recv(source = 0, tag = rank)
    
my_neighbors = comm.bcast(neighbors[rank,:], root = 0)    
neighbors[rank,i_east]
