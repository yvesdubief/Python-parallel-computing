"""
Parallel Hello World
"""

from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name() #optional only useful if running on different machines

sys.stdout.write(
    "Hello, I am process %d of %d on %s.\n" 
    % (rank, size, name))
