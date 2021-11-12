import h5py
import numpy as np
from scipy import integrate
import time

import tools as tls
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    inputfilename = "lorenz.in"
    f = open('lorenz.in','r')
    inputfile = f.readlines()
    f.close()
    N_trajectories = tls.parser(inputfile, "NUMBER OF TRAJECTORIES", "INT")
    Tintegration = tls.parser(inputfile, "INTEGRATION TIME", "FLOAT")
    Ntimesteps = tls.parser(inputfile, "NUMBER OF TIME STEPS", "INT")
    seed = np.int((time.time() - np.int(time.time()))*1e8)
    np.random.seed(seed)
    x0_all = -15 + 30 * np.random.random((N_trajectories, 3))
    print(x0_all[:,0])
    # distribution of tasks
    N_traj_proc = N_trajectories // size
    nresidual = N_trajectories - N_traj_proc * size
    tasks_per_proc = np.zeros(size,dtype = 'int')
    if nresidual == 0:
        tasks_per_proc[:] = N_traj_proc       
    else: 
        tasks_per_proc[:-1] = N_traj_proc
        tasks_per_proc[-1] = N_trajectories - N_traj_proc*(size - 1)
#     tasks_per_proc = comm.bcast(tasks_per_proc, root = 0)
else:
    N_trajectories = None
    Tintegration = None
    Ntimesteps = None
    tasks_per_proc = None
    
tasks_per_proc = comm.bcast(tasks_per_proc, root = 0)    
my_N_Trajectories = tasks_per_proc[rank]    
N_trajectories = comm.bcast(N_trajectories, root = 0)
Tintegration = comm.bcast(Tintegration, root = 0)
Ntimesteps = comm.bcast(Ntimesteps, root = 0)
print("tasks_per_proc",rank,tasks_per_proc)
if rank == 0:
    my_x0 = x0_all[:tasks_per_proc[0],:]
    if size > 1:
        for ip in range(1,size):
            nori = np.sum(tasks_per_proc[:ip])
            nend = nori + tasks_per_proc[ip]
            comm.send(x0_all[nori:nend,:],dest=ip,tag=ip)
            print("send",rank, ip, nori,nend)
else:
    my_x0 = comm.recv(source=0,tag=rank)
    

print(rank,my_x0[:,0],len(my_x0[:,0]))
sys.stdout.write(
    "my rank %d, number of tasks %d \n" 
    % (rank, my_N_Trajectories))
# print(rank,N_trajectories,Tintegration,Ntimesteps)

sigma=10.; beta=8./3; rho=28.0
def lorenz_deriv(x,t0):
    global sigma, beta, rho
    """Compute the time-derivative of a Lorentz system."""
    return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
t = np.linspace(0, Tintegration, Ntimesteps)
x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)
                  for x0i in my_x0])

if rank == 0:
    xt_all = np.zeros((N_trajectories,Ntimesteps,3))
    xt_all[:tasks_per_proc[0],:,:] = np.copy(x_t)
    if size > 1:
        for ip in range(1,size):
            nori = np.sum(tasks_per_proc[:ip])
            nend = np.sum(tasks_per_proc[:ip+1])
            xt_all[nori:nend,:,:] = comm.recv(source = ip, tag = ip)
    fname = 'output.h5'
    tls.write_xt_file(fname,xt_all,t,N_trajectories,Tintegration,Ntimesteps)
else:
    comm.send(x_t, dest = 0, tag = rank)
