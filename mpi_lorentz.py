"""
Parallel second derivative
"""

from mpi4py import MPI
import sys
import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames

sigma=10.; beta=8./3; rho=28.0
def lorenz_deriv(x,t0):
    global sigma, beta, rho
    """Compute the time-derivative of a Lorentz system."""
    return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]

def plot_attractor(x_t):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    for itraj in range(x_t.shape[0]):


        ax.plot3D(x_t[itraj,:,0], x_t[itraj,:,1], x_t[itraj,:,2])
    plt.savefig("proc_"+str(rank).zfill(3)+".png")
    plt.close(fig)
    
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

N_trajectories = 8
N_trajs_local = int(N_trajectories/size)
np.random.seed(rank)
x0 = -15 + 30 * np.random.random((N_trajs_local, 3))

# Solve for the trajectories
t = np.linspace(0, 4, 1000)
x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)
                  for x0i in x0])
plot_attractor(x_t)
