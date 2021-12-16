"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly = (2*np.pi, 2*np.pi)
Reynolds = 4.
Schmidt = 100.
n = 1
Lmax = 100.
Wi = 10
beta = 1.0
snapshots_directory = 'Re4L100Wi10b090Sc100'
ifreq = 100
polymers = False
dtmax = 1.0e-3
tmax = 0.25
# Create bases and domain
x_basis = de.Fourier('x', 1024, interval=(-Lx/2., Lx/2.), dealias=3/2)
y_basis = de.Fourier('y', 1024, interval=(-Ly/2., Ly/2.), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
if polymers:
    problem = de.IVP(domain, variables=['p','u','v','C11','C22','C12'])
    problem.parameters['uReinv'] = beta/Reynolds # (Rayleigh / Prandtl)**(-1/2)
    problem.parameters['pReinv'] = (1-beta)/Reynolds
else:
    problem = de.IVP(domain, variables=['p','u','v'])
    problem.parameters['Reinv'] = 1./Reynolds
problem.parameters['ReScinv'] = 1./(Reynolds*Schmidt) #(Rayleigh * Prandtl)**(-1/2)

problem.parameters['n'] = n
problem.substitutions['F'] = 'sin(n*y)'
if polymers:
    problem.parameters['L2m1'] = 1./Lmax**2
    problem.parameters['W'] = 1/Wi
    problem.substitutions['Ckk'] = 'C11+C22'
    problem.substitutions['fpt'] = '1/(1-Ckk*L2m1)'
    problem.substitutions['T11'] = 'W*(C11*fpt -1)'
    problem.substitutions['T22'] = 'W*(C22*fpt -1)'
    problem.substitutions['T12'] = 'W*(C12*fpt)'
    problem.substitutions['rhs11'] = '2*(C11*dx(u) + C12*dy(u)) - T11 + L2m1/C11**2'
    problem.substitutions['rhs22'] = '2*(C12*dx(v) + C22*dy(v)) - T22 + L2m1/C22**2'
    problem.substitutions['rhs12'] = 'C11*dx(v) + C22*dy(u) - T12'
problem.add_equation("dx(u) + dy(v) = 0", condition=('nx!=0 or ny!=0'))
problem.add_equation('p = 0', condition=('nx==0 and ny==0'))
# problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       = -(u*dx(b) + w*bz)")
if polymers:
    problem.add_equation("dt(u) - uReinv*(dx(dx(u)) + dy(dy(u))) + dx(p) = -(u*dx(u) + v*dy(u)) + F + pReinv*(dx(T11) + dy(T12))")
    problem.add_equation("dt(v) - uReinv*(dx(dx(v)) + dy(dy(v))) + dy(p) = -(u*dx(v) + v*dy(v)) + pReinv*(dx(T12) + dy(T22))")
    problem.add_equation("dt(C11) - ReScinv*(dx(dx(C11)) + dy(dy(C11))) = -(u*dx(C11) + v*dy(C11)) + rhs11 ")
    problem.add_equation("dt(C22) - ReScinv*(dx(dx(C22)) + dy(dy(C22))) = -(u*dx(C22) + v*dy(C22)) + rhs22 ")
    problem.add_equation("dt(C12) - ReScinv*(dx(dx(C12)) + dy(dy(C12))) = -(u*dx(C12) + v*dy(C12)) + rhs12 ")
else:
    problem.add_equation("dt(u) - Reinv*(dx(dx(u)) + dy(dy(u))) + dx(p) = -(u*dx(u) + v*dy(u)) + F")
    problem.add_equation("dt(v) - Reinv*(dx(dx(v)) + dy(dy(v))) + dy(p) = -(u*dx(v) + v*dy(v))")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    x, y = domain.all_grids()
    u = solver.state['u']
    v = solver.state['v']
    if polymers:
        C11 = solver.state['C11']
        C22 = solver.state['C22']
        C12 = solver.state['C12']


    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + perturbations damped at walls

    pert =  1e-2 * noise
    u['g'] = np.sin(1*y)+0.1*np.cos(4*x)*np.sin(4*y) + pert
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]
    pert =  1e-2 * noise
    v['g'] = -0.1*np.sin(4*x)*np.cos(4*y)+pert
    if polymers:
        C11['g'] = 1.
        C22['g'] = 1.
        C12['g'] = 0.

    # Timestepping and output
    dt = dtmax
    stop_sim_time = tmax
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = tmax
    fh_mode = 'overwrite'

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshots_directory, sim_dt=0.25, max_writes=50, mode=fh_mode)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=dtmax, threshold=0.05)
CFL.add_velocities(('u', 'v'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
if polymers:
    flow.add_property("sqrt(u*u + v*v) / uReinv", name='Re')
else:
    flow.add_property("sqrt(u*u + v*v) / Reinv", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    pit_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % ifreq == 0:
            pitE_time = time.time()
            tpit = (pitE_time - pit_time)/ifreq
            pit_time = time.time()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
            logger.info('cpu time/it = %e' % tpit)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
