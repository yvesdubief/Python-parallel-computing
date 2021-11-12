
import h5py
import numpy as np
from scipy import integrate
import time

def parser(inputfiledata,substring,datatype):
    for i in range(len(inputfiledata)):
        if inputfiledata[i].startswith(substring):
            if datatype == 'INT':
                data = np.int(inputfiledata[i][len(substring):-1])
            elif datatype == 'FLOAT':
                data = np.float(inputfiledata[i][len(substring):-1])
            elif datatype == "STR":
                data = inputfiledata[i][len(substring):-1].strip()
            else:
                print("parser function recognizes only 'INT and 'FLOAT' types")
    return data

def read_xt_file(fname):
    file = h5py.File(fname,"r+")
    x_t = file['x_t'][:]
    t = file['t'][:]
    N_trajectories = file['x_t'].attrs['Number of trajectories']
    Tintegration = file['x_t'].attrs['Integration time']
    Ntimesteps = file['x_t'].attrs['Number of time steps'] 
    return x_t,t,N_trajectories,Tintegration,Ntimesteps
def write_snapshot_file(fname,X,Y,p):
    file = h5py.File(fname,"w")
    Xset = file.create_dataset('X',data = X)
    Yset = file.create_dataset('Y',data = Y)
    pset = file.create_dataset('var',data = p)
    Xset.attrs['Lx'] = Lx
    file.close()
    return
