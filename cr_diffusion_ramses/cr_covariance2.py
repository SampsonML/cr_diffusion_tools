# ----
# Handy tools for CR diffusion science
# Matt Sampson
# ----

import os
import numpy as np
import matplotlib.pyplot as plt
import ramses_matt_io as ram
import yt

def cr_spatial_covariance(path, num, trial_name="tmp", group=1, normalize=True, sort=False):
    """
    Calculates the eigenvalues of the spatial covariance matrix
    for a given field from RAMSES data outputs
    
    Inputs
    ------
    filename: string -- path to the data
    
    Returns
    -------
    Ecr: 3 by 3 numpy array   -- the spatial covariance matrix of the CR energy density
    eigs: numpy array         -- the eigenvalues
    evecs: tuple              -- the eigenvectors
    t: float                  -- the time which calculation is made
    """
    
    # for now use yt to get time
    if num < 10:
        prefix = path + "/output_0000"
        suffix = str(num) + "/info_0000" + str(num) + ".txt"
    elif num < 100:
        prefix = path + "/output_000"
        suffix = str(num) + "/info_000" + str(num) + ".txt"
    else:
        prefix = path + "/output_00"
        suffix = str(num) + "/info_00" + str(num) + ".txt"
    filename = prefix + suffix
    ds = yt.load(filename)
    t = np.array( ds.current_time.in_units('s') )
    
    
    # create and save the data if needed otherwise directly load it in
    data_name = trial_name + "/snapshot_" +  str(num)
    if not os.path.exists(trial_name):
        os.makedirs(trial_name)
        # read in the data
        c=ram.rd_cell(path, num)
        ram.save_cell(c,data_name)
    elif os.path.isfile(data_name):
        # already there load it directly
        c=ram.load_cell(data_name)
    else:
        # directory there but snapshot not
        c=ram.rd_cell(path, num)
        ram.save_cell(c,data_name)

    
    # get the relevent parameters
    x = (c.x[0] - 0.5) * 20
    y = (c.x[1] - 0.5) * 20
    z = (c.x[2] - 0.5) * 20
    dx = (c.dx[0]) * 20
    group_idx = group + 18
    eps = 1e-13 # protect against singular matrices
    e = c.u[group_idx] + eps

    # in cgs 
    pc_to_cm = 3.086e18
    x = x * pc_to_cm
    y = y * pc_to_cm
    z = z * pc_to_cm
    
    # calculate the spatial covariance matrix
    Ecr = np.zeros((3, 3), dtype=float)

    for idx, rho_cr in enumerate(e):

        # column 1
        Ecr[0,0] += rho_cr * ((x[idx]) * (x[idx])) * dx**3
        Ecr[0,1] += rho_cr * ((x[idx]) * (y[idx])) * dx**3
        Ecr[0,2] += rho_cr * ((x[idx]) * (z[idx])) * dx**3 

        # column 2
        Ecr[1,1] += rho_cr * ((y[idx]) * (y[idx])) * dx**3
        Ecr[1,2] += rho_cr * ((y[idx]) * (z[idx])) * dx**3

        # column 3
        Ecr[2,2] += rho_cr * ((z[idx]) * (z[idx])) * dx**3

    # symmetric properties
    Ecr[1,0] = Ecr[0,1] 
    Ecr[2,0] = Ecr[0,2] 
    Ecr[2,1] = Ecr[1,2] 

    # Divide by total energy
    Ecr  /= np.sum(e * dx**3) 

    # calculate the eigenvalues
    #Ecr = Ecr / len(e) # divide by n_grid # TODO: remove this
    eigs, evecs = np.linalg.eig(Ecr)
    if sort: eigs = np.sort(eigs)[::-1] # sort in descending order for consistency
    
    return Ecr, eigs, evecs, t