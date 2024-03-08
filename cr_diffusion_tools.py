# ----
# Handy tools for CR diffusion science
# Matt Sampson
# ----

import numpy as np
import matplotlib.pyplot as plt
import yt

def cr_spatial_covariance(filename, group=1, normalize=True):
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

    # load the file in 
    group_name = "hydro_CRegy_0" + str(group)
    ds = yt.load(filename)
    t = np.array( ds.current_time.in_units('s') )
    
    # grab the cr energy density and x,y,z values
    data = ds.all_data()
    cr_data = np.array( data[("ramses", group_name)] ) 
    x_values = np.array( data[("ramses","x")] ) 
    y_values = np.array( data[("ramses","y")] ) 
    z_values = np.array( data[("ramses","z")] ) 
    
    # Cell volume
    dx = np.array( data[("ramses","dx")] )
    dy = np.array( data[("ramses","dx")] )
    dz = np.array( data[("ramses","dx")] )
    
    # ensure uniform domain
    assert dx[0] == dy[0] == dz[0], "Oh no! Mesh not uniform"
    assert np.min(dx) == np.max(dx), "Oh no! Mesh not uniform"
    
    # convert domain from yt's [0->L] to [-L/2 -> L/2]
    code_len =  np.array(ds.domain_width)
    box_len = np.array(ds.length_unit) * code_len
    x_mid = box_len[0] / 2
    y_mid = box_len[1] / 2
    z_mid = box_len[2] / 2
    x_values = x_values - x_mid
    y_values = y_values - y_mid
    z_values = z_values - z_mid

    # calculate the spatial covariance matrix
    Ecr = np.zeros((3, 3), dtype=float)

    for idx, rho_cr in enumerate(cr_data):

        # column 1
        Ecr[0,0] += rho_cr * ((x_values[idx]) * (x_values[idx])) 
        Ecr[0,1] += rho_cr * ((x_values[idx]) * (y_values[idx])) 
        Ecr[0,2] += rho_cr * ((x_values[idx]) * (z_values[idx])) 

        # column 2
        Ecr[1,1] += rho_cr * ((y_values[idx]) * (y_values[idx])) 
        Ecr[1,2] += rho_cr * ((y_values[idx]) * (z_values[idx])) 

        # column 3
        Ecr[2,2] += rho_cr * ((z_values[idx]) * (z_values[idx])) 

    # symmetric properties
    Ecr[1,0] = I_Ecr[0,1] 
    Ecr[2,0] = I_Ecr[0,2] 
    Ecr[2,1] = I_Ecr[1,2] 
    
    # Divide by total energy
    if normalize == True:
        E = cr_data * (dx[0] * 3.24078e-19)**3 # divide out the total CR energy
        Ecr /= np.sum(E)

    # calculate the eigenvalues
    eigs, evecs = np.linalg.eig(Ecr)
    
    return Ecr, eigs, evecs, t
