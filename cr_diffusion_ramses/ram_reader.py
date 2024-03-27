# ----
# Handy tools for CR diffusion science
# Matt Sampson
# ----

import os
import ramses_matt_io as ram

def ram_reader(path, num, trial_name="tmp", normalize=True, sort=False):
    """
    reader
    
    Inputs
    ------
    filename: string -- path to the data
    
    Returns
    -------
  
    """

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
        
    return c