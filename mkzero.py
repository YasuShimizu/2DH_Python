import numpy as np
def z0(fn,gxn,gyn,nx,ny):

    fn = np.zeros([nx+2, ny+2])
    gxn = np.zeros([nx+2, ny+2])
    gyn = np.zeros([nx+2, ny+2])
 
    return fn, gxn, gyn
