import numpy as np
from numba import jit
@jit
def ng_u(gux,guy,u,un,nx,ny,dx,dy):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            gux[i,j]=gux[i,j]+(un[i+1,j]-un[i-1,j]-u[i+1,j]+u[i-1,j])*0.5/dx
            guy[i,j]=guy[i,j]+(un[i,j+1]-un[i,j-1]-u[i,j+1]+u[i,j-1])*0.5/dy
    return gux,guy
@jit
def ng_v(gvx,gvy,v,vn,nx,ny,dx,dy):
    for i in np.arange(1,nx+1):
        for j in np.arange(2,ny):
            gvx[i,j]=gvx[i,j]+(vn[i+1,j]-vn[i-1,j]-v[i+1,j]+v[i-1,j])*0.5/dx
            gvy[i,j]=gvy[i,j]+(vn[i,j+1]-vn[i,j-1]-v[i,j+1]+v[i,j-1])*0.5/dy
    return gvx,gvy



