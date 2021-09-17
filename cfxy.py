import numpy as np

def cfxc(cfx,nx,ny,hs,un,g,snm,v_up,hs_up):
    for i in np.arange(0,nx+1):
       for j in np.arange(1,ny+1):
           cfx[i,j]=-g*snm**2*un[i,j]*np.sqrt(un[i,j]**2+v_up[i,j]**2)/hs_up[i,j]**(4./3.)
    return cfx

def cfyc(cfy,nx,ny,hs,vn,g,snm,u_vp,hs_vp):
    for i in np.arange(1,nx+1):
       for j in np.arange(1,ny):
           cfy[i,j]=-g*snm**2*vn[i,j]*np.sqrt(vn[i,j]**2+u_vp[i,j]**2)/hs_vp[i,j]**(4./3.)
    return cfy


