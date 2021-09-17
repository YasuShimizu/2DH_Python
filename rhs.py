import numpy as np
from numba import jit
@jit
def un_cal(un,u,nx,ny,dx,cfx,hn,g,dt):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            dhdx=(hn[i+1,j]-hn[i,j])/dx
            un[i,j]=u[i,j]+(cfx[i,j]-g*dhdx)*dt
    return un
@jit
def vn_cal(vn,v,nx,ny,dy,cfy,hn,g,dt):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            dhdy=(hn[i,j+1]-hn[i,j])/dy
            vn[i,j]=v[i,j]+(cfy[i,j]-g*dhdy)*dt
    return vn   
@jit
def qu_cal(qu,qc,un,nx,ny,dy,hs_up):
    for i in np.arange(0,nx+1):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            qu[i,j]=un[i,j]*dy*hs_up[i,j]
            qc[i]=qc[i]+qu[i,j]
    return qu,qc
@jit
def qv_cal(qv,vn,nx,ny,dx,hs_vp):
    for i in np.arange(0,nx+1):
       for j in np.arange(1,ny+1):
           qv[i,j]=vn[i,j]*dx*hs_vp[i,j]
    return qv
    