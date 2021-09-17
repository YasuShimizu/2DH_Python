import numpy as np

def v_up_c(v_up,v,nx,ny):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            v_up[i,j]=(v[i,j]+v[i,j-1]+v[i+1,j]+v[i+1,j-1])*.25
    return v_up

def hs_up_c(hs_up,hs,nx,ny):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            hs_up[i,j]=(hs[i,j]+hs[i+1,j])*.5
    return hs_up

def u_vp_c(u_vp,u,nx,ny):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            u_vp[i,j]=(u[i,j]+u[i-1,j]+u[i,j+1]+u[i-1,j+1])*.25
        u_vp[i,0]=(u[i,1]+u[i-1,1])*.5
        u_vp[i,ny]=(u[i,ny]+u[i-1,ny])*.5
    return u_vp

def hs_vp_c(hs_vp,hs,nx,ny):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            hs_vp[i,j]=(hs[i,j]+hs[i,j+1])*.5
        hs_vp[i,0]=hs[i,1]
        hs_vp[i,ny]=hs[i,ny]
    return hs_vp
