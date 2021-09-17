import numpy as np

def h_bound(h,hs,eta,nx,ny,j_west,j_east,j_hdown,h_down):
    if j_west==1: #Upstream Boundary
        h[0]=h[1];hs[0]=h[0]-eta[0] #Upstream Wall
    if j_east==1: #Downstream Boundary
        h[nx+1]=h[nx]; hs[nx+1]=h[nx+1]-eta[nx+1] #Downstream Wall
    else: #Downstream Free
        if j_hdown>=1: #Stage Given
            h[nx+1]=h_down; hs[nx+1]=h[nx+1]-eta[nx+1]
        else: #Free Downstream
            hs[nx+1]=hs[nx];h[nx+1]=hs[nx+1]+eta[nx+1] 
    return h,hs


def u_bound(u,nx,ny,j_west,j_east,ijh,u0):
    if j_west==1: #Upstream Velocity boundary
        u[0]=0. #Upstream Wall
    else:
        u[0]=u0
    if j_east==1: #Downstream 
        u[nx]=0.

    u[:,0]=u[:,1]  #Side Boundaries
    u[:,ny+1]=u[:,ny]       

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            if ijh[i,j]>0 or ijh[i+1,j]>0:
                u[i,j]=0.
    return u

def v_bound(v,nx,ny,ijh):
    v[:,0]=0,; v[:,ny]=0
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            if ijh[i,j]>0 or ijh[i,j+1]>0:
                v[i,j]=0.
    return v

def gbound_u(gux,guy,ijh,nx,ny):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            if ijh[i,j]>0 or ijh[i+1,j]>0:
                gux[i,j]=0.; guy[i,j]=0.
    return gux,guy

def gbound_v(gvx,gvy,ijh,nx,ny):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            if ijh[i,j]>0 or ijh[i,j+1]>0:
                gvx[i,j]=0.; gvy[i,j]=0.
    return gvx,gvy
