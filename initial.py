import numpy as np

def u_init(u,u0,nx,ny):
 for i in np.arange(0,nx+2):
     for j in np.arange(0,ny+2):
         u[i,j]=u0
 return u        
    
def h_init(h,hs,eta,hs0,nx,ny):
 for i in np.arange(0,nx+2):
     for j in np.arange(0,ny+2):
         hs[i,j]=hs0; h[i,j]=eta[i,j]+hs[i,j]
 return h,hs

def eta_init(eta,nx,ny,slope,dx,chl):
    eta0=chl*slope
    for i in np.arange(0,nx+2):
        eta_c=eta0-dx*float(i)*slope
        for j in np.arange(0,ny+2):
            eta[i,j]=eta_c
    return eta

def xe_center(x_center,eta_center,nx,slope,dx,chl):
    eta0=chl*slope
    for i in np.arange(0,nx+2):
        x_center[i]=dx*float(i)
        eta_center[i]=eta0-x_center[i]*slope
    return x_center,eta_center

def ep_init(ep,ep_x,nx,ny,snu_0):
    for i in np.arange(0,nx+2):
        for j in np.arange(0,ny+2):
            ep[i,j]=snu_0; ep_x[i,j]=snu_0
    return ep,ep_x
        
def diffs_init(gux,guy,gvx,gvy,u,v,nx,ny,dx,dy):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            gux[i,j]=(u[i+1,j]-u[i-1,j])/(2*dx)
            guy[i,j]=(u[i,j+1]-u[i,j-1])/(2*dy)
            gvx[i,j]=(v[i+1,j]-v[i-1,j])/(2*dx)
            gvy[i,j]=(v[i,j+1]-v[i,j-1])/(2*dy)
    return gux,guy,gvx,gvy




